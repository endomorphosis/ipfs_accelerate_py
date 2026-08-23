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

import hashlib
import json
import logging
import os
import re
import shlex
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

try:  # pragma: no cover - exercised by fail-closed platform checks
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    _fcntl = None

# Probe the real libc wrappers once. Tests may replace os.unlink later
# without changing whether this platform can unlink through a dir-fd.
_DIR_FD_OPEN = os.open in getattr(os, "supports_dir_fd", ())
_DIR_FD_STAT = os.stat in getattr(os, "supports_dir_fd", ())
_DIR_FD_UNLINK = os.unlink in getattr(os, "supports_dir_fd", ())

from ..merge.protected_recovery_fence import (
    FENCE_CONTENTION_BACKOFF_SECONDS,
    is_protected_recovery_fence_contention,
)
from ..runtime.event_log import append_jsonl_event, utc_now
from ..validation.validation_commands import validation_command_repository_root

_LOG = logging.getLogger(__name__)

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-consumed-no-progress@1"
)
DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-validation-retry@1"
)
DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-validation-retry-seed@1"
)
DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/portal-retry-deferral@1"
)
DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-protected-path-recovery@1"
)
DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_INTENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-protected-path-recovery-intent@1"
)
DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_GUARD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-protected-path-recovery-guard@1"
)
DATABASE_PORTAL_EXTERNAL_PROTECTED_CHECKOUT_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-external-protected-checkout-recovery@1"
)
DATABASE_PORTAL_INFLIGHT_PROCESS_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-inflight-process-recovery@1"
)
DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-validation-retry-seed-conflict-recovery@1"
)
DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON: Final[str] = (
    "Portal retry seed state conflicts with its source receipt"
)
DATABASE_PORTAL_POOLED_WORKTREE_CREATE_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-pooled-worktree-create-recovery@1"
)
DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON: Final[str] = (
    "pooled_worktree_create_failed"
)
DATABASE_PORTAL_POOLED_WORKTREE_CREATE_SOURCE_REASON: Final[str] = (
    "portal_provider_failed"
)
_PROTECTED_PATH_RECOVERY_INTENT_FILENAME: Final[str] = (
    "database-portal-protected-path-recovery-intent.json"
)
_PROTECTED_PATH_RECOVERY_FILENAME: Final[str] = (
    "database-portal-protected-path-recovery.json"
)
_EXTERNAL_PROTECTED_CHECKOUT_RECOVERY_FILENAME: Final[str] = (
    "database-portal-external-protected-checkout-recovery.json"
)
_INFLIGHT_PROCESS_RECOVERY_FILENAME: Final[str] = (
    "database-portal-inflight-process-recovery.json"
)
_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_FILENAME: Final[str] = (
    "database-portal-validation-retry-seed-conflict-recovery.json"
)
_POOLED_WORKTREE_CREATE_RECOVERY_FILENAME: Final[str] = (
    "database-portal-pooled-worktree-create-recovery.json"
)
_PAIRED_SUPERVISOR_PROTECTED_RECOVERY_OWNER: Final[str] = (
    "implementation_supervisor"
)
_EXTERNAL_PROTECTED_CHECKOUT_DEFERRAL_BACKOFF_SECONDS: Final[int] = 20
_INFLIGHT_PROCESS_DEFERRAL_BACKOFF_SECONDS: Final[int] = 20
_POOLED_WORKTREE_CREATE_DEFERRAL_BACKOFF_SECONDS: Final[int] = 30
_POOLED_WORKTREE_CREATE_FAILURE_PREFIX: Final[str] = (
    "failed to create pooled worktree"
)
_IMPLEMENTATION_PROTECTED_ACTIVE_FILENAME: Final[str] = (
    "implementation-protected-path-active.json"
)
_IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME: Final[str] = (
    "implementation-protected-path-incident.json"
)
_MAX_PROTECTED_PATH_RECOVERY_PATHS: Final[int] = 256
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
PROTECTED_CHECKOUT_SETUP_BLOCK_REASONS: Final[frozenset[str]] = frozenset(
    {
        "external_protected_checkout_recovery_required",
        "protected_recovery_owner_active",
        "protected_recovery_adoption_raced",
        "protected_checkout_recovery_required",
        "protected_checkout_recovery_failed",
        "supervisor_protected_recovery_owner_active",
        "supervisor_protected_recovery_adoption_raced",
        "supervisor_protected_recovery_journal_invalid",
        "checkout_mutation_protected_recovery_required",
    }
)


def is_protected_checkout_setup_block(reason: str) -> bool:
    """True when Portal/provider dispatch is blocked before any callback."""

    normalized = str(reason or "").strip().replace(" ", "_")
    if not normalized:
        return False
    return any(token in normalized for token in PROTECTED_CHECKOUT_SETUP_BLOCK_REASONS)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"completed", "complete", "done"})
_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_OPERATIONAL_PROJECTION_LINE = re.compile(
    r"(?mi)^-\s*completion\s+receipt\s*:\s*.*$"
)
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_DIAGNOSTIC_RECEIPT_BYTES: Final[int] = 256 * 1024
_MAX_CONTEXT_RECEIPT_BYTES: Final[int] = 256 * 1024
_MAX_FAILURE_LOG_BYTES: Final[int] = 128 * 1024
_MAX_CONSUMED_FAILURE_EVIDENCE_BYTES: Final[int] = 24 * 1024
_TASK_CONTRACT_MUTABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {"completion_receipt", "status"}
)
_ROOT_REPOSITORY_AUTHORITY: Final[str] = "ipfs_accelerate_py"
_MAX_REPOSITORY_PATH_BYTES: Final[int] = 1024
_MAX_TASK_IDENTITY_BYTES: Final[int] = 4096
_MAX_DATABASE_PORTAL_BACKOFF_SECONDS: Final[int] = 86_400
INFLIGHT_PROCESS_BACKOFF_SECONDS: Final[int] = 30
_INFLIGHT_PROCESS_SKIP_REASON: Final[str] = "inflight_process"
_MAX_DATABASE_PORTAL_TASK_ATTEMPTS: Final[int] = 10_000
_MAX_DATABASE_PORTAL_EVENT_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_DATABASE_PORTAL_EVENTS: Final[int] = 4096
# Closed post-dispatch reasons that consumed a provider attempt but produced
# no mergeable candidate.  These must retry while budget remains instead of
# being collapsed into untyped ``portal_provider_failed``.
DATABASE_PORTAL_CANDIDATE_RETRY_REASONS: Final[frozenset[str]] = frozenset(
    {
        "proposal_gate_failed",
        "proposal_validation_failed",
        "no_change_completion_not_allowed",
        "incomplete_expected_outputs",
        "expected_output_ignored_or_unstaged",
        "empty_or_no_change",
        "empty_patch_reserved_for_no_change_gate",
        "no_changes",
    }
)
# A sibling supervisor or daemon holds the shared checkout-mutation lock.
# Markdown Portal treats that as an unchanged deferral; the database path
# must not consume the claimed task as a terminal Portal failure.
DATABASE_PORTAL_CHECKOUT_CONTENTION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "external_protected_checkout_recovery_required",
        "protected_recovery_owner_active",
        "supervisor_protected_recovery_owner_active",
        "protected_recovery_adoption_raced",
        "checkout_mutation_lock_exists",
    }
)
DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS: Final[int] = (
    FENCE_CONTENTION_BACKOFF_SECONDS
)
_MAX_DATABASE_PORTAL_BINDING_BYTES: Final[int] = 64 * 1024
_MAX_DATABASE_PORTAL_PROJECTION_BYTES: Final[int] = 1024 * 1024
_POST_MERGE_RECOVERY_SCAN_LIMIT: Final[int] = 256
_MERGE_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-candidate@3"
)
_MERGE_TARGET_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
)
_POST_MERGE_DECLARED_OUTPUTS_MISSING_REASON: Final[str] = (
    "post_merge_declared_outputs_missing"
)
_CROSS_BOARD_COMPLETION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "cross_board_manual_completion_authority_metadata_invalid",
        "cross_board_manual_completion_authority_metadata_missing",
        "cross_board_manual_completion_authority_unavailable",
    }
)
_POST_MERGE_DECLARED_OUTPUT_COMPLETION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-declared-output-completion@1"
)
_POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "post-merge-declared-output-repair@1"
)
_POST_MERGE_DECLARED_OUTPUT_REPAIR_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_ids",
        "candidate_commit",
        "candidate_tree",
        "baseline_commit",
        "failed_integration_commit",
        "repair_parent_commit",
        "repair_commit",
        "repair_tree",
        "entries",
        "validation",
        "rollback_target",
        "receipt_id",
    }
)
_POST_MERGE_DECLARED_OUTPUT_REQUALIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "post-merge-declared-output-requalification@1"
)
_POST_MERGE_DECLARED_OUTPUT_REQUALIFICATION_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema",
            "task_ids",
            "candidate_commit",
            "source_repair_receipt_id",
            "source_repair_commit",
            "source_repair_receipt",
            "current_target_commit",
            "current_target_tree",
            "entries",
            "validation",
            "receipt_id",
        }
    )
)
_DATABASE_POST_MERGE_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-post-merge-declared-output-recovery@1"
)
_DATABASE_POST_MERGE_REQUALIFICATION_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-post-merge-declared-output-requalification-recovery@1"
)
_DATABASE_POST_MERGE_RECOVERY_PREAUTHORIZATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-post-merge-declared-output-recovery-preauthorization@1"
)
_POST_MERGE_RECOVERY_CURSOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-declared-output-recovery-cursor@1"
)
_POST_MERGE_RECOVERY_CURSOR_STAGES: Final[tuple[str, ...]] = (
    "completed_requests",
    "pending_requests",
    "quarantined_requests",
    "processing_requests",
)
_MAX_POST_MERGE_RECOVERY_CURSOR_BYTES: Final[int] = 64 * 1024
_POST_MERGE_COMPLETION_STATUSES: Final[frozenset[str]] = frozenset(
    {"merged", "already_merged", "deduplicated", "completed"}
)
_DATABASE_PORTAL_ATTEMPT_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "attempt_id",
        "claim_id",
        "task_cid",
        "canonical_task_key",
        "task_alias",
        "goal_cid",
        "plan_cid",
        "task_revision",
        "fencing_token",
        "fence_epoch",
        "lease_id",
        "task_body_digest",
        "task_contract_digest",
        "repository_tree_id",
        "projection_seed_digest",
        "projection_immutable_digest",
        "authoritative_task_store",
        "projection_authority",
        "binding_id",
    }
)


def _is_implementation_conflict(exc: BaseException) -> bool:
    """Return whether ``exc`` is a database implementation conflict.

    Running ``python -m ...implementation_daemon`` binds daemon classes to
    ``__main__``.  Relative imports of ``DatabaseImplementationConflictError``
    then see a different type, so identity-based ``except`` misses live
    preauthorization conflicts and fails the maintenance tick.
    """

    return type(exc).__name__ == "DatabaseImplementationConflictError"


class DatabasePortalBridgeError(RuntimeError):
    """A database claim could not obtain trustworthy Portal evidence."""


class DatabasePortalBridgeDeferred(DatabasePortalBridgeError):
    """Portal execution made bounded progress but is not yet acceptable."""

    def __init__(self, reason: str, *, backoff_seconds: int = 300) -> None:
        if (
            isinstance(backoff_seconds, bool)
            or not isinstance(backoff_seconds, int)
            or backoff_seconds < 0
            or backoff_seconds > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
        ):
            raise ValueError(
                "backoff_seconds must be an integer in "
                f"[0, {_MAX_DATABASE_PORTAL_BACKOFF_SECONDS}]"
            )
        reason_text = str(reason or "portal_execution_deferred").strip()
        super().__init__(reason_text or "portal_execution_deferred")
        self.reason = reason_text or "portal_execution_deferred"
        self.backoff_seconds = int(backoff_seconds)
        # A typed deferral occurs before the provider is admitted.  These
        # fields deliberately mirror the Portal result contract so the outer
        # database authority need not infer retry semantics from prose.
        self.attempt_consumed = False
        self.provider_dispatched = False


class DatabasePortalValidationRetry(DatabasePortalBridgeError):
    """A dispatched candidate failed only its authoritative validation.

    This is deliberately distinct from a pre-dispatch deferral.  It consumes
    an ordinary provider attempt and carries independently reproducible,
    identity-bound evidence.  Callers must not infer this disposition from a
    provider error string.
    """

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        value = dict(receipt)
        if (
            value.get("schema") != DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA
            or value.get("disposition") != "retry"
            or value.get("attempt_consumed") is not True
            or value.get("provider_dispatched") is not True
            or value.get("reason") != "declared_validation_failed"
        ):
            raise ValueError("validation retry receipt has an invalid disposition")
        backoff_seconds = value.get("backoff_seconds")
        if (
            isinstance(backoff_seconds, bool)
            or not isinstance(backoff_seconds, int)
            or backoff_seconds < 0
            or backoff_seconds > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
        ):
            raise ValueError("validation retry receipt has an invalid backoff")
        super().__init__("declared_validation_failed")
        self.reason = "declared_validation_failed"
        self.backoff_seconds = int(backoff_seconds)
        self.attempt_consumed = True
        self.provider_dispatched = True
        self.retry_receipt = value


class DatabasePortalCandidateRetry(DatabasePortalBridgeError):
    """A dispatched provider attempt produced an unusable candidate.

    Empty diffs, rejected proposals, and incomplete declared outputs consume
    the attempt and must retry from the failure-review addendum while the
    Portal attempt budget remains.  Callers must not infer this from generic
    provider error strings.
    """

    def __init__(self, reason: str, *, backoff_seconds: int = 0) -> None:
        if (
            isinstance(backoff_seconds, bool)
            or not isinstance(backoff_seconds, int)
            or backoff_seconds < 0
            or backoff_seconds > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
        ):
            raise ValueError(
                "backoff_seconds must be an integer in "
                f"[0, {_MAX_DATABASE_PORTAL_BACKOFF_SECONDS}]"
            )
        reason_text = str(reason or "").strip()
        if reason_text not in DATABASE_PORTAL_CANDIDATE_RETRY_REASONS:
            raise ValueError("candidate retry reason is not a closed retry code")
        super().__init__(reason_text)
        self.reason = reason_text
        self.backoff_seconds = int(backoff_seconds)
        self.attempt_consumed = True
        self.provider_dispatched = True


class DatabasePortalBridgeConsumedNoProgressError(DatabasePortalBridgeError):
    """One Portal attempt was consumed without an implementation candidate.

    The provider-effect state is deliberately unknown.  This exception seals
    only the durable no-progress outcome; it does not classify provider text
    or claim that a model call did or did not occur.
    """

    def __init__(
        self,
        message: str,
        *,
        failure_evidence: Mapping[str, Any],
    ) -> None:
        evidence = dict(failure_evidence)
        allowed = {
            "schema",
            "failure_kind",
            "failure_fingerprint",
            "diagnostic_failure_id",
            "diagnostic_receipt_id",
            "diagnostic_receipt_digest",
            "diagnostic_receipt_size",
            "context_receipt_id",
            "context_receipt_digest",
            "context_receipt_size",
            "log_digest",
            "log_size",
            "repository_id",
            "tree_id",
            "control_repository_tree_id",
            "task_cid",
            "task_contract_digest",
            "database_binding_id",
            "database_attempt_id",
            "database_claim_id",
            "database_lease_id",
            "database_fencing_token",
            "database_fence_epoch",
            "portal_task_id",
            "portal_attempt_number",
            "returncode",
            "attempt_consumed",
            "portal_provider_dispatched",
            "provider_effect_state",
            "implementation_commit_present",
            "implementation_candidate_present",
            "validation_state",
        }
        text_fields = (
            "failure_fingerprint",
            "diagnostic_failure_id",
            "diagnostic_receipt_id",
            "diagnostic_receipt_digest",
            "context_receipt_id",
            "context_receipt_digest",
            "log_digest",
            "repository_id",
            "tree_id",
            "control_repository_tree_id",
            "task_cid",
            "task_contract_digest",
            "database_binding_id",
            "database_attempt_id",
            "database_claim_id",
            "database_lease_id",
            "portal_task_id",
            "provider_effect_state",
            "validation_state",
        )
        if (
            set(evidence) != allowed
            or evidence.get("schema")
            != DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA
            or evidence.get("failure_kind") != "consumed_no_progress"
            or evidence.get("provider_effect_state")
            != "unknown_may_have_started"
            or evidence.get("attempt_consumed") is not True
            or type(evidence.get("portal_provider_dispatched")) is not bool
            or evidence.get("implementation_commit_present") is not False
            or evidence.get("implementation_candidate_present") is not False
            or evidence.get("validation_state") != "not_run"
            or isinstance(evidence.get("portal_attempt_number"), bool)
            or not isinstance(evidence.get("portal_attempt_number"), int)
            or int(evidence.get("portal_attempt_number") or 0) < 1
            or type(evidence.get("returncode")) is not int
            or evidence.get("returncode") == 0
            or not -(2**31) <= evidence.get("returncode") < 2**31
            or type(evidence.get("database_fencing_token")) is not int
            or evidence.get("database_fencing_token") < 1
            or type(evidence.get("database_fence_epoch")) is not int
            or evidence.get("database_fence_epoch") < 1
            or type(evidence.get("diagnostic_receipt_size")) is not int
            or not 1
            <= evidence.get("diagnostic_receipt_size")
            <= _MAX_DIAGNOSTIC_RECEIPT_BYTES
            or type(evidence.get("context_receipt_size")) is not int
            or not 1
            <= evidence.get("context_receipt_size")
            <= _MAX_CONTEXT_RECEIPT_BYTES
            or type(evidence.get("log_size")) is not int
            or not 0 <= evidence.get("log_size") <= _MAX_FAILURE_LOG_BYTES
            or any(
                not isinstance(evidence.get(key), str)
                or not str(evidence.get(key) or "")
                or len(str(evidence.get(key)).encode("utf-8")) > 1024
                or "\x00" in str(evidence.get(key))
                or "\n" in str(evidence.get(key))
                or "\r" in str(evidence.get(key))
                for key in text_fields
            )
            or not _SHA256_ID.fullmatch(
                str(evidence.get("failure_fingerprint") or "")
            )
            or not _SHA256_ID.fullmatch(
                str(evidence.get("diagnostic_receipt_digest") or "")
            )
            or not _SHA256_ID.fullmatch(
                str(evidence.get("context_receipt_digest") or "")
            )
            or not _SHA256_ID.fullmatch(str(evidence.get("log_digest") or ""))
            or not _SHA256_ID.fullmatch(
                str(evidence.get("task_contract_digest") or "")
            )
            or not _SHA256_ID.fullmatch(
                str(evidence.get("database_binding_id") or "")
            )
            or evidence.get("failure_fingerprint")
            != database_portal_consumed_no_progress_fingerprint(evidence)
            or len(_canonical_json(evidence))
            > _MAX_CONSUMED_FAILURE_EVIDENCE_BYTES
        ):
            raise ValueError("Portal consumed-no-progress evidence is invalid")
        super().__init__(message)
        self.failure_evidence = evidence


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


@dataclass(frozen=True)
class _DatabasePortalRecoveryProjection:
    """Verified ownership of one merge request by this database lane."""

    paths: DatabasePortalAttemptPaths
    binding: Mapping[str, Any]
    task_status: str


PortalDaemonFactory = Callable[[DatabasePortalAttemptPaths, str], Any]


class _ProtectedPathRecoveryAttemptCapability:
    """Bind recovery I/O to no-follow descriptors for one attempt directory."""

    def __init__(
        self,
        paths: DatabasePortalAttemptPaths,
        *,
        incident_present: bool,
    ) -> None:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        cloexec = getattr(os, "O_CLOEXEC", 0)
        if (
            _fcntl is None
            or not nofollow
            or not directory_flag
            or not _DIR_FD_OPEN
            or not _DIR_FD_STAT
            or not _DIR_FD_UNLINK
        ):
            raise DatabasePortalBridgeError(
                "protected-path recovery no-follow capability is unavailable"
            )
        self._root_path = paths.root
        self._root_fd = -1
        self._event_fd = -1
        self._event_lock_fd = -1
        self._closed = False
        self._expected: dict[str, tuple[int, int, int, int] | None] = {}
        self._fence_fds: dict[str, int] = {}
        self._fence_digests: dict[str, str] = {}
        self._event_digest = ""
        try:
            self._root_fd = os.open(
                paths.root,
                os.O_RDONLY | directory_flag | nofollow | cloexec,
            )
            root_metadata = os.fstat(self._root_fd)
            if not stat.S_ISDIR(root_metadata.st_mode):
                raise DatabasePortalBridgeError(
                    "protected-path recovery attempt capability is not a directory"
                )
            self._root_identity = (
                int(root_metadata.st_dev),
                int(root_metadata.st_ino),
                int(stat.S_IFMT(root_metadata.st_mode)),
                int(root_metadata.st_nlink),
            )
            self._event_fd = os.open(
                paths.events.name,
                os.O_RDWR | os.O_APPEND | nofollow | cloexec,
                dir_fd=self._root_fd,
            )
            event_metadata = os.fstat(self._event_fd)
            if (
                not stat.S_ISREG(event_metadata.st_mode)
                or event_metadata.st_nlink != 1
            ):
                raise DatabasePortalBridgeError(
                    "protected-path recovery event capability is not a private file"
                )
            self._expected[paths.events.name] = self._identity(event_metadata)
            self._event_lock_fd = os.open(
                f".{paths.events.name}.lock",
                os.O_RDWR | os.O_APPEND | os.O_CREAT | nofollow | cloexec,
                0o600,
                dir_fd=self._root_fd,
            )
            lock_metadata = os.fstat(self._event_lock_fd)
            if (
                not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_nlink != 1
            ):
                raise DatabasePortalBridgeError(
                    "protected-path recovery event lock is not a private file"
                )
            self._event_digest = _sha256_bytes(
                self._read_descriptor(
                    self._event_fd,
                    maximum=_MAX_DATABASE_PORTAL_EVENT_BYTES,
                )
            )
            self._bind_entry(
                _IMPLEMENTATION_PROTECTED_ACTIVE_FILENAME,
                required=True,
            )
            self._bind_entry(
                _IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME,
                required=incident_present,
            )
            if not self.verify():
                raise DatabasePortalBridgeError(
                    "protected-path recovery attempt capability changed during binding"
                )
        except OSError as exc:
            self.close()
            raise DatabasePortalBridgeError(
                "protected-path recovery no-follow capability could not be bound"
            ) from exc
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(stat.S_IFMT(metadata.st_mode)),
            int(metadata.st_nlink),
        )

    def _bind_entry(self, name: str, *, required: bool) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
            os, "O_CLOEXEC", 0
        )
        try:
            descriptor = os.open(name, flags, dir_fd=self._root_fd)
        except FileNotFoundError:
            if required:
                raise DatabasePortalBridgeError(
                    f"protected-path recovery artifact {name!r} disappeared"
                )
            self._expected[name] = None
            return
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise DatabasePortalBridgeError(
                    f"protected-path recovery artifact {name!r} is not private"
                )
            self._expected[name] = self._identity(metadata)
            self._fence_fds[name] = descriptor
            self._fence_digests[name] = _sha256_bytes(
                self._read_descriptor(descriptor, maximum=1024 * 1024)
            )
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _read_descriptor(descriptor: int, *, maximum: int) -> bytes:
        metadata = os.fstat(descriptor)
        if metadata.st_size < 0 or metadata.st_size > maximum:
            raise DatabasePortalBridgeError(
                "protected-path recovery artifact exceeds its read bound"
            )
        payload = os.pread(descriptor, metadata.st_size, 0)
        if len(payload) != metadata.st_size:
            raise DatabasePortalBridgeError(
                "protected-path recovery artifact changed during read"
            )
        return payload

    def verify(self) -> bool:
        """Return whether every bound name still denotes its admitted inode."""

        if self._closed or self._root_fd < 0:
            return False
        try:
            if self._identity(os.fstat(self._root_fd)) != self._root_identity:
                return False
            for name, expected in self._expected.items():
                try:
                    observed = os.stat(
                        name,
                        dir_fd=self._root_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if expected is not None:
                        return False
                    continue
                if expected is None or self._identity(observed) != expected:
                    return False
                descriptor = self._fence_fds.get(name)
                if descriptor is not None and (
                    self._identity(os.fstat(descriptor)) != expected
                    or _sha256_bytes(
                        self._read_descriptor(descriptor, maximum=1024 * 1024)
                    )
                    != self._fence_digests.get(name)
                ):
                    return False
            return True
        except (DatabasePortalBridgeError, OSError):
            return False

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Append one canonical event through the preopened no-follow file."""

        if not self.verify() or self._event_fd < 0 or _fcntl is None:
            raise DatabasePortalBridgeError(
                "protected-path recovery attempt capability is no longer current"
            )
        if self._event_lock_fd < 0:
            raise DatabasePortalBridgeError(
                "protected-path recovery event lock is unavailable"
            )
        _fcntl.flock(self._event_lock_fd, _fcntl.LOCK_EX)
        try:
            if not self.verify():
                raise DatabasePortalBridgeError(
                    "protected-path recovery attempt capability changed before append"
                )
            metadata = os.fstat(self._event_fd)
            if metadata.st_size < 1 or metadata.st_size > _MAX_DATABASE_PORTAL_EVENT_BYTES:
                raise DatabasePortalBridgeError(
                    "protected-path recovery event stream is empty or oversized"
                )
            encoded_stream = os.pread(self._event_fd, metadata.st_size, 0)
            if len(encoded_stream) != metadata.st_size or not encoded_stream.endswith(
                b"\n"
            ):
                raise DatabasePortalBridgeError(
                    "protected-path recovery event stream is not durably framed"
                )
            if _sha256_bytes(encoded_stream) != self._event_digest:
                raise DatabasePortalBridgeError(
                    "protected-path recovery event stream changed after binding"
                )
            try:
                events = [
                    json.loads(line)
                    for line in encoded_stream.splitlines()
                    if line.strip()
                ]
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeError(
                    "protected-path recovery event stream is malformed"
                ) from exc
            if not events or not isinstance(events[-1], Mapping):
                raise DatabasePortalBridgeError(
                    "protected-path recovery event stream has no predecessor"
                )
            expected_previous = ""
            expected_sequence = 1
            expected_stream = ""
            expected_snapshot = ""
            for observed_event in events:
                if not isinstance(observed_event, Mapping):
                    raise DatabasePortalBridgeError(
                        "protected-path recovery event stream contains a non-object"
                    )
                observed_body = dict(observed_event)
                observed_event_id = str(observed_body.pop("event_id", "") or "")
                derived_event_id = _sha256_bytes(
                    json.dumps(
                        observed_body,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                )
                observed_stream = str(observed_event.get("stream_id") or "")
                observed_snapshot = str(observed_event.get("snapshot_id") or "")
                if not expected_stream:
                    expected_stream = observed_stream
                    expected_snapshot = observed_snapshot
                if (
                    observed_event_id != derived_event_id
                    or observed_event.get("sequence") != expected_sequence
                    or str(observed_event.get("previous_event_id") or "")
                    != expected_previous
                    or observed_stream != expected_stream
                    or observed_snapshot != expected_snapshot
                ):
                    raise DatabasePortalBridgeError(
                        "protected-path recovery event chain is invalid"
                    )
                expected_previous = observed_event_id
                expected_sequence += 1
            predecessor = events[-1]
            sequence = predecessor.get("sequence")
            stream_id = str(predecessor.get("stream_id") or "")
            snapshot_id = str(predecessor.get("snapshot_id") or "")
            previous_event_id = str(predecessor.get("event_id") or "")
            if (
                type(sequence) is not int
                or sequence < 1
                or not stream_id
                or not snapshot_id
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", previous_event_id)
            ):
                raise DatabasePortalBridgeError(
                    "protected-path recovery predecessor event is invalid"
                )
            supplied = dict(payload)
            for reserved in (
                "stream_id",
                "snapshot_id",
                "sequence",
                "position",
                "event_id",
                "previous_event_id",
            ):
                supplied.pop(reserved, None)
            timestamp = supplied.pop("timestamp", None) or utc_now()
            supplied.pop("type", None)
            matching = [
                event
                for event in events
                if isinstance(event, Mapping)
                and event.get("type") == event_type
                and all(event.get(key) == value for key, value in supplied.items())
            ]
            if len(matching) > 1:
                raise DatabasePortalBridgeError(
                    "protected-path recovery event is duplicated"
                )
            if matching:
                return dict(matching[0])
            event = {
                "type": str(event_type),
                "timestamp": timestamp,
                **supplied,
                "stream_id": stream_id,
                "snapshot_id": snapshot_id,
                "sequence": sequence + 1,
                "previous_event_id": previous_event_id,
            }
            event["event_id"] = _sha256_bytes(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            line = json.dumps(
                event,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8") + b"\n"
            if metadata.st_size + len(line) > _MAX_DATABASE_PORTAL_EVENT_BYTES:
                raise DatabasePortalBridgeError(
                    "protected-path recovery event stream exceeds its bound"
                )
            view = memoryview(line)
            while view:
                written = os.write(self._event_fd, view)
                if written < 1:
                    raise DatabasePortalBridgeError(
                        "protected-path recovery event append made no progress"
                    )
                view = view[written:]
            os.fsync(self._event_fd)
            self._event_digest = _sha256_bytes(encoded_stream + line)
            return event
        finally:
            _fcntl.flock(self._event_lock_fd, _fcntl.LOCK_UN)

    def clear_fences(self) -> bool:
        """Unlink only the exact bound fence names through the attempt dir-fd."""

        if not self.verify():
            return False
        try:
            for name in (
                _IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME,
                _IMPLEMENTATION_PROTECTED_ACTIVE_FILENAME,
            ):
                expected = self._expected.get(name)
                if expected is None:
                    continue
                observed = os.stat(
                    name,
                    dir_fd=self._root_fd,
                    follow_symlinks=False,
                )
                if self._identity(observed) != expected:
                    return False
            # Validate the complete incident+active population before the
            # first unlink. The enclosing checkout-maintenance lease is the
            # cooperative writer exclusion boundary for these names.
            if not self.verify():
                return False
            for name in (
                _IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME,
                _IMPLEMENTATION_PROTECTED_ACTIVE_FILENAME,
            ):
                expected = self._expected.get(name)
                if expected is None:
                    continue
                os.unlink(name, dir_fd=self._root_fd)
                self._expected[name] = None
                descriptor = self._fence_fds.pop(name, -1)
                self._fence_digests.pop(name, None)
                if descriptor >= 0:
                    os.close(descriptor)
            os.fsync(self._root_fd)
        except OSError:
            return False
        return self.verify()

    def recovery_io(self) -> Mapping[str, Callable[..., Any]]:
        return {
            "verify": self.verify,
            "append_event": self.append_event,
            "clear_fences": self.clear_fences,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        descriptors = [
            *self._fence_fds.values(),
            self._event_lock_fd,
            self._event_fd,
            self._root_fd,
        ]
        self._fence_fds.clear()
        self._fence_digests.clear()
        for descriptor in descriptors:
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
        self._event_fd = -1
        self._event_lock_fd = -1
        self._root_fd = -1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def database_portal_task_contract_digest(record: Any) -> str:
    """Commit to the complete status-independent Portal task contract.

    ``TaskRecord.body`` is only one part of the execution contract.  The
    Portal projection also consumes the task's graph identity, ordering,
    declared outputs, acceptance policy, and validation commands.  Keep the
    mutable lifecycle fields (status, revision, and completion receipts) out
    of this digest so the same contract remains verifiable after quarantine.
    """

    def value(name: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(name, default)
        return getattr(record, name, default)

    raw_body = value("body", {})
    body = dict(raw_body) if isinstance(raw_body, Mapping) else {}
    contract_body = {
        str(key): item
        for key, item in body.items()
        if str(key) not in _TASK_CONTRACT_MUTABLE_FIELDS
    }

    raw_dependencies = value("dependencies", ()) or ()
    dependencies = [str(item) for item in raw_dependencies]

    def mappings(name: str) -> list[dict[str, Any]]:
        raw_items = value(name, ()) or ()
        return [dict(item) for item in raw_items if isinstance(item, Mapping)]

    contract = {
        "task_cid": str(value("task_cid", "") or ""),
        "task_alias": str(value("task_alias", "") or ""),
        "goal_cid": str(value("goal_cid", "") or ""),
        "plan_cid": str(value("plan_cid", "") or ""),
        "objective_id": str(value("objective_id", "") or ""),
        "priority": str(value("priority", "") or ""),
        "ordinal": int(value("ordinal", 0) or 0),
        "dependencies": dependencies,
        "outputs": mappings("outputs"),
        "acceptance": mappings("acceptance"),
        "validations": mappings("validations"),
        "body": contract_body,
    }
    if not contract["task_cid"]:
        raise DatabasePortalBridgeError("task contract has no canonical CID")
    return _sha256_bytes(_canonical_json(contract))


def database_portal_authoritative_repository_tree_id(
    task_source: Any,
    task_cid: str,
) -> str:
    """Resolve the persisted task tree, rejecting a divergent live view.

    ``DatabaseTaskSource.repository_tree_id`` is populated while materializing
    but is not itself persisted by that adapter.  The exact tree is persisted
    in the task identity, so cold-restart validation must prefer that identity
    while still rejecting a conflicting non-empty snapshot value.
    """

    snapshot = task_source.snapshot()
    snapshot_tree = str(
        getattr(snapshot, "repository_tree_id", "")
        or (
            snapshot.get("repository_tree_id", "")
            if isinstance(snapshot, Mapping)
            else ""
        )
    ).strip()
    identity_tree = ""
    intent = getattr(task_source, "intent", None)
    get_task = getattr(intent, "get_task", None)
    if callable(get_task):
        persisted = get_task(str(task_cid))
        identity = (
            persisted.get("identity")
            if isinstance(persisted, Mapping)
            and isinstance(persisted.get("identity"), Mapping)
            else {}
        )
        identity_tree = str(identity.get("repository_tree_id") or "").strip()
    if identity_tree and snapshot_tree and identity_tree != snapshot_tree:
        raise DatabasePortalBridgeError(
            "database task repository tree conflicts with persisted identity"
        )
    repository_tree_id = identity_tree or snapshot_tree
    if not repository_tree_id:
        raise DatabasePortalBridgeError(
            "database task source has no authoritative repository tree"
        )
    return repository_tree_id


def database_portal_consumed_no_progress_fingerprint(
    evidence: Mapping[str, Any],
) -> str:
    """Return the neutral circuit-breaker key for one sealed outcome."""

    material = {
        "schema": DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
        "failure_kind": str(evidence.get("failure_kind") or ""),
        "repository_id": str(evidence.get("repository_id") or ""),
        "tree_id": str(evidence.get("tree_id") or ""),
        "control_repository_tree_id": str(
            evidence.get("control_repository_tree_id") or ""
        ),
        "task_cid": str(evidence.get("task_cid") or ""),
        "task_contract_digest": str(
            evidence.get("task_contract_digest") or ""
        ),
        "diagnostic_failure_id": str(
            evidence.get("diagnostic_failure_id") or ""
        ),
        "diagnostic_receipt_id": str(
            evidence.get("diagnostic_receipt_id") or ""
        ),
        "context_receipt_id": str(evidence.get("context_receipt_id") or ""),
        "log_digest": str(evidence.get("log_digest") or ""),
        "returncode": evidence.get("returncode"),
        "provider_effect_state": str(
            evidence.get("provider_effect_state") or ""
        ),
    }
    return _sha256_bytes(_canonical_json(material))


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DatabasePortalBridgeError(
            f"could not read Portal attempt artifact {path.name!r}"
        ) from exc


def _bounded_file(path: Path, *, limit: int) -> bytes:
    """Read one bounded regular artifact without accepting truncation."""

    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("artifact is not a regular non-symlink file")
        size = path.stat().st_size
        if size > limit:
            raise OSError("artifact exceeds its byte limit")
        with path.open("rb") as handle:
            payload = handle.read(limit + 1)
        if len(payload) != size:
            raise OSError("artifact changed while read")
        return payload
    except OSError as exc:
        raise DatabasePortalBridgeError(
            f"could not read Portal attempt artifact {path.name!r}"
        ) from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_write_once(path: Path, payload: bytes) -> bool:
    """Publish immutable evidence atomically without replacing a first writer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


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


def _mapping_path(value: Mapping[str, Any]) -> str:
    candidates = tuple(
        value[key]
        for key in ("path", "output", "artifact_id", "fluent_id")
        if key in value and value[key] not in (None, "")
    )
    if not candidates:
        raise DatabasePortalBridgeError("task output mapping has no path identity")
    if any(type(candidate) is not str for candidate in candidates):
        raise DatabasePortalBridgeError("task output path identity is not a string")
    if len(set(candidates)) != 1:
        raise DatabasePortalBridgeError("task output mapping has ambiguous path identities")
    return candidates[0]


def _output_values(record: Any, body: Mapping[str, Any]) -> list[str]:
    raw = getattr(record, "outputs", ()) or body.get("outputs") or ()
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    selected: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            value = _mapping_path(item)
        elif type(item) is str:
            value = item
        else:
            raise DatabasePortalBridgeError("task output path identity is not a string")
        if value and value not in selected:
            selected.append(value)
    return selected


def _safe_output_path(value: Any) -> str:
    """Return one lossless repository-relative output path or fail closed."""

    if type(value) is not str:
        raise DatabasePortalBridgeError("task output path identity is not a string")
    path = PurePosixPath(value or ".")
    if (
        not value
        or value != value.strip()
        or len(value.encode("utf-8", errors="surrogatepass"))
        > _MAX_REPOSITORY_PATH_BYTES
        or "\\" in value
        or "," in value
        or path.is_absolute()
        or bool(PureWindowsPath(value).drive)
        or path == PurePosixPath(".")
        or path.as_posix() != value
        or ".." in path.parts
        or any(ord(character) < 32 for character in value)
    ):
        raise DatabasePortalBridgeError("task output path identity is unsafe or ambiguous")
    return path.as_posix()


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
                if not argv or any(type(part) is not str for part in argv):
                    raise DatabasePortalBridgeError(
                        "validation argv must contain exact nonempty strings"
                    )
                parts = tuple(argv)
                if any(not part or part != _line_value(part) for part in parts):
                    raise DatabasePortalBridgeError(
                        "validation argv contains noncanonical command text"
                    )
                # Database task sources preserve a Markdown shell command as
                # one argv item.  Re-joining that singleton would quote the
                # entire command and make the shell treat it as one executable
                # name.  Multi-item argv records remain losslessly joined.
                value = parts[0] if len(parts) == 1 else shlex.join(parts)
            else:
                if argv is not None:
                    raise DatabasePortalBridgeError(
                        "validation argv must be a sequence of exact strings"
                    )
                raw_value = item.get("command") or item.get("value")
                if (
                    type(raw_value) is not str
                    or not raw_value
                    or raw_value != _line_value(raw_value)
                ):
                    raise DatabasePortalBridgeError(
                        "validation command text is absent or noncanonical"
                    )
                value = raw_value
        else:
            if type(item) is not str or not item or item != _line_value(item):
                raise DatabasePortalBridgeError(
                    "validation command text is absent or noncanonical"
                )
            value = item
        if value and value not in selected:
            selected.append(value)
    return selected


def _safe_repository_path(value: Any) -> str:
    """Return one canonical relative repository path or fail closed."""

    if type(value) is not str:
        raise DatabasePortalBridgeError("owning repository metadata is not a string")
    selected = value.strip()
    path = PurePosixPath(selected or ".")
    if (
        not selected
        or selected != value
        or len(selected.encode("utf-8", errors="surrogatepass"))
        > _MAX_REPOSITORY_PATH_BYTES
        or "\\" in selected
        or path.is_absolute()
        or path.as_posix() != selected
        or ".." in path.parts
        or any(ord(character) < 32 for character in selected)
    ):
        raise DatabasePortalBridgeError("owning repository metadata is unsafe")
    return path.as_posix()


def _owning_repository(body: Mapping[str, Any]) -> str:
    """Read the owning-repository authority from consistent sealed fields."""

    raw_values: list[Any] = []
    for key in ("owning_repository", "owning repository"):
        if key in body and body[key] not in (None, ""):
            raw_values.append(body[key])
    markdown_metadata = body.get("markdown_metadata")
    if isinstance(markdown_metadata, Mapping):
        for key in ("owning_repository", "owning repository"):
            if key in markdown_metadata and markdown_metadata[key] not in (None, ""):
                raw_values.append(markdown_metadata[key])
    if not raw_values:
        return ""
    values = tuple(_safe_repository_path(value) for value in raw_values)
    if len(set(values)) != 1:
        raise DatabasePortalBridgeError("owning repository metadata is inconsistent")
    return values[0]


def _canonical_projection_identity(
    record: Any,
    body: Mapping[str, Any],
) -> tuple[str, str]:
    """Project the database task identity without creating new authority."""

    task_cid = getattr(record, "task_cid", None)
    if (
        type(task_cid) is not str
        or not task_cid
        or task_cid != _line_value(task_cid)
        or len(task_cid.encode("utf-8", errors="surrogatepass"))
        > _MAX_TASK_IDENTITY_BYTES
    ):
        raise DatabasePortalBridgeError(
            "database task CID is absent or noncanonical"
        )
    declared_cids = tuple(
        body[key]
        for key in ("canonical_task_cid", "canonical task cid")
        if key in body and body[key] not in (None, "")
    )
    if any(type(value) is not str or value != task_cid for value in declared_cids):
        raise DatabasePortalBridgeError(
            "database task body conflicts with its canonical CID"
        )

    declared_keys = tuple(
        body[key]
        for key in ("canonical_task_key", "canonical task key", "task_key")
        if key in body and body[key] not in (None, "")
    )
    if declared_keys:
        if (
            any(type(value) is not str for value in declared_keys)
            or len(set(declared_keys)) != 1
        ):
            raise DatabasePortalBridgeError(
                "database task body has an ambiguous canonical key"
            )
        task_key = declared_keys[0]
    else:
        # DuckDB task records currently persist the canonical CID but do not
        # require the historical semantic-key projection.  Derive only a
        # stable lookup key; the database CID remains the task authority.
        task_key = "task/v1/" + hashlib.sha256(task_cid.encode("utf-8")).hexdigest()
    if (
        not task_key
        or task_key != _line_value(task_key)
        or len(task_key.encode("utf-8", errors="surrogatepass"))
        > _MAX_TASK_IDENTITY_BYTES
    ):
        raise DatabasePortalBridgeError(
            "database task canonical key is absent or noncanonical"
        )
    return task_key, task_cid


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


def _projection_task_identity(
    record: Any,
    body: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the exact database identity admitted into a Portal projection.

    Portal's Markdown parser recognizes a canonical identity only when both
    the key and CID are present.  A database task CID rendered merely as
    descriptive metadata would otherwise be re-derived from the disposable
    projection and create a second identity for the same claimed task.
    """

    task_cid = str(getattr(record, "task_cid", "") or "").strip()
    if not task_cid or len(task_cid) > 1024 or any(character.isspace() for character in task_cid):
        raise DatabasePortalBridgeError("database task CID is not projection-safe")

    def claimed_values(*names: str) -> set[str]:
        selected = set(names)
        return {
            str(value).strip()
            for key, value in body.items()
            if str(key).strip().lower().replace("_", " ") in selected and value not in (None, "")
        }

    claimed_cids = claimed_values("task cid", "canonical task cid")
    if any(value != task_cid for value in claimed_cids):
        raise DatabasePortalBridgeError("database task body contradicts its authoritative task CID")

    claimed_keys = claimed_values("task key", "canonical task key")
    if len(claimed_keys) > 1:
        raise DatabasePortalBridgeError(
            "database task body contains contradictory canonical task keys"
        )
    task_key = next(iter(claimed_keys), task_cid)
    if not task_key or len(task_key) > 1024 or any(character.isspace() for character in task_key):
        raise DatabasePortalBridgeError("database task key is not projection-safe")
    return task_key, task_cid


def _projection_immutable_digest(text: str) -> str:
    normalized = _MUTABLE_PROJECTION_LINE.sub("- Status: <mutable>", text)
    return _sha256_bytes(normalized.encode("utf-8"))


def _projection_recovery_digest(text: str) -> str:
    """Ignore only mutable status and legacy operational receipt projection.

    Older bridge revisions projected the accelerator-owned status receipt into
    provider context.  Its replacement by the terminal blocked receipt must
    not invalidate semantic recovery, while every other projected task byte
    remains bound.
    """

    normalized = _MUTABLE_PROJECTION_LINE.sub("- Status: <mutable>", text)
    normalized = _OPERATIONAL_PROJECTION_LINE.sub("", normalized)
    normalized = "\n".join(
        line for line in normalized.splitlines() if line.strip()
    )
    return _sha256_bytes((normalized + "\n").encode("utf-8"))


def _projection_status(text: str) -> str:
    match = re.search(r"(?mi)^-\s*status\s*:\s*([^\r\n]+)$", text)
    return str(match.group(1) if match else "").strip().lower().replace("-", "_")


def _single_projection_field(text: str, label: str) -> str:
    matches = re.findall(
        rf"(?mi)^-\s*{re.escape(label)}\s*:\s*([^\r\n]*)$",
        text,
    )
    values = {str(match).strip() for match in matches}
    if len(values) != 1:
        raise DatabasePortalBridgeError(
            f"Portal task projection has an invalid {label!r} field"
        )
    return next(iter(values))


def verify_database_portal_attempt_projection(
    task_projection: Path | str,
    *,
    expected_task_alias: str = "",
    expected_task_cid: str = "",
    allowed_root: Path | str | None = None,
) -> dict[str, Any]:
    """Verify one immutable, database-authoritative attempt projection.

    The returned record is identity evidence only.  It grants no task,
    completion, merge, or policy authority.  This verifier exists so a merge
    candidate created by one fenced database attempt can be recognized by a
    later attempt without treating two disposable projection paths as two
    independent task boards.
    """

    supplied_projection = Path(task_projection)
    if supplied_projection.name != "task-projection.md":
        raise DatabasePortalBridgeError(
            "database Portal projection has a noncanonical filename"
        )
    if supplied_projection.is_symlink() or not supplied_projection.is_file():
        raise DatabasePortalBridgeError(
            "database Portal projection is not a regular non-symlink file"
        )
    try:
        projection = supplied_projection.resolve(strict=True)
        projection_size = projection.stat().st_size
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal projection is unavailable"
        ) from exc
    if projection_size > _MAX_DATABASE_PORTAL_PROJECTION_BYTES:
        raise DatabasePortalBridgeError(
            "database Portal projection exceeds the verification bound"
        )
    if allowed_root is not None:
        try:
            projection.relative_to(Path(allowed_root).resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal projection is outside the admitted root"
            ) from exc

    binding_path = projection.parent / "database-attempt-binding.json"
    if binding_path.is_symlink() or not binding_path.is_file():
        raise DatabasePortalBridgeError(
            "database Portal attempt binding is not a regular non-symlink file"
        )
    try:
        binding_size = binding_path.stat().st_size
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal attempt binding is unavailable"
        ) from exc
    if binding_size > _MAX_DATABASE_PORTAL_BINDING_BYTES:
        raise DatabasePortalBridgeError(
            "database Portal attempt binding exceeds the verification bound"
        )
    binding = dict(DatabasePortalExecutionBridge._read_binding(binding_path))
    if set(binding) != _DATABASE_PORTAL_ATTEMPT_BINDING_FIELDS:
        raise DatabasePortalBridgeError(
            "database Portal attempt binding fields are noncanonical"
        )
    if (
        binding.get("schema") != DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
        or binding.get("interface")
        != DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
        or binding.get("authoritative_task_store") != "duckdb"
        or binding.get("projection_authority") is not False
    ):
        raise DatabasePortalBridgeError(
            "database Portal attempt binding authority is invalid"
        )

    string_fields = (
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "goal_cid",
        "plan_cid",
        "lease_id",
    )
    if any(
        type(binding.get(field)) is not str
        or not str(binding[field]).strip()
        or str(binding[field]) != _line_value(binding[field])
        or len(str(binding[field]).encode("utf-8", errors="surrogatepass"))
        > _MAX_TASK_IDENTITY_BYTES
        for field in string_fields
    ):
        raise DatabasePortalBridgeError(
            "database Portal attempt binding identity is invalid"
        )
    if any(
        type(binding.get(field)) is not int or int(binding[field]) < 0
        for field in ("task_revision", "fencing_token", "fence_epoch")
    ):
        raise DatabasePortalBridgeError(
            "database Portal attempt binding fence is invalid"
        )
    digest_fields = (
        "task_body_digest",
        "projection_seed_digest",
        "projection_immutable_digest",
        "binding_id",
    )
    if any(
        type(binding.get(field)) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", binding[field]) is None
        for field in digest_fields
    ):
        raise DatabasePortalBridgeError(
            "database Portal attempt binding digest is invalid"
        )
    binding_body = dict(binding)
    binding_id = str(binding_body.pop("binding_id"))
    if binding_id != _sha256_bytes(_canonical_json(binding_body)):
        raise DatabasePortalBridgeError(
            "database Portal attempt binding identity does not verify"
        )
    expected_attempt_directory = hashlib.sha256(
        str(binding["attempt_id"]).encode("utf-8")
    ).hexdigest()[:24]
    if projection.parent.name != expected_attempt_directory:
        raise DatabasePortalBridgeError(
            "database Portal attempt directory is not identity-bound"
        )

    try:
        projection_text = projection.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatabasePortalBridgeError(
            "database Portal projection is unreadable"
        ) from exc
    if _projection_immutable_digest(projection_text) != str(
        binding["projection_immutable_digest"]
    ):
        raise DatabasePortalBridgeError(
            "database Portal projection immutable identity does not verify"
        )
    headers = _HEADER.findall(projection_text)
    task_alias = str(binding["task_alias"])
    task_cid = str(binding["task_cid"])
    if headers != [task_alias]:
        raise DatabasePortalBridgeError(
            "database Portal projection task alias does not verify"
        )
    projected_fields = {
        "Database task CID": task_cid,
        "Database attempt ID": str(binding["attempt_id"]),
        "Database claim ID": str(binding["claim_id"]),
        "Canonical task CID": task_cid,
        "Projection authority": "false",
    }
    if any(
        _single_projection_field(projection_text, label) != value
        for label, value in projected_fields.items()
    ):
        raise DatabasePortalBridgeError(
            "database Portal projection fields do not match its binding"
        )
    canonical_task_key = _single_projection_field(
        projection_text,
        "Canonical task key",
    )
    if (
        not canonical_task_key
        or canonical_task_key != _line_value(canonical_task_key)
        or len(
            canonical_task_key.encode("utf-8", errors="surrogatepass")
        )
        > _MAX_TASK_IDENTITY_BYTES
    ):
        raise DatabasePortalBridgeError(
            "database Portal projection canonical task key is invalid"
        )
    if expected_task_alias and task_alias != str(expected_task_alias):
        raise DatabasePortalBridgeError(
            "database Portal projection task alias changed"
        )
    if expected_task_cid and task_cid != str(expected_task_cid):
        raise DatabasePortalBridgeError(
            "database Portal projection task identity changed"
        )
    return {
        "verified": True,
        "binding_id": binding_id,
        "attempt_id": str(binding["attempt_id"]),
        "claim_id": str(binding["claim_id"]),
        "lease_id": str(binding["lease_id"]),
        "task_alias": task_alias,
        "task_cid": task_cid,
        "canonical_task_key": canonical_task_key,
        "goal_cid": str(binding["goal_cid"]),
        "plan_cid": str(binding["plan_cid"]),
        "task_revision": int(binding["task_revision"]),
        "fencing_token": int(binding["fencing_token"]),
        "fence_epoch": int(binding["fence_epoch"]),
        "projection_path": str(projection),
        "projection_immutable_digest": str(
            binding["projection_immutable_digest"]
        ),
        "projection_authority": False,
        "authoritative_task_store": "duckdb",
    }


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
                "attempt_consumed",
                "provider_dispatched",
                "backoff_seconds",
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
        repository_root: Path | str | None = None,
        worktree_root: Path | str | None = None,
        implementation_protected_paths: Sequence[str] = (),
        merge_queue: Any = None,
        merge_target_branch: str = "",
        worktree_submodule_paths: Sequence[str] = (),
        task_header_prefix: str = "## ",
        max_passes: int = 4,
        max_task_attempts: int = 0,
    ) -> None:
        if not callable(portal_factory):
            raise TypeError("portal_factory must be callable")
        if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        if (
            isinstance(max_task_attempts, bool)
            or not isinstance(max_task_attempts, int)
            or max_task_attempts < 0
            or max_task_attempts > _MAX_DATABASE_PORTAL_TASK_ATTEMPTS
        ):
            raise ValueError(
                "max_task_attempts must be an integer in "
                f"[0, {_MAX_DATABASE_PORTAL_TASK_ATTEMPTS}]"
            )
        self.task_source = task_source
        self.attempt_root = Path(attempt_root).absolute()
        self.portal_factory = portal_factory
        self.repository_root = (
            Path(repository_root).absolute() if repository_root is not None else None
        )
        self.worktree_root = (
            Path(worktree_root).absolute() if worktree_root is not None else None
        )
        self.implementation_protected_paths = tuple(
            sorted(
                _safe_repository_path(path)
                for path in (implementation_protected_paths or ())
            )
        )
        if len(set(self.implementation_protected_paths)) != len(
            self.implementation_protected_paths
        ):
            raise ValueError("implementation_protected_paths must be unique")
        self.merge_queue = merge_queue
        self.merge_target_branch = str(merge_target_branch or "").strip()
        if self.merge_queue is not None:
            from ..merge.checkout_lock import checkout_repository_id

            if self.repository_root is None or not self.merge_target_branch:
                raise ValueError(
                    "post-merge recovery requires repository_root and "
                    "merge_target_branch"
                )
            queue_branch = str(
                getattr(self.merge_queue, "target_branch", "") or ""
            ).strip()
            queue_repository_id = str(
                getattr(self.merge_queue, "target_repository_id", "") or ""
            ).strip()
            if (
                queue_branch != self.merge_target_branch
                or not queue_repository_id
                or queue_repository_id
                != checkout_repository_id(self.repository_root)
                or getattr(self.merge_queue, "require_target_binding", False)
                is not True
            ):
                raise ValueError(
                    "post-merge recovery requires an exact target-bound merge queue"
                )
            for operation in (
                "completed_requests",
                "pending_requests",
                "processing_requests",
                "quarantined_requests",
                "get",
            ):
                if not callable(getattr(self.merge_queue, operation, None)):
                    raise TypeError(
                        "post-merge recovery merge queue lacks "
                        f"{operation}()"
                    )
        self.worktree_submodule_paths = tuple(
            _safe_repository_path(path) for path in worktree_submodule_paths
        )
        if len(set(self.worktree_submodule_paths)) != len(
            self.worktree_submodule_paths
        ):
            raise ValueError("worktree_submodule_paths must be unique")
        self.task_header_prefix = str(task_header_prefix or "## ")
        self.max_passes = max_passes
        self.max_task_attempts = int(max_task_attempts)

    def _post_merge_recovery_cursor_path(self) -> Path:
        if self.merge_queue is None:
            raise DatabasePortalBridgeError("post-merge recovery queue is unavailable")
        binding = {
            "target_repository_id": str(
                getattr(self.merge_queue, "target_repository_id", "") or ""
            ),
            "target_branch": self.merge_target_branch,
            "attempt_root": str(self.attempt_root),
        }
        key = hashlib.sha256(_canonical_json(binding)).hexdigest()
        return (
            Path(self.merge_queue.queue_dir)
            / "train"
            / "post-merge-recovery-cursors"
            / f"{key}.json"
        )

    def _empty_post_merge_recovery_cursors(self) -> dict[str, str]:
        return {stage: "" for stage in _POST_MERGE_RECOVERY_CURSOR_STAGES}

    def _load_post_merge_recovery_cursors(self) -> dict[str, str]:
        """Load non-authoritative keyset progress, resetting invalid state."""

        if self.merge_queue is None:
            return self._empty_post_merge_recovery_cursors()
        path = self._post_merge_recovery_cursor_path()
        try:
            if path.stat().st_size > _MAX_POST_MERGE_RECOVERY_CURSOR_BYTES:
                return self._empty_post_merge_recovery_cursors()
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return self._empty_post_merge_recovery_cursors()
        if not isinstance(raw, Mapping):
            return self._empty_post_merge_recovery_cursors()
        value = dict(raw)
        state_id = str(value.pop("state_id", "") or "")
        cursors = value.get("cursors")
        expected_fields = {
            "schema",
            "target_repository_id",
            "target_branch",
            "attempt_root",
            "cursors",
        }
        if (
            set(value) != expected_fields
            or value.get("schema") != _POST_MERGE_RECOVERY_CURSOR_SCHEMA
            or value.get("target_repository_id")
            != str(getattr(self.merge_queue, "target_repository_id", "") or "")
            or value.get("target_branch") != self.merge_target_branch
            or value.get("attempt_root") != str(self.attempt_root)
            or not isinstance(cursors, Mapping)
            or set(cursors) != set(_POST_MERGE_RECOVERY_CURSOR_STAGES)
            or any(
                type(cursor) is not str
                or len(cursor.encode("utf-8", errors="surrogatepass")) > 4096
                or any(ord(character) < 32 for character in cursor)
                for cursor in cursors.values()
            )
        ):
            return self._empty_post_merge_recovery_cursors()
        from ..proof.formal_verification_contracts import content_identity

        if state_id != content_identity(value):
            return self._empty_post_merge_recovery_cursors()
        return {stage: str(cursors[stage]) for stage in _POST_MERGE_RECOVERY_CURSOR_STAGES}

    def _save_post_merge_recovery_cursors(
        self,
        cursors: Mapping[str, str],
    ) -> None:
        if self.merge_queue is None:
            return
        from ..proof.formal_verification_contracts import content_identity

        normalized = {
            stage: str(cursors.get(stage) or "")
            for stage in _POST_MERGE_RECOVERY_CURSOR_STAGES
        }
        body: dict[str, Any] = {
            "schema": _POST_MERGE_RECOVERY_CURSOR_SCHEMA,
            "target_repository_id": str(
                getattr(self.merge_queue, "target_repository_id", "") or ""
            ),
            "target_branch": self.merge_target_branch,
            "attempt_root": str(self.attempt_root),
            "cursors": normalized,
        }
        body["state_id"] = content_identity(body)
        _atomic_write(
            self._post_merge_recovery_cursor_path(),
            json.dumps(body, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )

    def _advance_post_merge_recovery_cursor(
        self,
        cursors: dict[str, str],
        stage: str,
        page: Sequence[Any],
    ) -> None:
        next_cursor = (
            str(getattr(page[-1], "request_id", "") or "") if page else ""
        )
        # The cursor is non-authoritative bookkeeping.  An idle maintenance
        # tick must not create durable filesystem churn while reporting zero
        # writes; preserve the write only for real progress or end-of-scan
        # wrap (non-empty -> empty).
        if cursors.get(stage, "") == next_cursor:
            return
        cursors[stage] = next_cursor
        self._save_post_merge_recovery_cursors(cursors)

    def _validation_repository_scope(self, body: Mapping[str, Any]) -> str:
        """Return the checked nested repository namespace for this task.

        Git mutation authority remains rooted at the accelerator checkout.
        Owner-relative outputs are projected into that root under this
        namespace, while validations enter the same configured repository.
        """

        owner = _owning_repository(body)
        if not owner or owner == _ROOT_REPOSITORY_AUTHORITY:
            return ""
        if owner not in self.worktree_submodule_paths:
            raise DatabasePortalBridgeError(
                f"owning repository {owner!r} is not a configured worktree submodule"
            )
        if self.repository_root is None:
            raise DatabasePortalBridgeError(
                "nested owning repository cannot be verified without repository_root"
            )
        try:
            root = self.repository_root.resolve(strict=True)
            candidate = (root / owner).resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                f"owning repository {owner!r} is unavailable or outside repository_root"
            ) from exc
        if not candidate.is_dir() or not (candidate / ".git").exists():
            raise DatabasePortalBridgeError(
                f"owning repository {owner!r} is not an initialized nested Git repository"
            )
        return owner

    @staticmethod
    def _scope_outputs(outputs: Sequence[str], repository: str) -> list[str]:
        """Project owner-relative paths into the superproject namespace.

        The owner is prepended exactly once.  A datasets-local package path
        such as ``ipfs_datasets_py/logic/api.py`` therefore intentionally
        becomes ``ipfs_datasets_py/ipfs_datasets_py/logic/api.py`` in the
        accelerator worktree.
        """

        scoped: list[str] = []
        for output in outputs:
            relative = _safe_output_path(output)
            projected = f"{repository}/{relative}" if repository else relative
            projected = _safe_output_path(projected)
            if projected not in scoped:
                scoped.append(projected)
        return scoped

    @staticmethod
    def _scope_validations(validations: Sequence[str], repository: str) -> list[str]:
        if not repository:
            return list(validations)
        if not validations:
            return []
        unscoped: list[str] = []
        for command in validations:
            command_root = validation_command_repository_root(command)
            if command_root is None:
                raise DatabasePortalBridgeError(
                    "nested-repository validation command has unsafe shell structure"
                )
            if command_root == "":
                unscoped.append(command)
            elif command_root == repository:
                if len(validations) != 1:
                    raise DatabasePortalBridgeError(
                        "multiple nested-repository validations must be unscoped"
                    )
                return [command]
            else:
                raise DatabasePortalBridgeError(
                    "validation command repository root conflicts with owning repository"
                )
        # The Markdown projection has one Validation field.  Emit exactly one
        # leading repository transition and fail fast across multiple typed
        # validation records; independently prefixing each record would make
        # the second ``cd`` relative to the already-entered nested repository.
        value = (
            f"cd {shlex.quote(repository)} && "
            + " && ".join(dict.fromkeys(unscoped))
        )
        if validation_command_repository_root(value) != repository:
            raise DatabasePortalBridgeError(
                "scoped validation command does not preserve repository authority"
            )
        return [value]

    def _paths(self, attempt: Any) -> DatabasePortalAttemptPaths:
        attempt_key = hashlib.sha256(str(attempt.attempt_id).encode("utf-8")).hexdigest()[:24]
        root = self.attempt_root / attempt_key
        return DatabasePortalAttemptPaths(
            root=root,
            task_projection=root / "task-projection.md",
            binding=root / "database-attempt-binding.json",
            state=root / "portal-task-state.json",
            strategy=root / "portal-strategy.json",
            events=root / "portal-events.jsonl",
            implementation_logs=root / "implementation-logs",
        )

    @staticmethod
    def _request_has_missing_output_recovery_lineage(request: Any) -> bool:
        """Accept only the exact quarantine class this maintenance path owns."""

        status = str(getattr(request, "status", "") or "").strip()
        metadata = getattr(request, "metadata", None)
        if not isinstance(metadata, Mapping):
            return False
        failure_reason = str(
            getattr(request, "failure_reason", "") or ""
        )
        if (
            status == "quarantined"
            and failure_reason
            == _POST_MERGE_DECLARED_OUTPUTS_MISSING_REASON
        ):
            return True
        if (
            status == "quarantined"
            and failure_reason in _CROSS_BOARD_COMPLETION_REASONS
        ):
            return True
        if status not in {"pending", "processing", "quarantined", "completed"}:
            return False
        revivals = metadata.get("revivals")
        if (
            not isinstance(revivals, Sequence)
            or isinstance(revivals, (str, bytes, bytearray, memoryview))
            or not revivals
            or not isinstance(revivals[-1], Mapping)
            or revivals[-1].get("previous_failure_reason")
            not in {
                _POST_MERGE_DECLARED_OUTPUTS_MISSING_REASON,
                *_CROSS_BOARD_COMPLETION_REASONS,
            }
        ):
            return False
        if status == "processing":
            return bool(
                str(getattr(request, "consumer_id", "") or "").startswith(
                    "merge-train:"
                )
                and str(getattr(request, "claim_token", "") or "")
            )
        if status == "quarantined":
            # Transport failure must not erase the sealed semantic origin.
            # These are the only generic terminal reasons produced while
            # recovering an abandoned merge-train claim.
            return failure_reason in {
                "merge train consumer exited on final attempt",
                "processing request exceeded max age",
            }
        if status == "completed":
            completion = metadata.get("completion")
            return bool(
                isinstance(completion, Mapping)
                and completion.get("schema")
                == _POST_MERGE_DECLARED_OUTPUT_COMPLETION_SCHEMA
                and completion.get("reason")
                == "post_merge_declared_outputs_repaired"
            )
        return True

    def _current_recovery_task_status(
        self,
        *,
        task_cid: str,
        task_alias: str,
    ) -> str:
        """Return an eligible canonical database status or an empty value."""

        getter = getattr(self.task_source, "get_task", None) or getattr(
            self.task_source,
            "get",
            None,
        )
        if not callable(getter):
            return ""
        try:
            record = getter(task_cid)
        except Exception as exc:
            # Attach/session failures must not look like "not our row".
            # Swallowing them advanced the recovery cursor past blocked
            # tasks and left the DuckDB frontier stuck.
            name = type(exc).__name__
            detail = str(exc)
            if (
                name
                in {
                    "DuckDBConnectionPolicyError",
                    "InvalidInputException",
                    "TimeoutError",
                }
                or "Authentication failed" in detail
                or "Authorization failed" in detail
                or "quack attach" in detail.lower()
            ):
                raise
            return ""
        if (
            record is None
            or str(getattr(record, "task_cid", "") or "") != task_cid
            or str(getattr(record, "task_alias", "") or "") != task_alias
        ):
            return ""
        status = str(getattr(record, "status", "") or "").strip().lower()
        # Once a fresh claim advances to in_progress (or completion lands), an
        # old completed queue row is historical evidence, not work to replay.
        return status if status in {"blocked", "retrying"} else ""

    def _owned_post_merge_recovery_projection(
        self,
        request: Any,
    ) -> _DatabasePortalRecoveryProjection | None:
        """Prove that one eligible request came from this lane's sealed attempt."""

        if self.merge_queue is None or not self._request_has_missing_output_recovery_lineage(
            request
        ):
            return None
        metadata = getattr(request, "metadata", None)
        if not isinstance(metadata, Mapping):
            return None
        task_alias = str(getattr(request, "task_id", "") or "")
        task_cid = str(getattr(request, "canonical_task_id", "") or "")
        task_key = str(getattr(request, "canonical_task_key", "") or "")
        commit_sha = str(getattr(request, "commit_sha", "") or "")
        queue_repository_id = str(
            getattr(self.merge_queue, "target_repository_id", "") or ""
        )
        if (
            metadata.get("schema") != _MERGE_CANDIDATE_SCHEMA
            or metadata.get("target_binding_schema")
            != _MERGE_TARGET_BINDING_SCHEMA
            or metadata.get("target_repository_id") != queue_repository_id
            or metadata.get("target_branch") != self.merge_target_branch
            or not task_alias
            or not task_cid
            or not task_key
            or re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None
            or metadata.get("implementation_commit") != commit_sha
            or metadata.get("task_header_prefix") != self.task_header_prefix
            or self.repository_root is None
            or metadata.get("repo_root") != str(self.repository_root)
            or metadata.get("completion_task_cids")
            != {task_alias: task_cid}
        ):
            return None

        task_payload = metadata.get("task")
        if not isinstance(task_payload, Mapping):
            return None
        task_metadata = task_payload.get("metadata")
        if (
            not isinstance(task_metadata, Mapping)
            or task_payload.get("task_id") != task_alias
            or task_payload.get("canonical_task_cid") != task_cid
            or task_payload.get("canonical_task_key") != task_key
            or task_metadata.get("database task cid") != task_cid
            or task_metadata.get("canonical task cid") != task_cid
            or task_metadata.get("canonical task key") != task_key
            or task_metadata.get("projection authority") != "false"
        ):
            return None

        raw_projection = metadata.get("todo_path")
        if type(raw_projection) is not str or not raw_projection:
            return None
        projection = Path(raw_projection)
        if (
            not projection.is_absolute()
            or str(projection) != raw_projection
            or projection.parent.parent != self.attempt_root
        ):
            return None
        root = projection.parent
        paths = DatabasePortalAttemptPaths(
            root=root,
            task_projection=projection,
            binding=root / "database-attempt-binding.json",
            state=root / "portal-task-state.json",
            strategy=root / "portal-strategy.json",
            events=root / "portal-events.jsonl",
            implementation_logs=root / "implementation-logs",
        )
        if any(
            metadata.get(key) != str(expected)
            for key, expected in (
                ("state_path", paths.state),
                ("strategy_path", paths.strategy),
                ("events_path", paths.events),
            )
        ):
            return None
        try:
            binding = verify_database_portal_attempt_projection(
                projection,
                expected_task_alias=task_alias,
                expected_task_cid=task_cid,
                allowed_root=self.attempt_root,
            )
        except (DatabasePortalBridgeError, OSError, TypeError, ValueError):
            return None
        if (
            binding.get("canonical_task_key") != task_key
            or task_metadata.get("database attempt id")
            != binding.get("attempt_id")
            or task_metadata.get("database claim id")
            != binding.get("claim_id")
        ):
            return None
        task_status = self._current_recovery_task_status(
            task_cid=task_cid,
            task_alias=task_alias,
        )
        if not task_status:
            return None
        return _DatabasePortalRecoveryProjection(
            paths=paths,
            binding=binding,
            task_status=task_status,
        )

    def _preauthorize_post_merge_recovery(
        self,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
        *,
        preauthorize: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        evidence_digest: Callable[[Mapping[str, Any]], str],
    ) -> None:
        """Require database authority before repair or current-tree validation."""

        binding = projection.binding
        source: dict[str, Any] = {
            "schema": _DATABASE_POST_MERGE_RECOVERY_PREAUTHORIZATION_SCHEMA,
            "request_id": str(getattr(request, "request_id", "") or ""),
            "task_cid": str(getattr(request, "canonical_task_id", "") or ""),
            "task_alias": str(getattr(request, "task_id", "") or ""),
            "candidate_commit": str(
                getattr(request, "commit_sha", "") or ""
            ),
            "source_attempt_id": str(binding.get("attempt_id") or ""),
            "source_claim_id": str(binding.get("claim_id") or ""),
            "source_lease_id": str(binding.get("lease_id") or ""),
            "source_fencing_token": binding.get("fencing_token"),
            "source_fence_epoch": binding.get("fence_epoch"),
            "source_binding_id": str(binding.get("binding_id") or ""),
            "source_projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
        }
        result = preauthorize(source)
        if not isinstance(result, Mapping):
            raise DatabasePortalBridgeError(
                "database post-merge preauthorization returned a non-object"
            )
        verified = dict(result)
        authorization_id = str(verified.pop("authorization_id", "") or "")
        expected = {
            **source,
            "authorized": True,
            "task_status": "blocked",
        }
        if (
            projection.task_status != "blocked"
            or verified != expected
            or authorization_id != evidence_digest(expected)
        ):
            raise DatabasePortalBridgeError(
                "database post-merge preauthorization is invalid"
            )

    def _post_merge_recovery_evidence(
        self,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
        *,
        evidence_digest: Callable[[Mapping[str, Any]], str],
    ) -> dict[str, Any] | None:
        """Compile the exact completed-row receipt into the database contract."""

        metadata = getattr(request, "metadata", None)
        completion = metadata.get("completion") if isinstance(metadata, Mapping) else None
        expected_completion_fields = {
            "schema",
            "status",
            "reason",
            "candidate_commit",
            "target_commit",
            "repair_receipt",
        }
        if (
            str(getattr(request, "status", "") or "") != "completed"
            or not isinstance(completion, Mapping)
            or set(completion) != expected_completion_fields
            or completion.get("schema")
            != _POST_MERGE_DECLARED_OUTPUT_COMPLETION_SCHEMA
            or completion.get("reason")
            != "post_merge_declared_outputs_repaired"
            or completion.get("status") not in _POST_MERGE_COMPLETION_STATUSES
            or completion.get("candidate_commit")
            != str(getattr(request, "commit_sha", "") or "")
        ):
            return None
        repair_receipt = completion.get("repair_receipt")
        target_commit = str(completion.get("target_commit") or "")
        task_alias = str(getattr(request, "task_id", "") or "")
        if (
            re.fullmatch(r"[0-9a-f]{40}", target_commit) is None
            or not isinstance(repair_receipt, Mapping)
            or repair_receipt.get("schema")
            != _POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA
            or repair_receipt.get("candidate_commit")
            != completion.get("candidate_commit")
            or repair_receipt.get("repair_commit") != target_commit
            or task_alias not in (repair_receipt.get("task_ids") or ())
            or not str(repair_receipt.get("receipt_id") or "")
        ):
            return None
        qualification_receipt = self._repair_receipt_for_current_target(
            repair_receipt,
            request=request,
            projection=projection,
        )
        if qualification_receipt is None:
            return None
        binding = projection.binding
        evidence: dict[str, Any] = {
            "request_id": str(getattr(request, "request_id", "") or ""),
            "task_cid": str(getattr(request, "canonical_task_id", "") or ""),
            "task_alias": task_alias,
            "candidate_commit": str(completion["candidate_commit"]),
            "source_attempt_id": str(binding.get("attempt_id") or ""),
            "source_claim_id": str(binding.get("claim_id") or ""),
            "source_lease_id": str(binding.get("lease_id") or ""),
            "source_fencing_token": binding.get("fencing_token"),
            "source_fence_epoch": binding.get("fence_epoch"),
            "source_binding_id": str(binding.get("binding_id") or ""),
            "source_projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
        }
        qualification_schema = str(
            qualification_receipt.get("schema") or ""
        )
        if qualification_schema == _POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA:
            evidence.update(
                schema=_DATABASE_POST_MERGE_RECOVERY_SCHEMA,
                repair_commit=str(
                    qualification_receipt.get("repair_commit") or ""
                ),
                repair_receipt_id=str(
                    qualification_receipt.get("receipt_id") or ""
                ),
                repair_receipt=dict(qualification_receipt),
            )
        elif (
            qualification_schema
            == _POST_MERGE_DECLARED_OUTPUT_REQUALIFICATION_SCHEMA
        ):
            evidence.update(
                schema=_DATABASE_POST_MERGE_REQUALIFICATION_RECOVERY_SCHEMA,
                qualified_target_commit=str(
                    qualification_receipt.get("current_target_commit") or ""
                ),
                requalification_receipt_id=str(
                    qualification_receipt.get("receipt_id") or ""
                ),
                requalification_receipt=dict(qualification_receipt),
            )
        else:
            return None
        if (
            not evidence["request_id"]
            or not evidence["source_attempt_id"]
            or not evidence["source_claim_id"]
            or not evidence["source_lease_id"]
            or type(evidence["source_fencing_token"]) is not int
            or type(evidence["source_fence_epoch"]) is not int
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(evidence["source_binding_id"]),
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(evidence["source_projection_immutable_digest"]),
            )
            is None
        ):
            return None
        evidence_id = evidence_digest(evidence)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", str(evidence_id or "")) is None:
            return None
        evidence["evidence_id"] = str(evidence_id)
        return evidence

    def _repair_receipt_current_target_identity(
        self,
        repair_receipt: Mapping[str, Any],
    ) -> tuple[str, str, bool] | None:
        """Verify source lineage and exact declared blobs at the live target.

        The returned boolean is true only when the live target is the recorded
        repair commit itself.  A descendant is merely eligible for fresh,
        uncached declared validation; ancestry and preserved output blobs do
        not by themselves authorize database recovery.
        """

        if self.repository_root is None or not self.merge_target_branch:
            return None
        repair_commit = str(repair_receipt.get("repair_commit") or "")
        repair_tree = str(repair_receipt.get("repair_tree") or "")
        candidate_commit = str(
            repair_receipt.get("candidate_commit") or ""
        )
        entries = repair_receipt.get("entries")
        if (
            re.fullmatch(r"[0-9a-f]{40}", repair_commit) is None
            or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", repair_tree)
            is None
            or re.fullmatch(r"[0-9a-f]{40}", candidate_commit) is None
            or not isinstance(entries, Sequence)
            or isinstance(entries, (str, bytes, bytearray, memoryview))
            or not entries
            or len(entries) > 4096
        ):
            return None

        def git(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *arguments],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )

        try:
            head = git(
                "rev-parse",
                "--verify",
                f"refs/heads/{self.merge_target_branch}^{{commit}}",
            )
            tree = git("rev-parse", "--verify", f"{repair_commit}^{{tree}}")
            ancestry = git(
                "merge-base",
                "--is-ancestor",
                candidate_commit,
                repair_commit,
            )
            current_ancestry = git(
                "merge-base",
                "--is-ancestor",
                repair_commit,
                head.stdout.strip(),
            )
            current_tree = git(
                "rev-parse",
                "--verify",
                f"{head.stdout.strip()}^{{tree}}",
            )
        except (OSError, subprocess.SubprocessError):
            return None
        current_head = head.stdout.strip()
        current_tree_id = current_tree.stdout.strip()
        if (
            head.returncode != 0
            or re.fullmatch(r"[0-9a-f]{40}", current_head) is None
            or tree.returncode != 0
            or tree.stdout.strip() != repair_tree
            or ancestry.returncode != 0
            or current_ancestry.returncode != 0
            or current_tree.returncode != 0
            or re.fullmatch(
                r"[0-9a-f]{40}(?:[0-9a-f]{24})?",
                current_tree_id,
            )
            is None
        ):
            return None

        observed_paths: set[str] = set()
        for raw_entry in entries:
            if not isinstance(raw_entry, Mapping):
                return None
            path = str(raw_entry.get("path") or "")
            if not path or path in observed_paths:
                return None
            try:
                safe_path = _safe_output_path(path)
                observed_items = [
                    subprocess.run(
                        [
                            "git",
                            "ls-tree",
                            "-z",
                            commit,
                            "--",
                            safe_path,
                        ],
                        cwd=self.repository_root,
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    for commit in (repair_commit, current_head)
                ]
            except (OSError, subprocess.SubprocessError):
                return None
            expected = (
                f"{raw_entry.get('mode')} "
                f"{raw_entry.get('object_type')} "
                f"{raw_entry.get('object_id')}\t{safe_path}\0"
            ).encode("utf-8")
            if any(
                item.returncode != 0 or item.stdout != expected
                for item in observed_items
            ):
                return None
            observed_paths.add(path)
        return current_head, current_tree_id, current_head == repair_commit

    @staticmethod
    def _post_merge_requalification_receipt_path(
        projection: _DatabasePortalRecoveryProjection,
        *,
        source_receipt_id: str,
        current_head: str,
        current_tree: str,
    ) -> Path:
        key = hashlib.sha256(
            _canonical_json(
                {
                    "source_repair_receipt_id": source_receipt_id,
                    "current_target_commit": current_head,
                    "current_target_tree": current_tree,
                }
            )
        ).hexdigest()
        return (
            projection.paths.root
            / "post-merge-declared-output-requalification"
            / f"{key}.json"
        )

    @staticmethod
    def _verified_post_merge_requalification_receipt(
        raw: Any,
        *,
        source_repair_receipt: Mapping[str, Any],
        task_alias: str,
        current_head: str,
        current_tree: str,
    ) -> dict[str, Any] | None:
        from ..proof.formal_verification_contracts import content_identity

        if not isinstance(raw, Mapping):
            return None
        value = dict(raw)
        receipt_id = str(value.pop("receipt_id", "") or "")
        source_receipt_id = str(
            source_repair_receipt.get("receipt_id") or ""
        )
        validations = value.get("validation")
        expected_validation_fields = {
            "task_id",
            "passed",
            "returncode",
            "validation_result_digests",
            "command_count",
            "log_sha256",
        }
        if (
            set(raw) != _POST_MERGE_DECLARED_OUTPUT_REQUALIFICATION_FIELDS
            or value.get("schema")
            != _POST_MERGE_DECLARED_OUTPUT_REQUALIFICATION_SCHEMA
            or value.get("task_ids") != [task_alias]
            or value.get("candidate_commit")
            != source_repair_receipt.get("candidate_commit")
            or value.get("source_repair_receipt_id") != source_receipt_id
            or value.get("source_repair_commit")
            != source_repair_receipt.get("repair_commit")
            or value.get("source_repair_receipt")
            != dict(source_repair_receipt)
            or value.get("current_target_commit") != current_head
            or value.get("current_target_tree") != current_tree
            or value.get("entries") != source_repair_receipt.get("entries")
            or not isinstance(validations, list)
            or len(validations) != 1
            or receipt_id != content_identity(value)
        ):
            return None
        validation = validations[0]
        digests = (
            validation.get("validation_result_digests")
            if isinstance(validation, Mapping)
            else None
        )
        command_count = (
            validation.get("command_count")
            if isinstance(validation, Mapping)
            else None
        )
        if (
            not isinstance(validation, Mapping)
            or set(validation) != expected_validation_fields
            or validation.get("task_id") != task_alias
            or validation.get("passed") is not True
            or validation.get("returncode") != 0
            or isinstance(command_count, bool)
            or not isinstance(command_count, int)
            or command_count < 1
            or not isinstance(digests, list)
            or len(digests) != command_count
            or any(
                re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(item)) is None
                for item in digests
            )
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(validation.get("log_sha256") or ""),
            )
            is None
        ):
            return None
        return {**value, "receipt_id": receipt_id}

    def _load_post_merge_requalification_receipt(
        self,
        path: Path,
        *,
        source_repair_receipt: Mapping[str, Any],
        task_alias: str,
        current_head: str,
        current_tree: str,
    ) -> dict[str, Any] | None:
        try:
            if path.stat().st_size > _MAX_DATABASE_PORTAL_PROJECTION_BYTES:
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        return self._verified_post_merge_requalification_receipt(
            raw,
            source_repair_receipt=source_repair_receipt,
            task_alias=task_alias,
            current_head=current_head,
            current_tree=current_tree,
        )

    def _requalify_descendant_repair_receipt(
        self,
        repair_receipt: Mapping[str, Any],
        *,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
        current_head: str,
        current_tree: str,
    ) -> dict[str, Any] | None:
        """Run the sealed task's declared validations at an exact descendant.

        Requalification is proposal-tier evidence for reopening the database
        task.  It runs no provider and grants no completion authority.  The
        canonical checkout lease guards creation of the detached validation
        worktree, while the merge-train consumer lease held by the caller
        prevents a competing queue integration from advancing the target.
        """

        if self.repository_root is None or self.merge_queue is None:
            return None
        task_alias = str(getattr(request, "task_id", "") or "")
        task_cid = str(getattr(request, "canonical_task_id", "") or "")
        receipt_task_ids = repair_receipt.get("task_ids")
        if (
            not task_alias
            or not task_cid
            or not isinstance(receipt_task_ids, list)
            or receipt_task_ids != [task_alias]
        ):
            # A database attempt projects exactly one task.  It cannot
            # revalidate additional receipt members it does not own.
            return None
        source_receipt_id = str(repair_receipt.get("receipt_id") or "")
        receipt_path = self._post_merge_requalification_receipt_path(
            projection,
            source_receipt_id=source_receipt_id,
            current_head=current_head,
            current_tree=current_tree,
        )
        if receipt_path.exists():
            # Immutable current-tree evidence is replayed verbatim.  Never run
            # a second validation (whose log identity could change the CAS
            # evidence) once the first valid receipt has been published.
            return self._load_post_merge_requalification_receipt(
                receipt_path,
                source_repair_receipt=repair_receipt,
                task_alias=task_alias,
                current_head=current_head,
                current_tree=current_tree,
            )

        portal = self.portal_factory(projection.paths, task_alias)
        if portal is None:
            raise DatabasePortalBridgeError(
                "portal_factory did not return a Portal-compatible daemon"
            )
        close = getattr(portal, "close_event_runtime", None) or getattr(
            portal,
            "close",
            None,
        )
        try:
            portal_queue = getattr(portal, "merge_queue", None)
            portal_repo_root = getattr(portal, "repo_root", None)
            portal_target = str(
                getattr(portal, "resolved_merge_target_branch", "") or ""
            )
            load_tasks = getattr(portal, "_load_tasks", None)
            run_validation = getattr(portal, "_run_validation_commands", None)
            run_mutation = getattr(
                portal,
                "_run_checkout_mutation_transaction",
                None,
            )
            cleanup_workspace = getattr(
                portal,
                "_cleanup_main_merge_workspace",
                None,
            )
            if (
                portal_queue is not self.merge_queue
                or portal_repo_root is None
                or Path(portal_repo_root).absolute() != self.repository_root
                or portal_target != self.merge_target_branch
                or not callable(load_tasks)
                or not callable(run_validation)
                or not callable(run_mutation)
                or not callable(cleanup_workspace)
            ):
                raise DatabasePortalBridgeError(
                    "Portal recovery daemon lacks current-tree validation authority"
                )
            try:
                tasks = list(load_tasks())
            except Exception:
                return None
            if (
                [str(getattr(task, "task_id", "") or "") for task in tasks]
                != [task_alias]
                or str(getattr(tasks[0], "canonical_task_cid", "") or "")
                != task_cid
                or not tuple(getattr(tasks[0], "validation", ()) or ())
            ):
                return None

            def validate_current_tree() -> dict[str, Any]:
                result: dict[str, Any] = {
                    "passed": False,
                    "reason": "current_tree_requalification_failed",
                    "validation": [],
                }
                validation_root = (
                    projection.paths.root
                    / "post-merge-declared-output-requalification"
                )
                validation_root.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(
                        prefix="worktree-",
                        dir=validation_root,
                    )
                )
                temporary.rmdir()
                workspace_added = False
                try:
                    before = subprocess.run(
                        [
                            "git",
                            "rev-parse",
                            "--verify",
                            f"refs/heads/{self.merge_target_branch}^{{commit}}",
                        ],
                        cwd=self.repository_root,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=10,
                    )
                    if before.returncode != 0 or before.stdout.strip() != current_head:
                        result["reason"] = "requalification_target_advanced"
                        return result
                    added = subprocess.run(
                        [
                            "git",
                            "worktree",
                            "add",
                            "--detach",
                            str(temporary),
                            current_head,
                        ],
                        cwd=self.repository_root,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=30,
                    )
                    if added.returncode != 0:
                        result["reason"] = (
                            "requalification_validation_worktree_unavailable"
                        )
                        return result
                    workspace_added = True
                    summaries: list[dict[str, Any]] = []
                    log_root = projection.paths.implementation_logs / (
                        "post-merge-declared-output-requalification"
                    )
                    log_root.mkdir(parents=True, exist_ok=True)
                    for task in tasks:
                        log_path = log_root / (
                            f"{task_alias}-{current_head[:16]}.log"
                        )
                        validation = run_validation(
                            temporary,
                            task,
                            log_path,
                            force_uncached=True,
                        )
                        command_results = (
                            validation.get("results")
                            if isinstance(validation, Mapping)
                            else None
                        )
                        if (
                            not isinstance(validation, Mapping)
                            or validation.get("passed") is not True
                            or validation.get("returncode") != 0
                            or not isinstance(command_results, list)
                            or not command_results
                            or not log_path.is_file()
                        ):
                            result["reason"] = (
                                "requalification_declared_validation_failed"
                            )
                            return result
                        result_digests = [
                            str(item.get("validation_result_digest") or "")
                            for item in command_results
                            if isinstance(item, Mapping)
                        ]
                        if (
                            len(result_digests) != len(command_results)
                            or any(
                                re.fullmatch(
                                    r"(?:sha256:)?[0-9a-f]{64}",
                                    item,
                                )
                                is None
                                for item in result_digests
                            )
                        ):
                            result["reason"] = (
                                "requalification_validation_evidence_invalid"
                            )
                            return result
                        summaries.append(
                            {
                                "task_id": task_alias,
                                "passed": True,
                                "returncode": 0,
                                "validation_result_digests": result_digests,
                                "command_count": len(command_results),
                                "log_sha256": hashlib.sha256(
                                    log_path.read_bytes()
                                ).hexdigest(),
                            }
                        )

                    workspace_head = subprocess.run(
                        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
                        cwd=temporary,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=10,
                    )
                    workspace_tree = subprocess.run(
                        ["git", "rev-parse", "--verify", "HEAD^{tree}"],
                        cwd=temporary,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=10,
                    )
                    workspace_status = subprocess.run(
                        [
                            "git",
                            "status",
                            "--porcelain=v1",
                            "-z",
                            "--untracked-files=all",
                        ],
                        cwd=temporary,
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    after = subprocess.run(
                        [
                            "git",
                            "rev-parse",
                            "--verify",
                            f"refs/heads/{self.merge_target_branch}^{{commit}}",
                        ],
                        cwd=self.repository_root,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=10,
                    )
                    if (
                        workspace_head.returncode != 0
                        or workspace_head.stdout.strip() != current_head
                        or workspace_tree.returncode != 0
                        or workspace_tree.stdout.strip() != current_tree
                        or workspace_status.returncode != 0
                        or workspace_status.stdout
                        or after.returncode != 0
                        or after.stdout.strip() != current_head
                    ):
                        result["reason"] = (
                            "requalification_validation_tree_changed"
                        )
                        return result
                    result.update(
                        passed=True,
                        reason="current_tree_requalified",
                        validation=summaries,
                    )
                    return result
                except (OSError, subprocess.SubprocessError):
                    result["reason"] = "requalification_validation_unavailable"
                    return result
                finally:
                    if workspace_added:
                        cleanup = cleanup_workspace(temporary, ephemeral=True)
                        if (
                            not isinstance(cleanup, Mapping)
                            or cleanup.get("cleaned") is not True
                        ):
                            result.update(
                                passed=False,
                                reason=(
                                    "requalification_validation_workspace_cleanup_failed"
                                ),
                            )

            transaction = run_mutation(
                task_id=task_alias,
                branch=self.merge_target_branch,
                operation="requalify_post_merge_declared_outputs",
                callback=validate_current_tree,
                failure_fields={"passed": False},
                extra={
                    "current_target_commit": current_head,
                    "source_repair_commit": str(
                        repair_receipt.get("repair_commit") or ""
                    ),
                },
            )
            validations = (
                transaction.get("validation")
                if isinstance(transaction, Mapping)
                else None
            )
            if (
                not isinstance(transaction, Mapping)
                or transaction.get("passed") is not True
                or not isinstance(validations, list)
                or len(validations) != len(tasks)
            ):
                return None
        finally:
            if callable(close):
                close()

        from ..proof.formal_verification_contracts import content_identity

        qualified: dict[str, Any] = {
            "schema": _POST_MERGE_DECLARED_OUTPUT_REQUALIFICATION_SCHEMA,
            "task_ids": [task_alias],
            "candidate_commit": str(
                repair_receipt.get("candidate_commit") or ""
            ),
            "source_repair_receipt_id": source_receipt_id,
            "source_repair_commit": str(
                repair_receipt.get("repair_commit") or ""
            ),
            "source_repair_receipt": dict(repair_receipt),
            "current_target_commit": current_head,
            "current_target_tree": current_tree,
            "entries": list(repair_receipt.get("entries") or ()),
            "validation": [dict(item) for item in validations],
        }
        qualified["receipt_id"] = content_identity(qualified)
        _atomic_write_once(
            receipt_path,
            json.dumps(qualified, indent=2, sort_keys=True).encode("utf-8")
            + b"\n",
        )
        return self._load_post_merge_requalification_receipt(
            receipt_path,
            source_repair_receipt=repair_receipt,
            task_alias=task_alias,
            current_head=current_head,
            current_tree=current_tree,
        )

    def _repair_receipt_for_current_target(
        self,
        repair_receipt: Mapping[str, Any],
        *,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
    ) -> dict[str, Any] | None:
        """Return exact or freshly requalified current-tree repair evidence."""

        from ..proof.formal_verification_contracts import content_identity

        if set(repair_receipt) != _POST_MERGE_DECLARED_OUTPUT_REPAIR_FIELDS:
            return None
        source = dict(repair_receipt)
        source_receipt_id = str(source.pop("receipt_id", "") or "")
        if (
            repair_receipt.get("schema")
            != _POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA
            or source_receipt_id != content_identity(source)
        ):
            return None
        identity = self._repair_receipt_current_target_identity(repair_receipt)
        if identity is None:
            return None
        current_head, current_tree, exact = identity
        if exact:
            return dict(repair_receipt)
        return self._requalify_descendant_repair_receipt(
            repair_receipt,
            request=request,
            projection=projection,
            current_head=current_head,
            current_tree=current_tree,
        )

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

    def _binding(self, attempt: Any, record: Any, seed: str) -> dict[str, Any]:
        body = dict(getattr(record, "body", {}) or {})
        canonical_task_key, canonical_task_cid = _projection_task_identity(
            record,
            body,
        )
        repository_tree_id = database_portal_authoritative_repository_tree_id(
            self.task_source,
            canonical_task_cid,
        )
        payload = {
            "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": canonical_task_cid,
            "canonical_task_key": canonical_task_key,
            "task_alias": str(
                getattr(record, "task_alias", "")
                or getattr(attempt, "task_alias", "")
                or attempt.task_cid
            ),
            "goal_cid": str(getattr(record, "goal_cid", "") or ""),
            "plan_cid": str(getattr(record, "plan_cid", "") or ""),
            "task_revision": int(getattr(record, "revision", 0) or 0),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "task_body_digest": _sha256_bytes(_canonical_json(body)),
            "task_contract_digest": database_portal_task_contract_digest(record),
            "repository_tree_id": repository_tree_id,
            "projection_seed_digest": _sha256_bytes(seed.encode("utf-8")),
            "projection_immutable_digest": _projection_immutable_digest(seed),
            "authoritative_task_store": "duckdb",
            "projection_authority": False,
        }
        payload["binding_id"] = _sha256_bytes(_canonical_json(payload))
        return payload

    def _render_projection(self, attempt: Any, record: Any) -> str:
        body = dict(getattr(record, "body", {}) or {})
        canonical_task_key, canonical_task_cid = _projection_task_identity(
            record,
            body,
        )
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
        repository_scope = self._validation_repository_scope(body)
        outputs = self._scope_outputs(_output_values(record, body), repository_scope)
        validations = self._scope_validations(
            _validation_values(record, body),
            repository_scope,
        )
        acceptance = _acceptance_value(record, body)
        priority = _line_value(getattr(record, "priority", "") or body.get("priority") or "P2")
        reserved = {
            "status",
            "completion",
            # Operational status receipts are accelerator-owned control-plane
            # state.  They are neither provider context nor semantic task
            # input, and status CASes must not invalidate an immutable Portal
            # attempt projection.
            "completion receipt",
            "completion_receipt",
            "priority",
            "track",
            "depends on",
            "depends_on",
            "outputs",
            "validation",
            "validations",
            "validation_commands",
            "acceptance",
            "task key",
            "task cid",
            "canonical task key",
            "canonical task cid",
            "canonical_task_key",
            "canonical_task_cid",
            "task_key",
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
            f"- Canonical Task Key: {canonical_task_key}",
            f"- Canonical Task CID: {canonical_task_cid}",
            f"- Database task CID: {_line_value(attempt.task_cid)}",
            f"- Database attempt ID: {_line_value(attempt.attempt_id)}",
            f"- Database claim ID: {_line_value(attempt.claim_id)}",
            f"- Database dependency CIDs: {_line_value(getattr(record, 'dependencies', ()))}",
            f"- Canonical task key: {canonical_task_key}",
            f"- Canonical task CID: {canonical_task_cid}",
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
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is unreadable"
            ) from exc
        if not isinstance(value, Mapping):
            raise DatabasePortalBridgeError("database Portal attempt binding is malformed")
        return value

    @staticmethod
    def _read_json_object(path: Path, *, noun: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(f"{noun} is unreadable") from exc
        if not isinstance(value, dict):
            raise DatabasePortalBridgeError(f"{noun} is not an object")
        return value

    def _ensure_attempt_projection(
        self, attempt: Any, record: Any
    ) -> tuple[DatabasePortalAttemptPaths, Mapping[str, Any]]:
        paths = self._paths(attempt)
        seed = self._render_projection(attempt, record)
        expected = self._binding(attempt, record, seed)
        paths.root.mkdir(parents=True, exist_ok=True)
        if paths.binding.exists():
            observed = self._read_binding(paths.binding)
            if observed != expected:
                raise DatabasePortalBridgeError(
                    "database Portal attempt binding changed across resume"
                )
        else:
            _atomic_write(
                paths.binding,
                json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
        if not paths.task_projection.exists():
            _atomic_write(paths.task_projection, seed.encode("utf-8"))
        self._verify_projection(paths, expected)
        return paths, expected

    @staticmethod
    def _verify_projection(paths: DatabasePortalAttemptPaths, binding: Mapping[str, Any]) -> str:
        try:
            text = paths.task_projection.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DatabasePortalBridgeError("Portal task projection is unreadable") from exc
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

    def _verified_recovery_binding(
        self,
        *,
        attempt: Any,
        record: Any,
        paths: DatabasePortalAttemptPaths,
    ) -> Mapping[str, Any]:
        """Rebind immutable attempt evidence after a control-status CAS.

        Blocking and retry transitions advance the DuckDB task revision and
        replace its operational status receipt.  They must not change the
        semantic task body, claim identity, or immutable projection.  This is
        the common recovery boundary used by every typed post-terminal repair.
        """

        if not (paths.binding.is_file() and paths.task_projection.is_file()):
            raise DatabasePortalBridgeError(
                "Portal recovery binding artifacts are incomplete"
            )
        seed = self._render_projection(attempt, record)
        expected_binding = self._binding(attempt, record, seed)
        observed_binding = self._read_binding(paths.binding)
        observed_body = dict(observed_binding)
        observed_binding_id = str(observed_body.pop("binding_id", "") or "")
        observed_revision = observed_body.get("task_revision")
        current_revision = int(getattr(record, "revision", 0) or 0)
        mutable_binding_fields = {
            "binding_id",
            "task_revision",
            "task_body_digest",
            "projection_seed_digest",
            "projection_immutable_digest",
        }
        stable_expected = {
            key: value
            for key, value in expected_binding.items()
            if key not in mutable_binding_fields
        }
        stable_observed = {
            key: value
            for key, value in observed_binding.items()
            if key not in mutable_binding_fields
        }
        observed_projection = self._verify_projection(paths, observed_binding)
        if (
            observed_binding_id != _sha256_bytes(_canonical_json(observed_body))
            or isinstance(observed_revision, bool)
            or not isinstance(observed_revision, int)
            or observed_revision < 1
            or current_revision < observed_revision
            or stable_observed != stable_expected
            or _projection_recovery_digest(observed_projection)
            != _projection_recovery_digest(seed)
        ):
            raise DatabasePortalBridgeError(
                "Portal recovery binding does not match the claim"
            )
        return observed_binding

    @staticmethod
    def _has_completion_event(
        paths: DatabasePortalAttemptPaths,
        alias: str,
        canonical_task_key: str,
        canonical_task_cid: str,
    ) -> bool:
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
                and str(event.get("canonical_task_key") or "") == canonical_task_key
                and str(event.get("canonical_task_cid") or "") == canonical_task_cid
            ):
                return True
        return False

    @staticmethod
    def _terminal_failure(result: Mapping[str, Any]) -> str:
        if result.get("blocked") is True:
            reason = str(result.get("reason") or "portal_execution_blocked")
            if reason in DATABASE_PORTAL_CHECKOUT_CONTENTION_REASONS:
                return ""
            if is_protected_recovery_fence_contention(reason):
                # Peer-owner recovery is a wait, not a task defect. The
                # typed-deferral classifier admits retry; do not CAS blocked.
                return ""
            return reason
        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return ""
        if implementation.get("deferred") is True:
            return str(implementation.get("reason") or "portal_execution_deferred")
        returncode = implementation.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool) and returncode != 0:
            return str(implementation.get("reason") or "portal_provider_failed")
        if implementation.get("skipped") is True:
            reason = str(implementation.get("reason") or "portal_execution_skipped")
            if reason == _INFLIGHT_PROCESS_SKIP_REASON:
                # A live implementer is a wait, not a task defect. Deferral
                # owns this reason; do not CAS blocked.
                return ""
            return reason
        return ""

    @staticmethod
    def _explicit_retryable_deferral(
        implementation: Mapping[str, Any],
    ) -> bool:
        """Admit only a closed, structured non-consuming deferral."""

        if (
            implementation.get("deferral_schema")
            != DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA
            or implementation.get("deferred") is not True
            or implementation.get("retryable") is not True
            or implementation.get("attempt_consumed") is not False
        ):
            return False
        kind = str(implementation.get("failure_kind") or "")
        provider_dispatched = implementation.get("provider_dispatched")
        if kind in {"lifecycle_setup", "lifecycle_race"}:
            return (
                provider_dispatched is False
                and implementation.get("provider_call_allowed")
                in (None, False)
            )
        if kind == "provider_capacity_backoff":
            retry_at = str(implementation.get("retry_at") or "")
            retry_after = implementation.get("retry_after_seconds")
            return bool(
                provider_dispatched is False
                and retry_at
                and type(retry_after) in {int, float}
                and not isinstance(retry_after, bool)
                and retry_after >= 0
            )
        if kind != "provider_capacity":
            return False
        returncode = implementation.get("returncode")
        retry_at = str(implementation.get("retry_at") or "")
        failure_class = str(implementation.get("failure_class") or "")
        providers = implementation.get("providers")
        return bool(
            type(provider_dispatched) is bool
            and type(returncode) is int
            and returncode != 0
            and retry_at
            and failure_class in {"transient_capacity", "hard_quota_exhausted"}
            and isinstance(providers, Sequence)
            and not isinstance(providers, (str, bytes, bytearray, memoryview))
            and any(str(item or "").strip() for item in providers)
        )

    @staticmethod
    def _consumed_no_progress_failure(
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        implementation: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Seal one consumed, validation-not-run, no-candidate outcome.

        Raw runner/provider text is never interpreted as a root cause.  The
        canonical context and diagnostic receipts establish the repository,
        tree, task and no-progress boundary; provider-effect state remains
        explicitly unknown.
        """

        returncode = implementation.get("returncode")
        validation = implementation.get("validation_result")
        commit_result = implementation.get("commit_result")
        merge_result = implementation.get("merge_result")
        board_completion = implementation.get("board_completion")
        if (
            type(returncode) is not int
            or returncode == 0
            or not -(2**31) <= returncode < 2**31
            or type(implementation.get("provider_dispatched")) is not bool
            or implementation.get("attempt_consumed") is not True
            or str(implementation.get("implementation_commit") or "")
            or str(implementation.get("completion_tree_id") or "")
            or not isinstance(commit_result, Mapping)
            or commit_result.get("committed") is not False
            or not isinstance(merge_result, Mapping)
            or merge_result.get("merged") is not False
            or not isinstance(board_completion, Mapping)
            or board_completion.get("complete") is not False
            or board_completion.get("pending_merge") is not False
            or not isinstance(validation, Mapping)
            or validation.get("attempted") is not False
            or validation.get("passed") is not True
            or str(validation.get("reason") or "") != "not_run"
            or type(validation.get("returncode")) is not int
            or validation.get("returncode") != 0
            or not isinstance(validation.get("results"), Sequence)
            or isinstance(
                validation.get("results"),
                (str, bytes, bytearray, memoryview),
            )
            or len(validation.get("results")) != 0
        ):
            return None

        task_id = str(implementation.get("task_id") or "").strip()
        canonical_task_cid = str(
            implementation.get("canonical_task_cid")
            or implementation.get("task_cid")
            or ""
        ).strip()
        portal_attempt = implementation.get("attempt")
        log_value = str(implementation.get("log_path") or "").strip()
        context_value = str(
            implementation.get("context_receipt_path") or ""
        ).strip()
        diagnostic_id = str(
            implementation.get("diagnostic_receipt_id") or ""
        ).strip()
        baseline = str(implementation.get("baseline_ref") or "").strip()
        if (
            task_id != str(binding.get("task_alias") or "")
            or canonical_task_cid != str(binding.get("task_cid") or "")
            or type(portal_attempt) is not int
            or portal_attempt < 1
            or not log_value
            or not context_value
            or not diagnostic_id
            or not baseline
        ):
            return None
        try:
            log_path = Path(log_value).expanduser().resolve(strict=True)
            context_path = Path(context_value).expanduser().resolve(strict=True)
            log_root = paths.implementation_logs.resolve(strict=True)
        except OSError:
            return None
        if log_root not in log_path.parents or log_root not in context_path.parents:
            return None
        try:
            raw_log = _bounded_file(
                log_path,
                limit=_MAX_FAILURE_LOG_BYTES,
            )
            raw_context = _bounded_file(
                context_path,
                limit=_MAX_CONTEXT_RECEIPT_BYTES,
            )
        except DatabasePortalBridgeError:
            return None

        try:
            diagnostic_paths = tuple(
                paths.implementation_logs.glob("*-diagnostic-receipt.json")
            )
        except OSError:
            return None
        if len(diagnostic_paths) != 1:
            return None
        diagnostic_path = diagnostic_paths[0]
        try:
            raw_diagnostic = _bounded_file(
                diagnostic_path,
                limit=_MAX_DIAGNOSTIC_RECEIPT_BYTES,
            )
            def reject_duplicate_keys(
                pairs: Sequence[tuple[str, Any]],
            ) -> dict[str, Any]:
                parsed: dict[str, Any] = {}
                for key, value in pairs:
                    if key in parsed:
                        raise ValueError(
                            "implementation diagnostic receipt has duplicate keys"
                        )
                    parsed[key] = value
                return parsed

            diagnostic_payload = json.loads(
                raw_diagnostic.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
            if not isinstance(diagnostic_payload, Mapping):
                return None
            from .implementation_daemon import ImplementationDiagnosticReceipt

            diagnostic = ImplementationDiagnosticReceipt.from_dict(
                diagnostic_payload
            )
            context_payload = json.loads(
                raw_context.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
            if not isinstance(context_payload, Mapping):
                return None
            from ..context.context_compiler import ContextCompilationReceipt

            context = ContextCompilationReceipt.from_dict(context_payload)
        except (DatabasePortalBridgeError, OSError, UnicodeDecodeError, ValueError):
            return None
        payload_receipt_id = diagnostic_payload.get("receipt_id")
        payload_failure_id = diagnostic_payload.get("failure_id")
        if (
            not isinstance(payload_receipt_id, str)
            or payload_receipt_id != diagnostic.receipt_id
            or payload_receipt_id != diagnostic_id
            or not isinstance(payload_failure_id, str)
            or payload_failure_id != diagnostic.failure_id
            or diagnostic.prior_decision_id != context.receipt_id
            or diagnostic.repository_id != context.repository_id
            or diagnostic.tree_id != context.tree_id
            or diagnostic.tree_id != baseline
            or context.objective_id != task_id
            or context.stage != "implementation"
            or isinstance(diagnostic.failure.get("returncode"), bool)
            or not isinstance(diagnostic.failure.get("returncode"), int)
            or diagnostic.failure.get("returncode") != returncode
            or diagnostic.changed_files
        ):
            return None
        projected_validation = {
            key: validation[key]
            for key in (
                "passed",
                "returncode",
                "reason",
                "reason_codes",
                "failed_commands",
                "failure_review",
            )
            if validation.get(key) not in (None, "", (), [], {})
        }
        if (
            diagnostic.failure.get("kind") != "implementation_failure"
            or diagnostic.failure.get("validation") != projected_validation
        ):
            return None

        signature_material = {
            "schema": DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
            "failure_kind": "consumed_no_progress",
            "repository_id": diagnostic.repository_id,
            "tree_id": baseline,
            "control_repository_tree_id": str(
                binding.get("repository_tree_id") or ""
            ),
            "task_cid": canonical_task_cid,
            "task_contract_digest": str(
                binding.get("task_contract_digest") or ""
            ),
            "diagnostic_failure_id": diagnostic.failure_id,
            "provider_effect_state": "unknown_may_have_started",
        }
        evidence = {
            **signature_material,
            "diagnostic_receipt_id": diagnostic_id,
            "diagnostic_receipt_digest": _sha256_bytes(raw_diagnostic),
            "diagnostic_receipt_size": len(raw_diagnostic),
            "context_receipt_id": context.receipt_id,
            "context_receipt_digest": _sha256_bytes(raw_context),
            "context_receipt_size": len(raw_context),
            "log_digest": _sha256_bytes(raw_log),
            "log_size": len(raw_log),
            "database_binding_id": str(binding.get("binding_id") or ""),
            "database_attempt_id": str(binding.get("attempt_id") or ""),
            "database_claim_id": str(binding.get("claim_id") or ""),
            "database_lease_id": str(binding.get("lease_id") or ""),
            "database_fencing_token": int(binding.get("fencing_token") or 0),
            "database_fence_epoch": int(binding.get("fence_epoch") or 0),
            "portal_task_id": task_id,
            "portal_attempt_number": portal_attempt,
            "returncode": returncode,
            "attempt_consumed": True,
            "portal_provider_dispatched": implementation.get(
                "provider_dispatched"
            ),
            "implementation_commit_present": False,
            "implementation_candidate_present": False,
            "validation_state": "not_run",
        }
        evidence["failure_fingerprint"] = (
            database_portal_consumed_no_progress_fingerprint(evidence)
        )
        try:
            return DatabasePortalBridgeConsumedNoProgressError(
                "portal_consumed_no_progress",
                failure_evidence=evidence,
            ).failure_evidence
        except ValueError:
            return None

    @staticmethod
    def _typed_deferral(
        result: Mapping[str, Any],
    ) -> tuple[str, int] | None:
        """Return exact Portal deferral data without parsing reason text."""

        blocked_reason = str(result.get("reason") or "").strip()
        if (
            result.get("blocked") is True
            and result.get("unchanged") is True
            and blocked_reason in DATABASE_PORTAL_CHECKOUT_CONTENTION_REASONS
        ):
            return (
                blocked_reason,
                DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS,
            )
        if result.get("blocked") is True:
            reason = str(result.get("reason") or "")
            if is_protected_recovery_fence_contention(reason):
                raw_backoff = result.get(
                    "backoff_seconds",
                    FENCE_CONTENTION_BACKOFF_SECONDS,
                )
                if (
                    isinstance(raw_backoff, bool)
                    or not isinstance(raw_backoff, int)
                    or raw_backoff < 0
                    or raw_backoff > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
                ):
                    raise DatabasePortalBridgeError(
                        "Portal fence deferral returned an invalid "
                        "backoff_seconds value"
                    )
                return (
                    reason or "external_protected_checkout_recovery_required",
                    int(raw_backoff),
                )
        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return None
        if (
            implementation.get("skipped") is True
            and str(implementation.get("reason") or "")
            == _INFLIGHT_PROCESS_SKIP_REASON
        ):
            raw_backoff = implementation.get(
                "backoff_seconds",
                INFLIGHT_PROCESS_BACKOFF_SECONDS,
            )
            if (
                isinstance(raw_backoff, bool)
                or not isinstance(raw_backoff, int)
                or raw_backoff < 0
                or raw_backoff > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
            ):
                raise DatabasePortalBridgeError(
                    "Portal inflight deferral returned an invalid "
                    "backoff_seconds value"
                )
            return (_INFLIGHT_PROCESS_SKIP_REASON, int(raw_backoff))
        # ``attempt_consumed=false``/``provider_dispatched=false`` also
        # describe a successful deterministic zero-provider closure.  Only
        # the explicit closed deferral signal grants retry semantics.
        if implementation.get("deferred") is not True:
            return None
        structured = DatabasePortalExecutionBridge._explicit_retryable_deferral(
            implementation
        )
        # Free-text ``deferred=true`` without a closed schema or an explicit
        # backoff is not retry authority.
        if not structured and "backoff_seconds" not in implementation:
            return None
        # Older typed deferrals predate the duration field.  They retain a
        # conservative bounded default instead of silently becoming a
        # zero-delay reconstruction loop.
        raw_backoff = implementation.get("backoff_seconds", 300)
        if (
            isinstance(raw_backoff, bool)
            or not isinstance(raw_backoff, int)
            or raw_backoff < 0
            or raw_backoff > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
        ):
            raise DatabasePortalBridgeError(
                "Portal deferral returned an invalid backoff_seconds value"
            )
        return (
            str(implementation.get("reason") or "portal_execution_deferred"),
            int(raw_backoff),
        )

    @staticmethod
    def _external_protected_checkout_deferral(
        result: Mapping[str, Any],
    ) -> tuple[str, int] | None:
        """Defer only a paired supervisor recovery journal, never a foreign one.

        The Portal child fail-closes when it sees a checkout-recovery lease
        it does not own.  That is correct: daemon and supervisor journals
        carry different guards and must not be interpreted with the other
        owner's schema.  The database authority may still wait when the
        owner tag names the paired supervisor, because that process already
        auto-adopts its own dead-owner journals.  Any other owner remains a
        terminal block.
        """

        if result.get("blocked") is not True:
            return None
        reason = str(result.get("reason") or "")
        if reason != "external_protected_checkout_recovery_required":
            return None
        recovery = result.get("protected_checkout_recovery")
        owner = (
            str(recovery.get("protected_recovery_owner") or "")
            if isinstance(recovery, Mapping)
            else str(result.get("protected_recovery_owner") or "")
        )
        if owner != _PAIRED_SUPERVISOR_PROTECTED_RECOVERY_OWNER:
            return None
        return (
            reason,
            _EXTERNAL_PROTECTED_CHECKOUT_DEFERRAL_BACKOFF_SECONDS,
        )

    @staticmethod
    def _inflight_process_deferral(
        result: Mapping[str, Any],
    ) -> tuple[str, int] | None:
        """Defer a live-worker skip instead of turning it into a terminal block.

        Portal ``run_once`` reports ``skipped``/``inflight_process`` when an
        implementation runner for this attempt still looks live.  That is a
        stable wait, not a failed provider.  Mapping it through
        ``_terminal_failure`` burned the task into ``blocked``.
        """

        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return None
        if implementation.get("skipped") is not True:
            return None
        reason = str(implementation.get("reason") or "")
        if reason != "inflight_process":
            return None
        return (reason, _INFLIGHT_PROCESS_DEFERRAL_BACKOFF_SECONDS)

    @staticmethod
    def _worktree_lifecycle_claim_deferral(
        result: Mapping[str, Any],
    ) -> tuple[str, int] | None:
        """Defer a leftover worktree lifecycle claim instead of terminalizing it.

        Portal reports ``skipped``/``worktree_lifecycle_claim_exists`` when a
        prior attempt still holds the fenced workspace.  That is a wait, not
        a failed provider.  Mapping it through ``_terminal_failure`` burned
        the typed deferral budget after leftover seed-conflict recovery.
        """

        implementation = result.get("implementation_result")
        payload = implementation if isinstance(implementation, Mapping) else result
        if not isinstance(payload, Mapping):
            return None
        reason = str(payload.get("reason") or "")
        if reason not in {
            "worktree_lifecycle_claim_exists",
            "worktree_lifecycle_active_transition_failed",
            "worktree_lifecycle_transition_failed",
        }:
            return None
        if payload.get("deferred") is True:
            return None
        if payload.get("skipped") is not True and payload.get("lifecycle_race") is not True:
            return None
        backoff = payload.get("backoff_seconds", 30)
        if (
            isinstance(backoff, bool)
            or not isinstance(backoff, int)
            or backoff < 0
            or backoff > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
        ):
            backoff = 30
        return (reason, int(backoff))

    @staticmethod
    def _pooled_worktree_create_deferral(
        result: Mapping[str, Any],
    ) -> tuple[str, int] | None:
        """Defer a failed pooled ``git worktree add`` instead of terminalizing it.

        Portal historically mapped cold checkout interrupts to returncode 1
        and ``portal_provider_failed``.  That is infrastructure, not a
        dispatched provider failure.
        """

        implementation = result.get("implementation_result")
        payload = implementation if isinstance(implementation, Mapping) else result
        if not isinstance(payload, Mapping):
            return None
        if payload.get("deferred") is True:
            return None
        if payload.get("provider_dispatched") is True:
            return None
        exception = payload.get("exception_result")
        if not isinstance(exception, Mapping):
            return None
        if str(exception.get("phase") or "") != "worktree_setup":
            return None
        message = str(exception.get("message") or "")
        if not message.startswith(_POOLED_WORKTREE_CREATE_FAILURE_PREFIX):
            return None
        return (
            DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON,
            _POOLED_WORKTREE_CREATE_DEFERRAL_BACKOFF_SECONDS,
        )

    @staticmethod
    def _looks_like_validation_retry(
        implementation: Mapping[str, Any],
    ) -> bool:
        """Select only the closed post-dispatch validation-failure shape."""

        validation = implementation.get("validation_result")
        reason = str(validation.get("reason") or "") if isinstance(validation, Mapping) else ""
        return bool(
            implementation.get("returncode") not in (None, 0)
            and implementation.get("attempt_consumed") is True
            and implementation.get("provider_dispatched") is True
            and isinstance(validation, Mapping)
            and validation.get("attempted") is True
            and validation.get("passed") is False
            and reason
            in {
                "declared_validation_failed",
                "validation_command_failed",
            }
        )

    @classmethod
    def _candidate_retry_reason(
        cls,
        implementation: Mapping[str, Any],
    ) -> str:
        """Return the closed retry code for an unusable dispatched candidate."""

        if cls._looks_like_validation_retry(implementation):
            return ""
        if implementation.get("returncode") in (None, 0):
            return ""
        if implementation.get("attempt_consumed") is not True:
            return ""
        if implementation.get("provider_dispatched") is not True:
            return ""
        validation = implementation.get("validation_result")
        commit_result = implementation.get("commit_result")
        observed = [
            implementation.get("reason"),
            validation.get("reason") if isinstance(validation, Mapping) else None,
            validation.get("error") if isinstance(validation, Mapping) else None,
            commit_result.get("reason") if isinstance(commit_result, Mapping) else None,
        ]
        for value in observed:
            text = str(value or "").strip()
            if text in DATABASE_PORTAL_CANDIDATE_RETRY_REASONS:
                return text
        return ""

    @staticmethod
    def _verified_event_chain(paths: DatabasePortalAttemptPaths) -> list[dict[str, Any]]:
        """Read one bounded attempt-local event chain without repairing it."""

        try:
            size = paths.events.stat().st_size
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "validation retry has no durable Portal event stream"
            ) from exc
        if size <= 0 or size > _MAX_DATABASE_PORTAL_EVENT_BYTES:
            raise DatabasePortalBridgeError(
                "validation retry Portal event stream exceeds its closed bound"
            )
        try:
            raw_lines = paths.events.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "validation retry Portal event stream is unreadable"
            ) from exc
        if not raw_lines or len(raw_lines) > _MAX_DATABASE_PORTAL_EVENTS:
            raise DatabasePortalBridgeError(
                "validation retry Portal event population is outside its closed bound"
            )

        events: list[dict[str, Any]] = []
        prior_event_id = ""
        stream_id = ""
        snapshot_id = ""
        for ordinal, line in enumerate(raw_lines, start=1):
            try:
                event = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeError(
                    "validation retry Portal event stream contains invalid JSON"
                ) from exc
            if not isinstance(event, dict):
                raise DatabasePortalBridgeError(
                    "validation retry Portal event stream contains a non-object"
                )
            body = dict(event)
            claimed_event_id = str(body.pop("event_id", "") or "")
            try:
                encoded = json.dumps(
                    body,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError, RecursionError) as exc:
                raise DatabasePortalBridgeError(
                    "validation retry Portal event is not canonical JSON"
                ) from exc
            expected_event_id = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
            current_stream = str(event.get("stream_id") or "")
            current_snapshot = str(event.get("snapshot_id") or "")
            sequence = event.get("sequence")
            if (
                claimed_event_id != expected_event_id
                or not current_stream
                or not current_snapshot
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != ordinal
                or str(event.get("previous_event_id") or "") != prior_event_id
                or (stream_id and current_stream != stream_id)
                or (snapshot_id and current_snapshot != snapshot_id)
            ):
                raise DatabasePortalBridgeError(
                    "validation retry Portal event chain failed identity verification"
                )
            stream_id = current_stream
            snapshot_id = current_snapshot
            prior_event_id = claimed_event_id
            events.append(event)
        return events

    def _current_protected_path_digests(
        self,
        protected_paths: Sequence[str],
    ) -> dict[str, str]:
        """Bind protected content to the current shared checkout without links."""

        if self.repository_root is None:
            raise DatabasePortalBridgeError(
                "protected-path recovery requires repository_root"
            )
        if (
            not protected_paths
            or len(protected_paths) > _MAX_PROTECTED_PATH_RECOVERY_PATHS
            or len(set(protected_paths)) != len(protected_paths)
        ):
            raise DatabasePortalBridgeError(
                "protected-path recovery population is outside its closed bound"
            )
        try:
            root = self.repository_root.resolve(strict=True)
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "protected-path recovery repository is unavailable"
            ) from exc
        digests: dict[str, str] = {}
        for raw_relative in protected_paths:
            relative = _safe_repository_path(raw_relative)
            if relative == ".":
                raise DatabasePortalBridgeError(
                    "protected-path recovery refuses the repository root"
                )
            candidate = root / relative
            try:
                current = root
                for component in PurePosixPath(relative).parts:
                    current = current / component
                    metadata = current.stat(follow_symlinks=False)
                    if stat.S_ISLNK(metadata.st_mode):
                        raise DatabasePortalBridgeError(
                            "protected-path recovery refuses symlink components"
                        )
                    if current != candidate and not stat.S_ISDIR(metadata.st_mode):
                        raise DatabasePortalBridgeError(
                            "protected-path recovery has a non-directory ancestor"
                        )
                    if current != candidate and (current / ".git").exists():
                        raise DatabasePortalBridgeError(
                            "protected-path recovery refuses submodule paths"
                        )
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise DatabasePortalBridgeError(
                        "protected-path recovery requires singly linked regular files"
                    )
                candidate.resolve(strict=True).relative_to(root)
            except (OSError, ValueError, RuntimeError) as exc:
                raise DatabasePortalBridgeError(
                    "protected-path recovery path escapes the shared checkout"
                ) from exc
            digests[relative] = _sha256_file(candidate)
        return digests

    def _disposed_workspace_path(self, value: Any) -> str:
        """Return one absent, canonical workspace below this repository."""

        if (
            self.repository_root is None
            or self.worktree_root is None
            or type(value) is not str
            or not value
        ):
            raise DatabasePortalBridgeError(
                "protected-path recovery requires an exact workspace path"
            )
        raw = Path(value)
        try:
            root = self.repository_root.resolve(strict=True)
            worktree_root = self.worktree_root.resolve(strict=True)
            resolved = raw.resolve(strict=False)
            resolved.relative_to(root)
            resolved.relative_to(worktree_root)
            if not raw.is_absolute() or raw != resolved or resolved == root:
                raise DatabasePortalBridgeError(
                    "protected-path recovery workspace is not canonical and bounded"
                )
            try:
                raw.lstat()
            except FileNotFoundError:
                pass
            else:
                raise DatabasePortalBridgeError(
                    "protected-path recovery workspace has not been disposed"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "protected-path recovery workspace is unavailable or unbounded"
            ) from exc
        return str(resolved)

    def _verify_protected_path_attempt_boundary(
        self,
        paths: DatabasePortalAttemptPaths,
    ) -> None:
        """Reject linked or escaped attempt artifacts before recovery writes."""

        try:
            configured_root = self.attempt_root
            attempt_root = configured_root.resolve(strict=True)
            attempt_dir = paths.root.resolve(strict=True)
            attempt_dir.relative_to(attempt_root)
            if configured_root != attempt_root or paths.root != attempt_dir:
                raise DatabasePortalBridgeError(
                    "protected-path recovery attempt root is linked or noncanonical"
                )
            for directory in (attempt_root, attempt_dir):
                metadata = directory.stat(follow_symlinks=False)
                if not stat.S_ISDIR(metadata.st_mode) or directory.is_symlink():
                    raise DatabasePortalBridgeError(
                        "protected-path recovery attempt boundary is not a directory"
                    )
            entries = list(attempt_dir.iterdir())
        except (OSError, RuntimeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "protected-path recovery attempt boundary is unavailable"
            ) from exc
        if len(entries) > 4096:
            raise DatabasePortalBridgeError(
                "protected-path recovery attempt population exceeds its bound"
            )
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise DatabasePortalBridgeError(
                    "protected-path recovery attempt artifact is unreadable"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise DatabasePortalBridgeError(
                    "protected-path recovery refuses linked attempt artifacts"
                )
            if stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise DatabasePortalBridgeError(
                        "protected-path recovery refuses hard-linked attempt artifacts"
                    )
            elif not stat.S_ISDIR(metadata.st_mode):
                raise DatabasePortalBridgeError(
                    "protected-path recovery refuses special attempt artifacts"
                )

    @staticmethod
    def _protected_path_identity_digests(
        scope: Mapping[str, Any],
        protected_paths: Sequence[str],
    ) -> dict[str, str]:
        paths = scope.get("paths")
        if not isinstance(paths, Mapping) or set(map(str, paths)) != set(
            protected_paths
        ):
            raise DatabasePortalBridgeError(
                "protected-path recovery snapshot population is incomplete"
            )
        digests: dict[str, str] = {}
        for relative in protected_paths:
            identity = paths.get(relative)
            if (
                not isinstance(identity, Mapping)
                or identity.get("state") != "present"
                or identity.get("kind") != "regular_file"
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(identity.get("sha256") or "")
                )
            ):
                raise DatabasePortalBridgeError(
                    "protected-path recovery snapshot has an unsafe identity"
                )
            digests[relative] = f"sha256:{identity['sha256']}"
        return digests

    def _build_protected_path_recovery_intent(
        self,
        *,
        attempt: Any,
        record: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Prove one workspace-disposal incident is not a protected edit."""

        incident_path = paths.root / _IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME
        active_path = paths.root / _IMPLEMENTATION_PROTECTED_ACTIVE_FILENAME
        incident = self._read_json_object(
            incident_path,
            noun="protected-path incident",
        )
        active = self._read_json_object(
            active_path,
            noun="protected-path active snapshot",
        )
        alias = str(binding.get("task_alias") or "")
        if (
            incident.get("schema") != "implementation-protected-path-incident-v1"
            or incident.get("reason") != "implementation_protected_path_mutated"
            or incident.get("requires_operator_clearance") is not True
            or incident.get("shared_checkout_restored") is not False
            or active.get("schema") != "implementation-protected-path-active-v1"
            or active.get("ephemeral_worktree") is not True
            or incident.get("task_id") != alias
            or active.get("task_id") != alias
            or incident.get("workspace_path") != active.get("workspace_path")
            or incident.get("attempt") != active.get("attempt")
        ):
            raise DatabasePortalBridgeError(
                "protected-path incident does not match its active attempt"
            )
        portal_attempt = incident.get("attempt")
        if (
            isinstance(portal_attempt, bool)
            or not isinstance(portal_attempt, int)
            or portal_attempt < 1
        ):
            raise DatabasePortalBridgeError(
                "protected-path incident has no exact Portal attempt"
            )
        protected = active.get("protected_paths")
        if (
            not isinstance(protected, list)
            or not all(type(item) is str for item in protected)
        ):
            raise DatabasePortalBridgeError(
                "protected-path active snapshot has no closed path population"
            )
        protected_paths = tuple(
            sorted(_safe_repository_path(item) for item in protected)
        )
        if len(set(protected_paths)) != len(protected_paths):
            raise DatabasePortalBridgeError(
                "protected-path active snapshot contains duplicate paths"
            )
        if (
            not self.implementation_protected_paths
            or protected_paths != self.implementation_protected_paths
        ):
            raise DatabasePortalBridgeError(
                "protected-path active population differs from configuration"
            )
        body = dict(getattr(record, "body", {}) or {})
        repository_scope = self._validation_repository_scope(body)
        output_paths = self._scope_outputs(
            _output_values(record, body),
            repository_scope,
        )
        for output in output_paths:
            output_path = PurePosixPath(output)
            for protected_path in map(PurePosixPath, protected_paths):
                if (
                    output_path == protected_path
                    or output_path in protected_path.parents
                    or protected_path in output_path.parents
                ):
                    raise DatabasePortalBridgeError(
                        "task output scope intersects a protected path"
                    )
        snapshot = active.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise DatabasePortalBridgeError(
                "protected-path active snapshot has no identity map"
            )
        workspace_scope = snapshot.get("workspace")
        shared_scope = snapshot.get("shared_checkout")
        if not isinstance(workspace_scope, Mapping) or not isinstance(
            shared_scope, Mapping
        ):
            raise DatabasePortalBridgeError(
                "protected-path recovery requires both snapshot scopes"
            )
        normalized_workspace = self._disposed_workspace_path(
            incident.get("workspace_path")
        )
        assert self.repository_root is not None
        if (
            workspace_scope.get("root") != normalized_workspace
            or shared_scope.get("root")
            != str(self.repository_root.resolve(strict=True))
        ):
            raise DatabasePortalBridgeError(
                "protected-path snapshot roots do not match the configured checkout"
            )
        shared_digests = self._protected_path_identity_digests(
            shared_scope,
            protected_paths,
        )
        workspace_digests = self._protected_path_identity_digests(
            workspace_scope,
            protected_paths,
        )
        if shared_digests != workspace_digests:
            raise DatabasePortalBridgeError(
                "protected paths differed before ephemeral workspace disposal"
            )
        current_digests = self._current_protected_path_digests(protected_paths)
        if current_digests != shared_digests:
            raise DatabasePortalBridgeError(
                "shared protected content changed since the active snapshot"
            )
        mutations = incident.get("mutations")
        if not isinstance(mutations, list) or not mutations:
            raise DatabasePortalBridgeError(
                "protected-path incident has no mutation evidence"
            )
        mutated_paths: list[str] = []
        workspace_identities = workspace_scope.get("paths")
        assert isinstance(workspace_identities, Mapping)
        for mutation in mutations:
            if not isinstance(mutation, Mapping):
                raise DatabasePortalBridgeError(
                    "protected-path incident has malformed mutation evidence"
                )
            relative = str(mutation.get("path") or "")
            if (
                mutation.get("scope") != "workspace"
                or mutation.get("change") != "deleted"
                or relative not in protected_paths
                or mutation.get("after") != {"state": "missing"}
                or mutation.get("before") != workspace_identities.get(relative)
            ):
                raise DatabasePortalBridgeError(
                    "protected-path incident is not a pure workspace disposal"
                )
            mutated_paths.append(relative)
        incident_paths = incident.get("protected_paths")
        if (
            len(set(mutated_paths)) != len(mutated_paths)
            or not isinstance(incident_paths, list)
            or sorted(incident_paths) != sorted(mutated_paths)
        ):
            raise DatabasePortalBridgeError(
                "protected-path incident mutation population is inconsistent"
            )

        events = self._verified_event_chain(paths)
        mutation_events = [
            event
            for event in events
            if event.get("type") == "implementation_protected_path_mutated"
            and event.get("task_id") == alias
            and event.get("attempt") == portal_attempt
            and event.get("workspace_path") == incident.get("workspace_path")
            and event.get("mutations") == mutations
        ]
        if len(mutation_events) != 1:
            raise DatabasePortalBridgeError(
                "protected-path incident has no unique durable mutation event"
            )
        event = mutation_events[0]
        clearance_basis = {
            "kind": "auto-clear-protected-path-stall",
            "task_id": alias,
            "attempt": int(portal_attempt),
            "workspace_path": normalized_workspace,
            "mutated_paths": sorted(mutated_paths),
            "scopes": ["workspace"],
            "changes": ["deleted"],
            "class_codes": ["workspace_protected_deletion"],
            "latched_at": str(incident.get("latched_at") or ""),
        }
        clearance_id = _sha256_bytes(_canonical_json(clearance_basis))
        intent = {
            "schema": DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_INTENT_SCHEMA,
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "portal_attempt": int(portal_attempt),
            "binding_id": str(binding.get("binding_id") or ""),
            "workspace_path": normalized_workspace,
            "incident_digest": _sha256_bytes(_canonical_json(incident)),
            "active_snapshot_digest": _sha256_bytes(_canonical_json(active)),
            "protected_paths": list(protected_paths),
            "mutated_paths": sorted(mutated_paths),
            "shared_path_digests": shared_digests,
            "clearance_id": clearance_id,
            "mutation_event_id": str(event.get("event_id") or ""),
            "event_stream_id": str(event.get("stream_id") or ""),
        }
        intent["intent_id"] = _sha256_bytes(_canonical_json(intent))
        return intent

    @staticmethod
    def _protected_path_recovery_guard(
        intent: Mapping[str, Any],
    ) -> dict[str, Any]:
        guard = {
            "schema": DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_GUARD_SCHEMA,
            "task_id": str(intent.get("task_alias") or ""),
            "attempt": int(intent.get("portal_attempt") or 0),
            "workspace_path": str(intent.get("workspace_path") or ""),
            "clearance_id": str(intent.get("clearance_id") or ""),
            "incident_digest": str(intent.get("incident_digest") or ""),
            "active_snapshot_digest": str(
                intent.get("active_snapshot_digest") or ""
            ),
            "protected_paths": list(intent.get("protected_paths") or []),
            "mutated_paths": list(intent.get("mutated_paths") or []),
            "class_codes": ["workspace_protected_deletion"],
            "shared_path_digests": dict(
                intent.get("shared_path_digests") or {}
            ),
        }
        guard["guard_id"] = _sha256_bytes(_canonical_json(guard))
        return guard

    def _verify_protected_path_recovery_intent(
        self,
        *,
        attempt: Any,
        binding: Mapping[str, Any],
        intent: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_fields = {
            "schema",
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "lease_id",
            "attempt_number",
            "fencing_token",
            "fence_epoch",
            "portal_attempt",
            "binding_id",
            "workspace_path",
            "incident_digest",
            "active_snapshot_digest",
            "protected_paths",
            "mutated_paths",
            "shared_path_digests",
            "clearance_id",
            "mutation_event_id",
            "event_stream_id",
            "intent_id",
        }
        body = dict(intent)
        intent_id = body.pop("intent_id", None)
        protected_paths = intent.get("protected_paths")
        mutated_paths = intent.get("mutated_paths")
        shared_digests = intent.get("shared_path_digests")
        if (
            set(intent) != expected_fields
            or intent.get("schema")
            != DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_INTENT_SCHEMA
            or intent_id != _sha256_bytes(_canonical_json(body))
            or intent.get("task_cid") != str(attempt.task_cid)
            or intent.get("task_alias") != str(binding.get("task_alias") or "")
            or intent.get("attempt_id") != str(attempt.attempt_id)
            or intent.get("claim_id") != str(attempt.claim_id)
            or intent.get("lease_id")
            != str(getattr(attempt, "lease_id", "") or "")
            or any(
                isinstance(intent.get(field), bool)
                or not isinstance(intent.get(field), int)
                for field in (
                    "attempt_number",
                    "fencing_token",
                    "fence_epoch",
                )
            )
            or intent.get("attempt_number") != int(attempt.attempt_number)
            or intent.get("fencing_token") != int(attempt.fencing_token)
            or intent.get("fence_epoch") != int(attempt.fence_epoch)
            or intent.get("binding_id") != str(binding.get("binding_id") or "")
            or isinstance(intent.get("portal_attempt"), bool)
            or not isinstance(intent.get("portal_attempt"), int)
            or int(intent.get("portal_attempt") or 0) < 1
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(intent.get("incident_digest") or "")
            )
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(intent.get("active_snapshot_digest") or ""),
            )
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(intent.get("clearance_id") or "")
            )
            or not isinstance(protected_paths, list)
            or not all(type(item) is str for item in protected_paths)
            or protected_paths != sorted(set(protected_paths))
            or not protected_paths
            or not isinstance(mutated_paths, list)
            or not all(type(item) is str for item in mutated_paths)
            or not mutated_paths
            or mutated_paths != sorted(set(mutated_paths))
            or not set(mutated_paths).issubset(set(protected_paths))
            or not isinstance(shared_digests, Mapping)
            or set(map(str, shared_digests)) != set(protected_paths)
            or any(
                not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or ""))
                for value in shared_digests.values()
            )
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(intent.get("mutation_event_id") or ""),
            )
            or not re.fullmatch(
                r"event-log:sha256:[0-9a-f]{64}",
                str(intent.get("event_stream_id") or ""),
            )
        ):
            raise DatabasePortalBridgeError(
                "protected-path recovery intent is malformed or foreign"
            )
        if self._disposed_workspace_path(intent.get("workspace_path")) != intent.get(
            "workspace_path"
        ):
            raise DatabasePortalBridgeError(
                "protected-path recovery workspace identity changed"
            )
        current = self._current_protected_path_digests(protected_paths)
        if current != dict(shared_digests):
            raise DatabasePortalBridgeError(
                "protected content changed after recovery was prepared"
            )
        return dict(intent)

    def _finalize_protected_path_recovery_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        intent: Mapping[str, Any],
    ) -> dict[str, Any]:
        verified_intent = self._verify_protected_path_recovery_intent(
            attempt=attempt,
            binding=binding,
            intent=intent,
        )
        clearance_id = str(verified_intent["clearance_id"])
        clearance_path = paths.root / (
            "implementation-protected-path-auto-clearance-"
            f"{clearance_id.removeprefix('sha256:')[:16]}.json"
        )
        clearance = self._read_json_object(
            clearance_path,
            noun="protected-path auto-clearance receipt",
        )
        clearance_basis = {
            "kind": "auto-clear-protected-path-stall",
            "task_id": str(clearance.get("task_id") or ""),
            "attempt": clearance.get("attempt"),
            "workspace_path": str(clearance.get("workspace_path") or ""),
            "mutated_paths": list(clearance.get("mutated_paths") or []),
            "scopes": list(clearance.get("scopes") or []),
            "changes": list(clearance.get("changes") or []),
            "class_codes": list(clearance.get("class_codes") or []),
            "latched_at": str(clearance.get("incident_latched_at") or ""),
        }
        if (
            clearance.get("schema")
            != "implementation-protected-path-auto-clearance-v1"
            or clearance.get("clearance_id") != clearance_id
            or clearance.get("reason")
            != "ephemeral_workspace_protected_deletions_shared_intact"
            or clearance.get("task_id") != verified_intent["task_alias"]
            or clearance.get("attempt") != verified_intent["portal_attempt"]
            or clearance.get("workspace_path") != verified_intent["workspace_path"]
            or clearance.get("mutated_paths")
            != verified_intent["mutated_paths"]
            or clearance.get("scopes") != ["workspace"]
            or clearance.get("changes") != ["deleted"]
            or clearance.get("class_codes")
            != ["workspace_protected_deletion"]
            or clearance.get("shared_protected_paths_present")
            != verified_intent["mutated_paths"]
            or _sha256_bytes(_canonical_json(clearance_basis)) != clearance_id
        ):
            raise DatabasePortalBridgeError(
                "protected-path auto-clearance receipt is not the prepared repair"
            )
        if (
            (paths.root / _IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME).exists()
            or (paths.root / _IMPLEMENTATION_PROTECTED_ACTIVE_FILENAME).exists()
        ):
            raise DatabasePortalBridgeError(
                "protected-path fence remains active after auto-clearance"
            )
        events = self._verified_event_chain(paths)
        mutation_events = [
            event
            for event in events
            if event.get("event_id") == verified_intent["mutation_event_id"]
        ]
        clearance_events = [
            event
            for event in events
            if event.get("type")
            == "implementation_protected_path_incident_auto_cleared"
            and event.get("clearance_id") == clearance_id
            and event.get("task_id") == verified_intent["task_alias"]
            and event.get("attempt") == verified_intent["portal_attempt"]
            and event.get("mutated_paths") == verified_intent["mutated_paths"]
            and event.get("class_codes") == ["workspace_protected_deletion"]
        ]
        if len(mutation_events) != 1 or len(clearance_events) != 1:
            raise DatabasePortalBridgeError(
                "protected-path recovery has no unique durable event pair"
            )
        mutation_event = mutation_events[0]
        event_mutations = mutation_event.get("mutations")
        if not isinstance(event_mutations, list) or sorted(
            str(item.get("path") or "")
            for item in event_mutations
            if isinstance(item, Mapping)
        ) != verified_intent["mutated_paths"] or any(
            not isinstance(item, Mapping)
            or item.get("scope") != "workspace"
            or item.get("change") != "deleted"
            or item.get("after") != {"state": "missing"}
            or not isinstance(item.get("before"), Mapping)
            or (
                f"sha256:{item['before'].get('sha256', '')}"
                != verified_intent["shared_path_digests"].get(
                    str(item.get("path") or "")
                )
            )
            for item in event_mutations
        ):
            raise DatabasePortalBridgeError(
                "protected-path mutation event is not the prepared disposal"
            )
        clearance_event = clearance_events[0]
        if (
            mutation_event.get("type") != "implementation_protected_path_mutated"
            or mutation_event.get("stream_id")
            != verified_intent["event_stream_id"]
            or mutation_event.get("task_id") != verified_intent["task_alias"]
            or mutation_event.get("attempt")
            != verified_intent["portal_attempt"]
            or mutation_event.get("workspace_path")
            != verified_intent["workspace_path"]
            or clearance_event.get("reason")
            != "ephemeral_workspace_protected_deletions_shared_intact"
            or clearance_event.get("cleared") is not True
            or clearance_event.get("auto") is not True
            or clearance_event.get("blocked") is not False
            or clearance_event.get("workspace_path")
            != verified_intent["workspace_path"]
            or clearance_event.get("stream_id")
            != verified_intent["event_stream_id"]
        ):
            raise DatabasePortalBridgeError(
                "protected-path recovery event stream changed"
            )
        receipt = {
            "schema": DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA,
            "disposition": "retry",
            "reason": "ephemeral_workspace_protected_deletions_recovered",
            "task_cid": str(attempt.task_cid),
            "task_alias": str(binding.get("task_alias") or ""),
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "portal_attempt": int(verified_intent["portal_attempt"]),
            "binding_id": str(binding.get("binding_id") or ""),
            "workspace_path": str(verified_intent["workspace_path"]),
            "incident_digest": str(verified_intent["incident_digest"]),
            "active_snapshot_digest": str(
                verified_intent["active_snapshot_digest"]
            ),
            "clearance_id": clearance_id,
            "clearance_receipt_digest": _sha256_file(clearance_path),
            "protected_paths": list(verified_intent["protected_paths"]),
            "mutated_paths": list(verified_intent["mutated_paths"]),
            "class_codes": ["workspace_protected_deletion"],
            "shared_path_digests": dict(
                verified_intent["shared_path_digests"]
            ),
            "event_stream_id": str(verified_intent["event_stream_id"]),
            "mutation_event_id": str(verified_intent["mutation_event_id"]),
            "clearance_event_id": str(clearance_event.get("event_id") or ""),
            "events_digest": _sha256_file(paths.events),
            "backoff_seconds": 0,
            # Conservatively consume one implementation slot.  This does not
            # assert that a remote provider ran; it prevents a cleanup race
            # from becoming an unbounded free retry loop.
            "attempt_consumed": True,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def _verify_protected_path_recovery_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_fields = {
            "schema",
            "disposition",
            "reason",
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "lease_id",
            "attempt_number",
            "fencing_token",
            "fence_epoch",
            "portal_attempt",
            "binding_id",
            "workspace_path",
            "incident_digest",
            "active_snapshot_digest",
            "clearance_id",
            "clearance_receipt_digest",
            "protected_paths",
            "mutated_paths",
            "class_codes",
            "shared_path_digests",
            "event_stream_id",
            "mutation_event_id",
            "clearance_event_id",
            "events_digest",
            "backoff_seconds",
            "attempt_consumed",
            "receipt_id",
        }
        intent_path = paths.root / _PROTECTED_PATH_RECOVERY_INTENT_FILENAME
        if set(receipt) != expected_fields or not intent_path.is_file():
            raise DatabasePortalBridgeError(
                "protected-path recovery receipt is malformed or foreign"
            )
        intent = self._read_json_object(
            intent_path,
            noun="protected-path recovery intent",
        )
        expected = self._finalize_protected_path_recovery_receipt(
            attempt=attempt,
            paths=paths,
            binding=binding,
            intent=intent,
        )
        if dict(receipt) != expected:
            raise DatabasePortalBridgeError(
                "protected-path recovery receipt changed after finalization"
            )
        return expected

    def recover_protected_path_retry(self, attempt: Any) -> Mapping[str, Any]:
        """Automatically rearm only a proved ephemeral-workspace disposal.

        The protected-path guard remains fail closed for content edits,
        symlinks, shared-checkout mutations, output-scope overlap, missing
        evidence, and live workspaces.  A durable intent closes the crash gap
        between clearing the attempt-local fence and the DuckDB status CAS.
        """

        record = self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        self._verify_protected_path_attempt_boundary(paths)
        if not paths.events.is_file():
            raise DatabasePortalBridgeError(
                "protected-path recovery has no durable Portal event stream"
            )
        binding = self._verified_recovery_binding(
            attempt=attempt,
            record=record,
            paths=paths,
        )
        final_path = paths.root / _PROTECTED_PATH_RECOVERY_FILENAME
        if final_path.is_file():
            return self._verify_protected_path_recovery_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                receipt=self._read_json_object(
                    final_path,
                    noun="protected-path recovery receipt",
                ),
            )

        intent_path = paths.root / _PROTECTED_PATH_RECOVERY_INTENT_FILENAME
        incident_path = paths.root / _IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME
        active_path = paths.root / _IMPLEMENTATION_PROTECTED_ACTIVE_FILENAME
        if incident_path.is_file():
            prepared = self._build_protected_path_recovery_intent(
                attempt=attempt,
                record=record,
                paths=paths,
                binding=binding,
            )
            if intent_path.exists():
                observed_intent = self._read_json_object(
                    intent_path,
                    noun="protected-path recovery intent",
                )
                if observed_intent != prepared:
                    raise DatabasePortalBridgeError(
                        "protected-path recovery intent changed across resume"
                    )
            else:
                _atomic_write(
                    intent_path,
                    json.dumps(prepared, indent=2, sort_keys=True).encode("utf-8")
                    + b"\n",
                )
            self._verify_protected_path_attempt_boundary(paths)
            capability = _ProtectedPathRecoveryAttemptCapability(
                paths,
                incident_present=True,
            )
            try:
                daemon = self.portal_factory(
                    paths,
                    str(binding.get("task_alias") or attempt.task_cid),
                )
                reconcile = getattr(
                    daemon,
                    "_reconcile_implementation_protected_path_fence",
                    None,
                )
                if not callable(reconcile):
                    raise DatabasePortalBridgeError(
                        "Portal executor has no protected-path reconciler"
                    )
                try:
                    result = reconcile(
                        protected_path_recovery_guard=(
                            self._protected_path_recovery_guard(prepared)
                        ),
                        protected_path_recovery_io=capability.recovery_io(),
                    )
                finally:
                    close = getattr(daemon, "close_event_runtime", None) or getattr(
                        daemon, "close", None
                    )
                    if callable(close):
                        close()
            finally:
                capability.close()
            if (
                not isinstance(result, Mapping)
                or result.get("blocked") is not False
                or result.get("auto") is not True
                or result.get("clearance_id") != prepared["clearance_id"]
                or result.get("class_codes")
                != ["workspace_protected_deletion"]
                or result.get("mutated_paths") != prepared["mutated_paths"]
            ):
                raise DatabasePortalBridgeError(
                    "protected-path incident was not eligible for automatic recovery"
                )
            intent = prepared
        else:
            if not intent_path.is_file():
                raise DatabasePortalBridgeError(
                    "protected-path incident and recovery intent are absent"
                )
            intent = self._read_json_object(
                intent_path,
                noun="protected-path recovery intent",
            )
            self._verify_protected_path_attempt_boundary(paths)
            intent = self._verify_protected_path_recovery_intent(
                attempt=attempt,
                binding=binding,
                intent=intent,
            )
            if active_path.is_file():
                capability = _ProtectedPathRecoveryAttemptCapability(
                    paths,
                    incident_present=False,
                )
                try:
                    daemon = self.portal_factory(
                        paths,
                        str(binding.get("task_alias") or attempt.task_cid),
                    )
                    reconcile = getattr(
                        daemon,
                        "_reconcile_implementation_protected_path_fence",
                        None,
                    )
                    if not callable(reconcile):
                        raise DatabasePortalBridgeError(
                            "Portal executor has no protected-path reconciler"
                        )
                    try:
                        result = reconcile(
                            protected_path_recovery_guard=(
                                self._protected_path_recovery_guard(intent)
                            ),
                            protected_path_recovery_io=capability.recovery_io(),
                        )
                    finally:
                        close = getattr(
                            daemon, "close_event_runtime", None
                        ) or getattr(daemon, "close", None)
                        if callable(close):
                            close()
                finally:
                    capability.close()
                if not isinstance(result, Mapping) or result.get("blocked") is not False:
                    raise DatabasePortalBridgeError(
                        "protected-path recovery could not finish fence cleanup"
                    )

        self._verify_protected_path_attempt_boundary(paths)
        receipt = self._finalize_protected_path_recovery_receipt(
            attempt=attempt,
            paths=paths,
            binding=binding,
            intent=intent,
        )
        _atomic_write(
            final_path,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        self._verify_protected_path_attempt_boundary(paths)
        return receipt

    def _checkout_mutation_lock_path(self) -> Path:
        if self.repository_root is None:
            raise DatabasePortalBridgeError(
                "external checkout recovery has no repository root"
            )
        from ..merge.checkout_lock import checkout_mutation_lock_path

        return checkout_mutation_lock_path(self.repository_root)

    def _finalize_external_protected_checkout_recovery_receipt(
        self,
        *,
        attempt: Any,
        lock_path: Path,
    ) -> dict[str, Any]:
        receipt = {
            "schema": DATABASE_PORTAL_EXTERNAL_PROTECTED_CHECKOUT_RECOVERY_SCHEMA,
            "disposition": "retry",
            "reason": "external_protected_checkout_lock_absent",
            "source_reason": "external_protected_checkout_recovery_required",
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "lock_path": str(lock_path),
            "lock_present": False,
            "backoff_seconds": 0,
            "attempt_consumed": False,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def _verify_external_protected_checkout_recovery_receipt(
        self,
        *,
        attempt: Any,
        lock_path: Path,
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_fields = {
            "schema",
            "disposition",
            "reason",
            "source_reason",
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "lease_id",
            "attempt_number",
            "fencing_token",
            "fence_epoch",
            "lock_path",
            "lock_present",
            "backoff_seconds",
            "attempt_consumed",
            "receipt_id",
        }
        if set(receipt) != expected_fields:
            raise DatabasePortalBridgeError(
                "external checkout recovery receipt is malformed or foreign"
            )
        expected = self._finalize_external_protected_checkout_recovery_receipt(
            attempt=attempt,
            lock_path=lock_path,
        )
        if dict(receipt) != expected:
            raise DatabasePortalBridgeError(
                "external checkout recovery receipt changed after finalization"
            )
        return expected

    def recover_external_protected_checkout(self, attempt: Any) -> Mapping[str, Any]:
        """Rearm only after the shared checkout mutation lock is gone.

        This recovery never reads another owner's signed journal.  The
        lock path is derived from the configured repository root; absence
        of that file is the closed proof that the crash-window leftover
        has cleared.  A still-present lock, including a paired supervisor
        journal that has not finished, stays blocked.
        """

        self._record_for_attempt(self.task_source, attempt)
        lock_path = self._checkout_mutation_lock_path()
        try:
            lock_present = lock_path.exists()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "external checkout recovery could not observe the mutation lock"
            ) from exc
        if lock_present:
            raise DatabasePortalBridgeError(
                "external checkout recovery requires the checkout mutation "
                "lock to be absent"
            )
        paths = self._paths(attempt)
        final_path = paths.root / _EXTERNAL_PROTECTED_CHECKOUT_RECOVERY_FILENAME
        if final_path.is_file():
            return self._verify_external_protected_checkout_recovery_receipt(
                attempt=attempt,
                lock_path=lock_path,
                receipt=self._read_json_object(
                    final_path,
                    noun="external checkout recovery receipt",
                ),
            )
        receipt = self._finalize_external_protected_checkout_recovery_receipt(
            attempt=attempt,
            lock_path=lock_path,
        )
        _atomic_write(
            final_path,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
            + b"\n",
        )
        return self._verify_external_protected_checkout_recovery_receipt(
            attempt=attempt,
            lock_path=lock_path,
            receipt=receipt,
        )

    def _observe_live_inflight_implementation(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
    ) -> Mapping[str, Any] | None:
        """Ask the attempt-local Portal executor whether a runner is still live.

        This reuses the Portal daemon's own inflight detector rather than
        inventing a second process schema.  Missing the detector fails closed.
        """

        alias = str(getattr(attempt, "task_alias", "") or attempt.task_cid)
        daemon = self.portal_factory(paths, alias)
        if daemon is None:
            raise DatabasePortalBridgeError(
                "inflight-process recovery portal factory returned no executor"
            )
        try:
            inspect = getattr(daemon, "_find_live_inflight_implementation", None)
            if not callable(inspect):
                raise DatabasePortalBridgeError(
                    "Portal executor has no inflight-process detector"
                )
            observed = inspect()
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(
                daemon, "close", None
            )
            if callable(close):
                close()
        if observed is None:
            return None
        if not isinstance(observed, Mapping):
            raise DatabasePortalBridgeError(
                "inflight-process detector returned a non-object observation"
            )
        return observed

    def _finalize_inflight_process_recovery_receipt(
        self,
        *,
        attempt: Any,
    ) -> dict[str, Any]:
        receipt = {
            "schema": DATABASE_PORTAL_INFLIGHT_PROCESS_RECOVERY_SCHEMA,
            "disposition": "retry",
            "reason": "inflight_process_absent",
            "source_reason": "inflight_process",
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "live_runner_present": False,
            "backoff_seconds": 0,
            "attempt_consumed": False,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def _verify_inflight_process_recovery_receipt(
        self,
        *,
        attempt: Any,
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_fields = {
            "schema",
            "disposition",
            "reason",
            "source_reason",
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "lease_id",
            "attempt_number",
            "fencing_token",
            "fence_epoch",
            "live_runner_present",
            "backoff_seconds",
            "attempt_consumed",
            "receipt_id",
        }
        if set(receipt) != expected_fields:
            raise DatabasePortalBridgeError(
                "inflight-process recovery receipt is malformed or foreign"
            )
        expected = self._finalize_inflight_process_recovery_receipt(attempt=attempt)
        if dict(receipt) != expected:
            raise DatabasePortalBridgeError(
                "inflight-process recovery receipt changed after finalization"
            )
        return expected

    def recover_inflight_process(self, attempt: Any) -> Mapping[str, Any]:
        """Rearm only after this attempt's implementation runner is gone.

        A live worker stays blocked.  Absence is proved by the same Portal
        inflight detector that produced the original skip, bound to this
        attempt's private event stream.
        """

        self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        observed = self._observe_live_inflight_implementation(
            attempt=attempt,
            paths=paths,
        )
        if observed is not None:
            raise DatabasePortalBridgeError(
                "inflight-process recovery requires the implementation "
                "runner to be absent"
            )
        final_path = paths.root / _INFLIGHT_PROCESS_RECOVERY_FILENAME
        if final_path.is_file():
            return self._verify_inflight_process_recovery_receipt(
                attempt=attempt,
                receipt=self._read_json_object(
                    final_path,
                    noun="inflight-process recovery receipt",
                ),
            )
        receipt = self._finalize_inflight_process_recovery_receipt(attempt=attempt)
        _atomic_write(
            final_path,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
            + b"\n",
        )
        return self._verify_inflight_process_recovery_receipt(
            attempt=attempt,
            receipt=receipt,
        )

    def _safe_progressed_ref_name(self, name: str) -> bool:
        """Accept only closed implementation or rescue ref names."""

        if (
            not (
                name.startswith("rescue/")
                or name.startswith("implementation/")
            )
            or ".." in name
            or "@{" in name
            or "\\" in name
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", name)
        ):
            return False
        try:
            checked = subprocess.run(
                ["git", "check-ref-format", f"refs/heads/{name}"],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return checked.returncode == 0

    def _git_commit_object_exists(self, commit: str) -> bool:
        """Prove a claimed commit exists in this repository's object store."""

        if self.repository_root is None:
            return False
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            return False
        try:
            result = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def _validation_retry_seed_state_is_compatible(
        self,
        *,
        current_state: Mapping[str, Any],
        state_seed: Mapping[str, Any],
    ) -> bool:
        """Accept identity-bound Portal progress over a validation-retry seed.

        Portal ``run_once`` mutates attempt counts, returncode, branch, and
        commit after the seed is projected.  Exact equality then terminalizes
        a live resume.  Foreign task identity, malformed counters, unsafe
        ref names, and invented commits stay fail-closed.
        """

        if not isinstance(current_state, Mapping):
            return False
        for key in (
            "last_implementation_task_id",
            "last_implementation_task_key",
            "last_implementation_task_cid",
        ):
            if current_state.get(key) != state_seed.get(key):
                return False
        branch = current_state.get("last_implementation_branch")
        if type(branch) is not str or not self._safe_progressed_ref_name(branch):
            return False
        returncode = current_state.get("last_implementation_returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            return False
        for count_key in (
            "implementation_attempts",
            "implementation_attempts_by_cid",
        ):
            seed_counts = state_seed.get(count_key)
            observed_counts = current_state.get(count_key)
            if not isinstance(seed_counts, Mapping) or not isinstance(
                observed_counts, Mapping
            ):
                return False
            if set(observed_counts) != set(seed_counts):
                return False
            for identity, seed_count in seed_counts.items():
                observed = observed_counts.get(identity)
                if (
                    isinstance(seed_count, bool)
                    or not isinstance(seed_count, int)
                    or isinstance(observed, bool)
                    or not isinstance(observed, int)
                    or observed < seed_count
                ):
                    return False
        commit = current_state.get("last_implementation_commit")
        seed_commit = state_seed.get("last_implementation_commit")
        if type(commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", commit):
            return False
        if commit == seed_commit:
            return True
        return self._git_commit_object_exists(commit)

    def _validation_retry_seed_event(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
    ) -> Mapping[str, Any]:
        """Bind this attempt to its exact durable validation-retry seed event."""

        alias = str(getattr(attempt, "task_alias", "") or "")
        task_cid = str(attempt.task_cid)
        matching = [
            event
            for event in self._verified_event_chain(paths)
            if event.get("type") == "database_portal_validation_retry_seeded"
            and event.get("task_id") == alias
            and event.get("canonical_task_cid") == task_cid
            and str(event.get("target_database_attempt_id") or "")
            == str(attempt.attempt_id)
        ]
        if len(matching) != 1:
            raise DatabasePortalBridgeError(
                "validation-retry seed-conflict recovery has no exact seed event"
            )
        return matching[0]

    def _state_seed_from_validation_retry_seed_event(
        self,
        *,
        attempt: Any,
        seed_event: Mapping[str, Any],
    ) -> dict[str, Any]:
        alias = str(getattr(attempt, "task_alias", "") or "")
        task_cid = str(attempt.task_cid)
        receipt = seed_event.get("validation_retry_receipt")
        source_portal_attempt = (
            receipt.get("portal_attempt") if isinstance(receipt, Mapping) else None
        )
        if (
            seed_event.get("task_id") != alias
            or seed_event.get("canonical_task_cid") != task_cid
            or type(seed_event.get("canonical_task_key") or "") is not str
            or not str(seed_event.get("canonical_task_key") or "")
            or isinstance(source_portal_attempt, bool)
            or not isinstance(source_portal_attempt, int)
            or source_portal_attempt < 1
        ):
            raise DatabasePortalBridgeError(
                "validation-retry seed-conflict recovery seed event is foreign"
            )
        return {
            "implementation_attempts": {alias: source_portal_attempt},
            "implementation_attempts_by_cid": {task_cid: source_portal_attempt},
            "last_implementation_task_id": alias,
            "last_implementation_task_key": str(
                seed_event.get("canonical_task_key") or ""
            ),
            "last_implementation_task_cid": task_cid,
            "last_implementation_returncode": 1,
            "last_implementation_branch": str(
                seed_event.get("rescue_branch") or ""
            ),
            "last_implementation_commit": str(
                seed_event.get("implementation_commit") or ""
            ),
        }

    def _observe_validation_retry_seed_conflict_state(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
    ) -> dict[str, Any]:
        """Prove leftover seed-conflict state is identity-bound Portal progress."""

        if not paths.state.is_file() or not paths.events.is_file():
            raise DatabasePortalBridgeError(
                "validation-retry seed-conflict recovery artifacts are incomplete"
            )
        seed_event = self._validation_retry_seed_event(
            attempt=attempt,
            paths=paths,
        )
        try:
            current_state = json.loads(paths.state.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "Portal retry seed state is unreadable"
            ) from exc
        state_seed = self._state_seed_from_validation_retry_seed_event(
            attempt=attempt,
            seed_event=seed_event,
        )
        if not self._validation_retry_seed_state_is_compatible(
            current_state=current_state,
            state_seed=state_seed,
        ):
            raise DatabasePortalBridgeError(
                "validation-retry seed-conflict recovery requires "
                "identity-bound progressed Portal state"
            )
        return {
            "seed_id": str(seed_event.get("seed_id") or ""),
            "seed_commit": str(seed_event.get("implementation_commit") or ""),
            "seed_rescue_branch": str(seed_event.get("rescue_branch") or ""),
            "observed_commit": str(
                current_state.get("last_implementation_commit") or ""
            ),
            "observed_branch": str(
                current_state.get("last_implementation_branch") or ""
            ),
        }

    def _finalize_validation_retry_seed_conflict_recovery_receipt(
        self,
        *,
        attempt: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        receipt = {
            "schema": (
                DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_SCHEMA
            ),
            "disposition": "retry",
            "reason": "validation_retry_seed_state_progressed",
            "source_reason": DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON,
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "seed_id": str(observation.get("seed_id") or ""),
            "seed_commit": str(observation.get("seed_commit") or ""),
            "seed_rescue_branch": str(observation.get("seed_rescue_branch") or ""),
            "observed_commit": str(observation.get("observed_commit") or ""),
            "observed_branch": str(observation.get("observed_branch") or ""),
            "identity_bound": True,
            "backoff_seconds": 0,
            "attempt_consumed": False,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def _verify_validation_retry_seed_conflict_recovery_receipt(
        self,
        *,
        attempt: Any,
        observation: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_fields = {
            "schema",
            "disposition",
            "reason",
            "source_reason",
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "lease_id",
            "attempt_number",
            "fencing_token",
            "fence_epoch",
            "seed_id",
            "seed_commit",
            "seed_rescue_branch",
            "observed_commit",
            "observed_branch",
            "identity_bound",
            "backoff_seconds",
            "attempt_consumed",
            "receipt_id",
        }
        if set(receipt) != expected_fields:
            raise DatabasePortalBridgeError(
                "validation-retry seed-conflict recovery receipt is malformed "
                "or foreign"
            )
        expected = self._finalize_validation_retry_seed_conflict_recovery_receipt(
            attempt=attempt,
            observation=observation,
        )
        if dict(receipt) != expected:
            raise DatabasePortalBridgeError(
                "validation-retry seed-conflict recovery receipt changed after "
                "finalization"
            )
        return expected

    def recover_validation_retry_seed_conflict(
        self, attempt: Any
    ) -> Mapping[str, Any]:
        """Rearm only identity-bound portal-progressed validation-retry state.

        The leftover block is an exact-equality false alarm: Portal advanced
        the private attempt state after the seed was projected.  Foreign
        identity, a missing seed event, or a commit absent from this
        repository stays blocked.
        """

        self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        observation = self._observe_validation_retry_seed_conflict_state(
            attempt=attempt,
            paths=paths,
        )
        final_path = paths.root / _VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_FILENAME
        if final_path.is_file():
            return self._verify_validation_retry_seed_conflict_recovery_receipt(
                attempt=attempt,
                observation=observation,
                receipt=self._read_json_object(
                    final_path,
                    noun="validation-retry seed-conflict recovery receipt",
                ),
            )
        receipt = self._finalize_validation_retry_seed_conflict_recovery_receipt(
            attempt=attempt,
            observation=observation,
        )
        _atomic_write(
            final_path,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
            + b"\n",
        )
        return self._verify_validation_retry_seed_conflict_recovery_receipt(
            attempt=attempt,
            observation=observation,
            receipt=receipt,
        )

    def _observe_pooled_worktree_create_failure(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
    ) -> dict[str, Any]:
        """Prove the leftover terminal block was a pre-dispatch worktree add."""

        if not paths.events.is_file():
            raise DatabasePortalBridgeError(
                "pooled-worktree create recovery artifacts are incomplete"
            )
        alias = str(getattr(attempt, "task_alias", "") or "")
        task_cid = str(attempt.task_cid)
        finished = [
            event
            for event in self._verified_event_chain(paths)
            if event.get("type") == "implementation_finished"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or event.get("task_cid") or "")
            in {"", task_cid}
        ]
        if not finished:
            raise DatabasePortalBridgeError(
                "pooled-worktree create recovery has no implementation_finished event"
            )
        last = finished[-1]
        exception = last.get("exception_result")
        if not isinstance(exception, Mapping):
            raise DatabasePortalBridgeError(
                "pooled-worktree create recovery is not a worktree-setup failure"
            )
        message = str(exception.get("message") or "")
        worktree_path = str(
            last.get("worktree_path") or exception.get("worktree_path") or ""
        )
        if (
            last.get("provider_dispatched") is not False
            or str(exception.get("phase") or "") != "worktree_setup"
            or not message.startswith(_POOLED_WORKTREE_CREATE_FAILURE_PREFIX)
        ):
            raise DatabasePortalBridgeError(
                "pooled-worktree create recovery requires a pre-dispatch "
                "worktree-setup failure"
            )
        if worktree_path:
            candidate = Path(worktree_path)
            if candidate.exists():
                raise DatabasePortalBridgeError(
                    "pooled-worktree create recovery requires the leftover "
                    "worktree path to be absent"
                )
        return {
            "worktree_path": worktree_path,
            "worktree_present": False,
            "exception_type": str(exception.get("exception_type") or ""),
            "phase": "worktree_setup",
        }

    def _finalize_pooled_worktree_create_recovery_receipt(
        self,
        *,
        attempt: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        receipt = {
            "schema": DATABASE_PORTAL_POOLED_WORKTREE_CREATE_RECOVERY_SCHEMA,
            "disposition": "retry",
            "reason": DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON,
            "source_reason": DATABASE_PORTAL_POOLED_WORKTREE_CREATE_SOURCE_REASON,
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "worktree_path": str(observation.get("worktree_path") or ""),
            "worktree_present": False,
            "identity_bound": True,
            "backoff_seconds": 0,
            "attempt_consumed": False,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def _verify_pooled_worktree_create_recovery_receipt(
        self,
        *,
        attempt: Any,
        observation: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_fields = {
            "schema",
            "disposition",
            "reason",
            "source_reason",
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "lease_id",
            "attempt_number",
            "fencing_token",
            "fence_epoch",
            "worktree_path",
            "worktree_present",
            "identity_bound",
            "backoff_seconds",
            "attempt_consumed",
            "receipt_id",
        }
        if set(receipt) != expected_fields:
            raise DatabasePortalBridgeError(
                "pooled-worktree create recovery receipt is malformed or foreign"
            )
        expected = self._finalize_pooled_worktree_create_recovery_receipt(
            attempt=attempt,
            observation=observation,
        )
        if dict(receipt) != expected:
            raise DatabasePortalBridgeError(
                "pooled-worktree create recovery receipt changed after finalization"
            )
        return expected

    def recover_pooled_worktree_create(self, attempt: Any) -> Mapping[str, Any]:
        """Rearm a leftover pooled-worktree create interrupt.

        The historical failed phase stays immutable.  Absence of the leftover
        checkout path plus the attempt-local worktree-setup exception is the
        closed proof that the crash-window leftover cleared.
        """

        self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        observation = self._observe_pooled_worktree_create_failure(
            attempt=attempt,
            paths=paths,
        )
        final_path = paths.root / _POOLED_WORKTREE_CREATE_RECOVERY_FILENAME
        if final_path.is_file():
            return self._verify_pooled_worktree_create_recovery_receipt(
                attempt=attempt,
                observation=observation,
                receipt=self._read_json_object(
                    final_path,
                    noun="pooled-worktree create recovery receipt",
                ),
            )
        receipt = self._finalize_pooled_worktree_create_recovery_receipt(
            attempt=attempt,
            observation=observation,
        )
        _atomic_write(
            final_path,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
            + b"\n",
        )
        return self._verify_pooled_worktree_create_recovery_receipt(
            attempt=attempt,
            observation=observation,
            receipt=receipt,
        )

    def _preserved_commit_exists(
        self,
        *,
        commit: str,
        rescue_branch: str,
    ) -> bool:
        """Independently bind the claimed rescue ref to the preserved commit."""

        if self.repository_root is None:
            return False
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            return False
        if (
            not rescue_branch.startswith("rescue/")
            or ".." in rescue_branch
            or "@{" in rescue_branch
            or "\\" in rescue_branch
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", rescue_branch)
        ):
            return False
        try:
            checked = subprocess.run(
                ["git", "check-ref-format", f"refs/heads/{rescue_branch}"],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            resolved = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"refs/heads/{rescue_branch}^{{commit}}",
                ],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return (
            checked.returncode == 0
            and resolved.returncode == 0
            and resolved.stdout.strip() == commit
        )

    def _validation_retry_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        implementation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Reproduce the one post-dispatch failure class eligible to retry."""

        if self.max_task_attempts <= 0:
            return None
        attempt_number = getattr(attempt, "attempt_number", 0)
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            return None
        if implementation is not None and not self._looks_like_validation_retry(
            implementation
        ):
            return None

        events = self._verified_event_chain(paths)
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        matching_finished = [
            (index, event)
            for index, event in enumerate(events)
            if event.get("type") == "implementation_finished"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or event.get("task_cid") or "")
            == task_cid
        ]
        if not matching_finished:
            return None
        finished_index, finished = matching_finished[-1]
        if not self._looks_like_validation_retry(finished):
            return None
        portal_attempt = finished.get("attempt")
        # The outer coordination attempt number is a monotone database fence,
        # not this current-schema retry budget.  Legacy attempts can therefore
        # legitimately make it much larger than max_task_attempts.  Portal's
        # independently replayed per-task attempt counter is the bounded
        # generation and is carried into the next private attempt state.
        if (
            isinstance(portal_attempt, bool)
            or not isinstance(portal_attempt, int)
            or portal_attempt < 1
            or portal_attempt >= self.max_task_attempts
        ):
            return None
        if implementation is not None:
            for field in (
                "attempt",
                "returncode",
                "attempt_consumed",
                "provider_dispatched",
                "implementation_commit",
            ):
                if implementation.get(field) != finished.get(field):
                    return None

        validation = finished.get("validation_result")
        assert isinstance(validation, Mapping)
        proposal_gate = validation.get("proposal_gate")
        review = validation.get("failure_review")
        dag = validation.get("validation_dag_receipt")
        preservation = finished.get("failed_preservation_result")
        if not all(
            isinstance(value, Mapping)
            for value in (proposal_gate, review, dag, preservation)
        ):
            return None
        assert isinstance(proposal_gate, Mapping)
        assert isinstance(review, Mapping)
        assert isinstance(dag, Mapping)
        assert isinstance(preservation, Mapping)
        board_completion = finished.get("board_completion")
        merge_result = finished.get("merge_result")
        preservation_commit = preservation.get("commit_result")
        if not all(
            isinstance(value, Mapping)
            for value in (board_completion, merge_result, preservation_commit)
        ):
            return None
        assert isinstance(board_completion, Mapping)
        assert isinstance(merge_result, Mapping)
        assert isinstance(preservation_commit, Mapping)

        returncode = validation.get("returncode")
        coverage_errors = validation.get("coverage_errors")
        reason_codes = review.get("reason_codes")
        nodes = dag.get("nodes")
        if (
            isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode == 0
            or validation.get("auto_rescue_terminal") is not True
            or validation.get("completion_authoritative") is not False
            or validation.get("merge_eligible") is not False
            or coverage_errors not in ([], ())
            or proposal_gate.get("attempted") is not True
            or proposal_gate.get("accepted") is not True
            or proposal_gate.get("reason_codes") not in ([], ())
            or review.get("decision") != "guide_rescue"
            or set(str(item) for item in (reason_codes or ()))
            != {"validation_command_failed"}
            or review.get("denied_paths") not in ([], ())
            or review.get("out_of_scope_paths") not in ([], ())
            or review.get("contract_gap_paths") not in ([], ())
            or review.get("missing_expected_outputs") not in ([], ())
            or review.get("justified_paths") not in ([], ())
            or dag.get("passed") is not False
            or dag.get("coverage_complete") is not True
            or dag.get("uncovered_impact") is not False
            or not isinstance(nodes, Sequence)
            or isinstance(nodes, (str, bytes, bytearray, memoryview))
            or not any(
                isinstance(node, Mapping)
                and node.get("mandatory") is True
                and node.get("selected") is True
                and node.get("disposition") == "failed"
                and isinstance(node.get("returncode"), int)
                and not isinstance(node.get("returncode"), bool)
                and int(node.get("returncode")) != 0
                and bool(str(node.get("result_digest") or ""))
                for node in nodes
            )
            or finished.get("protected_path_violation") is not None
            or board_completion.get("complete") is True
            or merge_result.get("merged") is True
            or merge_result.get("queued") is True
        ):
            return None
        for container in (finished, validation, proposal_gate, review):
            for field in (
                "denied_effects",
                "forbidden_effects",
                "unauthorized_effects",
            ):
                if container.get(field) not in (None, [], ()):
                    return None

        proposal_id = str(proposal_gate.get("proposal_id") or "")
        proposal_receipt_id = str(proposal_gate.get("receipt_id") or "")
        proposal_policy_id = str(proposal_gate.get("policy_id") or "")
        validation_receipt_id = str(dag.get("receipt_id") or "")
        review_receipt_id = str(review.get("receipt_id") or "")
        commit = str(finished.get("implementation_commit") or "")
        preserved_commit = str(preservation.get("preserved_commit") or "")
        rescue_branch = str(preservation.get("rescue_branch") or "")
        changed_paths = tuple(str(item) for item in (proposal_gate.get("changed_paths") or ()))
        if (
            not all(
                (
                    proposal_id,
                    proposal_receipt_id,
                    proposal_policy_id,
                    validation_receipt_id,
                    review_receipt_id,
                )
            )
            or not changed_paths
            or len(set(changed_paths)) != len(changed_paths)
            or dag.get("proposal_receipt_id") != proposal_receipt_id
            or dag.get("objective_id") != task_cid
            or tuple(dag.get("changed_paths") or ()) != changed_paths
            or preservation.get("preserved") is not True
            or preservation.get("implementation_commit") != commit
            or preserved_commit != commit
            or preservation_commit.get("committed") is not True
            or preservation_commit.get("commit") != commit
            or not self._preserved_commit_exists(
                commit=commit,
                rescue_branch=rescue_branch,
            )
        ):
            return None

        preservation_matches = [
            (index, event)
            for index, event in enumerate(events[:finished_index])
            if event.get("type") == "failed_validation_worktree_preserved"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or "") == task_cid
            and event.get("attempt") == finished.get("attempt")
            and event.get("preserved") is True
            and event.get("implementation_commit") == commit
            and event.get("preserved_commit") == commit
            and event.get("rescue_branch") == rescue_branch
        ]
        if not preservation_matches:
            return None
        preservation_index, preservation_event = preservation_matches[-1]
        proposal_matches = [
            (index, event)
            for index, event in enumerate(events[:preservation_index])
            if event.get("type") == "implementation_proposal_validated"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or "") == task_cid
            and event.get("attempted") is True
            and event.get("accepted") is True
            and event.get("reason_codes") in ([], ())
            and event.get("proposal_id") == proposal_id
            and event.get("receipt_id") == proposal_receipt_id
            and event.get("policy_id") == proposal_policy_id
            and tuple(event.get("changed_paths") or ()) == changed_paths
        ]
        if not proposal_matches:
            return None
        proposal_index, proposal_event = proposal_matches[-1]
        output_matches = [
            (index, event)
            for index, event in enumerate(events[:proposal_index])
            if event.get("type") == "implementation_expected_outputs_checked"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or "") == task_cid
            and event.get("proposal_id") == proposal_id
            and event.get("passed") is True
            and event.get("issues") in ([], ())
            and tuple(event.get("expected_paths") or ()) == changed_paths
            and tuple(event.get("staged_paths") or ()) == changed_paths
            and event.get("force_staged_paths") in ([], ())
        ]
        if not output_matches:
            return None
        _output_index, output_event = output_matches[-1]

        receipt = {
            "schema": DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
            "disposition": "retry",
            "reason": "declared_validation_failed",
            "task_cid": task_cid,
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "portal_attempt": int(portal_attempt),
            "typed_retry_generation": int(portal_attempt),
            "retry_budget_basis": "portal_attempt",
            "legacy_database_attempts_excluded": True,
            "max_task_attempts": int(self.max_task_attempts),
            "remaining_task_attempts": int(
                self.max_task_attempts - portal_attempt
            ),
            "attempt_consumed": True,
            "provider_dispatched": True,
            "backoff_seconds": 0,
            "implementation_commit": commit,
            "rescue_branch": rescue_branch,
            "binding_id": str(binding.get("binding_id") or ""),
            "events_digest": _sha256_file(paths.events),
            "event_stream_id": str(finished.get("stream_id") or ""),
            "expected_output_event_id": str(output_event.get("event_id") or ""),
            "proposal_event_id": str(proposal_event.get("event_id") or ""),
            "preservation_event_id": str(preservation_event.get("event_id") or ""),
            "implementation_event_id": str(finished.get("event_id") or ""),
            "proposal_id": proposal_id,
            "proposal_receipt_id": proposal_receipt_id,
            "proposal_policy_id": proposal_policy_id,
            "validation_receipt_id": validation_receipt_id,
            "failure_review_receipt_id": review_receipt_id,
            "changed_paths": list(changed_paths),
            "authoritative_validation_executed": True,
            "proposal_policy_accepted": True,
            "output_policy_passed": True,
            "denial_findings": [],
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def recover_validation_retry(self, attempt: Any) -> Mapping[str, Any]:
        """Reproduce retry evidence for a previously terminalized attempt.

        This reads only the attempt-local projection and immutable event
        evidence.  It does not update a task, claim, queue, or execution row.
        """

        record = self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        if not (
            paths.binding.is_file()
            and paths.task_projection.is_file()
            and paths.events.is_file()
        ):
            raise DatabasePortalBridgeError(
                "validation retry recovery artifacts are incomplete"
            )
        seed = self._render_projection(attempt, record)
        expected_binding = self._binding(attempt, record, seed)
        observed_binding = self._read_binding(paths.binding)
        observed_body = dict(observed_binding)
        observed_binding_id = str(observed_body.pop("binding_id", "") or "")
        observed_revision = observed_body.get("task_revision")
        current_revision = int(getattr(record, "revision", 0) or 0)
        # Control status transitions legitimately advance the task revision
        # after this immutable attempt binding was written.  All semantic
        # body and claim fields remain exact; only a positive historical task
        # revision not newer than the current control record is accepted.
        stable_expected = {
            key: value
            for key, value in expected_binding.items()
            if key
            not in {
                "binding_id",
                "task_revision",
                "task_body_digest",
                "projection_seed_digest",
                "projection_immutable_digest",
            }
        }
        stable_observed = {
            key: value
            for key, value in observed_binding.items()
            if key
            not in {
                "binding_id",
                "task_revision",
                "task_body_digest",
                "projection_seed_digest",
                "projection_immutable_digest",
            }
        }
        observed_projection = self._verify_projection(paths, observed_binding)
        if (
            observed_binding_id
            != _sha256_bytes(_canonical_json(observed_body))
            or isinstance(observed_revision, bool)
            or not isinstance(observed_revision, int)
            or observed_revision < 1
            or current_revision < observed_revision
            or stable_observed != stable_expected
            or _projection_recovery_digest(observed_projection)
            != _projection_recovery_digest(seed)
        ):
            raise DatabasePortalBridgeError(
                "validation retry recovery binding does not match the claim"
            )
        receipt = self._validation_retry_receipt(
            attempt=attempt,
            paths=paths,
            binding=observed_binding,
        )
        if receipt is None:
            raise DatabasePortalBridgeError(
                "attempt is not eligible for typed validation retry recovery"
            )
        return receipt

    def _validation_retry_seed_from_record(
        self,
        *,
        attempt: Any,
        record: Any,
    ) -> dict[str, Any] | None:
        """Recover the prior retry receipt carried through the claim CAS."""

        body = dict(getattr(record, "body", {}) or {})
        status_receipt = body.get("completion_receipt")
        if not isinstance(status_receipt, Mapping):
            return None
        seed = status_receipt.get("validation_retry_seed")
        if seed is None:
            return None
        if (
            status_receipt.get("operation") != "database_claim"
            or status_receipt.get("attempt_id") != str(attempt.attempt_id)
            or status_receipt.get("claim_id") != str(attempt.claim_id)
            or status_receipt.get("attempt_number")
            != int(attempt.attempt_number)
            or status_receipt.get("fencing_token")
            != int(attempt.fencing_token)
            or status_receipt.get("fence_epoch") != int(attempt.fence_epoch)
            or status_receipt.get("lease_id")
            != str(getattr(attempt, "lease_id", "") or "")
            or not isinstance(seed, Mapping)
        ):
            raise DatabasePortalBridgeError(
                "database claim carries a malformed validation retry seed"
            )
        value = dict(seed)
        receipt_id = value.pop("receipt_id", None)
        changed_paths = seed.get("changed_paths")
        source_attempt_number = seed.get("attempt_number")
        target_attempt_number = getattr(attempt, "attempt_number", 0)
        source_portal_attempt = seed.get("portal_attempt")
        scoped_outputs = self._scope_outputs(
            _output_values(record, body),
            self._validation_repository_scope(body),
        )
        if (
            seed.get("schema") != DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA
            or seed.get("disposition") != "retry"
            or seed.get("reason") != "declared_validation_failed"
            or seed.get("task_cid") != str(attempt.task_cid)
            or seed.get("task_alias")
            != str(getattr(attempt, "task_alias", "") or "")
            or seed.get("attempt_consumed") is not True
            or seed.get("provider_dispatched") is not True
            or seed.get("proposal_policy_accepted") is not True
            or seed.get("output_policy_passed") is not True
            or seed.get("denial_findings") != []
            or seed.get("max_task_attempts") != self.max_task_attempts
            or isinstance(source_attempt_number, bool)
            or not isinstance(source_attempt_number, int)
            or isinstance(target_attempt_number, bool)
            or not isinstance(target_attempt_number, int)
            or target_attempt_number < 1
            or target_attempt_number <= source_attempt_number
            or str(seed.get("attempt_id") or "")
            == str(attempt.attempt_id)
            or status_receipt.get("validation_retry_source_attempt_id")
            != str(seed.get("attempt_id") or "")
            or isinstance(source_portal_attempt, bool)
            or not isinstance(source_portal_attempt, int)
            or source_portal_attempt < 1
            or source_portal_attempt >= self.max_task_attempts
            or seed.get("typed_retry_generation") != source_portal_attempt
            or seed.get("retry_budget_basis") != "portal_attempt"
            or seed.get("legacy_database_attempts_excluded") is not True
            or seed.get("remaining_task_attempts")
            != self.max_task_attempts - source_portal_attempt
            or not isinstance(changed_paths, list)
            or changed_paths != scoped_outputs
            or receipt_id != _sha256_bytes(_canonical_json(value))
            or not self._preserved_commit_exists(
                commit=str(seed.get("implementation_commit") or ""),
                rescue_branch=str(seed.get("rescue_branch") or ""),
            )
        ):
            raise DatabasePortalBridgeError(
                "database claim validation retry seed failed verification"
            )
        return dict(seed)

    def _initialize_validation_retry_seed(
        self,
        *,
        attempt: Any,
        record: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Project a checked prior candidate into the new private Portal state."""

        seed = self._validation_retry_seed_from_record(
            attempt=attempt,
            record=record,
        )
        if seed is None:
            return None
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        _canonical_task_key, canonical_task_cid = _canonical_projection_identity(
            record,
            dict(getattr(record, "body", {}) or {}),
        )
        if canonical_task_cid != task_cid:
            raise DatabasePortalBridgeError(
                "validation retry seed task identity changed"
            )
        source_portal_attempt = seed.get("portal_attempt")
        if (
            isinstance(source_portal_attempt, bool)
            or not isinstance(source_portal_attempt, int)
            or source_portal_attempt < 1
        ):
            raise DatabasePortalBridgeError(
                "validation retry seed has no exact Portal attempt"
            )
        seed_body = {
            "schema": DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA,
            "task_id": alias,
            "canonical_task_key": _canonical_task_key,
            "canonical_task_cid": task_cid,
            "source_database_attempt_id": str(seed.get("attempt_id") or ""),
            "target_database_attempt_id": str(attempt.attempt_id),
            "target_claim_id": str(attempt.claim_id),
            "source_retry_receipt_id": str(seed.get("receipt_id") or ""),
            "implementation_commit": str(seed.get("implementation_commit") or ""),
            "rescue_branch": str(seed.get("rescue_branch") or ""),
            "changed_paths": list(seed.get("changed_paths") or ()),
            "validation_retry_receipt": dict(seed),
            "completion_authoritative": False,
        }
        seed_body["seed_id"] = _sha256_bytes(_canonical_json(seed_body))

        existing_seed_event: Mapping[str, Any] | None = None
        if paths.events.exists():
            for event in self._verified_event_chain(paths):
                if (
                    event.get("type")
                    == "database_portal_validation_retry_seeded"
                    and event.get("seed_id") == seed_body["seed_id"]
                ):
                    existing_seed_event = event
                    break
            if existing_seed_event is None:
                raise DatabasePortalBridgeError(
                    "Portal attempt event stream predates its required retry seed"
                )
        else:
            existing_seed_event = append_jsonl_event(
                paths.events,
                "database_portal_validation_retry_seeded",
                seed_body,
            )

        state_seed = {
            "implementation_attempts": {alias: source_portal_attempt},
            "implementation_attempts_by_cid": {
                task_cid: source_portal_attempt,
            },
            "last_implementation_task_id": alias,
            "last_implementation_task_key": _canonical_task_key,
            "last_implementation_task_cid": task_cid,
            "last_implementation_returncode": 1,
            "last_implementation_branch": str(seed.get("rescue_branch") or ""),
            "last_implementation_commit": str(
                seed.get("implementation_commit") or ""
            ),
        }
        if paths.state.exists():
            try:
                current_state = json.loads(paths.state.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeError(
                    "Portal retry seed state is unreadable"
                ) from exc
            exact_seed = isinstance(current_state, Mapping) and all(
                current_state.get(key) == value
                for key, value in state_seed.items()
            )
            if not exact_seed and not self._validation_retry_seed_state_is_compatible(
                current_state=current_state,
                state_seed=state_seed,
            ):
                raise DatabasePortalBridgeError(
                    DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON
                )
        else:
            _atomic_write(
                paths.state,
                json.dumps(state_seed, indent=2, sort_keys=True).encode("utf-8")
                + b"\n",
            )
        return {
            "seed_id": str(seed_body["seed_id"]),
            "seed_event_id": str(existing_seed_event.get("event_id") or ""),
            "source_retry_receipt_id": str(seed.get("receipt_id") or ""),
            "implementation_commit": str(seed.get("implementation_commit") or ""),
            "rescue_branch": str(seed.get("rescue_branch") or ""),
            "portal_attempt": source_portal_attempt,
        }

    def _acceptance_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        alias = str(binding.get("task_alias") or "")
        projection_text = self._verify_projection(paths, binding)
        if _projection_status(projection_text) not in _TERMINAL_STATUSES:
            raise DatabasePortalBridgeDeferred("Portal task projection is not complete")
        if not self._has_completion_event(
            paths,
            alias,
            str(binding.get("canonical_task_key") or ""),
            str(binding.get("task_cid") or ""),
        ):
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
        evidence_digest = _sha256_bytes(_canonical_json(evidence))
        receipt = {
            "schema": self.RECEIPT_SCHEMA,
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
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def _execution_route_binding(
        self,
        *,
        attempt: Any,
        record: Any,
    ) -> dict[str, Any]:
        """Recover and validate the route fixed before the shared claim CAS."""

        body = getattr(attempt, "body", None)
        raw = (
            body.get("execution_route_binding")
            if isinstance(body, Mapping)
            else None
        )
        policy = getattr(self.task_source, "execution_route_policy", None)
        validate = getattr(
            self.task_source,
            "validate_execution_route_binding",
            None,
        )
        if raw is None and policy is None:
            return {}
        if not isinstance(raw, Mapping) or not callable(validate):
            raise DatabasePortalBridgeError(
                "database attempt has no valid launch execution-route binding"
            )
        try:
            validated = validate(
                raw,
                task=record,
                allow_claim_revision=True,
            )
        except Exception as exc:
            raise DatabasePortalBridgeError(
                "database attempt execution-route binding is no longer authoritative"
            ) from exc
        if not isinstance(validated, Mapping) or not validated:
            raise DatabasePortalBridgeError(
                "database attempt execution-route validation returned no binding"
            )
        return dict(validated)

    def run_provider(self, attempt: Any) -> Mapping[str, Any]:
        """Run bounded real Portal passes and return only accepted evidence."""

        record = self._record_for_attempt(self.task_source, attempt)
        execution_route_binding = self._execution_route_binding(
            attempt=attempt,
            record=record,
        )
        paths, binding = self._ensure_attempt_projection(attempt, record)
        self._initialize_validation_retry_seed(
            attempt=attempt,
            record=record,
            paths=paths,
            binding=binding,
        )
        summaries: list[Mapping[str, Any]] = []
        daemon = self.portal_factory(
            paths,
            str(binding.get("task_alias") or attempt.task_cid),
        )
        if daemon is None or not callable(getattr(daemon, "run_once", None)):
            raise DatabasePortalBridgeError(
                "portal_factory did not return a Portal-compatible daemon"
            )
        if execution_route_binding:
            bind_route = getattr(
                daemon,
                "bind_launch_task_execution_route",
                None,
            )
            if not callable(bind_route):
                raise DatabasePortalBridgeError(
                    "Portal daemon cannot bind the admitted task execution route"
                )
            bind_route(execution_route_binding)
        try:
            for _pass_index in range(self.max_passes):
                projection = self._verify_projection(paths, binding)
                if _projection_status(
                    projection
                ) in _TERMINAL_STATUSES and self._has_completion_event(
                    paths,
                    str(binding.get("task_alias") or ""),
                    str(binding.get("canonical_task_key") or ""),
                    str(binding.get("task_cid") or ""),
                ):
                    return self._acceptance_receipt(
                        attempt=attempt,
                        paths=paths,
                        binding=binding,
                        summaries=summaries,
                    )
                raw_result = daemon.run_once()
                if not isinstance(raw_result, Mapping):
                    raise DatabasePortalBridgeError("Portal daemon returned a non-object result")
                summary = _bounded_portal_result(raw_result)
                summaries.append(summary)
                self._verify_projection(paths, binding)
                deferral = self._typed_deferral(raw_result)
                if deferral is not None:
                    reason, backoff_seconds = deferral
                    raise DatabasePortalBridgeDeferred(
                        reason,
                        backoff_seconds=backoff_seconds,
                    )
                external_deferral = self._external_protected_checkout_deferral(
                    raw_result
                )
                if external_deferral is not None:
                    reason, backoff_seconds = external_deferral
                    raise DatabasePortalBridgeDeferred(
                        reason,
                        backoff_seconds=backoff_seconds,
                    )
                inflight_deferral = self._inflight_process_deferral(raw_result)
                if inflight_deferral is not None:
                    reason, backoff_seconds = inflight_deferral
                    raise DatabasePortalBridgeDeferred(
                        reason,
                        backoff_seconds=backoff_seconds,
                    )
                lifecycle_deferral = self._worktree_lifecycle_claim_deferral(
                    raw_result
                )
                if lifecycle_deferral is not None:
                    reason, backoff_seconds = lifecycle_deferral
                    raise DatabasePortalBridgeDeferred(
                        reason,
                        backoff_seconds=backoff_seconds,
                    )
                pooled_worktree_deferral = self._pooled_worktree_create_deferral(
                    raw_result
                )
                if pooled_worktree_deferral is not None:
                    reason, backoff_seconds = pooled_worktree_deferral
                    raise DatabasePortalBridgeDeferred(
                        reason,
                        backoff_seconds=backoff_seconds,
                    )
                implementation = raw_result.get("implementation_result")
                if (
                    isinstance(implementation, Mapping)
                    and self._looks_like_validation_retry(implementation)
                ):
                    retry_receipt = self._validation_retry_receipt(
                        attempt=attempt,
                        paths=paths,
                        binding=binding,
                        implementation=implementation,
                    )
                    if retry_receipt is not None:
                        raise DatabasePortalValidationRetry(retry_receipt)
                if isinstance(implementation, Mapping):
                    candidate_reason = self._candidate_retry_reason(implementation)
                    portal_attempt = implementation.get("attempt")
                    durable_attempt = getattr(attempt, "attempt_number", 0)
                    local_attempt = (
                        portal_attempt
                        if type(portal_attempt) is int
                        else 0
                    )
                    bounded_attempt = max(
                        durable_attempt if type(durable_attempt) is int else 0,
                        local_attempt,
                    )
                    # Portal-local attempt counters reset on every database
                    # claim. Bound retries with the durable claim number so
                    # empty Codex candidates cannot spin forever at attempt 1.
                    if (
                        candidate_reason
                        and self.max_task_attempts > 0
                        and 1 <= bounded_attempt < self.max_task_attempts
                    ):
                        raise DatabasePortalCandidateRetry(candidate_reason)
                failure = self._terminal_failure(raw_result)
                if failure:
                    implementation = raw_result.get("implementation_result")
                    if isinstance(
                        implementation, Mapping
                    ) and self._explicit_retryable_deferral(implementation):
                        raise DatabasePortalBridgeDeferred(failure)
                    if (
                        raw_result.get("blocked") is True
                        and is_protected_checkout_setup_block(failure)
                        and failure
                        != "external_protected_checkout_recovery_required"
                    ):
                        # A leftover supervisor/daemon recovery journal is
                        # setup contention, not a dispatched provider outcome.
                        # External recovery is narrower: only the paired
                        # implementation supervisor may grant wait authority,
                        # and that exact owner was handled above.  A foreign
                        # journal remains a terminal ownership conflict.
                        raise DatabasePortalBridgeDeferred(failure)
                    if isinstance(implementation, Mapping):
                        consumed_no_progress = (
                            self._consumed_no_progress_failure(
                                paths,
                                binding,
                                implementation,
                            )
                        )
                        if consumed_no_progress is not None:
                            raise DatabasePortalBridgeConsumedNoProgressError(
                                "portal_consumed_no_progress",
                                failure_evidence=consumed_no_progress,
                            )
                    raise DatabasePortalBridgeError(failure)
            return self._acceptance_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                summaries=summaries,
            )
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(daemon, "close", None)
            if callable(close):
                close()

    def recover_post_merge_declared_outputs(
        self,
        database_daemon: Any,
    ) -> Mapping[str, Any] | None:
        """Repair one owned quarantine and rearm its exact database task.

        This is not an ordinary merge-queue consumer.  It can see only a
        missing-output quarantine whose request points back to this lane's
        immutable database-attempt projection.  A completed row is replayed
        first so a crash between queue settlement and the DuckDB status CAS is
        idempotently recoverable without invoking the merge callback again.
        """

        if self.merge_queue is None:
            return None
        digest = getattr(
            database_daemon,
            "_database_portal_evidence_digest",
            None,
        )
        recover = getattr(
            database_daemon,
            "recover_blocked_post_merge_declared_outputs",
            None,
        )
        preauthorize = getattr(
            database_daemon,
            "preauthorize_post_merge_declared_output_recovery",
            None,
        )
        if (
            not callable(digest)
            or not callable(recover)
            or not callable(preauthorize)
        ):
            raise DatabasePortalBridgeError(
                "database daemon lacks post-merge recovery authority"
            )
        # A completed queue row can outlive the database attempt that created
        # it.  Only an exact latest-attempt conflict is a stale-row signal;
        # malformed authority evidence must still fail the maintenance tick.
        from ..merge.merge_train import MergeTrain

        train = MergeTrain(
            repo_root=self.repository_root,
            queue=self.merge_queue,
            target_branch=self.merge_target_branch,
            max_attempts=int(
                getattr(self.merge_queue, "max_attempts", 3)
            ),
        )

        # Queue completion precedes the database CAS.  Replay at most one
        # keyset page per stage and tick.  Durable, target-and-lane-bound
        # non-authoritative cursors let reconstructed --once bridges continue
        # without an unbounded scan under the canonical consumer lease.
        cursors = self._load_post_merge_recovery_cursors()

        def completion_recovery_page(cursor: str) -> Sequence[Any]:
            return self.merge_queue.completed_requests(
                limit=_POST_MERGE_RECOVERY_SCAN_LIMIT,
                completion_schema=(
                    _POST_MERGE_DECLARED_OUTPUT_COMPLETION_SCHEMA
                ),
                completion_reason=(
                    "post_merge_declared_outputs_repaired"
                ),
                before_request_id=cursor,
            )

        completion_page = completion_recovery_page(
            cursors["completed_requests"]
        )

        def replay_completed_page() -> Mapping[str, Any] | None:
            for snapshot in completion_page:
                snapshot_projection = (
                    self._owned_post_merge_recovery_projection(snapshot)
                )
                if snapshot_projection is None:
                    continue
                completed = self.merge_queue.get(
                    str(getattr(snapshot, "request_id", "") or "")
                )
                projection = (
                    self._owned_post_merge_recovery_projection(completed)
                    if completed is not None
                    else None
                )
                if projection is None:
                    continue
                try:
                    self._preauthorize_post_merge_recovery(
                        completed,
                        projection,
                        preauthorize=preauthorize,
                        evidence_digest=digest,
                    )
                except Exception as exc:
                    if not _is_implementation_conflict(exc):
                        raise
                    continue
                evidence = self._post_merge_recovery_evidence(
                    completed,
                    projection,
                    evidence_digest=digest,
                )
                if evidence is None:
                    continue
                try:
                    result = recover(evidence)
                except Exception as exc:
                    if not _is_implementation_conflict(exc):
                        raise
                    continue
                if not isinstance(result, Mapping):
                    raise DatabasePortalBridgeError(
                        "database post-merge recovery returned a non-object"
                    )
                return dict(result)
            return None

        if completion_page:
            acquired, replay_result = train.run_under_consumer_lease(
                replay_completed_page
            )
            if not acquired:
                return None
            self._advance_post_merge_recovery_cursor(
                cursors,
                "completed_requests",
                completion_page,
            )
            if replay_result is not None:
                return dict(replay_result)
        else:
            self._advance_post_merge_recovery_cursor(
                cursors,
                "completed_requests",
                (),
            )

        selected: Any = None
        selected_projection: _DatabasePortalRecoveryProjection | None = None
        for snapshot_name in (
            "pending_requests",
            "quarantined_requests",
            "processing_requests",
        ):
            snapshot = getattr(self.merge_queue, snapshot_name)
            page = snapshot(
                limit=_POST_MERGE_RECOVERY_SCAN_LIMIT,
                after_request_id=cursors[snapshot_name],
            )
            owned_conflict = False
            for request in page:
                projection = self._owned_post_merge_recovery_projection(
                    request
                )
                if projection is None:
                    continue
                try:
                    self._preauthorize_post_merge_recovery(
                        request,
                        projection,
                        preauthorize=preauthorize,
                        evidence_digest=digest,
                    )
                except Exception as exc:
                    if not _is_implementation_conflict(exc):
                        raise
                    owned_conflict = True
                    _LOG.warning(
                        "post-merge recovery preauthorization conflict "
                        "request_id=%s task_id=%s: %s",
                        getattr(request, "request_id", ""),
                        getattr(request, "task_id", ""),
                        exc,
                    )
                    continue
                selected = request
                selected_projection = projection
                break
            if selected is None and not owned_conflict:
                self._advance_post_merge_recovery_cursor(
                    cursors,
                    snapshot_name,
                    page,
                )
            if selected is not None:
                break
        if selected is None or selected_projection is None:
            return None

        selected_request_id = str(
            getattr(selected, "request_id", "") or ""
        )

        def exact_owned_request(request: Any) -> bool:
            if (
                str(getattr(request, "request_id", "") or "")
                != selected_request_id
            ):
                return False
            projection = self._owned_post_merge_recovery_projection(request)
            if projection is None:
                return False
            try:
                self._preauthorize_post_merge_recovery(
                    request,
                    projection,
                    preauthorize=preauthorize,
                    evidence_digest=digest,
                )
            except Exception as exc:
                if not _is_implementation_conflict(exc):
                    raise
                return False
            return True

        @contextmanager
        def configured_processor(recovery_train: Any) -> Any:
            current = self.merge_queue.get(selected_request_id)
            current_projection = (
                self._owned_post_merge_recovery_projection(current)
                if current is not None
                else None
            )
            if current_projection is None:
                raise DatabasePortalBridgeError(
                    "selected recovery request lost its sealed projection"
                )
            # Recheck after the exact row is claimed and while the canonical
            # consumer lease is held.  No Portal or validation authority is
            # constructed if database control advanced since discovery.
            # Conflicts may be ``__main__.DatabaseImplementationConflictError``
            # when the daemon is launched with ``-m``; the caller catches them
            # by type name.
            self._preauthorize_post_merge_recovery(
                current,
                current_projection,
                preauthorize=preauthorize,
                evidence_digest=digest,
            )
            portal = self.portal_factory(
                current_projection.paths,
                str(getattr(current, "task_id", "") or ""),
            )
            if portal is None:
                raise DatabasePortalBridgeError(
                    "portal_factory did not return a Portal-compatible daemon"
                )
            close = getattr(
                portal,
                "close_event_runtime",
                None,
            ) or getattr(portal, "close", None)
            try:
                merge_callback = getattr(
                    portal,
                    "_merge_train_callback",
                    None,
                )
                portal_queue = getattr(portal, "merge_queue", None)
                portal_repo_root = getattr(portal, "repo_root", None)
                portal_target = str(
                    getattr(
                        portal,
                        "resolved_merge_target_branch",
                        "",
                    )
                    or ""
                )
                if (
                    not callable(merge_callback)
                    or portal_queue is not self.merge_queue
                    or portal_repo_root is None
                    or self.repository_root is None
                    or Path(portal_repo_root).absolute()
                    != self.repository_root
                    or portal_target != self.merge_target_branch
                ):
                    raise DatabasePortalBridgeError(
                        "Portal recovery daemon is not bound to the selected target"
                    )
                from ..proof.formal_verification_policy import (
                    FormalVerificationPolicy,
                    default_formal_verification_policy,
                )

                proof_gate = getattr(portal, "proof_gate", None)
                raw_policy = getattr(
                    portal,
                    "formal_verification_policy",
                    None,
                )
                if raw_policy is None:
                    policy = (
                        default_formal_verification_policy()
                        if proof_gate is not None
                        else None
                    )
                elif isinstance(raw_policy, FormalVerificationPolicy):
                    policy = raw_policy
                elif isinstance(raw_policy, Mapping):
                    policy = FormalVerificationPolicy.from_dict(raw_policy)
                else:
                    raise TypeError(
                        "Portal formal_verification_policy is invalid"
                    )
                recovery_train.merge_callback = merge_callback
                recovery_train.formal_verification_policy = policy
                recovery_train.proof_gate = proof_gate
                recovery_train.proof_gate_callback = proof_gate
                portal_proof_cache = getattr(
                    portal,
                    "proof_cache_dir",
                    None,
                )
                if portal_proof_cache is not None:
                    recovery_train.proof_cache_dir = Path(
                        portal_proof_cache
                    )
                if policy is not None:
                    recovery_train.proof_cache_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    recovery_train.proof_gate_pin_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    recovery_train.proof_gate_attempt_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                recovery_train.decision_runtime = getattr(
                    portal,
                    "decision_runtime",
                    None,
                )
                recovery_train.decision_runtime_cancellation = getattr(
                    portal,
                    "implementation_cancelled",
                    None,
                )
                yield
            finally:
                if callable(close):
                    close()

        database_result: dict[str, Any] | None = None

        def rearm_after_queue_settlement(
            _claimed: Any,
            _train_result: Mapping[str, Any],
        ) -> None:
            nonlocal database_result
            completed = self.merge_queue.get(selected_request_id)
            projection = (
                self._owned_post_merge_recovery_projection(completed)
                if completed is not None
                else None
            )
            if completed is not None and projection is not None:
                try:
                    self._preauthorize_post_merge_recovery(
                        completed,
                        projection,
                        preauthorize=preauthorize,
                        evidence_digest=digest,
                    )
                except Exception as exc:
                    if not _is_implementation_conflict(exc):
                        raise
                    return
            evidence = (
                self._post_merge_recovery_evidence(
                    completed,
                    projection,
                    evidence_digest=digest,
                )
                if completed is not None and projection is not None
                else None
            )
            if evidence is None:
                return
            try:
                result = recover(evidence)
            except Exception as exc:
                if not _is_implementation_conflict(exc):
                    raise
                return
            if not isinstance(result, Mapping):
                raise DatabasePortalBridgeError(
                    "database post-merge recovery returned a non-object"
                )
            database_result = dict(result)

        try:
            train_result = train.recover_one_integrated_quarantine(
                request_filter=exact_owned_request,
                request_id=selected_request_id,
                processor_context=configured_processor,
                after_process=rearm_after_queue_settlement,
                allow_post_merge_declared_output_recovery=True,
            )
        except Exception as exc:
            if not _is_implementation_conflict(exc):
                raise
            return None
        if database_result is not None:
            return database_result
        if train_result is None:
            return None
        return {
            "schema": _DATABASE_POST_MERGE_RECOVERY_SCHEMA,
            "attempted": True,
            "recovered": False,
            "reason": "post_merge_repair_not_completed",
            "request_id": str(getattr(selected, "request_id", "") or ""),
            "merge_status": str(
                train_result.get("status")
                or train_result.get("reason")
                or ""
            )
            if isinstance(train_result, Mapping)
            else "invalid_result",
            "write_count": 0,
        }

    @staticmethod
    def _require_accepted_provider(attempt: Any, provider_result: Mapping[str, Any]) -> str:
        if (
            provider_result.get("schema") != DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
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
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise DatabasePortalBridgeError(
                "database effect rejected malformed Portal evidence identity"
            )
        return digest

    def apply_effect(self, attempt: Any, provider_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind the already-applied Portal effect to the database phase."""

        digest = self._require_accepted_provider(attempt, provider_result)
        return {
            "status": "applied",
            "effect": "portal-supervised-accepted-effect",
            "effect_key": f"portal:{attempt.task_cid}:{attempt.attempt_id}",
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(provider_result.get("receipt_id") or ""),
            "evidence_digest": digest,
        }

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
        return {
            "outcome": "passed",
            "evidence_digest": digest,
            "argv": ["portal-supervisor-gates"],
            "validator": self.INTERFACE,
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(effect_result.get("portal_receipt_id") or ""),
        }


__all__ = (
    "DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA",
    "DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA",
    "DATABASE_PORTAL_RETRY_DEFERRAL_SCHEMA",
    "DATABASE_PORTAL_EXTERNAL_PROTECTED_CHECKOUT_RECOVERY_SCHEMA",
    "DATABASE_PORTAL_INFLIGHT_PROCESS_RECOVERY_SCHEMA",
    "DATABASE_PORTAL_POOLED_WORKTREE_CREATE_FAILED_REASON",
    "DATABASE_PORTAL_POOLED_WORKTREE_CREATE_RECOVERY_SCHEMA",
    "DATABASE_PORTAL_POOLED_WORKTREE_CREATE_SOURCE_REASON",
    "DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_REASON",
    "DATABASE_PORTAL_VALIDATION_RETRY_SEED_CONFLICT_RECOVERY_SCHEMA",
    "DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_INTENT_SCHEMA",
    "DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA",
    "PROTECTED_CHECKOUT_SETUP_BLOCK_REASONS",
    "DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA",
    "DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA",
    "DatabasePortalAttemptPaths",
    "DATABASE_PORTAL_CANDIDATE_RETRY_REASONS",
    "DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS",
    "DATABASE_PORTAL_CHECKOUT_CONTENTION_REASONS",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeConsumedNoProgressError",
    "DatabasePortalBridgeError",
    "DatabasePortalCandidateRetry",
    "DatabasePortalExecutionBridge",
    "DatabasePortalValidationRetry",
    "PortalDaemonFactory",
    "database_portal_authoritative_repository_tree_id",
    "database_portal_consumed_no_progress_fingerprint",
    "is_protected_checkout_setup_block",
    "database_portal_task_contract_digest",
    "verify_database_portal_attempt_projection",
)
