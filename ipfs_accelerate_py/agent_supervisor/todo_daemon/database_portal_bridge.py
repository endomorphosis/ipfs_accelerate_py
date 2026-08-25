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
import math
import os
import re
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

from ..merge.merge_queue import FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA
from ..runtime.event_log import append_jsonl_event
from ..validation.validation_commands import validation_command_repository_root

_LOG = logging.getLogger(__name__)

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-completion-binding@1"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
_VRIF_BENCHMARK_TASK_ALIAS: Final[str] = "VRIF-030"
_VRIF_BENCHMARK_OUTPUT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
        "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl",
        "test/api/residual_intelligence/test_benchmark.py",
    }
)
_VRIF_BENCHMARK_MANIFEST_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "program_identifier",
    "status",
    "owner_task",
    "source_revision",
    "partitions",
    "required_case_kinds",
    "task_families",
    "training_admission",
    "weights_committed",
    "large_corpus_committed",
    "promotion_evidence",
    "benchmark_freeze",
)
_VRIF_BENCHMARK_CASE_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "family",
    "partition",
    "kind",
    "hidden_test",
    "group_id",
    "input_identity",
    "input_disposition",
    "expected_outcome",
    "case_id",
)
_VRIF_TERMINAL_TASK_ALIAS: Final[str] = "VRIF-032"
_VRIF_TERMINAL_OUTPUT_PATHS: Final[frozenset[str]] = frozenset(
    {
        (
            "docs/architecture/residual_intelligence_inventory/"
            "final_release_report.json"
        ),
        (
            "docs/architecture/residual_intelligence_inventory/"
            "final_release_report.md"
        ),
        "test/api/residual_intelligence/test_release_report.py",
    }
)
_VRIF_TERMINAL_REPORT_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "start_tree",
    "end_tree",
    "corpus_admission_id",
    "expert_dispositions",
    "before",
    "after",
    "costs",
    "promotion_eligible",
    "rollback_target",
    "gaps",
    "producer_artifacts",
    "files_symbols",
    "corpus_rights_splits",
    "architecture_tokenizer_checkpoint",
    "proof_validation",
    "drift",
    "rollback_blocker_eligibility",
)
_VRIF_BENCHMARK_MANIFEST_PATH: Final[str] = (
    "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
)
_VRIF_BENCHMARK_CASES_PATH: Final[str] = (
    "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl"
)
_VRIF_BENCHMARK_TEST_PATH: Final[str] = (
    "test/api/residual_intelligence/test_benchmark.py"
)
_VRIF_RELEASE_REPORT_JSON_PATH: Final[str] = (
    "docs/architecture/residual_intelligence_inventory/final_release_report.json"
)
_VRIF_RELEASE_REPORT_MARKDOWN_PATH: Final[str] = (
    "docs/architecture/residual_intelligence_inventory/final_release_report.md"
)
_VRIF_SEMANTIC_BLOB_MAX_BYTES: Final[int] = 8 * 1024 * 1024
_DATABASE_PORTAL_COMPLETION_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_cid",
        "attempt_id",
        "binding_id",
        "portal_receipt_id",
        "evidence_digest",
        "baseline_commit",
        "baseline_tree",
        "implementation_commit",
        "completion_event_id",
        "receipt_id",
    }
)
DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-validation-retry@1"
)
DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-validation-retry-seed@1"
)
DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-capacity-retry@1"
)
DATABASE_PORTAL_CAPACITY_RETRY_SEED_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-capacity-retry-seed@1"
)
DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-consumed-attempt-retry@1"
)
DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SEED_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-consumed-attempt-retry-seed@1"
)
DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-protected-path-preservation@1"
)
DATABASE_PORTAL_PROTECTED_RECONCILIATION_SELF_LOCK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-protected-reconciliation-self-lock@1"
)
DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-post-merge-completion-recovery-seed@1"
)
DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-post-merge-completion-recovery-seed@2"
)
DATABASE_POST_MERGE_COMPLETION_LINEAGE_FAILURE_REASON: Final[str] = (
    "Portal completion lacks one exact implementation commit"
)
DATABASE_POST_MERGE_COMPLETION_EVALUATED_BASELINE_MISSING_REASON: Final[str] = (
    "Portal completion lacks one exact evaluated baseline"
)
DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON: Final[str] = (
    "post-merge completion recovery seed target generation changed"
)
_DATABASE_POST_MERGE_COMPLETION_RECOVERY_TERMINAL_REASONS: Final[
    frozenset[str]
] = frozenset(
    {
        DATABASE_POST_MERGE_COMPLETION_LINEAGE_FAILURE_REASON,
        DATABASE_POST_MERGE_COMPLETION_EVALUATED_BASELINE_MISSING_REASON,
        DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON,
    }
)
_DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
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
        "source_task_revision",
        "portal_attempt",
        "attempt_consumed",
        "provider_dispatched",
        "completion_authoritative",
        "local_recovery_required",
        "mutation_scopes",
        "protected_paths",
        "baseline_commit",
        "implementation_commit",
        "preserved_commit",
        "rescue_branch",
        "original_branch",
        "original_worktree_path",
        "binding_id",
        "events_digest",
        "event_stream_id",
        "implementation_started_event_id",
        "protected_mutation_event_id",
        "preservation_event_id",
        "implementation_finished_event_id",
        "protected_path_violation_digest",
        "preservation_digest",
        "receipt_id",
    }
)
_PROTECTED_PATH_PRESERVATION_EVENT_CHAIN: Final[tuple[str, ...]] = (
    "task_selected",
    "implementation_protected_path_snapshot_recorded",
    "implementation_started",
    "pre_implementation_kernel_evaluated",
    "implementation_protected_path_mutated",
    "cleanup_finished",
    "protected_path_interrupted_worktree_preserved",
    "implementation_finished",
    "daemon_pass",
)
_CONSUMED_ATTEMPT_TERMINAL_EVENT_CHAIN: Final[tuple[str, ...]] = (
    "task_selected",
    "implementation_protected_path_snapshot_recorded",
    "implementation_started",
    "pre_implementation_kernel_evaluated",
    "implementation_protected_path_snapshot_cleared",
    "worktree_pool_lease_released",
    "implementation_finished",
    "daemon_pass",
)
_PORTAL_EVENT_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "previous_event_id",
        "sequence",
        "snapshot_id",
        "stream_id",
        "timestamp",
        "type",
    }
)
_CONSUMED_ATTEMPT_TERMINAL_EVENT_FIELDS: Final[
    Mapping[str, frozenset[str]]
] = {
    "task_selected": _PORTAL_EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "board_namespace",
            "canonical_task_cid",
            "canonical_task_key",
            "task_id",
            "title",
            "track",
        }
    ),
    "implementation_protected_path_snapshot_recorded": (
        _PORTAL_EVENT_ENVELOPE_FIELDS
        | frozenset(
            {
                "attempt",
                "board_namespace",
                "canonical_task_cid",
                "canonical_task_key",
                "protected_paths",
                "task_id",
                "workspace_path",
            }
        )
    ),
    "implementation_started": _PORTAL_EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "attempt",
            "baseline_ref",
            "board_namespace",
            "branch",
            "cache_hit",
            "canonical_task_cid",
            "canonical_task_key",
            "checkpoint_directory",
            "command",
            "execution_mode",
            "log_path",
            "outputs",
            "provider_dispatched",
            "saved_duration_seconds",
            "setup_duration_seconds",
            "task_id",
            "timeout_policy",
            "workspace_setup",
            "worktree_lifecycle",
            "worktree_path",
        }
    ),
    "pre_implementation_kernel_evaluated": _PORTAL_EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "analytical_candidate_count",
            "attempt",
            "board_namespace",
            "canonical_task_cid",
            "canonical_task_key",
            "disposition",
            "event",
            "interface",
            "kernel_receipt",
            "provider_authorized",
            "provider_hook_count",
            "reason_code",
            "receipt_cid",
            "residual_packet_cid",
            "skip_provider",
            "task_id",
        }
    ),
    "implementation_protected_path_snapshot_cleared": (
        _PORTAL_EVENT_ENVELOPE_FIELDS
        | frozenset(
            {
                "attempt",
                "board_namespace",
                "canonical_task_cid",
                "canonical_task_key",
                "reason",
                "task_id",
            }
        )
    ),
    "worktree_pool_lease_released": _PORTAL_EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "attempted",
            "base_commit",
            "base_ref",
            "branch",
            "cache_hit",
            "cache_key",
            "dependency_paths",
            "entry_id",
            "estimated_seconds_saved",
            "handoff_reason",
            "invalidation_reason",
            "invalidation_reasons",
            "lifecycle_finalize",
            "pooled",
            "reason",
            "released",
            "reused",
            "setup_seconds",
            "setup_time_saved_seconds",
            "worktree_path",
        }
    ),
    "implementation_finished": _PORTAL_EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "attempt",
            "attempt_consumed",
            "baseline_ref",
            "board_completion",
            "board_namespace",
            "branch",
            "cache_hit",
            "canonical_task_cid",
            "canonical_task_key",
            "cleanup_result",
            "commit_result",
            "diagnostic_receipt_id",
            "failed_preservation_result",
            "implementation_commit",
            "lifecycle_finalize",
            "log_path",
            "merge_result",
            "provider_dispatched",
            "returncode",
            "saved_duration_seconds",
            "setup_duration_seconds",
            "task_cid",
            "task_id",
            "validation_result",
            "workspace_setup",
            "worktree_path",
        }
    ),
    "daemon_pass": _PORTAL_EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "active_task_id",
            "attempt_limited_task_ids",
            "blocked_count",
            "completed_count",
            "completion_receipt_task_ids",
            "eligible_ready_count",
            "execution_slice_task_cids_by_id",
            "execution_slice_task_statuses",
            "manual_completion_authority_affected_goal_ids",
            "manual_completion_authority_dependency_task_ids",
            "manual_completion_authority_required_task_ids",
            "manual_completion_authority_revalidation_only",
            "manual_completion_authority_task_ids",
            "manual_completion_renewal_quarantined_task_ids",
            "manual_completion_revalidation_only_task_ids",
            "manual_completion_revalidation_task_ids",
            "max_task_attempts",
            "ordinary_provider_dispatch_allowed",
            "projection_delta_keys",
            "protected_path_conflicts",
            "quarantined_manual_completion_status_task_ids",
            "ready_count",
            "released_retry_budget_strategy_block_task_ids",
            "retry_budget_rearmed_task_ids",
            "retry_budget_reset_deferred_task_ids",
            "retry_budget_reset_task_ids",
            "selectable_ready_count",
            "selection_idle_reason",
            "shared_active_merge_task_ids",
            "shared_completed_task_ids",
            "strict_deprioritized_ready_count",
            "virgin_task_transfer",
            "waiting_count",
        }
    ),
}
_CONSUMED_ATTEMPT_SEED_EVENT_FIELDS: Final[
    Mapping[str, frozenset[str]]
] = {
    "database_portal_validation_retry_seeded": (
        _PORTAL_EVENT_ENVELOPE_FIELDS
        | frozenset(
            {
                "schema",
                "task_id",
                "canonical_task_key",
                "canonical_task_cid",
                "source_database_attempt_id",
                "target_database_attempt_id",
                "target_claim_id",
                "source_retry_receipt_id",
                "implementation_commit",
                "rescue_branch",
                "changed_paths",
                "validation_retry_receipt",
                "completion_authoritative",
                "seed_id",
            }
        )
    ),
    "database_portal_capacity_retry_seeded": _PORTAL_EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "schema",
            "task_id",
            "canonical_task_cid",
            "source_database_attempt_id",
            "target_database_attempt_id",
            "target_claim_id",
            "source_retry_receipt_id",
            "portal_attempt",
            "capacity_retry_receipt",
            "completion_authoritative",
            "seed_id",
        }
    ),
    "database_portal_consumed_attempt_retry_seeded": (
        _PORTAL_EVENT_ENVELOPE_FIELDS
        | frozenset(
            {
                "schema",
                "task_id",
                "canonical_task_cid",
                "source_database_attempt_id",
                "target_database_attempt_id",
                "target_claim_id",
                "source_retry_receipt_id",
                "portal_attempt",
                "consumed_attempt_retry_receipt",
                "completion_authoritative",
                "seed_id",
            }
        )
    ),
}
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_OPERATIONAL_PROJECTION_LINE = re.compile(
    r"(?mi)^-\s*completion\s+receipt\s*:\s*.*$"
)
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_ROOT_REPOSITORY_AUTHORITY: Final[str] = "ipfs_accelerate_py"
_MAX_REPOSITORY_PATH_BYTES: Final[int] = 1024
_MAX_TASK_IDENTITY_BYTES: Final[int] = 4096
_MAX_DATABASE_PORTAL_BACKOFF_SECONDS: Final[int] = 86_400
_MAX_DATABASE_PORTAL_CAPACITY_BACKOFF_SECONDS: Final[int] = 31 * 86_400
_MAX_DATABASE_PORTAL_TASK_ATTEMPTS: Final[int] = 10_000
_MAX_DATABASE_PORTAL_EVENT_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_DATABASE_PORTAL_EVENTS: Final[int] = 4096
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
_FALSE_POSITIVE_COMPLETION_LINEAGE_UNPROVEN_REASON: Final[str] = (
    "false_positive_completion_integration_lineage_unproven"
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
_POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "post-merge-callback-integration-requalification@1"
)
_POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        "schema",
        "task_ids",
        "task_cid",
        "request_id",
        "candidate_commit",
        "baseline_commit",
        "integration_commit",
        "source_event_id",
        "source_event_digest",
        "source_validation_result_digest",
        "queue_validation_proof_digest",
        "train_dedupe_key",
        "train_receipt_id",
        "train_receipt",
        "current_target_commit",
        "current_target_tree",
        "entries",
        "validation",
        "receipt_id",
    }
)
_POST_MERGE_COMPLETION_RECOVERY_SEED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_cid",
        "task_alias",
        "attempt_id",
        "attempt_number",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "source_task_revision",
        "request_id",
        "candidate_commit",
        "qualified_target_commit",
        "qualification_kind",
        "qualification_receipt_id",
        "queue_source_attempt_id",
        "queue_source_claim_id",
        "queue_source_lease_id",
        "queue_source_fencing_token",
        "queue_source_fence_epoch",
        "queue_source_binding_id",
        "queue_source_projection_immutable_digest",
        "recovery_evidence_id",
        "terminal_reason",
        "seed_id",
    }
)
_POST_MERGE_COMPLETION_RECOVERY_SEED_V2_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {*_POST_MERGE_COMPLETION_RECOVERY_SEED_FIELDS, "recovery_control_revision"}
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
_DATABASE_POST_MERGE_CALLBACK_INTEGRATION_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-post-merge-callback-integration-recovery@1"
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
    "priority_task_cids",
    "completed_requests",
    "pending_requests",
    "quarantined_requests",
    "processing_requests",
)
_POST_MERGE_COMPLETION_RECOVERY_TASK_PAGE_SIZE: Final[int] = 32
_MAX_POST_MERGE_COMPLETION_RECOVERY_TASK_CIDS: Final[int] = 1_000
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


class DatabasePortalCapacityRetry(DatabasePortalBridgeError):
    """One exactly proved post-dispatch dual-provider capacity retry."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        value = dict(receipt)
        backoff_seconds = value.get("backoff_seconds")
        if (
            value.get("schema") != DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA
            or value.get("disposition") != "retry"
            or value.get("reason") != "dual_provider_capacity_exhausted"
            or value.get("attempt_consumed") is not True
            or value.get("provider_dispatched") is not True
            or isinstance(backoff_seconds, bool)
            or not isinstance(backoff_seconds, int)
            or backoff_seconds < 0
            or backoff_seconds
            > _MAX_DATABASE_PORTAL_CAPACITY_BACKOFF_SECONDS
        ):
            raise ValueError("capacity retry receipt has an invalid disposition")
        super().__init__("dual_provider_capacity_exhausted")
        self.reason = "dual_provider_capacity_exhausted"
        self.backoff_seconds = int(backoff_seconds)
        self.retry_not_before_ms = int(
            value.get("retry_not_before_ms") or 0
        )
        self.attempt_consumed = True
        self.provider_dispatched = True
        self.retry_receipt = value


class DatabasePortalConsumedAttemptTerminal(DatabasePortalBridgeError):
    """Replay one exact legacy consumed-attempt terminal disposition.

    The receipt is preserved for independent recovery, but this exception is
    intentionally *not* a retry subtype.  The outer database daemon therefore
    reproduces the historical generic ``portal_provider_failed`` failed phase
    before its guarded supersession recovery runs.
    """

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        value = dict(receipt)
        if (
            value.get("schema")
            != DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA
            or value.get("disposition") != "retry"
            or value.get("reason") != "unclassified_post_dispatch_failure"
            or value.get("attempt_consumed") is not True
            or value.get("provider_dispatched") is not True
            or value.get("implementation_returncode") != 1
        ):
            raise ValueError(
                "consumed-attempt receipt has an invalid terminal disposition"
            )
        super().__init__("portal_provider_failed")
        self.reason = "portal_provider_failed"
        self.attempt_consumed = True
        self.provider_dispatched = True
        self.retry_receipt = value


class DatabasePortalProtectedPathPreserved(DatabasePortalBridgeError):
    """Replay one exact post-dispatch protected-path preservation.

    This is neither a pre-dispatch deferral nor an ordinary consumed retry.
    The provider already ran, while the external protected-path fence restored
    Portal's attempt counter and preserved the candidate for zero-provider
    validation.  Outer authorities must therefore keep this receipt distinct
    from both retry classes.
    """

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        value = dict(receipt)
        commit = str(value.get("preserved_commit") or "")
        rescue_branch = str(value.get("rescue_branch") or "")
        if (
            set(value)
            != _DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_FIELDS
            or value.get("schema")
            != DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA
            or value.get("disposition") != "protected_candidate_preserved"
            or value.get("reason")
            != "implementation_protected_path_mutated"
            or value.get("attempt_consumed") is not False
            or value.get("provider_dispatched") is not True
            or value.get("completion_authoritative") is not False
            or value.get("local_recovery_required") is not True
            or value.get("mutation_scopes") != ["shared_checkout"]
            or value.get("implementation_commit") != commit
            or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or not rescue_branch.startswith("rescue/")
            or not rescue_branch.endswith("-protected-path-interrupted")
        ):
            raise ValueError(
                "protected-path preservation receipt has an invalid disposition"
            )
        receipt_id = str(value.get("receipt_id") or "")
        if receipt_id != _content_addressed_record(
            value,
            identity_field="receipt_id",
        ):
            raise ValueError(
                "protected-path preservation receipt identity is invalid"
            )
        super().__init__("implementation_protected_path_mutated")
        self.reason = "implementation_protected_path_mutated"
        self.attempt_consumed = False
        self.provider_dispatched = True
        self.preservation_receipt = value
        # Compatibility with the outer daemon's existing typed-exception
        # receipt accessor.  This remains a preservation disposition, not an
        # authorization to dispatch or consume an ordinary retry.
        self.retry_receipt = value


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


def _strict_json_bytes(value: bytes, *, noun: str) -> Any:
    """Decode one bounded UTF-8 JSON value without duplicate object keys."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DatabasePortalBridgeError(
                    f"{noun} contains duplicate key {key!r}"
                )
            result[key] = item
        return result

    try:
        text = value.decode("utf-8")
        return json.loads(text, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatabasePortalBridgeError(f"{noun} is not strict UTF-8 JSON") from exc


def _git_blob_at_commit(
    repository_root: Path,
    *,
    commit: str,
    path: str,
    max_bytes: int = _VRIF_SEMANTIC_BLOB_MAX_BYTES,
) -> bytes:
    """Read one exact bounded regular Git blob from an immutable commit."""

    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise DatabasePortalBridgeError("VRIF semantic commit identity is malformed")
    safe_path = _safe_output_path(path)

    try:
        tree_entry = subprocess.run(
            ["git", "ls-tree", "-z", commit, "--", safe_path],
            cwd=repository_root,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DatabasePortalBridgeError(
            f"VRIF semantic artifact is unavailable: {safe_path}"
        ) from exc
    match = re.fullmatch(
        rb"(100644|100755) blob ([0-9a-f]{40})\t([^\0]+)\0",
        tree_entry.stdout,
    )
    try:
        observed_path = match.group(3).decode("utf-8") if match is not None else ""
    except UnicodeDecodeError as exc:
        raise DatabasePortalBridgeError(
            f"VRIF semantic artifact path is malformed: {safe_path}"
        ) from exc
    if (
        tree_entry.returncode != 0
        or match is None
        or observed_path != safe_path
    ):
        raise DatabasePortalBridgeError(
            f"VRIF semantic artifact is absent or not a regular blob: {safe_path}"
        )
    object_id = match.group(2).decode("ascii")

    def cat_file(mode: str) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                ["git", "cat-file", mode, object_id],
                cwd=repository_root,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DatabasePortalBridgeError(
                f"VRIF semantic artifact is unavailable: {safe_path}"
            ) from exc

    object_type = cat_file("-t")
    object_size = cat_file("-s")
    try:
        type_text = object_type.stdout.decode("ascii").strip()
        size = int(object_size.stdout.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError) as exc:
        raise DatabasePortalBridgeError(
            f"VRIF semantic artifact metadata is malformed: {safe_path}"
        ) from exc
    if (
        object_type.returncode != 0
        or object_size.returncode != 0
        or type_text != "blob"
        or isinstance(size, bool)
        or size < 0
        or size > max_bytes
    ):
        raise DatabasePortalBridgeError(
            f"VRIF semantic artifact is absent, non-blob, or oversized: {safe_path}"
        )
    payload = cat_file("blob")
    if payload.returncode != 0 or len(payload.stdout) != size:
        raise DatabasePortalBridgeError(
            f"VRIF semantic artifact changed during read: {safe_path}"
        )
    return bytes(payload.stdout)


def _content_addressed_record(
    value: Mapping[str, Any],
    *,
    identity_field: str,
) -> str:
    body = {
        key: item for key, item in value.items() if key != identity_field
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
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
    if len(matches) != 1:
        raise DatabasePortalBridgeError(
            f"Portal task projection has an invalid {label!r} field"
        )
    return str(matches[0]).strip()


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

    def _priority_repaired_completion_requests(
        self,
        task_cids: Sequence[str],
    ) -> tuple[Any, ...]:
        """Resolve exact blocked-task completion rows outside the fair cursor.

        The database owner supplies the bounded latest-terminal task
        identities.  Each identity is queried directly in the target-bound
        durable queue with exact completion and reopen predicates.  Zero or
        multiple rows fail closed for that task.
        """

        if self.merge_queue is None:
            return ()
        completed = getattr(self.merge_queue, "completed_requests", None)
        normalized = tuple(dict.fromkeys(str(item or "") for item in task_cids))
        if (
            not callable(completed)
            or not normalized
            or len(normalized) > 32
            or any(not item for item in normalized)
        ):
            return ()
        selected: list[Any] = []
        for task_cid in normalized:
            repair_matches = tuple(
                completed(
                    limit=2,
                    completion_schema=(
                        _POST_MERGE_DECLARED_OUTPUT_COMPLETION_SCHEMA
                    ),
                    completion_reason="post_merge_declared_outputs_repaired",
                    canonical_task_id=task_cid,
                    reopen_schema=FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA,
                    reopen_reason="declared_outputs_not_on_target",
                )
                or ()
            )
            callback_candidates = tuple(
                completed(
                    limit=256,
                    canonical_task_id=task_cid,
                )
                or ()
            )
            callback_matches = tuple(
                item
                for item in callback_candidates
                if isinstance(getattr(item, "metadata", None), Mapping)
                and item.metadata.get("schema")
                == _MERGE_CANDIDATE_SCHEMA
                and "completion" not in item.metadata
            )
            if len(repair_matches) > 1 or len(callback_matches) > 1:
                continue
            by_request_id = {
                str(getattr(item, "request_id", "") or ""): item
                for item in (*repair_matches, *callback_matches)
                if str(getattr(item, "request_id", "") or "")
            }
            matches = tuple(by_request_id.values())
            if (
                len(matches) == 1
                and str(
                    getattr(matches[0], "canonical_task_id", "") or ""
                )
                == task_cid
            ):
                selected.append(matches[0])
        return tuple(selected)

    @staticmethod
    def _priority_recovery_task_cid_page(
        task_cids: Sequence[str],
        *,
        after_task_cid: str,
    ) -> tuple[tuple[str, ...], str]:
        """Select one fair bounded keyset page from owner-proved task CIDs."""

        normalized = tuple(str(item or "") for item in task_cids)
        if (
            len(normalized)
            > _MAX_POST_MERGE_COMPLETION_RECOVERY_TASK_CIDS
            or any(type(item) is not str for item in task_cids)
            or normalized != tuple(sorted(set(normalized)))
            or any(not item for item in normalized)
        ):
            raise DatabasePortalBridgeError(
                "database completion-recovery task identities are malformed"
            )
        cursor = str(after_task_cid or "")
        page = tuple(
            item for item in normalized if not cursor or item > cursor
        )[:_POST_MERGE_COMPLETION_RECOVERY_TASK_PAGE_SIZE]
        return page, (page[-1] if page else "")

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
        if status == "completed" and "completion" not in metadata:
            # Pre-fix metadata-only shortcuts completed without a callback
            # completion contract or quarantine/revival lineage.  Admission
            # here remains read-only; the recovery path later requires the
            # exact train receipt, blocked database attempt, target identity,
            # non-ancestry, and candidate-vs-target output mismatch.
            return True
        false_reopen = metadata.get("false_positive_completion_reopen")
        false_reopen_lineage = bool(
            isinstance(false_reopen, Mapping)
            and false_reopen.get("schema")
            == FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA
            and false_reopen.get("reason")
            == "declared_outputs_not_on_target"
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(false_reopen.get("train_receipt_id") or ""),
            )
            is not None
        )
        if false_reopen_lineage:
            if status == "pending":
                return True
            if status == "processing":
                return bool(
                    str(getattr(request, "consumer_id", "") or "").startswith(
                        "merge-train:"
                    )
                    and str(getattr(request, "claim_token", "") or "")
                )
            if status == "quarantined":
                return failure_reason in {
                    _POST_MERGE_DECLARED_OUTPUTS_MISSING_REASON,
                    _FALSE_POSITIVE_COMPLETION_LINEAGE_UNPROVEN_REASON,
                    "merge train consumer exited on final attempt",
                    "processing request exceeded max age",
                }
            completion = metadata.get("completion")
            return bool(
                status == "completed"
                and isinstance(completion, Mapping)
                and completion.get("schema")
                == _POST_MERGE_DECLARED_OUTPUT_COMPLETION_SCHEMA
                and completion.get("reason")
                == "post_merge_declared_outputs_repaired"
            )
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
                "completion" not in metadata
                or (
                    isinstance(completion, Mapping)
                    and completion.get("schema")
                    == _POST_MERGE_DECLARED_OUTPUT_COMPLETION_SCHEMA
                    and completion.get("reason")
                    == "post_merge_declared_outputs_repaired"
                )
            )
        return True

    def _current_recovery_task_status(
        self,
        *,
        task_cid: str,
        task_alias: str,
        allowed_statuses: frozenset[str] = frozenset({"blocked", "retrying"}),
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
        # Ordinary recovery stops once a fresh claim advances.  The dedicated
        # completion-recovery seed passes ``in_progress`` explicitly so the
        # successor can reproduce its queue evidence without widening the
        # maintenance scanner's authority.
        return status if status in allowed_statuses else ""

    def _owned_post_merge_recovery_projection(
        self,
        request: Any,
        *,
        allowed_task_statuses: frozenset[str] = frozenset(
            {"blocked", "retrying"}
        ),
        allow_shared_lane_source: bool = False,
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
            or any(part in {"", ".", ".."} for part in projection.parts[1:])
        ):
            return None
        source_attempt_root = projection.parent.parent
        verification_root = self.attempt_root
        if source_attempt_root != self.attempt_root:
            if not allow_shared_lane_source:
                return None
            try:
                raw_shared_state_root = self.attempt_root.parent.parent
                if (
                    raw_shared_state_root.is_symlink()
                    or source_attempt_root.parent.parent
                    != raw_shared_state_root
                ):
                    return None
                shared_state_root = raw_shared_state_root.resolve(strict=True)
                current_attempt_root = self.attempt_root.resolve(strict=True)
                source_attempt_root_resolved = source_attempt_root.resolve(
                    strict=True
                )
                current_relative = current_attempt_root.relative_to(
                    shared_state_root
                )
                source_relative = source_attempt_root_resolved.relative_to(
                    shared_state_root
                )
            except (OSError, RuntimeError, ValueError):
                return None
            current_lane_match = re.fullmatch(
                r"lane-([0-9]+)",
                current_relative.parts[0] if current_relative.parts else "",
            )
            source_lane_match = re.fullmatch(
                r"lane-([0-9]+)",
                source_relative.parts[0] if source_relative.parts else "",
            )
            current_attempt_match = re.fullmatch(
                r"([a-z0-9_]+)_lane_([0-9]+)_database_portal_attempts",
                current_relative.parts[1] if len(current_relative.parts) > 1 else "",
            )
            source_attempt_match = re.fullmatch(
                r"([a-z0-9_]+)_lane_([0-9]+)_database_portal_attempts",
                source_relative.parts[1] if len(source_relative.parts) > 1 else "",
            )
            if (
                len(current_relative.parts) != 2
                or len(source_relative.parts) != 2
                or current_lane_match is None
                or source_lane_match is None
                or current_attempt_match is None
                or source_attempt_match is None
                or current_lane_match.group(1)
                != current_attempt_match.group(2)
                or source_lane_match.group(1) != source_attempt_match.group(2)
                or current_attempt_match.group(1)
                != source_attempt_match.group(1)
                or any(
                    path.is_symlink()
                    for path in (
                        self.attempt_root.parent,
                        self.attempt_root,
                        source_attempt_root.parent,
                        source_attempt_root,
                        projection.parent,
                        projection,
                    )
                )
                or re.fullmatch(r"[0-9a-f]{24}", projection.parent.name)
                is None
            ):
                return None
            verification_root = source_attempt_root_resolved
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
                allowed_root=verification_root,
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
            allowed_statuses=allowed_task_statuses,
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
        train: Any | None = None,
    ) -> dict[str, Any] | None:
        """Compile the exact completed-row receipt into the database contract."""

        metadata = getattr(request, "metadata", None)
        completion = metadata.get("completion") if isinstance(metadata, Mapping) else None
        if completion is None:
            return self._post_merge_callback_integration_evidence(
                request,
                projection,
                evidence_digest=evidence_digest,
                train=train,
            )
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

    def _reopen_false_positive_completed_request(
        self,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
        *,
        train: Any,
        target_commit: str,
    ) -> dict[str, Any] | None:
        """Reopen one receipt-proved declared-output shortcut mistake.

        The historical shortcut completed some Portal rows after checking only
        that declared paths existed.  Recovery is closed over the exact train
        receipt, current target commit, sealed database-attempt projection, and
        a literal candidate-vs-target Git comparison.  It performs only the
        queue CAS here; the ordinary pending recovery stage subsequently routes
        the same row through the Portal repair callback and database rearm.
        """

        metadata = getattr(request, "metadata", None)
        if (
            str(getattr(request, "status", "") or "") != "completed"
            or not isinstance(metadata, Mapping)
            or "completion" in metadata
        ):
            return None
        canonical = str(
            getattr(request, "canonical_identity", "") or ""
        )
        candidate = str(getattr(request, "commit_sha", "") or "")
        dedupe_key = str(getattr(request, "dedupe_key", "") or "")
        make_key = getattr(train, "_dedupe_key", None)
        read_receipt = getattr(train, "_read_receipt", None)
        outputs_match_state = getattr(
            train,
            "portal_declared_outputs_match_commit_state",
            None,
        )
        ancestor_state = getattr(train, "_is_ancestor_state", None)
        if not all(
            callable(operation)
            for operation in (
                make_key,
                read_receipt,
                outputs_match_state,
                ancestor_state,
            )
        ):
            raise DatabasePortalBridgeError(
                "merge train lacks false-completion recovery verification"
            )
        receipt_key = str(make_key(canonical, candidate) or "")
        if not receipt_key or receipt_key != dedupe_key:
            return None
        receipt = read_receipt(receipt_key)
        if not isinstance(receipt, Mapping):
            return None
        receipt = dict(receipt)
        if (
            receipt.get("status") != "already_merged"
            or receipt.get("reason")
            != "declared_outputs_already_on_target"
            or receipt.get("mutation_short_circuited") is not True
        ):
            return None
        target = str(target_commit or "").strip().casefold()
        receipt_target = str(receipt.get("target_commit") or "")
        receipt_target_ancestry = ancestor_state(receipt_target, target)
        receipt_target_outputs = outputs_match_state(
            request,
            receipt_target,
        )
        current_target_outputs = outputs_match_state(request, target)
        candidate_ancestry = ancestor_state(candidate, target)
        if (
            not target
            or not receipt_target
            or receipt.get("merge_commit") != receipt_target
            or receipt_target_ancestry is not True
            or receipt_target_outputs is not False
            or current_target_outputs is not False
            or candidate_ancestry is not False
        ):
            return None
        reopen = getattr(
            self.merge_queue,
            "reopen_false_positive_completion",
            None,
        )
        if not callable(reopen):
            raise DatabasePortalBridgeError(
                "merge queue lacks false-completion recovery authority"
            )
        reopened = reopen(request, completion_receipt=receipt)
        if reopened is None:
            raise DatabasePortalBridgeError(
                "false completed merge request disappeared during recovery"
            )
        reopened_metadata = getattr(reopened, "metadata", None)
        reopen_receipt = (
            reopened_metadata.get("false_positive_completion_reopen")
            if isinstance(reopened_metadata, Mapping)
            else None
        )
        if (
            str(getattr(reopened, "request_id", "") or "")
            != str(getattr(request, "request_id", "") or "")
            or str(getattr(reopened, "status", "") or "") != "pending"
            or not isinstance(reopen_receipt, Mapping)
            or reopen_receipt.get("reason") != "declared_outputs_not_on_target"
        ):
            raise DatabasePortalBridgeError(
                "false completed merge request did not reopen exactly"
            )
        return {
            "schema": _DATABASE_POST_MERGE_RECOVERY_SCHEMA,
            "attempted": True,
            "recovered": False,
            "reason": "false_positive_completion_reopened",
            "request_id": str(getattr(request, "request_id", "") or ""),
            "task_cid": str(
                getattr(request, "canonical_task_id", "") or ""
            ),
            "task_alias": str(getattr(request, "task_id", "") or ""),
            "candidate_commit": candidate,
            "target_commit": target,
            "write_count": 1,
        }

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

    def _callback_integration_source_evidence(
        self,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
        *,
        train: Any,
    ) -> dict[str, Any] | None:
        """Verify one legacy callback integration that missed reconciliation.

        This is intentionally closed over the historical schema-v3 shape.  A
        completed row with a successful receipt is not enough: the queue row,
        full train receipt, queued Portal source, later bare completion, Git
        lineage, and every declared output blob must all agree exactly.
        """

        if self.repository_root is None or self.merge_queue is None:
            return None
        metadata = getattr(request, "metadata", None)
        task_alias = str(getattr(request, "task_id", "") or "")
        task_cid = str(getattr(request, "canonical_task_id", "") or "")
        task_key = str(getattr(request, "canonical_task_key", "") or "")
        request_id = str(getattr(request, "request_id", "") or "")
        candidate = str(getattr(request, "commit_sha", "") or "")
        canonical = str(getattr(request, "canonical_identity", "") or "")
        dedupe_key = str(getattr(request, "dedupe_key", "") or "")
        completion_task_cids = (
            metadata.get("completion_task_cids")
            if isinstance(metadata, Mapping)
            else None
        )
        task_payload = metadata.get("task") if isinstance(metadata, Mapping) else None
        outputs = task_payload.get("outputs") if isinstance(task_payload, Mapping) else None
        validation_proof = (
            metadata.get("validation_proof")
            if isinstance(metadata, Mapping)
            else None
        )
        baseline = str(metadata.get("baseline_ref") or "") if isinstance(metadata, Mapping) else ""
        candidate_tree = str(metadata.get("candidate_tree") or "") if isinstance(metadata, Mapping) else ""
        events_path = str(metadata.get("events_path") or "") if isinstance(metadata, Mapping) else ""
        if (
            str(getattr(request, "status", "") or "") != "completed"
            or not isinstance(metadata, Mapping)
            or metadata.get("schema") != _MERGE_CANDIDATE_SCHEMA
            or "completion" in metadata
            or not request_id
            or not task_alias
            or not task_cid
            or not task_key
            or canonical != task_key
            or re.fullmatch(r"[0-9a-f]{40}", candidate) is None
            or re.fullmatch(r"[0-9a-f]{40}", baseline) is None
            or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", candidate_tree)
            is None
            or metadata.get("implementation_commit") != candidate
            or completion_task_cids != {task_alias: task_cid}
            or not isinstance(task_payload, Mapping)
            or task_payload.get("task_id") != task_alias
            or task_payload.get("canonical_task_cid") != task_cid
            or task_payload.get("canonical_task_key") != task_key
            or not isinstance(outputs, list)
            or not outputs
            or len(outputs) > 4096
            or len(set(str(item) for item in outputs)) != len(outputs)
            or any(not isinstance(item, str) or not item for item in outputs)
            or not isinstance(validation_proof, Mapping)
            or validation_proof.get("attempted") is not True
            or validation_proof.get("passed") is not True
            or validation_proof.get("returncode") != 0
            or validation_proof.get("target_commit") != candidate
        ):
            return None
        try:
            if Path(events_path).resolve() != projection.paths.events.resolve():
                return None
        except (OSError, RuntimeError, ValueError):
            return None

        make_key = getattr(train, "_dedupe_key", None)
        read_receipt = getattr(train, "_read_receipt", None)
        receipt_path = getattr(train, "_receipt_path", None)
        if (
            not callable(make_key)
            or not callable(read_receipt)
            or not callable(receipt_path)
        ):
            raise DatabasePortalBridgeError(
                "merge train lacks callback integration recovery verification"
            )
        receipt_key = str(make_key(canonical, candidate) or "")
        if not receipt_key or receipt_key != dedupe_key:
            return None
        try:
            if (
                receipt_path(receipt_key).stat().st_size
                > _MAX_DATABASE_PORTAL_PROJECTION_BYTES
            ):
                return None
        except (FileNotFoundError, OSError):
            return None
        receipt = read_receipt(receipt_key)
        top_fields = {
            "acceptance_pending",
            "accepted",
            "callback_owned_integration",
            "canonical_task_id",
            "commit_sha",
            "distributed_publication_admission",
            "finished_at",
            "integrated",
            "merge_commit",
            "merge_result",
            "merged",
            "request_id",
            "started_at",
            "status",
            "target_branch",
            "target_commit",
            "task_id",
        }
        if not isinstance(receipt, Mapping) or set(receipt) != top_fields:
            return None
        receipt = dict(receipt)
        integration = str(receipt.get("target_commit") or "")
        started_at = receipt.get("started_at")
        finished_at = receipt.get("finished_at")
        admission = receipt.get("distributed_publication_admission")
        merge_result = receipt.get("merge_result")
        proof = (
            merge_result.get("integration_commit_proof")
            if isinstance(merge_result, Mapping)
            else None
        )
        invariant = (
            merge_result.get("post_merge_declared_output_invariant")
            if isinstance(merge_result, Mapping)
            else None
        )
        todo = (
            merge_result.get("todo_update_result")
            if isinstance(merge_result, Mapping)
            else None
        )
        completion_receipts = (
            todo.get("completion_receipts") if isinstance(todo, Mapping) else None
        )
        member = (
            completion_receipts[0]
            if isinstance(completion_receipts, list)
            and len(completion_receipts) == 1
            else None
        )
        checks = invariant.get("checks") if isinstance(invariant, Mapping) else None
        observed_check_paths = (
            [str(item.get("path") or "") for item in checks]
            if isinstance(checks, list)
            and all(isinstance(item, Mapping) for item in checks)
            else []
        )
        if (
            receipt.get("status") != "merged"
            or receipt.get("accepted") is not True
            or receipt.get("integrated") is not True
            or receipt.get("merged") is not True
            or receipt.get("callback_owned_integration") is not True
            or receipt.get("acceptance_pending") is not False
            or receipt.get("request_id") != request_id
            or receipt.get("task_id") != task_alias
            or receipt.get("canonical_task_id") != canonical
            or receipt.get("commit_sha") != candidate
            or receipt.get("target_branch") != self.merge_target_branch
            or receipt.get("merge_commit") != integration
            or re.fullmatch(r"[0-9a-f]{40}", integration) is None
            or isinstance(started_at, bool)
            or not isinstance(started_at, (int, float))
            or isinstance(finished_at, bool)
            or not isinstance(finished_at, (int, float))
            or not math.isfinite(float(started_at))
            or not math.isfinite(float(finished_at))
            or float(started_at) > float(finished_at)
            or not isinstance(admission, Mapping)
            or set(admission)
            != {"schema", "admitted", "distributed", "request_id", "status"}
            or admission.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/distributed-lane-admission@1"
            or admission.get("admitted") is not True
            or admission.get("distributed") is not False
            or admission.get("request_id") != request_id
            or admission.get("status") != "local"
            or not isinstance(merge_result, Mapping)
            or merge_result.get("attempted") is not True
            or merge_result.get("merged") is not True
            or merge_result.get("returncode") != 0
            or merge_result.get("merge_commit") != integration
            or merge_result.get("target_branch") != self.merge_target_branch
            or not isinstance(proof, Mapping)
            or set(proof)
            != {
                "implementation_commit",
                "integration_commit",
                "integration_ref",
                "passed",
                "reasons",
                "target_branch",
            }
            or proof.get("implementation_commit") != candidate
            or proof.get("integration_commit") != integration
            or proof.get("integration_ref") != integration
            or proof.get("passed") is not True
            or proof.get("reasons") != []
            or proof.get("target_branch") != self.merge_target_branch
            or not isinstance(invariant, Mapping)
            or invariant.get("passed") is not True
            or invariant.get("reason") != "declared_outputs_tracked"
            or invariant.get("mode") != "repository_tree"
            or invariant.get("repository_ref") != integration
            or invariant.get("task_ids") != [task_alias]
            or invariant.get("missing_outputs") != []
            or invariant.get("unsafe_outputs") != []
            or invariant.get("untracked_outputs") != []
            or observed_check_paths != outputs
            or any(
                set(item)
                != {
                    "exists",
                    "path",
                    "reason",
                    "repository",
                    "repository_ref",
                    "task_id",
                    "tracked",
                    "tracked_path",
                }
                or item.get("exists") is not True
                or item.get("tracked") is not True
                or item.get("reason") != "declared_output_tracked"
                or item.get("repository") != "."
                or item.get("repository_ref") != integration
                or item.get("task_id") != task_alias
                or item.get("tracked_path") != item.get("path")
                for item in (checks or ())
            )
            or not isinstance(todo, Mapping)
            or set(todo)
            != {
                "already_completed_task_ids",
                "commit_result",
                "completion_reason",
                "completion_receipts",
                "inserted_status_task_ids",
                "missing_status_task_ids",
                "missing_task_ids",
                "path",
                "task_id",
                "updated",
                "updated_checkbox_task_ids",
                "updated_task_ids",
            }
            or todo.get("task_id") != task_alias
            or todo.get("updated") is not True
            or todo.get("completion_reason") != "single_task"
            or todo.get("updated_task_ids") != [task_alias]
            or todo.get("missing_task_ids") != []
            or todo.get("missing_status_task_ids") != []
            or not isinstance(member, Mapping)
            or set(member)
            != {
                "board_namespace",
                "canonical_task_cid",
                "canonical_task_key",
                "schema",
                "status",
                "task_id",
            }
            or member.get("schema")
            != "ipfs_accelerate_py.agent_supervisor.member_completion_receipt@1"
            or member.get("status") != "succeeded"
            or member.get("task_id") != task_alias
            or member.get("canonical_task_cid") != task_cid
            or member.get("canonical_task_key") != task_key
        ):
            return None

        try:
            events = self._verified_event_chain(projection.paths)
        except DatabasePortalBridgeError:
            return None

        def event_request_id(event: Mapping[str, Any]) -> str:
            direct = str(event.get("request_id") or "")
            event_merge = event.get("merge_result")
            return direct or (
                str(event_merge.get("request_id") or "")
                if isinstance(event_merge, Mapping)
                else ""
            )

        request_sources = [
            event
            for event in events
            if event.get("type") == "implementation_finished"
            and event_request_id(event) == request_id
        ]
        if len(request_sources) != 1:
            return None
        source_event = request_sources[0]
        event_merge = source_event.get("merge_result")
        event_validation = source_event.get("validation_result")
        event_board = source_event.get("board_completion")
        source_sequence = source_event.get("sequence")
        source_attempt = source_event.get("attempt")
        source_event_id = str(source_event.get("event_id") or "")
        completions = [
            event
            for event in events
            if event.get("type") == "task_completed"
            and event.get("task_id") == task_alias
            and event.get("canonical_task_cid") == task_cid
        ]
        reconciliations = [
            event
            for event in events
            if event.get("type") == "merge_reconciled"
            and (
                event_request_id(event) == request_id
                or event.get("completion_source_event_id") == source_event_id
                or (
                    event.get("task_id") == task_alias
                    and event.get("canonical_task_cid") == task_cid
                    and event.get("implementation_commit") == candidate
                )
            )
        ]
        completion_sequence = completions[0].get("sequence") if len(completions) == 1 else None
        if (
            source_event.get("task_id") != task_alias
            or source_event.get("canonical_task_cid") != task_cid
            or source_event.get("canonical_task_key") != task_key
            or isinstance(source_attempt, bool)
            or not isinstance(source_attempt, int)
            or source_attempt < 1
            or source_event.get("attempt_consumed") is not True
            or source_event.get("provider_dispatched") is not True
            or source_event.get("returncode") != 0
            or source_event.get("baseline_ref") != baseline
            or source_event.get("implementation_commit") != candidate
            or re.fullmatch(r"sha256:[0-9a-f]{64}", source_event_id) is None
            or isinstance(source_sequence, bool)
            or not isinstance(source_sequence, int)
            or not isinstance(event_validation, Mapping)
            or event_validation.get("attempted") is not True
            or event_validation.get("passed") is not True
            or event_validation.get("returncode") != 0
            or not isinstance(event_merge, Mapping)
            or event_merge.get("queued") is not True
            or event_merge.get("merged") is not False
            or event_merge.get("reason") != "merge_queued"
            or event_merge.get("request_id") != request_id
            or event_merge.get("implementation_commit") != candidate
            or event_merge.get("completion_task_cids") != {task_alias: task_cid}
            or not isinstance(event_board, Mapping)
            or event_board.get("complete") is not False
            or event_board.get("pending_merge") is not True
            or event_board.get("reason") != "merge_queued_awaiting_integration"
            or len(completions) != 1
            or completions[0].get("reason") != "task_became_completed"
            or completions[0].get("completion_receipt_repair") is not False
            or isinstance(completion_sequence, bool)
            or not isinstance(completion_sequence, int)
            or completion_sequence <= source_sequence
            or reconciliations
        ):
            return None

        def git(*arguments: str) -> subprocess.CompletedProcess[Any]:
            return subprocess.run(
                ["git", *arguments],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                timeout=10,
            )

        try:
            head = git(
                "rev-parse",
                "--verify",
                f"refs/heads/{self.merge_target_branch}^{{commit}}",
            )
            head_text = head.stdout.decode().strip() if isinstance(head.stdout, bytes) else str(head.stdout).strip()
            current_tree_result = git("rev-parse", "--verify", f"{head_text}^{{tree}}")
            current_tree = current_tree_result.stdout.decode().strip() if isinstance(current_tree_result.stdout, bytes) else str(current_tree_result.stdout).strip()
            candidate_tree_result = git("rev-parse", "--verify", f"{candidate}^{{tree}}")
            candidate_tree_text = candidate_tree_result.stdout.decode().strip() if isinstance(candidate_tree_result.stdout, bytes) else str(candidate_tree_result.stdout).strip()
            parents = git("rev-list", "--parents", "-n", "1", candidate)
            parent_text = parents.stdout.decode().strip() if isinstance(parents.stdout, bytes) else str(parents.stdout).strip()
            candidate_integration = git("merge-base", "--is-ancestor", candidate, integration)
            integration_current = git("merge-base", "--is-ancestor", integration, head_text)
        except (OSError, subprocess.SubprocessError):
            return None
        if (
            head.returncode != 0
            or re.fullmatch(r"[0-9a-f]{40}", head_text) is None
            or current_tree_result.returncode != 0
            or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", current_tree) is None
            or candidate_tree_result.returncode != 0
            or candidate_tree_text != candidate_tree
            or parents.returncode != 0
            or parent_text.split() != [candidate, baseline]
            or candidate_integration.returncode != 0
            or integration_current.returncode != 0
        ):
            return None
        entries: list[dict[str, Any]] = []
        for path in outputs:
            try:
                safe_path = _safe_output_path(path)
                observed = [
                    git("ls-tree", "-z", commit, "--", safe_path)
                    for commit in (candidate, integration, head_text)
                ]
            except (DatabasePortalBridgeError, OSError, subprocess.SubprocessError):
                return None
            if any(item.returncode != 0 for item in observed):
                return None
            raw_entry = observed[0].stdout
            if not isinstance(raw_entry, bytes):
                raw_entry = str(raw_entry).encode("utf-8")
            if not raw_entry or any(item.stdout != observed[0].stdout for item in observed[1:]):
                return None
            match = re.fullmatch(
                rb"([0-9]{6}) (blob) ([0-9a-f]{40}(?:[0-9a-f]{24})?)\t([^\0]+)\0",
                raw_entry,
            )
            if match is None or match.group(4).decode("utf-8") != safe_path:
                return None
            entries.append(
                {
                    "path": safe_path,
                    "mode": match.group(1).decode("ascii"),
                    "object_type": "blob",
                    "object_id": match.group(3).decode("ascii"),
                }
            )
        canonical_train_receipt = _canonical_json(receipt)
        if len(canonical_train_receipt) > _MAX_DATABASE_PORTAL_PROJECTION_BYTES:
            return None
        return {
            "task_ids": [task_alias],
            "task_cid": task_cid,
            "request_id": request_id,
            "candidate_commit": candidate,
            "baseline_commit": baseline,
            "integration_commit": integration,
            "source_event_id": source_event_id,
            "source_event_digest": _sha256_bytes(_canonical_json(source_event)),
            "source_validation_result_digest": _sha256_bytes(
                _canonical_json(event_validation)
            ),
            "queue_validation_proof_digest": _sha256_bytes(
                _canonical_json(validation_proof)
            ),
            "train_dedupe_key": receipt_key,
            "train_receipt_id": _sha256_bytes(canonical_train_receipt),
            # Formal content identities reject floats.  Preserve the complete
            # legacy receipt, including float timestamps, as exact bounded
            # canonical JSON and hash those UTF-8 bytes independently.
            "train_receipt": canonical_train_receipt.decode("utf-8"),
            "current_target_commit": head_text,
            "current_target_tree": current_tree,
            "entries": entries,
        }

    @staticmethod
    def _post_merge_callback_integration_receipt_path(
        projection: _DatabasePortalRecoveryProjection,
        *,
        train_receipt_id: str,
        current_head: str,
        current_tree: str,
    ) -> Path:
        key = hashlib.sha256(
            _canonical_json(
                {
                    "train_receipt_id": train_receipt_id,
                    "current_target_commit": current_head,
                    "current_target_tree": current_tree,
                }
            )
        ).hexdigest()
        return (
            projection.paths.root
            / "post-merge-callback-integration-requalification"
            / f"{key}.json"
        )

    @staticmethod
    def _verified_post_merge_callback_integration_receipt(
        raw: Any,
        *,
        source: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        from ..proof.formal_verification_contracts import content_identity

        if not isinstance(raw, Mapping):
            return None
        value = dict(raw)
        receipt_id = str(value.pop("receipt_id", "") or "")
        validation = value.get("validation")
        expected_source_fields = {
            key for key in _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_FIELDS
            if key not in {"schema", "validation", "receipt_id"}
        }
        if (
            set(raw) != _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_FIELDS
            or value.get("schema")
            != _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_SCHEMA
            or any(value.get(key) != source.get(key) for key in expected_source_fields)
            or not isinstance(validation, list)
            or len(validation) != 1
            or receipt_id != content_identity(value)
        ):
            return None
        item = validation[0]
        digests = item.get("validation_result_digests") if isinstance(item, Mapping) else None
        command_count = item.get("command_count") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "task_id",
                "passed",
                "returncode",
                "validation_result_digests",
                "command_count",
                "log_sha256",
            }
            or item.get("task_id") != source.get("task_ids", [""])[0]
            or item.get("passed") is not True
            or item.get("returncode") != 0
            or isinstance(command_count, bool)
            or not isinstance(command_count, int)
            or command_count < 1
            or not isinstance(digests, list)
            or len(digests) != command_count
            or any(re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(digest)) is None for digest in digests)
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("log_sha256") or "")) is None
        ):
            return None
        return {**value, "receipt_id": receipt_id}

    def _load_post_merge_callback_integration_receipt(
        self,
        path: Path,
        *,
        source: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        try:
            if path.stat().st_size > _MAX_DATABASE_PORTAL_PROJECTION_BYTES:
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        return self._verified_post_merge_callback_integration_receipt(
            raw,
            source=source,
        )

    def _requalify_callback_integration(
        self,
        source: Mapping[str, Any],
        *,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
    ) -> dict[str, Any] | None:
        """Run one fresh uncached declared validation at the bound target."""

        if self.repository_root is None or self.merge_queue is None:
            return None
        task_alias = str(getattr(request, "task_id", "") or "")
        task_cid = str(getattr(request, "canonical_task_id", "") or "")
        current_head = str(source.get("current_target_commit") or "")
        current_tree = str(source.get("current_target_tree") or "")
        if source.get("task_ids") != [task_alias] or source.get("task_cid") != task_cid:
            return None
        path = self._post_merge_callback_integration_receipt_path(
            projection,
            train_receipt_id=str(source.get("train_receipt_id") or ""),
            current_head=current_head,
            current_tree=current_tree,
        )
        if path.exists():
            return self._load_post_merge_callback_integration_receipt(
                path,
                source=source,
            )
        portal = self.portal_factory(projection.paths, task_alias)
        if portal is None:
            raise DatabasePortalBridgeError(
                "portal_factory did not return a Portal-compatible daemon"
            )
        close = getattr(portal, "close_event_runtime", None) or getattr(portal, "close", None)
        try:
            load_tasks = getattr(portal, "_load_tasks", None)
            run_validation = getattr(portal, "_run_validation_commands", None)
            run_mutation = getattr(portal, "_run_checkout_mutation_transaction", None)
            cleanup_workspace = getattr(portal, "_cleanup_main_merge_workspace", None)
            portal_root = getattr(portal, "repo_root", None)
            if (
                getattr(portal, "merge_queue", None) is not self.merge_queue
                or portal_root is None
                or Path(portal_root).absolute() != self.repository_root
                or str(getattr(portal, "resolved_merge_target_branch", "") or "")
                != self.merge_target_branch
                or not callable(load_tasks)
                or not callable(run_validation)
                or not callable(run_mutation)
                or not callable(cleanup_workspace)
            ):
                raise DatabasePortalBridgeError(
                    "Portal recovery daemon lacks callback requalification authority"
                )
            tasks = list(load_tasks())
            if (
                len(tasks) != 1
                or str(getattr(tasks[0], "task_id", "") or "") != task_alias
                or str(getattr(tasks[0], "canonical_task_cid", "") or "") != task_cid
                or not tuple(getattr(tasks[0], "validation", ()) or ())
            ):
                return None

            def validate() -> dict[str, Any]:
                result: dict[str, Any] = {"passed": False, "validation": []}
                root = projection.paths.root / "post-merge-callback-integration-requalification"
                root.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix="worktree-", dir=root))
                temporary.rmdir()
                added = False
                try:
                    before = subprocess.run(
                        ["git", "rev-parse", "--verify", f"refs/heads/{self.merge_target_branch}^{{commit}}"],
                        cwd=self.repository_root,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=10,
                    )
                    if before.returncode != 0 or before.stdout.strip() != current_head:
                        return result
                    worktree = subprocess.run(
                        ["git", "worktree", "add", "--detach", str(temporary), current_head],
                        cwd=self.repository_root,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=30,
                    )
                    if worktree.returncode != 0:
                        return result
                    added = True
                    log_root = projection.paths.implementation_logs / "post-merge-callback-integration-requalification"
                    log_root.mkdir(parents=True, exist_ok=True)
                    log_path = log_root / f"{task_alias}-{current_head[:16]}.log"
                    validation = run_validation(
                        temporary,
                        tasks[0],
                        log_path,
                        force_uncached=True,
                    )
                    command_results = validation.get("results") if isinstance(validation, Mapping) else None
                    digests = (
                        [str(item.get("validation_result_digest") or "") for item in command_results]
                        if isinstance(command_results, list)
                        and all(isinstance(item, Mapping) for item in command_results)
                        else []
                    )
                    head = subprocess.run(
                        ["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=temporary,
                        capture_output=True, check=False, text=True, timeout=10,
                    )
                    tree = subprocess.run(
                        ["git", "rev-parse", "--verify", "HEAD^{tree}"], cwd=temporary,
                        capture_output=True, check=False, text=True, timeout=10,
                    )
                    status = subprocess.run(
                        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                        cwd=temporary, capture_output=True, check=False, timeout=10,
                    )
                    after = subprocess.run(
                        ["git", "rev-parse", "--verify", f"refs/heads/{self.merge_target_branch}^{{commit}}"],
                        cwd=self.repository_root, capture_output=True, check=False, text=True, timeout=10,
                    )
                    if (
                        not isinstance(validation, Mapping)
                        or validation.get("passed") is not True
                        or validation.get("returncode") != 0
                        or not isinstance(command_results, list)
                        or not command_results
                        or len(digests) != len(command_results)
                        or any(re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", item) is None for item in digests)
                        or not log_path.is_file()
                        or head.returncode != 0
                        or head.stdout.strip() != current_head
                        or tree.returncode != 0
                        or tree.stdout.strip() != current_tree
                        or status.returncode != 0
                        or status.stdout
                        or after.returncode != 0
                        or after.stdout.strip() != current_head
                    ):
                        return result
                    result.update(
                        passed=True,
                        validation=[
                            {
                                "task_id": task_alias,
                                "passed": True,
                                "returncode": 0,
                                "validation_result_digests": digests,
                                "command_count": len(command_results),
                                "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                            }
                        ],
                    )
                    return result
                except (OSError, subprocess.SubprocessError):
                    return result
                finally:
                    if added:
                        cleanup = cleanup_workspace(temporary, ephemeral=True)
                        if not isinstance(cleanup, Mapping) or cleanup.get("cleaned") is not True:
                            result["passed"] = False

            transaction = run_mutation(
                task_id=task_alias,
                branch=self.merge_target_branch,
                operation="requalify_post_merge_callback_integration",
                callback=validate,
                failure_fields={"passed": False},
                extra={
                    "current_target_commit": current_head,
                    "source_integration_commit": str(source.get("integration_commit") or ""),
                },
            )
            validations = transaction.get("validation") if isinstance(transaction, Mapping) else None
            if (
                not isinstance(transaction, Mapping)
                or transaction.get("passed") is not True
                or not isinstance(validations, list)
                or len(validations) != 1
            ):
                return None
        finally:
            if callable(close):
                close()
        from ..proof.formal_verification_contracts import content_identity

        qualified = {
            "schema": _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_SCHEMA,
            **dict(source),
            "validation": [dict(item) for item in validations],
        }
        qualified["receipt_id"] = content_identity(qualified)
        _atomic_write_once(
            path,
            json.dumps(qualified, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        return self._load_post_merge_callback_integration_receipt(
            path,
            source=source,
        )

    def _post_merge_callback_integration_evidence(
        self,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
        *,
        evidence_digest: Callable[[Mapping[str, Any]], str],
        train: Any | None = None,
    ) -> dict[str, Any] | None:
        if train is None:
            if self.repository_root is None or self.merge_queue is None:
                return None
            from ..merge.merge_train import MergeTrain

            train = MergeTrain(
                repo_root=self.repository_root,
                queue=self.merge_queue,
                target_branch=self.merge_target_branch,
                max_attempts=int(getattr(self.merge_queue, "max_attempts", 3)),
            )
        source = self._callback_integration_source_evidence(
            request,
            projection,
            train=train,
        )
        if source is None:
            return None
        qualification = self._requalify_callback_integration(
            source,
            request=request,
            projection=projection,
        )
        if qualification is None:
            return None
        binding = projection.binding
        evidence: dict[str, Any] = {
            "schema": _DATABASE_POST_MERGE_CALLBACK_INTEGRATION_RECOVERY_SCHEMA,
            "request_id": str(getattr(request, "request_id", "") or ""),
            "task_cid": str(getattr(request, "canonical_task_id", "") or ""),
            "task_alias": str(getattr(request, "task_id", "") or ""),
            "candidate_commit": str(getattr(request, "commit_sha", "") or ""),
            "source_attempt_id": str(binding.get("attempt_id") or ""),
            "source_claim_id": str(binding.get("claim_id") or ""),
            "source_lease_id": str(binding.get("lease_id") or ""),
            "source_fencing_token": binding.get("fencing_token"),
            "source_fence_epoch": binding.get("fence_epoch"),
            "source_binding_id": str(binding.get("binding_id") or ""),
            "source_projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
            "qualified_target_commit": str(
                qualification.get("current_target_commit") or ""
            ),
            "callback_requalification_receipt_id": str(
                qualification.get("receipt_id") or ""
            ),
            "callback_requalification_receipt": dict(qualification),
        }
        if (
            not evidence["request_id"]
            or not evidence["source_attempt_id"]
            or not evidence["source_claim_id"]
            or not evidence["source_lease_id"]
            or type(evidence["source_fencing_token"]) is not int
            or type(evidence["source_fence_epoch"]) is not int
            or re.fullmatch(r"sha256:[0-9a-f]{64}", evidence["source_binding_id"])
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                evidence["source_projection_immutable_digest"],
            )
            is None
        ):
            return None
        evidence_id = evidence_digest(evidence)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", str(evidence_id or "")) is None:
            return None
        evidence["evidence_id"] = str(evidence_id)
        return evidence

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
        payload = {
            "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
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
            "plan_cid": str(getattr(record, "plan_cid", "") or ""),
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
        payload["binding_id"] = _sha256_bytes(_canonical_json(payload))
        return payload

    def _render_projection(self, attempt: Any, record: Any) -> str:
        body = dict(getattr(record, "body", {}) or {})
        canonical_task_key, canonical_task_cid = _canonical_projection_identity(
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
        priority = _line_value(
            getattr(record, "priority", "") or body.get("priority") or "P2"
        )
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
            "canonical task key",
            "canonical_task_key",
            "canonical task cid",
            "canonical_task_cid",
            "task key",
            "task_key",
            "scope expansion policy",
            "scope_expansion_policy",
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
            f"- Canonical task key: {canonical_task_key}",
            f"- Canonical task CID: {canonical_task_cid}",
            # The canonical database revision owns only its typed output/effect
            # paths.  Validation imports are readable context, never an
            # implicit mutation grant for this private Portal projection.
            "- Scope expansion policy: exact",
            "- Projection authority: false",
        ]
        if alias == _VRIF_BENCHMARK_TASK_ALIAS and set(outputs) == set(
            _VRIF_BENCHMARK_OUTPUT_PATHS
        ):
            lines.extend(
                (
                    "- Root benchmark contract: owner-exact no-training freeze",
                    (
                        "- Root benchmark authority: "
                        "scripts/run_agent_supervisor_residual_intelligence.py::"
                        "_vrif_frozen_benchmark_contract"
                    ),
                    (
                        "- Producer construction API: "
                        "ipfs_accelerate_py.agent_supervisor.residual_intelligence."
                        "benchmark.build_frozen_benchmark_contract; its returned "
                        "cases and benchmark_freeze match the independent owner"
                    ),
                    (
                        "- Benchmark binding authority: reconstruct "
                        "base_frozen_bindings exactly as "
                        "_vrif_terminal_report_evidence does; the executable "
                        "fixture is test/api/residual_intelligence/"
                        "test_goal_authority.py"
                    ),
                    "- Required benchmark manifest keys: "
                    + ", ".join(_VRIF_BENCHMARK_MANIFEST_FIELDS),
                    "- Required benchmark case keys: "
                    + ", ".join(_VRIF_BENCHMARK_CASE_FIELDS),
                    (
                        "- Benchmark population contract: exactly 96 cases, "
                        "one paired partition/kind entry for each of 24 task "
                        "families; do not construct the legacy Cartesian "
                        "384-case population"
                    ),
                    (
                        "- Benchmark lineage contract: source_revision and "
                        "benchmark_freeze source bind the Portal baseline "
                        "commit and tree before these three outputs change"
                    ),
                    (
                        "- Benchmark disposition: payload unavailable because "
                        "training is unavailable; every expected outcome is "
                        "CAPABILITY_UNAVAILABLE and the paired evaluation is "
                        "all-abstain/not-run"
                    ),
                    (
                        "- Validation-policy binding: freeze the final tracked "
                        "blob identity of test/api/residual_intelligence/"
                        "test_benchmark.py"
                    ),
                    (
                        "- Construction order: finish the candidate "
                        "test_benchmark.py bytes first, then hash that exact "
                        "tracked payload for validation_policy before "
                        "regenerating manifest.json and every cases.jsonl row"
                    ),
                    (
                        "- Deterministic recovery command: resolve the Portal "
                        "baseline with git rev-parse HEAD, then run "
                        "scripts/materialize_vrif_frozen_benchmark.py "
                        "--baseline-commit <the resolved 40-hex commit> "
                        "--write; inspect and retain the resulting three-file "
                        "declared-output patch"
                    ),
                    (
                        "- Retry no-change rule: the checked-in freeze predates "
                        "this Portal baseline and is not acceptable merely "
                        "because its historical self-consistency tests pass; "
                        "an empty patch is a terminal implementation failure"
                    ),
                    (
                        "- Independent-validation requirement: self-consistency "
                        "through load_frozen_benchmark is insufficient; "
                        "test_benchmark.py must independently reconstruct the "
                        "owner base_frozen_bindings and prove exact equality "
                        "with the owner-derived freeze and 96-case schedule"
                    ),
                    (
                        "- Dynamic identity rule: recompute every schedule, "
                        "binding, case, baseline, and freeze identity for this "
                        "Portal baseline/candidate; never copy an earlier "
                        "freeze identifier"
                    ),
                )
            )
        if alias == _VRIF_TERMINAL_TASK_ALIAS and set(outputs) == set(
            _VRIF_TERMINAL_OUTPUT_PATHS
        ):
            # VRIF's root owner consumes a stricter terminal artifact than the
            # historical release-model test used to describe.  Put that
            # already-reviewed contract in the immutable provider projection
            # so a retry cannot silently reproduce the obsolete report shape.
            # This grants no additional edit path: the bridge-owned exact
            # scope above remains the mutation authority.
            lines.extend(
                (
                    "- Root completion contract: owner-exact VRIF terminal report",
                    "- Required machine report keys: "
                    + ", ".join(_VRIF_TERMINAL_REPORT_FIELDS),
                    (
                        "- Root contract authority: "
                        "scripts/run_agent_supervisor_residual_intelligence.py::"
                        "_vrif_terminal_report_evidence"
                    ),
                    (
                        "- Root contract fixture: "
                        "test/api/residual_intelligence/test_goal_authority.py"
                    ),
                    (
                        "- Human report contract: render byte-for-byte with "
                        "scripts/run_agent_supervisor_residual_intelligence.py::"
                        "_vrif_release_report_markdown"
                    ),
                    (
                        "- Human-report validation: replace substring-only "
                        "checks in test_release_report.py with exact UTF-8 byte "
                        "equality between final_release_report.md and "
                        "_vrif_release_report_markdown(parsed_machine_report)"
                    ),
                    (
                        "- Retry lineage rule: derive end_tree and "
                        "drift.evaluated_tree from this attempt's Portal "
                        "baseline tree, refresh every VRIF-030 producer/freeze "
                        "identity, and modify all three declared outputs "
                        "relative to the retry baseline"
                    ),
                    (
                        "- Producer artifact contract: bind the declared "
                        "VRIF-028 through VRIF-031 current tracked blobs into "
                        "the exact producer_artifacts bundle"
                    ),
                    (
                        "- Unavailable release disposition: every expert is "
                        "CAPABILITY_UNAVAILABLE; costs are exactly tokens=0 "
                        "and break_even=0; promotion_eligible is false"
                    ),
                    (
                        "- Forbidden release claims: learned, verified, safe, "
                        "autonomous, token-efficient, production-ready"
                    ),
                    (
                        "- Required not-run set: gpu_live_qualification, "
                        "promotion, training"
                    ),
                )
            )
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

    @staticmethod
    def _exact_callback_reconciliation_for_completion_source(
        reconciliation: Mapping[str, Any],
        source: Mapping[str, Any],
        *,
        alias: str,
        task_cid: str,
    ) -> bool:
        """Verify the complete callback handoff, not merely its landed SHA."""

        from ..proof.formal_verification_contracts import content_identity

        source_merge = source.get("merge_result")
        source_validation = source.get("validation_result")
        source_board = source.get("board_completion")
        source_type = str(source.get("type") or "")
        source_branch = str(source.get("branch") or "")
        if (
            not isinstance(source_merge, Mapping)
            or not isinstance(source_validation, Mapping)
            or not isinstance(source_board, Mapping)
        ):
            return False
        request_id = str(source_merge.get("request_id") or "")
        baseline = str(source.get("baseline_ref") or "")
        implementation = str(source.get("implementation_commit") or "")
        source_event_id = str(source.get("event_id") or "")
        source_attempt = source.get("attempt")
        completion_task_cids = source_merge.get("completion_task_cids")
        common_source_required_fields = {
            "attempt",
            "attempt_consumed",
            "baseline_ref",
            "board_completion",
            "branch",
            "canonical_task_cid",
            "canonical_task_key",
            "implementation_commit",
            "merge_result",
            "provider_dispatched",
            "returncode",
            "task_id",
            "validation_result",
        } | set(_PORTAL_EVENT_ENVELOPE_FIELDS)
        common_source_fields = common_source_required_fields | {
            "board_namespace",
            "task_source_identity",
        }
        implementation_source_fields = common_source_fields | {
            "cache_hit",
            "cleanup_result",
            "commit_result",
            "completion_receipt_degraded",
            "deferred",
            "dependency_preflight",
            "diagnostic_receipt_id",
            "exception_result",
            "failed_preservation_result",
            "failure_kind",
            "implementation_result",
            "infrastructure_failure",
            "lifecycle_finalize",
            "log_path",
            "protected_path_violation",
            "provider_call_allowed",
            "reason",
            "saved_duration_seconds",
            "setup_duration_seconds",
            "submodule_init_failures",
            "task_cid",
            "task_execution_receipt_id",
            "task_execution_receipt_path",
            "termination_result",
            "timeout_result",
            "todo_update_result",
            "workspace_setup",
            "worktree_path",
        }
        projected_source_fields = common_source_fields | {
            "database_portal_merge_continuation_source",
            "merge_queue_synchronous_source",
            "reason",
        }
        source_fields = (
            implementation_source_fields
            if source_type == "implementation_finished"
            else projected_source_fields
            if source_type == "worktree_reconciliation_candidate_queued"
            else set()
        )
        if (
            not source_fields
            or not common_source_required_fields <= set(source)
            or not set(source) <= source_fields
            or not request_id
            or re.fullmatch(r"[0-9a-f]{40}", baseline) is None
            or re.fullmatch(r"[0-9a-f]{40}", implementation) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", source_event_id) is None
            or type(source_attempt) is not int
            or source_attempt < 1
            or source_merge.get("queued") is not True
            or source_merge.get("merged") is not False
            or source_merge.get("attempted") is not False
            or source_merge.get("reason") != "merge_queued"
            or source_merge.get("request_id") != request_id
            or source_merge.get("branch") != source_branch
            or source_merge.get("implementation_commit") != implementation
            or source_merge.get("canonical_task_cid") != task_cid
            or source_merge.get("canonical_task_key")
            != str(source.get("canonical_task_key") or "")
            or not str(source_merge.get("target_repository_id") or "")
            or not str(source_merge.get("target_branch") or "")
            or not isinstance(completion_task_cids, Mapping)
            or str(completion_task_cids.get(alias) or "") != task_cid
            or source.get("returncode") != 0
            or source_validation.get("attempted") is not True
            or source_validation.get("passed") is not True
            or type(source_validation.get("returncode")) is not int
            or source_validation.get("returncode") != 0
            or source_board
            != {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            }
        ):
            return False
        if source_type == "implementation_finished":
            if (
                source.get("attempt_consumed") is not True
                or type(source.get("provider_dispatched")) is not bool
            ):
                return False
        else:
            reason = str(source.get("reason") or "")
            provenance_key = (
                "database_portal_merge_continuation_source"
                if reason
                == "database_portal_merge_continuation_source_projected"
                else "merge_queue_synchronous_source"
                if reason == "merge_queue_synchronous_source_projected"
                else ""
            )
            provenance = source.get(provenance_key) if provenance_key else None
            if (
                not provenance_key
                or source.get("attempt_consumed") is not False
                or source.get("provider_dispatched") is not False
                or not isinstance(provenance, Mapping)
            ):
                return False
            provenance_body = dict(provenance)
            source_projection_id = str(
                provenance_body.pop("source_projection_id", "") or ""
            )
            expected_schema = (
                "ipfs_accelerate_py.agent_supervisor."
                "database-portal-merge-continuation-source@1"
                if provenance_key
                == "database_portal_merge_continuation_source"
                else (
                    "ipfs_accelerate_py.agent_supervisor."
                    "merge-queue-synchronous-source@1"
                )
            )
            continuation_provenance_fields = {
                "canonical_task_key",
                "consumer_attempt_id",
                "consumer_binding_id",
                "goal_cid",
                "plan_cid",
                "producer_attempt_id",
                "producer_binding_id",
                "producer_completion_source_event_id",
                "producer_events_path",
                "producer_portal_attempt",
                "producer_projection_immutable_digest",
                "producer_projection_path",
                "producer_provider_dispatched",
                "producer_state_path",
                "producer_strategy_path",
                "request_id",
                "schema",
                "task_cid",
                "task_id",
            }
            synchronous_provenance_fields = {
                "baseline_ref",
                "branch",
                "canonical_task_key",
                "implementation_commit",
                "merge_candidate_enqueued_event_id",
                "portal_attempt",
                "request_id",
                "schema",
                "task_cid",
                "task_id",
                "validation_repository_tree_id",
                "validation_target_commit",
                "validation_target_tree",
            }
            expected_provenance_fields = (
                continuation_provenance_fields
                if provenance_key
                == "database_portal_merge_continuation_source"
                else synchronous_provenance_fields
            )
            if (
                source_validation
                != {"attempted": True, "passed": True, "returncode": 0}
                or set(provenance_body) != expected_provenance_fields
                or provenance_body.get("schema") != expected_schema
                or provenance_body.get("request_id") != request_id
                or provenance_body.get("task_id") != alias
                or provenance_body.get("task_cid") != task_cid
                or provenance_body.get("canonical_task_key")
                != str(source.get("canonical_task_key") or "")
                or source_projection_id != content_identity(provenance_body)
            ):
                return False
            if provenance_key == "merge_queue_synchronous_source":
                validation_tree = str(
                    provenance_body.get("validation_target_tree") or ""
                )
                if (
                    provenance_body.get("portal_attempt") != source_attempt
                    or provenance_body.get("branch") != source_branch
                    or provenance_body.get("baseline_ref") != baseline
                    or provenance_body.get("implementation_commit")
                    != implementation
                    or provenance_body.get("validation_target_commit")
                    != implementation
                    or re.fullmatch(r"[0-9a-f]{40}", validation_tree) is None
                    or provenance_body.get("validation_repository_tree_id")
                    != f"git-tree:{validation_tree}"
                    or re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(
                            provenance_body.get(
                                "merge_candidate_enqueued_event_id"
                            )
                            or ""
                        ),
                    )
                    is None
                ):
                    return False
            else:
                producer_projection = Path(
                    str(
                        provenance_body.get("producer_projection_path")
                        or ""
                    )
                )
                producer_root = producer_projection.parent
                if (
                    provenance_body.get("producer_portal_attempt")
                    != source_attempt
                    or type(
                        provenance_body.get("producer_provider_dispatched")
                    )
                    is not bool
                    or re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(
                            provenance_body.get(
                                "producer_completion_source_event_id"
                            )
                            or ""
                        ),
                    )
                    is None
                    or not str(provenance_body.get("goal_cid") or "")
                    or not str(provenance_body.get("plan_cid") or "")
                    or not str(
                        provenance_body.get("producer_binding_id") or ""
                    )
                    or not str(
                        provenance_body.get("consumer_binding_id") or ""
                    )
                    or provenance_body.get("producer_binding_id")
                    == provenance_body.get("consumer_binding_id")
                    or not str(
                        provenance_body.get("producer_attempt_id") or ""
                    )
                    or not str(
                        provenance_body.get("consumer_attempt_id") or ""
                    )
                    or provenance_body.get("producer_attempt_id")
                    == provenance_body.get("consumer_attempt_id")
                    or producer_projection.name != "task-projection.md"
                    or Path(
                        str(provenance_body.get("producer_state_path") or "")
                    )
                    != producer_root / "portal-task-state.json"
                    or Path(
                        str(
                            provenance_body.get("producer_strategy_path")
                            or ""
                        )
                    )
                    != producer_root / "portal-strategy.json"
                    or Path(
                        str(provenance_body.get("producer_events_path") or "")
                    )
                    != producer_root / "portal-events.jsonl"
                    or re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(
                            provenance_body.get(
                                "producer_projection_immutable_digest"
                            )
                            or ""
                        ),
                    )
                    is None
                ):
                    return False
        expected_task_cids = {
            str(task_id): str(cid)
            for task_id, cid in completion_task_cids.items()
        }
        if (
            not expected_task_cids
            or any(not task_id or not cid for task_id, cid in expected_task_cids.items())
        ):
            return False

        integration = str(reconciliation.get("merge_commit") or "")
        target = str(reconciliation.get("target_commit") or "")
        branch = str(reconciliation.get("branch") or "")
        candidate_key = content_identity(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "merge-queue-reconciled-candidate@1"
                ),
                "task_id": alias,
                "task_cid": task_cid,
                "request_id": request_id,
                "baseline_ref": baseline,
                "implementation_commit": implementation,
                "completion_source_event_id": source_event_id,
            }
        )
        merge_result = reconciliation.get("merge_result")
        expected_merge_result = {
            "attempted": True,
            "merged": True,
            "queued": False,
            "reason": "merge_queue_callback_completed",
            "request_id": request_id,
            "merge_commit": integration,
            "target_commit": integration,
        }
        reconciliation_required_fields = {
            "attempt",
            "baseline_ref",
            "branch",
            "canonical_task_cid",
            "completion_receipt_evidence",
            "completion_source_event_id",
            "completion_task_cids",
            "implementation_commit",
            "integration_commit_proof",
            "landed_commit",
            "merge_commit",
            "merge_result",
            "post_merge_declared_output_invariant",
            "reason",
            "reconciled_candidate_key",
            "request_id",
            "resolved",
            "target_commit",
            "task_id",
        }
        reconciliation_optional_fields = {
            "board_namespace",
            "canonical_task_key",
            "database_portal_merge_continuation_source",
            "task_source_identity",
        }
        reconciliation_fields = set(reconciliation)
        if (
            not reconciliation_required_fields <= reconciliation_fields
            or not reconciliation_fields
            <= (
                reconciliation_required_fields
                | reconciliation_optional_fields
                | set(_PORTAL_EVENT_ENVELOPE_FIELDS)
            )
            or reconciliation.get("resolved") is not True
            or str(reconciliation.get("reason") or "")
            != "merge_queue_callback_completed"
            or str(reconciliation.get("task_id") or "") != alias
            or str(reconciliation.get("canonical_task_cid") or "") != task_cid
            or reconciliation.get("attempt") != source_attempt
            or branch != source_branch
            or (
                "canonical_task_key" in reconciliation
                and reconciliation.get("canonical_task_key")
                != source.get("canonical_task_key")
            )
            or ("board_namespace" in reconciliation)
            != ("board_namespace" in source)
            or (
                "board_namespace" in reconciliation
                and (
                    not str(reconciliation.get("board_namespace") or "")
                    or reconciliation.get("board_namespace")
                    != source.get("board_namespace")
                )
            )
            or ("task_source_identity" in reconciliation)
            != ("task_source_identity" in source)
            or (
                "task_source_identity" in reconciliation
                and (
                    not isinstance(
                        reconciliation.get("task_source_identity"), Mapping
                    )
                    or reconciliation.get("task_source_identity")
                    != source.get("task_source_identity")
                )
            )
            or (
                "database_portal_merge_continuation_source" in reconciliation
                and reconciliation.get(
                    "database_portal_merge_continuation_source"
                )
                != source.get("database_portal_merge_continuation_source")
            )
            or str(reconciliation.get("request_id") or "") != request_id
            or str(reconciliation.get("completion_source_event_id") or "")
            != source_event_id
            or str(reconciliation.get("baseline_ref") or "") != baseline
            or str(reconciliation.get("implementation_commit") or "")
            != implementation
            or str(reconciliation.get("landed_commit") or "") != implementation
            or re.fullmatch(r"[0-9a-f]{40}", integration) is None
            or target != integration
            or reconciliation.get("completion_task_cids") != expected_task_cids
            or str(reconciliation.get("reconciled_candidate_key") or "")
            != candidate_key
            or merge_result != expected_merge_result
        ):
            return False

        proof = reconciliation.get("integration_commit_proof")
        if (
            not isinstance(proof, Mapping)
            or set(proof)
            != {
                "implementation_commit",
                "integration_commit",
                "integration_ref",
                "passed",
                "reasons",
                "target_branch",
            }
            or proof.get("passed") is not True
            or proof.get("reasons") != []
            or str(proof.get("implementation_commit") or "") != implementation
            or str(proof.get("integration_commit") or "") != integration
            or str(proof.get("integration_ref") or "") != integration
            or str(proof.get("target_branch") or "")
            != str(source_merge.get("target_branch") or "")
        ):
            return False

        invariant = reconciliation.get("post_merge_declared_output_invariant")
        if (
            not isinstance(invariant, Mapping)
            or set(invariant)
            != {
                "checks",
                "missing_outputs",
                "mode",
                "passed",
                "reason",
                "repository_ref",
                "task_ids",
                "unsafe_outputs",
                "untracked_outputs",
            }
            or invariant.get("passed") is not True
            or invariant.get("mode") != "repository_tree"
            or invariant.get("reason") != "declared_outputs_tracked"
            or str(invariant.get("repository_ref") or "") != integration
            or invariant.get("missing_outputs") != []
            or invariant.get("unsafe_outputs") != []
            or invariant.get("untracked_outputs") != []
            or not isinstance(invariant.get("checks"), list)
            or invariant.get("task_ids") != list(expected_task_cids)
        ):
            return False
        for check in invariant["checks"]:
            if (
                not isinstance(check, Mapping)
                or set(check)
                != {
                    "exists",
                    "path",
                    "reason",
                    "repository",
                    "repository_ref",
                    "task_id",
                    "tracked",
                    "tracked_path",
                }
                or check.get("exists") is not True
                or check.get("tracked") is not True
                or check.get("reason") != "declared_output_tracked"
                or not str(check.get("path") or "")
                or not str(check.get("repository") or "")
                or check.get("tracked_path") != check.get("path")
                or str(check.get("repository_ref") or "") != integration
                or str(check.get("task_id") or "") not in expected_task_cids
            ):
                return False

        receipt_evidence = reconciliation.get("completion_receipt_evidence")
        if not isinstance(receipt_evidence, Mapping):
            return False
        receipt_body = dict(receipt_evidence)
        receipt_id = str(receipt_body.pop("receipt_id", "") or "")
        receipts = receipt_body.get("completion_receipts")
        if (
            set(receipt_evidence)
            != {
                "schema",
                "request_id",
                "completion_source_event_id",
                "integration_commit",
                "completion_task_cids",
                "completion_receipts",
                "receipt_id",
            }
            or receipt_body.get("schema")
            != (
                "ipfs_accelerate_py.agent_supervisor."
                "merge-queue-callback-completion-receipt@1"
            )
            or receipt_body.get("request_id") != request_id
            or receipt_body.get("completion_source_event_id") != source_event_id
            or receipt_body.get("integration_commit") != integration
            or receipt_body.get("completion_task_cids") != expected_task_cids
            or not isinstance(receipts, list)
            or receipt_id != content_identity(receipt_body)
        ):
            return False
        receipt_cids: dict[str, str] = {}
        for receipt in receipts:
            if (
                not isinstance(receipt, Mapping)
                or set(receipt)
                != {
                    "board_namespace",
                    "canonical_task_cid",
                    "canonical_task_key",
                    "schema",
                    "status",
                    "task_id",
                }
                or receipt.get("schema")
                != (
                    "ipfs_accelerate_py.agent_supervisor."
                    "member_completion_receipt@1"
                )
                or receipt.get("status") != "succeeded"
                or not str(receipt.get("board_namespace") or "")
                or not str(receipt.get("canonical_task_key") or "")
            ):
                return False
            receipt_cids[str(receipt.get("task_id") or "")] = str(
                receipt.get("canonical_task_cid") or ""
            )
        return (
            len(receipts) == len(expected_task_cids)
            and receipt_cids == expected_task_cids
        )

    @classmethod
    def _completion_event_evidence(
        cls,
        paths: DatabasePortalAttemptPaths,
        *,
        alias: str,
        task_cid: str,
    ) -> Mapping[str, Any] | None:
        """Bind projected completion to one exact implementation commit.

        Portal's canonical ``task_completed`` event binds the task revision,
        but older producers do not copy the implementation commit into that
        event.  In that case the same verified event chain must contain a
        successful, task-bound implementation/reconciliation event before the
        completion event.  Conflicting commit evidence fails closed.
        """

        events = cls._verified_event_chain(paths)
        completions = [
            (index, event)
            for index, event in enumerate(events)
            if (
                event.get("type") == "task_completed"
                and str(event.get("task_id") or "") == alias
                and str(event.get("canonical_task_cid") or "") == task_cid
            )
        ]
        if not completions:
            return None
        if len(completions) != 1:
            raise DatabasePortalBridgeError(
                "Portal completion event evidence is ambiguous"
            )
        completion_index, completion = completions[0]
        completion_event_id = str(completion.get("event_id") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", completion_event_id):
            raise DatabasePortalBridgeError(
                "Portal completion event identity is malformed"
            )

        completion_commits: set[str] = set()
        direct_commit = str(completion.get("implementation_commit") or "")
        if direct_commit:
            if not re.fullmatch(r"[0-9a-f]{40}", direct_commit):
                raise DatabasePortalBridgeError(
                    "Portal completion event implementation commit is malformed"
                )
            completion_commits.add(direct_commit)

        reconciled_commits: set[str] = set()
        reconciled_commit_indices: dict[str, list[int]] = {}
        callback_reconciliations: list[tuple[int, Mapping[str, Any]]] = []
        callback_reconciliation_indices: dict[str, list[int]] = {}
        sources: list[dict[str, Any]] = []
        for event_index, event in enumerate(events[:completion_index]):
            if (
                str(event.get("task_id") or "") != alias
                or str(event.get("canonical_task_cid") or "") != task_cid
            ):
                continue
            event_type = str(event.get("type") or "")
            merge_result = event.get("merge_result")
            validation_result = event.get("validation_result")
            admissible = False
            if event_type == "implementation_finished":
                admissible = bool(
                    isinstance(validation_result, Mapping)
                    and validation_result.get("passed") is True
                    and isinstance(merge_result, Mapping)
                )
            elif event_type == "worktree_reconciliation_candidate_queued":
                admissible = bool(
                    event.get("attempt_consumed") is False
                    and event.get("provider_dispatched") is False
                    and isinstance(validation_result, Mapping)
                    and validation_result.get("passed") is True
                    and isinstance(merge_result, Mapping)
                    and merge_result.get("queued") is True
                )
            elif event_type == "merge_reconciled":
                if (
                    event.get("resolved") is True
                    and isinstance(merge_result, Mapping)
                    and merge_result.get("merged") is True
                ):
                    if (
                        str(event.get("reason") or "")
                        == "merge_queue_callback_completed"
                    ):
                        callback_reconciliations.append((event_index, event))
                        continue
                    reconciled_commit = str(
                        event.get("implementation_commit") or ""
                    )
                    if not re.fullmatch(r"[0-9a-f]{40}", reconciled_commit):
                        raise DatabasePortalBridgeError(
                            "Portal reconciled completion has no exact commit"
                        )
                    completion_commits.add(reconciled_commit)
                    reconciled_commits.add(reconciled_commit)
                    reconciled_commit_indices.setdefault(
                        reconciled_commit,
                        [],
                    ).append(event_index)
                continue
            if not admissible:
                continue
            implementation_commit = str(
                event.get("implementation_commit")
                or (
                    event.get("completion_tree_id")
                    if event.get("authority_revalidation_only") is True
                    else ""
                )
                or ""
            )
            baseline_commit = str(event.get("baseline_ref") or "")
            source_event_id = str(event.get("event_id") or "")
            portal_attempt = event.get("attempt")
            if (
                not re.fullmatch(r"[0-9a-f]{40}", implementation_commit)
                or not re.fullmatch(r"[0-9a-f]{40}", baseline_commit)
                or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}", source_event_id
                )
                or isinstance(portal_attempt, bool)
                or not isinstance(portal_attempt, int)
                or portal_attempt < 1
            ):
                raise DatabasePortalBridgeError(
                    "Portal completion source lacks exact commit lineage"
                )
            sources.append(
                {
                    "implementation_commit": implementation_commit,
                    "baseline_commit": baseline_commit,
                    "completion_source_event_id": source_event_id,
                    "completion_source_event_type": event_type,
                    "completion_source_portal_attempt": portal_attempt,
                    "_source_event_index": event_index,
                    "_source_event": event,
                    "_source_merged": merge_result.get("merged") is True,
                    "_source_queued": merge_result.get("queued") is True,
                }
            )

        source_by_event_id = {
            str(source.get("completion_source_event_id") or ""): source
            for source in sources
        }
        if len(source_by_event_id) != len(sources):
            raise DatabasePortalBridgeError(
                "Portal completion source event identity is ambiguous"
            )

        def exact_projected_source_history(
            source_record: Mapping[str, Any],
        ) -> bool:
            """Bind a local deterministic projection to its prior enqueue."""

            source_event = source_record.get("_source_event")
            if not isinstance(source_event, Mapping):
                return False
            if source_event.get("type") != (
                "worktree_reconciliation_candidate_queued"
            ):
                return True
            if source_event.get("reason") == (
                "database_portal_merge_continuation_source_projected"
            ):
                # The exact, content-addressed continuation provenance above
                # is the sealed local replay record.  Reopening producer
                # history here would make completion depend on mutable
                # cross-attempt availability.
                return True
            provenance = source_event.get("merge_queue_synchronous_source")
            source_merge = source_event.get("merge_result")
            if not isinstance(provenance, Mapping) or not isinstance(
                source_merge, Mapping
            ):
                return False
            source_index = int(source_record.get("_source_event_index") or 0)
            enqueue_event_id = str(
                provenance.get("merge_candidate_enqueued_event_id") or ""
            )
            enqueue_matches = [
                event
                for event in events[:source_index]
                if event.get("type") == "merge_candidate_enqueued"
                and str(event.get("event_id") or "") == enqueue_event_id
            ]
            if len(enqueue_matches) != 1:
                return False
            enqueue = enqueue_matches[0]
            return bool(
                str(enqueue.get("task_id") or "") == alias
                and str(enqueue.get("canonical_task_cid") or "") == task_cid
                and enqueue.get("canonical_task_key")
                == source_event.get("canonical_task_key")
                and enqueue.get("attempt") == source_event.get("attempt")
                and enqueue.get("request_id") == source_merge.get("request_id")
                and enqueue.get("branch") == source_event.get("branch")
                and enqueue.get("baseline_ref")
                == source_event.get("baseline_ref")
                and enqueue.get("implementation_commit")
                == source_event.get("implementation_commit")
                and enqueue.get("attempted") is False
                and enqueue.get("queued") is True
                and enqueue.get("merged") is False
                and enqueue.get("reason") == "merge_queued"
                and enqueue.get("completion_task_cids")
                == source_merge.get("completion_task_cids")
                and enqueue.get("target_repository_id")
                == source_merge.get("target_repository_id")
                and enqueue.get("target_branch")
                == source_merge.get("target_branch")
            )

        for reconciliation_index, reconciliation in callback_reconciliations:
            reconciliation_source_id = str(
                reconciliation.get("completion_source_event_id") or ""
            )
            reconciliation_source = source_by_event_id.get(
                reconciliation_source_id
            )
            if (
                reconciliation_source is None
                or reconciliation_index
                <= int(reconciliation_source["_source_event_index"])
                or not exact_projected_source_history(reconciliation_source)
                or not cls._exact_callback_reconciliation_for_completion_source(
                    reconciliation,
                    reconciliation_source["_source_event"],
                    alias=alias,
                    task_cid=task_cid,
                )
            ):
                raise DatabasePortalBridgeError(
                    "Portal callback reconciliation binding is invalid"
                )
            if callback_reconciliation_indices.get(reconciliation_source_id):
                raise DatabasePortalBridgeError(
                    "Portal callback reconciliation binding is ambiguous"
                )
            callback_reconciliation_indices.setdefault(
                reconciliation_source_id,
                [],
            ).append(reconciliation_index)
            reconciled_commit = str(
                reconciliation.get("implementation_commit") or ""
            )
            completion_commits.add(reconciled_commit)
            reconciled_commits.add(reconciled_commit)
            reconciled_commit_indices.setdefault(
                reconciled_commit,
                [],
            ).append(reconciliation_index)

        source_commits = {
            source["implementation_commit"] for source in sources
        }
        if not completion_commits:
            completion_commits = set(source_commits)
        if len(completion_commits) != 1:
            raise DatabasePortalBridgeError(
                "Portal completion lacks one exact implementation commit"
            )
        implementation_commit = next(iter(completion_commits))
        matching_sources = [
            source
            for source in sources
            if source["implementation_commit"] == implementation_commit
        ]
        if len(matching_sources) == 2:
            queued_sources = [
                source
                for source in matching_sources
                if source.get("_source_queued") is True
                and source.get("_source_event", {}).get("type")
                == "worktree_reconciliation_candidate_queued"
            ]
            terminal_sources = [
                source
                for source in matching_sources
                if source.get("_source_merged") is True
                and source.get("_source_event", {}).get("type")
                == "implementation_finished"
            ]
            if len(queued_sources) == 1 and len(terminal_sources) == 1:
                queued_source = queued_sources[0]
                terminal_source = terminal_sources[0]
                queued_event = queued_source["_source_event"]
                terminal_event = terminal_source["_source_event"]
                queued_merge = queued_event.get("merge_result")
                terminal_merge = terminal_event.get("merge_result")
                terminal_validation = terminal_event.get("validation_result")
                reconciliation_positions = callback_reconciliation_indices.get(
                    str(queued_source.get("completion_source_event_id") or ""),
                    [],
                )
                reconciliation_event = (
                    events[reconciliation_positions[0]]
                    if len(reconciliation_positions) == 1
                    else {}
                )
                ordered_confirmation = bool(
                    len(reconciliation_positions) == 1
                    and queued_source["_source_event_index"]
                    < reconciliation_positions[0]
                    < terminal_source["_source_event_index"]
                    and terminal_source["baseline_commit"]
                    == queued_source["baseline_commit"]
                    and terminal_source["completion_source_portal_attempt"]
                    == queued_source["completion_source_portal_attempt"]
                    and terminal_event.get("returncode") == 0
                    and terminal_event.get("attempt_consumed") is True
                    and type(terminal_event.get("provider_dispatched")) is bool
                    and terminal_event.get("branch")
                    == queued_event.get("branch")
                    and terminal_event.get("canonical_task_key")
                    == queued_event.get("canonical_task_key")
                    and isinstance(terminal_validation, Mapping)
                    and terminal_validation.get("attempted") is True
                    and terminal_validation.get("passed") is True
                    and type(terminal_validation.get("returncode")) is int
                    and terminal_validation.get("returncode") == 0
                    and terminal_event.get("board_completion")
                    == {
                        "complete": True,
                        "pending_merge": False,
                        "reason": "merged_into_target",
                    }
                    and isinstance(queued_merge, Mapping)
                    and isinstance(terminal_merge, Mapping)
                    and terminal_merge.get("attempted") is True
                    and terminal_merge.get("merged") is True
                    and terminal_merge.get("queued") is False
                    and terminal_merge.get("request_id")
                    == queued_merge.get("request_id")
                    and terminal_merge.get("branch")
                    == queued_merge.get("branch")
                    and terminal_merge.get("implementation_commit")
                    == queued_merge.get("implementation_commit")
                    and terminal_merge.get("canonical_task_key")
                    == queued_merge.get("canonical_task_key")
                    and terminal_merge.get("canonical_task_cid")
                    == queued_merge.get("canonical_task_cid")
                    and terminal_merge.get("completion_task_cids")
                    == queued_merge.get("completion_task_cids")
                    and terminal_merge.get("target_repository_id")
                    == queued_merge.get("target_repository_id")
                    and terminal_merge.get("target_branch")
                    == queued_merge.get("target_branch")
                    and terminal_merge.get("merge_commit")
                    == reconciliation_event.get("merge_commit")
                    and terminal_merge.get("target_commit")
                    == reconciliation_event.get("target_commit")
                )
                if ordered_confirmation:
                    # The producer appends this terminal confirmation after
                    # the synchronous callback returns.  It supersedes only
                    # the one reconciled deterministic source it confirms.
                    matching_sources = [terminal_source]
        source_handoff_admitted = False
        if len(matching_sources) == 1:
            selected_source = matching_sources[0]
            source_handoff_admitted = bool(
                selected_source.get("_source_merged") is True
                or (
                    selected_source.get("_source_queued") is True
                    and (
                        bool(
                            callback_reconciliation_indices.get(
                                str(
                                    selected_source.get(
                                        "completion_source_event_id"
                                    )
                                    or ""
                                )
                            )
                        )
                        or (
                            implementation_commit in reconciled_commits
                            and any(
                                reconciliation_index
                                > selected_source["_source_event_index"]
                                for reconciliation_index in (
                                    reconciled_commit_indices.get(
                                        implementation_commit,
                                        (),
                                    )
                                )
                            )
                        )
                    )
                )
            )
        if (
            len(matching_sources) != 1
            or not source_handoff_admitted
        ):
            raise DatabasePortalBridgeError(
                "Portal completion lacks one exact evaluated baseline"
            )
        source = matching_sources[-1]
        source.pop("_source_event_index", None)
        source.pop("_source_event", None)
        source.pop("_source_merged", None)
        source.pop("_source_queued", None)
        claimed_source_event_id = str(
            completion.get("completion_source_event_id") or ""
        )
        claimed_baseline_commit = str(
            completion.get("baseline_commit") or ""
        )
        claimed_portal_attempt = completion.get("attempt")
        if (
            claimed_source_event_id
            and claimed_source_event_id
            != source["completion_source_event_id"]
        ) or (
            claimed_baseline_commit
            and claimed_baseline_commit != source["baseline_commit"]
        ) or (
            claimed_portal_attempt is not None
            and claimed_portal_attempt
            != source["completion_source_portal_attempt"]
        ):
            raise DatabasePortalBridgeError(
                "Portal completion event conflicts with its source lineage"
            )
        return {
            **source,
            "completion_event_id": completion_event_id,
        }

    @classmethod
    def _ensure_protected_recovery_completion_event(
        cls,
        paths: DatabasePortalAttemptPaths,
        *,
        alias: str,
        task_cid: str,
        baseline_commit: str,
        implementation_commit: str,
        queue_reconciliation_proven: bool = False,
    ) -> Mapping[str, Any]:
        """Repair the exact private completion event after a zero-provider merge.

        Portal's merge callback durably updates the private task projection,
        but the ordinary next daemon pass is what normally appends
        ``task_completed``.  Protected recovery cannot call ``run_once`` (that
        could dispatch a provider), so it may append only this closed repair
        after proving the exact zero-provider source event and commit lineage.
        """

        existing = cls._completion_event_evidence(
            paths,
            alias=alias,
            task_cid=task_cid,
        )
        if existing is not None:
            if (
                str(existing.get("baseline_commit") or "")
                != baseline_commit
                or str(existing.get("implementation_commit") or "")
                != implementation_commit
            ):
                raise DatabasePortalBridgeError(
                    "protected recovery completion event changed lineage"
                )
            return existing

        sources: list[Mapping[str, Any]] = []
        for event in cls._verified_event_chain(paths):
            if (
                str(event.get("task_id") or "") != alias
                or str(event.get("canonical_task_cid") or "") != task_cid
                or str(event.get("baseline_ref") or "") != baseline_commit
                or str(event.get("implementation_commit") or "")
                != implementation_commit
                or event.get("attempt_consumed") is not False
                or event.get("provider_dispatched") is not False
            ):
                continue
            validation_result = event.get("validation_result")
            merge_result = event.get("merge_result")
            event_type = str(event.get("type") or "")
            if (
                not isinstance(validation_result, Mapping)
                or validation_result.get("passed") is not True
                or not isinstance(merge_result, Mapping)
                or (
                    event_type == "implementation_finished"
                    and merge_result.get("merged") is not True
                )
                or (
                    event_type
                    == "worktree_reconciliation_candidate_queued"
                    and merge_result.get("queued") is not True
                )
                or event_type
                not in {
                    "implementation_finished",
                    "worktree_reconciliation_candidate_queued",
                }
            ):
                continue
            sources.append(event)
        if len(sources) != 1:
            raise DatabasePortalBridgeError(
                "protected recovery lacks one exact completion source event"
            )
        source_event_id = str(sources[0].get("event_id") or "")
        source_portal_attempt = sources[0].get("attempt")
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", source_event_id)
            or isinstance(source_portal_attempt, bool)
            or not isinstance(source_portal_attempt, int)
            or source_portal_attempt < 1
        ):
            raise DatabasePortalBridgeError(
                "protected recovery completion source identity is malformed"
            )
        source_event_type = str(sources[0].get("type") or "")
        if source_event_type == "worktree_reconciliation_candidate_queued":
            if not queue_reconciliation_proven:
                raise DatabasePortalBridgeError(
                    "protected recovery queued completion lacks exact "
                    "reconciliation proof"
                )
            source_merge_result = sources[0].get("merge_result")
            source_request_id = str(
                source_merge_result.get("request_id")
                if isinstance(source_merge_result, Mapping)
                else ""
            )
            if not source_request_id:
                raise DatabasePortalBridgeError(
                    "protected recovery queued completion has no request "
                    "identity"
                )
            append_jsonl_event(
                paths.events,
                "merge_reconciled",
                {
                    "task_id": alias,
                    "canonical_task_cid": task_cid,
                    "attempt": source_portal_attempt,
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                    "baseline_ref": baseline_commit,
                    "implementation_commit": implementation_commit,
                    "completion_source_event_id": source_event_id,
                    "resolved": True,
                    "reason": "protected_recovery_merge_completed",
                    "merge_result": {
                        "attempted": True,
                        "merged": True,
                        "queued": False,
                        "request_id": source_request_id,
                        "reason": "protected_recovery_merge_completed",
                    },
                },
            )
        append_jsonl_event(
            paths.events,
            "task_completed",
            {
                "task_id": alias,
                "canonical_task_cid": task_cid,
                "implementation_commit": implementation_commit,
                "baseline_commit": baseline_commit,
                "completion_source_event_id": source_event_id,
                "attempt": source_portal_attempt,
                "completion_receipt_repair": True,
                "reason": "protected_recovery_merge_completed",
            },
        )
        repaired = cls._completion_event_evidence(
            paths,
            alias=alias,
            task_cid=task_cid,
        )
        if (
            repaired is None
            or str(repaired.get("baseline_commit") or "")
            != baseline_commit
            or str(repaired.get("implementation_commit") or "")
            != implementation_commit
            or str(repaired.get("completion_source_event_id") or "")
            != source_event_id
        ):
            raise DatabasePortalBridgeError(
                "protected recovery completion event repair failed"
            )
        return repaired

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
    def _typed_deferral(
        result: Mapping[str, Any],
    ) -> tuple[str, int] | None:
        """Return exact Portal deferral data without parsing reason text."""

        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return None
        # ``attempt_consumed=false``/``provider_dispatched=false`` also
        # describe a successful deterministic zero-provider closure.  Only
        # the explicit closed deferral signal grants retry semantics.
        if implementation.get("deferred") is not True:
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

    def _capacity_retry_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        implementation: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Bind one exact protected capacity proof to this database claim."""

        if not paths.events.is_file():
            return None
        try:
            events = self._verified_event_chain(paths)
        except DatabasePortalBridgeError:
            return None
        event_matches = [
            (index, event)
            for index, event in enumerate(events)
            if event.get("type")
            == "implementation_post_dispatch_capacity_retry"
            and event.get("task_id")
            == str(binding.get("task_alias") or "")
            and event.get("canonical_task_cid") == str(attempt.task_cid)
            and isinstance(event.get("post_dispatch_capacity_retry"), Mapping)
        ]
        if len(event_matches) != 1:
            return None
        authoritative_index, authoritative = event_matches[0]
        trailing = events[authoritative_index + 1 :]
        if (
            len(trailing) > 1
            or any(event.get("type") != "daemon_pass" for event in trailing)
            or any(
                str(event.get("active_task_id") or "")
                not in {"", str(binding.get("task_alias") or "")}
                for event in trailing
            )
        ):
            # A later task/implementation disposition supersedes this event;
            # an old capacity rejection must never override later success.
            return None
        closed_fields = (
            "task_id",
            "canonical_task_cid",
            "attempt",
            "returncode",
            "retryable",
            "deferred",
            "attempt_consumed",
            "provider_dispatched",
            "typed_deferral_slot_consumed",
            "reason",
            "failure_class",
            "providers",
            "post_dispatch_capacity_retry",
            "quota_probe_receipt",
            "route_outcome",
            "codex_capacity_receipt",
        )
        if (
            implementation.get("post_dispatch_capacity_retry") is not None
            and any(
                implementation.get(name) != authoritative.get(name)
                for name in closed_fields
            )
        ):
            return None
        proof = authoritative.get("post_dispatch_capacity_retry")
        primary = authoritative.get("quota_probe_receipt")
        outcome = authoritative.get("route_outcome")
        capacity = authoritative.get("codex_capacity_receipt")
        if not all(
            isinstance(item, Mapping)
            for item in (proof, primary, outcome, capacity)
        ):
            return None
        proof = dict(proof)
        primary = dict(primary)
        outcome = dict(outcome)
        capacity = dict(capacity)
        proof_fields = {
            "schema",
            "task_id",
            "attempt",
            "task_revision_cid",
            "logical_attempt_id",
            "invocation_binding_id",
            "route_id",
            "decision_id",
            "primary_receipt_id",
            "route_outcome_id",
            "capacity_receipt_id",
            "fallback_provider_id",
            "fallback_model_id",
            "fallback_reasoning_effort",
            "fallback_returncode",
            "provider_dispatched",
            "attempt_consumed",
            "observed_at_ms",
            "retry_not_before_ms",
            "proof_id",
        }
        capacity_fields = {
            "schema",
            "source",
            "failure_class",
            "reason_code",
            "primary_receipt_id",
            "nonce",
            "route_id",
            "invocation_binding_id",
            "logical_attempt_id",
            "fallback_provider_id",
            "fallback_model_id",
            "fallback_reasoning_effort",
            "fallback_returncode",
            "outcome_decision",
            "decision_id",
            "provider_dispatched",
            "candidate_activity_observed",
            "attempt_consumed",
            "completion_authority",
            "observed_at_ms",
            "retry_not_before_ms",
            "evidence_kind",
            "evidence_sha256",
            "evidence_bytes",
            "evidence_overflow",
            "receipt_id",
        }
        portal_attempt = authoritative.get("attempt")
        returncode = authoritative.get("returncode")
        route_plan = outcome.get("route_plan")
        observed_at_ms = capacity.get("observed_at_ms")
        retry_not_before_ms = capacity.get("retry_not_before_ms")
        if (
            authoritative.get("retryable") is not True
            or authoritative.get("deferred") is not False
            or authoritative.get("reason")
            != "provider_capacity_exhausted"
            or authoritative.get("failure_class")
            != "dual_provider_capacity_exhausted"
            or authoritative.get("providers") != ["grok", "codex"]
            or authoritative.get("attempt_consumed") is not True
            or authoritative.get("provider_dispatched") is not True
            or authoritative.get("typed_deferral_slot_consumed") is not False
            or isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode == 0
            or isinstance(portal_attempt, bool)
            or not isinstance(portal_attempt, int)
            or portal_attempt < 1
            or portal_attempt > self.max_task_attempts
            or set(proof) != proof_fields
            or proof.get("schema")
            != (
                "ipfs_accelerate_py.agent_supervisor."
                "post-dispatch-capacity-retry-proof@1"
            )
            or proof.get("proof_id")
            != _sha256_bytes(
                _canonical_json(
                    {
                        key: value
                        for key, value in proof.items()
                        if key != "proof_id"
                    }
                )
            )
            or proof.get("task_id")
            != str(binding.get("task_alias") or "")
            or proof.get("attempt") != portal_attempt
            or proof.get("task_revision_cid") != str(attempt.task_cid)
            or proof.get("provider_dispatched") is not True
            or proof.get("attempt_consumed") is not True
            or set(capacity) != capacity_fields
            or capacity.get("schema")
            != (
                "ipfs_accelerate_py.agent_supervisor."
                "codex-terminal-capacity-receipt@1"
            )
            or capacity.get("source") != "grok_cli_runner"
            or capacity.get("failure_class") != "usage_limit"
            or capacity.get("reason_code") != "codex_usage_limit_reached"
            or capacity.get("provider_dispatched") is not True
            or capacity.get("candidate_activity_observed") is not False
            or capacity.get("attempt_consumed") is not True
            or capacity.get("completion_authority") is not False
            or capacity.get("evidence_overflow") is not False
            or capacity.get("fallback_provider_id") != "codex"
            or capacity.get("fallback_model_id") != "gpt-5.6-terra"
            or capacity.get("fallback_reasoning_effort")
            not in {"medium", "high"}
            or capacity.get("fallback_returncode") != returncode
            or capacity.get("outcome_decision") != "fallback_failed"
            or capacity.get("receipt_id")
            != _content_addressed_record(
                capacity,
                identity_field="receipt_id",
            )
            or outcome.get("outcome_id")
            != _content_addressed_record(outcome, identity_field="outcome_id")
            or primary.get("receipt_id")
            != _content_addressed_record(primary, identity_field="receipt_id")
            or not isinstance(route_plan, Mapping)
            or route_plan.get("fallback_provider_id") != "codex"
            or route_plan.get("fallback_model_id") != "gpt-5.6-terra"
            or route_plan.get("fallback_reasoning_effort")
            != capacity.get("fallback_reasoning_effort")
            or outcome.get("decision") != "fallback_failed"
            or outcome.get("fallback_dispatched") is not True
            or outcome.get("fallback_returncode") != returncode
            or outcome.get("fallback_capacity_receipt") != capacity
            or outcome.get("preflight_receipt_id")
            != primary.get("receipt_id")
            or proof.get("primary_receipt_id")
            != primary.get("receipt_id")
            or proof.get("route_outcome_id") != outcome.get("outcome_id")
            or proof.get("capacity_receipt_id")
            != capacity.get("receipt_id")
            or proof.get("logical_attempt_id")
            != capacity.get("logical_attempt_id")
            or proof.get("invocation_binding_id")
            != capacity.get("invocation_binding_id")
            or outcome.get("invocation_binding_id")
            != capacity.get("invocation_binding_id")
            or proof.get("route_id") != capacity.get("route_id")
            or route_plan.get("route_id") != capacity.get("route_id")
            or proof.get("decision_id") != capacity.get("decision_id")
            or outcome.get("decision_id") != capacity.get("decision_id")
            or proof.get("fallback_provider_id") != "codex"
            or proof.get("fallback_model_id") != "gpt-5.6-terra"
            or proof.get("fallback_reasoning_effort")
            != capacity.get("fallback_reasoning_effort")
            or proof.get("fallback_returncode") != returncode
            or proof.get("observed_at_ms") != observed_at_ms
            or proof.get("retry_not_before_ms") != retry_not_before_ms
            or isinstance(observed_at_ms, bool)
            or not isinstance(observed_at_ms, int)
            or observed_at_ms <= 0
            or isinstance(retry_not_before_ms, bool)
            or not isinstance(retry_not_before_ms, int)
            or (
                retry_not_before_ms != 0
                and not (
                    observed_at_ms < retry_not_before_ms
                    <= observed_at_ms
                    + _MAX_DATABASE_PORTAL_CAPACITY_BACKOFF_SECONDS * 1000
                )
            )
        ):
            return None
        now_ms = int(time.time() * 1000)
        if retry_not_before_ms:
            backoff_seconds = min(
                _MAX_DATABASE_PORTAL_CAPACITY_BACKOFF_SECONDS,
                max(0, (retry_not_before_ms - now_ms + 999) // 1000),
            )
        else:
            backoff_seconds = min(
                _MAX_DATABASE_PORTAL_BACKOFF_SECONDS,
                max(15 * 60, 15 * 60 * (2 ** min(portal_attempt - 1, 6))),
            )
        receipt: dict[str, Any] = {
            "schema": DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA,
            "disposition": "retry",
            "reason": "dual_provider_capacity_exhausted",
            "task_cid": str(attempt.task_cid),
            "task_alias": str(binding.get("task_alias") or ""),
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "portal_attempt": portal_attempt,
            "ordinary_retry_generation": portal_attempt,
            "max_task_attempts": int(self.max_task_attempts),
            "remaining_task_attempts": int(
                self.max_task_attempts - portal_attempt
            ),
            "attempt_consumed": True,
            "provider_dispatched": True,
            "backoff_seconds": int(backoff_seconds),
            "retry_not_before_ms": retry_not_before_ms,
            "binding_id": str(binding.get("binding_id") or ""),
            "events_digest": _sha256_file(paths.events),
            "event_stream_id": str(authoritative.get("stream_id") or ""),
            "implementation_event_id": str(
                authoritative.get("event_id") or ""
            ),
            "post_dispatch_capacity_proof": proof,
            "primary_receipt": primary,
            "route_outcome": outcome,
            "codex_capacity_receipt": capacity,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    @staticmethod
    def _looks_like_validation_retry(
        implementation: Mapping[str, Any],
    ) -> bool:
        """Select only the closed post-dispatch validation-failure shape."""

        validation = implementation.get("validation_result")
        return bool(
            implementation.get("returncode") not in (None, 0)
            and implementation.get("attempt_consumed") is True
            and implementation.get("provider_dispatched") is True
            and isinstance(validation, Mapping)
            and validation.get("attempted") is True
            and validation.get("passed") is False
            and validation.get("reason") == "declared_validation_failed"
        )

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

    def _preserved_commit_descends_from(
        self,
        *,
        baseline_commit: str,
        preserved_commit: str,
    ) -> bool:
        """Bind a preserved candidate to its exact immutable baseline."""

        if (
            self.repository_root is None
            or re.fullmatch(r"[0-9a-f]{40}", baseline_commit) is None
            or re.fullmatch(r"[0-9a-f]{40}", preserved_commit) is None
            or baseline_commit == preserved_commit
        ):
            return False
        try:
            baseline = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"{baseline_commit}^{{commit}}",
                ],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            candidate = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"{preserved_commit}^{{commit}}",
                ],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            ancestry = subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    baseline_commit,
                    preserved_commit,
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
            baseline.returncode == 0
            and baseline.stdout.strip() == baseline_commit
            and candidate.returncode == 0
            and candidate.stdout.strip() == preserved_commit
            and ancestry.returncode == 0
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

    def _protected_path_preservation_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Bind one exact post-dispatch, externally interrupted candidate.

        A shared-checkout protected path can change after provider dispatch but
        before proposal validation.  Portal preserves the candidate and
        restores its ordinary-attempt counter.  On crash replay this terminal
        must be recognized before any carried retry seed is reinitialized;
        otherwise a successor state can reach another provider dispatch.

        Near-shape evidence fails closed.  In particular, workspace-scoped
        mutations, incomplete preservation, a substituted rescue ref, or a
        later execution event never degrade into an ordinary Portal pass.
        """

        events = self._verified_event_chain(paths)
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        preservation_type = "protected_path_interrupted_worktree_preserved"
        protected_reason = "implementation_protected_path_mutated"
        marker_present = any(
            event.get("type") == preservation_type
            or (
                event.get("type") == "implementation_finished"
                and (
                    event.get("reason") == protected_reason
                    or (
                        isinstance(
                            event.get("protected_path_violation"),
                            Mapping,
                        )
                        and event["protected_path_violation"].get("reason")
                        == protected_reason
                    )
                    or (
                        isinstance(
                            event.get("failed_preservation_result"),
                            Mapping,
                        )
                        and isinstance(
                            event["failed_preservation_result"].get(
                                "protected_path_violation"
                            ),
                            Mapping,
                        )
                        and event["failed_preservation_result"][
                            "protected_path_violation"
                        ].get("reason")
                        == protected_reason
                    )
                )
            )
            for event in events
        )
        if not marker_present:
            return None

        seed_event: Mapping[str, Any] | None = None
        terminal_events = events
        if events and str(events[0].get("type") or "") in (
            _CONSUMED_ATTEMPT_SEED_EVENT_FIELDS
        ):
            seed_event = events[0]
            terminal_events = events[1:]
        if tuple(
            str(event.get("type") or "") for event in terminal_events
        ) != _PROTECTED_PATH_PRESERVATION_EVENT_CHAIN:
            raise DatabasePortalBridgeError(
                "protected-path preservation event chain is not exact"
            )
        (
            selected_event,
            snapshot_event,
            started_event,
            kernel_event,
            mutation_event,
            cleanup_event,
            preserved_event,
            finished_event,
            daemon_pass_event,
        ) = terminal_events

        seed_receipt: Mapping[str, Any] | None = None
        if seed_event is not None:
            seed_type = str(seed_event.get("type") or "")
            seed_receipt_field = {
                "database_portal_validation_retry_seeded": (
                    "validation_retry_receipt"
                ),
                "database_portal_capacity_retry_seeded": (
                    "capacity_retry_receipt"
                ),
                "database_portal_consumed_attempt_retry_seeded": (
                    "consumed_attempt_retry_receipt"
                ),
            }[seed_type]
            seed_schema = {
                "database_portal_validation_retry_seeded": (
                    DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA
                ),
                "database_portal_capacity_retry_seeded": (
                    DATABASE_PORTAL_CAPACITY_RETRY_SEED_SCHEMA
                ),
                "database_portal_consumed_attempt_retry_seeded": (
                    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SEED_SCHEMA
                ),
            }[seed_type]
            nested_schema = {
                "database_portal_validation_retry_seeded": (
                    DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA
                ),
                "database_portal_capacity_retry_seeded": (
                    DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA
                ),
                "database_portal_consumed_attempt_retry_seeded": (
                    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA
                ),
            }[seed_type]
            candidate_seed_receipt = seed_event.get(seed_receipt_field)
            seed_identity_body = {
                key: value
                for key, value in seed_event.items()
                if key not in _PORTAL_EVENT_ENVELOPE_FIELDS
                and key != "seed_id"
            }
            if (
                set(seed_event)
                != _CONSUMED_ATTEMPT_SEED_EVENT_FIELDS[seed_type]
                or seed_event.get("schema") != seed_schema
                or seed_event.get("task_id") != alias
                or seed_event.get("canonical_task_cid") != task_cid
                or seed_event.get("target_database_attempt_id")
                != str(attempt.attempt_id)
                or seed_event.get("target_claim_id") != str(attempt.claim_id)
                or not str(
                    seed_event.get("source_database_attempt_id") or ""
                )
                or seed_event.get("source_database_attempt_id")
                == str(attempt.attempt_id)
                or seed_event.get("completion_authoritative") is not False
                or not isinstance(candidate_seed_receipt, Mapping)
                or candidate_seed_receipt.get("schema") != nested_schema
                or candidate_seed_receipt.get("receipt_id")
                != seed_event.get("source_retry_receipt_id")
                or candidate_seed_receipt.get("attempt_id")
                != seed_event.get("source_database_attempt_id")
                or candidate_seed_receipt.get("task_cid") != task_cid
                or candidate_seed_receipt.get("task_alias") != alias
                or candidate_seed_receipt.get("attempt_consumed") is not True
                or candidate_seed_receipt.get("provider_dispatched") is not True
                or seed_event.get("seed_id")
                != _sha256_bytes(_canonical_json(seed_identity_body))
            ):
                raise DatabasePortalBridgeError(
                    "protected-path preservation retry seed failed verification"
                )
            seed_receipt = candidate_seed_receipt

        portal_attempt = finished_event.get("attempt")
        source_revision = binding.get("task_revision")
        baseline_commit = str(started_event.get("baseline_ref") or "")
        branch = str(started_event.get("branch") or "")
        workspace_path = str(started_event.get("worktree_path") or "")
        preservation = finished_event.get("failed_preservation_result")
        violation = finished_event.get("protected_path_violation")
        validation = finished_event.get("validation_result")
        commit_result = (
            preservation.get("commit_result")
            if isinstance(preservation, Mapping)
            else None
        )
        cleanup_result = (
            preservation.get("cleanup_result")
            if isinstance(preservation, Mapping)
            else None
        )
        preserved_commit = str(
            preservation.get("preserved_commit")
            if isinstance(preservation, Mapping)
            else ""
        )
        implementation_commit = str(
            preservation.get("implementation_commit")
            if isinstance(preservation, Mapping)
            else ""
        )
        rescue_branch = str(
            preservation.get("rescue_branch")
            if isinstance(preservation, Mapping)
            else ""
        )
        preservation_fields = {
            "task_id",
            "attempt",
            "branch",
            "worktree_path",
            "started_at",
            "finished_at",
            "preserved",
            "rescue_branch",
            "implementation_commit",
            "preserved_commit",
            "commit_result",
            "cleanup_result",
            "pruned_seeded_context",
            "protected_path_violation",
        }
        preservation_event_body = {
            key: preserved_event.get(key) for key in preservation_fields
        }
        mutations = (
            violation.get("mutations")
            if isinstance(violation, Mapping)
            else None
        )
        mutation_paths: list[str] = []
        if isinstance(mutations, list):
            for mutation in mutations:
                if not isinstance(mutation, Mapping):
                    mutation_paths = []
                    break
                path = str(mutation.get("path") or "")
                try:
                    safe_path = _safe_repository_path(path)
                except DatabasePortalBridgeError:
                    mutation_paths = []
                    break
                mutation_paths.append(safe_path)
        protected_paths = (
            violation.get("protected_paths")
            if isinstance(violation, Mapping)
            else None
        )
        expected_rescue_branch = (
            "rescue/"
            + (
                branch.removeprefix("implementation/")
                .strip("/")
                .replace(" ", "-")
                or "implementation-attempt"
            )
            + "-protected-path-interrupted"
        )
        event_identity_values = (
            str(started_event.get("event_id") or ""),
            str(mutation_event.get("event_id") or ""),
            str(preserved_event.get("event_id") or ""),
            str(finished_event.get("event_id") or ""),
            str(binding.get("binding_id") or ""),
        )
        cleanup_event_body = {
            key: value
            for key, value in cleanup_event.items()
            if key not in _PORTAL_EVENT_ENVELOPE_FIELDS
        }
        seed_portal_attempt = (
            seed_receipt.get("portal_attempt")
            if isinstance(seed_receipt, Mapping)
            else None
        )
        if (
            selected_event.get("task_id") != alias
            or selected_event.get("canonical_task_cid") != task_cid
            or snapshot_event.get("task_id") != alias
            or snapshot_event.get("canonical_task_cid") != task_cid
            or snapshot_event.get("attempt") != portal_attempt
            or snapshot_event.get("workspace_path") != workspace_path
            or started_event.get("task_id") != alias
            or started_event.get("canonical_task_cid") != task_cid
            or started_event.get("attempt") != portal_attempt
            or started_event.get("provider_dispatched") is not False
            or kernel_event.get("task_id") != alias
            or kernel_event.get("canonical_task_cid") != task_cid
            or kernel_event.get("attempt") != portal_attempt
            or not branch.startswith("implementation/")
            or not workspace_path
            or isinstance(portal_attempt, bool)
            or not isinstance(portal_attempt, int)
            or portal_attempt < 1
            or (
                self.max_task_attempts > 0
                and portal_attempt > self.max_task_attempts
            )
            or isinstance(source_revision, bool)
            or not isinstance(source_revision, int)
            or source_revision < 1
            or (
                seed_event is not None
                and (
                    isinstance(seed_portal_attempt, bool)
                    or not isinstance(seed_portal_attempt, int)
                    or seed_portal_attempt < 1
                    or seed_portal_attempt + 1 != portal_attempt
                    or (
                        seed_event.get("portal_attempt") is not None
                        and seed_event.get("portal_attempt")
                        != seed_portal_attempt
                    )
                )
            )
            or mutation_event.get("task_id") != alias
            or mutation_event.get("canonical_task_cid") != task_cid
            or mutation_event.get("attempt") != portal_attempt
            or mutation_event.get("reason") != protected_reason
            or mutation_event.get("workspace_path") != workspace_path
            or not isinstance(preservation, Mapping)
            or set(preservation) != preservation_fields
            or any(
                preservation.get(key) != value
                for key, value in preservation_event_body.items()
            )
            or preserved_event.get("task_id") != alias
            or preserved_event.get("canonical_task_cid") != task_cid
            or preserved_event.get("attempt") != portal_attempt
            or preserved_event.get("branch") != branch
            or preserved_event.get("worktree_path") != workspace_path
            or preservation.get("preserved") is not True
            or not isinstance(commit_result, Mapping)
            or commit_result.get("committed") is not True
            or commit_result.get("commit") != preserved_commit
            or implementation_commit != preserved_commit
            or finished_event.get("task_id") != alias
            or finished_event.get("task_cid") != task_cid
            or finished_event.get("canonical_task_cid") != task_cid
            or finished_event.get("attempt") != portal_attempt
            or finished_event.get("branch") != branch
            or finished_event.get("baseline_ref") != baseline_commit
            or finished_event.get("worktree_path") != workspace_path
            or finished_event.get("reason") != protected_reason
            or finished_event.get("deferred") is not True
            or finished_event.get("returncode") != 1
            or finished_event.get("attempt_consumed") is not False
            or finished_event.get("provider_dispatched") is not True
            or finished_event.get("implementation_commit")
            != preserved_commit
            or finished_event.get("commit_result") != commit_result
            or finished_event.get("cleanup_result") != cleanup_result
            or finished_event.get("merge_result")
            != {"merged": False, "reason": "not_attempted"}
            or finished_event.get("board_completion")
            != {
                "complete": False,
                "pending_merge": False,
                "reason": "implementation_or_validation_failed",
            }
            or not isinstance(validation, Mapping)
            or validation.get("attempted") is not False
            or validation.get("passed") is not False
            or validation.get("returncode") != 1
            or validation.get("results") != []
            or validation.get("reason") != protected_reason
            or validation.get("protected_path_violation") != violation
            or not isinstance(violation, Mapping)
            or violation.get("reason") != protected_reason
            or violation.get("verification_deferred") is True
            or violation.get("workspace_path") != workspace_path
            or violation.get("task_id") != alias
            or violation.get("attempt") != portal_attempt
            or not isinstance(mutations, list)
            or not mutations
            or not mutation_paths
            or len(set(mutation_paths)) != len(mutation_paths)
            or {
                str(mutation.get("scope") or "")
                for mutation in mutations
                if isinstance(mutation, Mapping)
            }
            != {"shared_checkout"}
            or protected_paths != sorted(mutation_paths)
            or mutation_event.get("mutations") != mutations
            or mutation_event.get("protected_paths") != protected_paths
            or mutation_event.get("shared_checkout_restored") is not False
            or rescue_branch != expected_rescue_branch
            or not isinstance(cleanup_result, Mapping)
            or cleanup_result.get("cleaned") is not True
            or cleanup_event_body != cleanup_result
            or daemon_pass_event.get("active_task_id") not in {None, ""}
            or re.fullmatch(
                r"event-log:sha256:[0-9a-f]{64}",
                str(finished_event.get("stream_id") or ""),
            )
            is None
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                for value in event_identity_values
            )
            or not self._preserved_commit_exists(
                commit=preserved_commit,
                rescue_branch=rescue_branch,
            )
            or not self._preserved_commit_descends_from(
                baseline_commit=baseline_commit,
                preserved_commit=preserved_commit,
            )
        ):
            raise DatabasePortalBridgeError(
                "protected-path preservation terminal failed verification"
            )

        receipt: dict[str, Any] = {
            "schema": DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA,
            "disposition": "protected_candidate_preserved",
            "reason": protected_reason,
            "task_cid": task_cid,
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "source_task_revision": int(source_revision),
            "portal_attempt": int(portal_attempt),
            "attempt_consumed": False,
            "provider_dispatched": True,
            "completion_authoritative": False,
            "local_recovery_required": True,
            "mutation_scopes": ["shared_checkout"],
            "protected_paths": list(protected_paths),
            "baseline_commit": baseline_commit,
            "implementation_commit": preserved_commit,
            "preserved_commit": preserved_commit,
            "rescue_branch": rescue_branch,
            "original_branch": branch,
            "original_worktree_path": workspace_path,
            "binding_id": str(binding.get("binding_id") or ""),
            "events_digest": _sha256_file(paths.events),
            "event_stream_id": str(finished_event.get("stream_id") or ""),
            "implementation_started_event_id": str(
                started_event.get("event_id") or ""
            ),
            "protected_mutation_event_id": str(
                mutation_event.get("event_id") or ""
            ),
            "preservation_event_id": str(
                preserved_event.get("event_id") or ""
            ),
            "implementation_finished_event_id": str(
                finished_event.get("event_id") or ""
            ),
            "protected_path_violation_digest": _sha256_bytes(
                _canonical_json(violation)
            ),
            "preservation_digest": _sha256_bytes(
                _canonical_json(preservation)
            ),
        }
        receipt["receipt_id"] = _content_addressed_record(
            receipt,
            identity_field="receipt_id",
        )
        return receipt

    def _consumed_attempt_retry_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Bind one legacy, unclassified post-dispatch failure to its evidence.

        This recovery deliberately makes no provider-capacity claim.  It is
        eligible only for the closed historical event shape where Portal
        charged an ordinary attempt, dispatched the provider, produced no
        candidate, skipped validation, and recorded no later disposition.
        """

        attempt_number = getattr(attempt, "attempt_number", 0)
        source_revision = binding.get("task_revision")
        if (
            self.max_task_attempts <= 1
            or isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
            or isinstance(source_revision, bool)
            or not isinstance(source_revision, int)
            or source_revision < 1
        ):
            return None

        events = self._verified_event_chain(paths)
        seed_event: Mapping[str, Any] | None = None
        terminal_events = events
        first_type = str(events[0].get("type") or "") if events else ""
        if first_type in _CONSUMED_ATTEMPT_SEED_EVENT_FIELDS:
            seed_event = events[0]
            terminal_events = events[1:]
            seed_receipt_field = {
                "database_portal_validation_retry_seeded": (
                    "validation_retry_receipt"
                ),
                "database_portal_capacity_retry_seeded": (
                    "capacity_retry_receipt"
                ),
                "database_portal_consumed_attempt_retry_seeded": (
                    "consumed_attempt_retry_receipt"
                ),
            }[first_type]
            seed_schema = {
                "database_portal_validation_retry_seeded": (
                    DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA
                ),
                "database_portal_capacity_retry_seeded": (
                    DATABASE_PORTAL_CAPACITY_RETRY_SEED_SCHEMA
                ),
                "database_portal_consumed_attempt_retry_seeded": (
                    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SEED_SCHEMA
                ),
            }[first_type]
            nested_schema = {
                "database_portal_validation_retry_seeded": (
                    DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA
                ),
                "database_portal_capacity_retry_seeded": (
                    DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA
                ),
                "database_portal_consumed_attempt_retry_seeded": (
                    DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA
                ),
            }[first_type]
            seed_receipt = seed_event.get(seed_receipt_field)
            seed_identity_body = {
                key: value
                for key, value in seed_event.items()
                if key not in _PORTAL_EVENT_ENVELOPE_FIELDS
                and key != "seed_id"
            }
            if (
                set(seed_event)
                != _CONSUMED_ATTEMPT_SEED_EVENT_FIELDS[first_type]
                or seed_event.get("schema") != seed_schema
                or seed_event.get("task_id")
                != str(binding.get("task_alias") or "")
                or seed_event.get("canonical_task_cid")
                != str(attempt.task_cid)
                or seed_event.get("target_database_attempt_id")
                != str(attempt.attempt_id)
                or seed_event.get("target_claim_id") != str(attempt.claim_id)
                or not str(seed_event.get("source_database_attempt_id") or "")
                or seed_event.get("source_database_attempt_id")
                == str(attempt.attempt_id)
                or seed_event.get("completion_authoritative") is not False
                or not isinstance(seed_receipt, Mapping)
                or seed_receipt.get("schema") != nested_schema
                or seed_receipt.get("receipt_id")
                != seed_event.get("source_retry_receipt_id")
                or seed_receipt.get("attempt_id")
                != seed_event.get("source_database_attempt_id")
                or seed_receipt.get("task_cid") != str(attempt.task_cid)
                or seed_receipt.get("task_alias")
                != str(binding.get("task_alias") or "")
                or seed_receipt.get("attempt_consumed") is not True
                or seed_receipt.get("provider_dispatched") is not True
                or seed_event.get("seed_id")
                != _sha256_bytes(_canonical_json(seed_identity_body))
            ):
                return None

        if (
            tuple(
                str(event.get("type") or "") for event in terminal_events
            )
            != _CONSUMED_ATTEMPT_TERMINAL_EVENT_CHAIN
            or any(
                set(event)
                != _CONSUMED_ATTEMPT_TERMINAL_EVENT_FIELDS[event_type]
                for event_type, event in zip(
                    _CONSUMED_ATTEMPT_TERMINAL_EVENT_CHAIN,
                    terminal_events,
                    strict=True,
                )
            )
        ):
            return None
        (
            selected,
            protected_snapshot,
            started,
            kernel,
            protected_clear,
            pool_release,
            finished,
            daemon_pass,
        ) = terminal_events
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        portal_attempt = finished.get("attempt")
        returncode = finished.get("returncode")
        baseline_commit = str(started.get("baseline_ref") or "")
        branch = str(started.get("branch") or "")
        canonical_task_key = str(selected.get("canonical_task_key") or "")
        board_namespace = str(selected.get("board_namespace") or "")
        workspace_path = str(started.get("worktree_path") or "")
        expected_validation = {
            "attempted": False,
            "passed": True,
            "reason": "not_run",
            "results": [],
            "returncode": 0,
        }
        expected_board_completion = {
            "complete": False,
            "pending_merge": False,
            "reason": "implementation_or_validation_failed",
        }
        if (
            selected.get("task_id") != alias
            or selected.get("canonical_task_cid") != task_cid
            or selected.get("track") != "implementation"
            or not str(selected.get("title") or "")
            or not canonical_task_key
            or not board_namespace
            or any(
                event.get("task_id") != alias
                or event.get("canonical_task_cid") != task_cid
                or event.get("canonical_task_key") != canonical_task_key
                or event.get("board_namespace") != board_namespace
                for event in (
                    protected_snapshot,
                    started,
                    kernel,
                    protected_clear,
                    finished,
                )
            )
            or any(
                event.get("attempt") != portal_attempt
                for event in (
                    protected_snapshot,
                    started,
                    kernel,
                    protected_clear,
                    finished,
                )
            )
            or not isinstance(
                protected_snapshot.get("protected_paths"), list
            )
            or protected_snapshot.get("workspace_path") != workspace_path
            or started.get("task_id") != alias
            or started.get("canonical_task_cid") != task_cid
            or started.get("attempt") != portal_attempt
            or started.get("provider_dispatched") is not False
            or not branch
            or not workspace_path
            or not re.fullmatch(r"[0-9a-f]{40}", baseline_commit)
            or kernel.get("event") != "pre_implementation_kernel_evaluated"
            or kernel.get("interface")
            != "ImplementationDaemon@pre_implementation_kernel"
            or kernel.get("disposition") != "abstain_review"
            or kernel.get("reason_code") != "no_analytical_close"
            or kernel.get("provider_authorized") is not False
            or kernel.get("skip_provider") is not True
            or kernel.get("analytical_candidate_count") != 0
            or kernel.get("provider_hook_count") != 0
            or not isinstance(kernel.get("kernel_receipt"), Mapping)
            or protected_clear.get("reason")
            != "failed_agent_terminal_check_unchanged"
            or pool_release.get("attempted") is not True
            or pool_release.get("released") is not True
            or pool_release.get("pooled") is not True
            or pool_release.get("handoff_reason")
            != "implementation_command_failed"
            or pool_release.get("reason") != "clean_prepared_workspace"
            or pool_release.get("branch") != branch
            or pool_release.get("base_commit") != baseline_commit
            or pool_release.get("worktree_path") != workspace_path
            or finished.get("task_id") != alias
            or finished.get("task_cid") != task_cid
            or finished.get("canonical_task_cid") != task_cid
            or finished.get("branch") != branch
            or finished.get("baseline_ref") != baseline_commit
            or isinstance(portal_attempt, bool)
            or not isinstance(portal_attempt, int)
            or not 1 <= portal_attempt < self.max_task_attempts
            or isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode != 1
            or finished.get("attempt_consumed") is not True
            or finished.get("provider_dispatched") is not True
            or finished.get("validation_result") != expected_validation
            or finished.get("implementation_commit") != ""
            or finished.get("commit_result") != {"committed": False}
            or finished.get("merge_result")
            != {"merged": False, "reason": "not_attempted"}
            or finished.get("board_completion") != expected_board_completion
            or finished.get("failed_preservation_result") != {}
            or finished.get("worktree_path") != workspace_path
            or finished.get("log_path") != started.get("log_path")
            or finished.get("workspace_setup") != started.get("workspace_setup")
            or daemon_pass.get("active_task_id") != ""
            or daemon_pass.get("max_task_attempts") != self.max_task_attempts
            or daemon_pass.get("ordinary_provider_dispatch_allowed") is not True
        ):
            return None

        if seed_event is not None:
            seed_receipt_field = {
                "database_portal_validation_retry_seeded": (
                    "validation_retry_receipt"
                ),
                "database_portal_capacity_retry_seeded": (
                    "capacity_retry_receipt"
                ),
                "database_portal_consumed_attempt_retry_seeded": (
                    "consumed_attempt_retry_receipt"
                ),
            }[str(seed_event.get("type") or "")]
            seed_receipt = seed_event[seed_receipt_field]
            seed_portal_attempt = seed_receipt.get("portal_attempt")
            if (
                isinstance(seed_portal_attempt, bool)
                or not isinstance(seed_portal_attempt, int)
                or seed_portal_attempt != portal_attempt - 1
                or seed_receipt.get("max_task_attempts")
                != self.max_task_attempts
            ):
                return None

        started_event_id = str(started.get("event_id") or "")
        finished_event_id = str(finished.get("event_id") or "")
        binding_id = str(binding.get("binding_id") or "")
        if any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in (started_event_id, finished_event_id, binding_id)
        ):
            return None
        receipt: dict[str, Any] = {
            "schema": DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA,
            "disposition": "retry",
            "reason": "unclassified_post_dispatch_failure",
            "failure_class": "unclassified_post_dispatch_failure",
            "provider_capacity_classification": "unproven",
            "capacity_retry_proven": False,
            "task_cid": task_cid,
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "attempt_number": int(attempt_number),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "source_task_revision": int(source_revision),
            "portal_attempt": int(portal_attempt),
            "ordinary_retry_generation": int(portal_attempt),
            "retry_budget_basis": "portal_attempt",
            "legacy_database_attempts_excluded": True,
            "max_task_attempts": int(self.max_task_attempts),
            "remaining_task_attempts": int(
                self.max_task_attempts - portal_attempt
            ),
            "attempt_consumed": True,
            "provider_dispatched": True,
            "backoff_seconds": 0,
            "retry_not_before_ms": 0,
            "binding_id": binding_id,
            "events_digest": _sha256_file(paths.events),
            "event_stream_id": str(finished.get("stream_id") or ""),
            "implementation_started_event_id": started_event_id,
            "implementation_finished_event_id": finished_event_id,
            "baseline_commit": baseline_commit,
            "implementation_returncode": int(returncode),
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def _recovery_attempt_binding(
        self,
        attempt: Any,
        *,
        recovery_name: str,
    ) -> tuple[DatabasePortalAttemptPaths, Mapping[str, Any]]:
        """Verify immutable attempt artifacts against the current control row."""

        record = self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        if not (
            paths.binding.is_file()
            and paths.task_projection.is_file()
            and paths.events.is_file()
        ):
            raise DatabasePortalBridgeError(
                f"{recovery_name} artifacts are incomplete"
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
                f"{recovery_name} binding does not match the claim"
            )
        return paths, observed_binding

    def recover_validation_retry(self, attempt: Any) -> Mapping[str, Any]:
        """Reproduce retry evidence for a previously terminalized attempt.

        This reads only the attempt-local projection and immutable event
        evidence.  It does not update a task, claim, queue, or execution row.
        """

        paths, observed_binding = self._recovery_attempt_binding(
            attempt,
            recovery_name="validation retry recovery",
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

    def recover_protected_path_preservation(
        self,
        attempt: Any,
    ) -> Mapping[str, Any]:
        """Recover one exact preserved candidate without provider work.

        The returned receipt is immutable source-attempt evidence for the
        database daemon's protected-preservation transition.  This method is
        read-only and never initializes a successor Portal attempt.
        """

        paths, observed_binding = self._recovery_attempt_binding(
            attempt,
            recovery_name="protected-path preservation recovery",
        )
        receipt = self._protected_path_preservation_receipt(
            attempt=attempt,
            paths=paths,
            binding=observed_binding,
        )
        if receipt is None:
            raise DatabasePortalBridgeError(
                "attempt is not eligible for protected-path preservation "
                "recovery"
            )
        return receipt

    @staticmethod
    def _protected_preservation_recovery_identity(
        *,
        attempt: Any,
        binding: Mapping[str, Any],
        seed: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str, str]:
        """Return the deterministic identity for one zero-provider replay."""

        alias = str(binding.get("task_alias") or "")
        identity = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-protected-preservation-recovery@1"
            ),
            "source_receipt_id": str(seed.get("receipt_id") or ""),
            "source_attempt_id": str(seed.get("attempt_id") or ""),
            "source_claim_id": str(seed.get("claim_id") or ""),
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "baseline_commit": str(seed.get("baseline_commit") or ""),
            "preserved_commit": str(seed.get("preserved_commit") or ""),
            "rescue_branch": str(seed.get("rescue_branch") or ""),
            "target_attempt_id": str(attempt.attempt_id),
            "target_claim_id": str(attempt.claim_id),
            "target_attempt_number": int(attempt.attempt_number),
            "target_fencing_token": int(attempt.fencing_token),
            "target_fence_epoch": int(attempt.fence_epoch),
            "target_lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "target_binding_id": str(binding.get("binding_id") or ""),
        }
        recovery_key = _sha256_bytes(_canonical_json(identity))
        recovery_digest = recovery_key.removeprefix("sha256:")
        safe_alias = re.sub(r"[^a-z0-9._-]+", "-", alias.lower()).strip("-")
        safe_alias = safe_alias or "protected-task"
        recovery_branch = (
            f"implementation/{safe_alias}-protected-{recovery_digest[:20]}"
        )
        return identity, recovery_key, recovery_branch

    def recover_protected_reconciliation_self_lock(
        self,
        attempt: Any,
        preservation_seed: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Prove the historical nested verification self-lock without retrying it.

        This recovery is deliberately narrower than an ordinary ``not_attempted``
        retry.  It admits only a content-addressed attempt-local event chain in
        which every declared validation command passed, the zero-provider
        protected replay owned the same checkout lease that the final verifier
        timed out on, and no merge was attempted.  The method is read-only.
        """

        paths, binding = self._recovery_attempt_binding(
            attempt,
            recovery_name="protected reconciliation self-lock recovery",
        )
        try:
            seed = DatabasePortalProtectedPathPreserved(
                preservation_seed
            ).retry_receipt
        except ValueError as exc:
            raise DatabasePortalBridgeError(
                "protected reconciliation self-lock seed failed verification"
            ) from exc
        if (
            seed.get("task_cid") != str(attempt.task_cid)
            or seed.get("task_alias")
            != str(getattr(attempt, "task_alias", "") or "")
            or str(seed.get("attempt_id") or "")
            == str(attempt.attempt_id)
            or not self._preserved_commit_exists(
                commit=str(seed.get("preserved_commit") or ""),
                rescue_branch=str(seed.get("rescue_branch") or ""),
            )
            or not self._preserved_commit_descends_from(
                baseline_commit=str(seed.get("baseline_commit") or ""),
                preserved_commit=str(seed.get("preserved_commit") or ""),
            )
        ):
            raise DatabasePortalBridgeError(
                "protected reconciliation self-lock seed is not the exact "
                "preserved candidate"
            )
        _identity, recovery_key, recovery_branch = (
            self._protected_preservation_recovery_identity(
                attempt=attempt,
                binding=binding,
                seed=seed,
            )
        )
        from ..merge.checkout_lock import checkout_mutation_lock_path

        expected_lock_path = checkout_mutation_lock_path(self.repository_root)
        events = self._verified_event_chain(paths)
        expected_event_types = (
            "implementation_task_claim_lock_cleared",
            "implementation_protected_path_snapshot_recorded",
            "worktree_reconciliation_validation_started",
            "implementation_expected_outputs_checked",
            "implementation_proposal_validated",
            "implementation_protected_path_verification_lock_timeout",
            "worktree_reconciliation_validation_finished",
            "cleanup_finished",
        )
        observed_event_types = tuple(
            str(event.get("type") or "") for event in events
        )
        if observed_event_types != expected_event_types:
            raise DatabasePortalBridgeError(
                "protected reconciliation self-lock event chain does not "
                "match the exact historical population"
            )
        claim_cleared = events[0]
        snapshot = events[1]
        expected_outputs = events[3]
        proposal = events[4]
        selected: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
            event_type: []
            for event_type in (
                "worktree_reconciliation_validation_started",
                "implementation_protected_path_verification_lock_timeout",
                "worktree_reconciliation_validation_finished",
                "cleanup_finished",
            )
        }
        for index, event in enumerate(events):
            event_type = str(event.get("type") or "")
            if event_type in selected:
                selected[event_type].append((index, event))
        if any(len(matches) != 1 for matches in selected.values()):
            raise DatabasePortalBridgeError(
                "protected reconciliation self-lock event chain is absent or ambiguous"
            )
        started_index, started = selected[
            "worktree_reconciliation_validation_started"
        ][0]
        timeout_index, timeout = selected[
            "implementation_protected_path_verification_lock_timeout"
        ][0]
        finished_index, finished = selected[
            "worktree_reconciliation_validation_finished"
        ][0]
        cleanup_index, cleanup = selected["cleanup_finished"][0]
        if not started_index < timeout_index < finished_index < cleanup_index:
            raise DatabasePortalBridgeError(
                "protected reconciliation self-lock event order is invalid"
            )
        later_authority = {
            "implementation_started",
            "implementation_finished",
            "task_completed",
            "worktree_reconciliation_candidate_queued",
            "worktree_reconciliation_candidate_merged",
            "worktree_reconciliation_validation_started",
            "worktree_reconciliation_validation_finished",
        }
        if any(
            str(event.get("type") or "") in later_authority
            for event in events[finished_index + 1 :]
        ):
            raise DatabasePortalBridgeError(
                "protected reconciliation self-lock was superseded by later authority"
            )

        lock = timeout.get("lock")
        protected = finished.get("protected_path_violation")
        validation = finished.get("validation_result")
        validation_protected = (
            validation.get("protected_path_violation")
            if isinstance(validation, Mapping)
            else None
        )
        results = validation.get("results") if isinstance(validation, Mapping) else None
        stages = validation.get("stages") if isinstance(validation, Mapping) else None
        validation_dag = (
            validation.get("validation_dag_receipt")
            if isinstance(validation, Mapping)
            else None
        )
        proposal_gate = (
            validation.get("proposal_gate")
            if isinstance(validation, Mapping)
            else None
        )
        mutations = timeout.get("mutations")
        protected_paths = timeout.get("protected_paths")
        snapshot_paths = snapshot.get("protected_paths")
        waited_seconds = lock.get("waited_seconds") if isinstance(lock, Mapping) else None
        owner_pid = lock.get("lock_owner_pid") if isinstance(lock, Mapping) else None
        violation_fields = {
            "task_id",
            "reason",
            "attempt",
            "workspace_path",
            "protected_paths",
            "mutations",
            "lock",
            "verification_deferred",
            "shared_checkout_restored",
        }
        timeout_violation = {
            field: timeout.get(field) for field in violation_fields
        }
        proposal_gate_fields = {
            "accepted",
            "attempted",
            "changed_paths",
            "completion_authoritative",
            "policy_id",
            "proof_authoritative",
            "proposal_id",
            "reason_codes",
            "receipt_id",
            "repository_tree_id",
        }
        exact_lock = bool(
            isinstance(lock, Mapping)
            and set(lock)
            == {
                "acquired",
                "lock_owner_branch",
                "lock_owner_operation",
                "lock_owner_pid",
                "lock_owner_task_id",
                "lock_path",
                "reason",
                "waited_seconds",
            }
            and lock.get("acquired") is False
            and lock.get("reason") == "lock_exists"
            and lock.get("lock_owner_branch") == recovery_branch
            and lock.get("lock_owner_operation")
            == "reconcile_protected_preservation_candidate"
            and lock.get("lock_owner_task_id")
            == str(getattr(attempt, "task_alias", "") or "")
            and isinstance(owner_pid, int)
            and not isinstance(owner_pid, bool)
            and owner_pid > 0
            and isinstance(waited_seconds, (int, float))
            and not isinstance(waited_seconds, bool)
            and math.isfinite(float(waited_seconds))
            and float(waited_seconds) > 0
            and bool(str(lock.get("lock_path") or ""))
            and Path(str(lock.get("lock_path") or "")) == expected_lock_path
        )
        exact_mutations = bool(
            isinstance(protected_paths, list)
            and protected_paths
            and len(protected_paths) == len(set(protected_paths))
            and isinstance(mutations, list)
            and mutations
            and len(mutations) == len(protected_paths)
            and {str(item.get("path") or "") for item in mutations}
            == set(protected_paths)
            and all(
                isinstance(item, Mapping)
                and item.get("change") == "verification_inconclusive"
                and item.get("scope") == "shared_checkout"
                and item.get("after")
                == {
                    "error": (
                        "implementation_protected_path_verification_lock_timeout"
                    ),
                    "state": "error",
                }
                for item in mutations
            )
        )
        exact_validation = bool(
            isinstance(validation, Mapping)
            and validation.get("attempted") is True
            and validation.get("passed") is False
            and validation.get("returncode") == 1
            and validation.get("reason") == "implementation_protected_path_mutated"
            and isinstance(results, list)
            and results
            and all(
                isinstance(result, Mapping)
                and result.get("returncode") == 0
                and result.get("timed_out") is False
                for result in results
            )
            and isinstance(stages, list)
            and stages
            and all(
                isinstance(stage, Mapping) and stage.get("passed") is True
                for stage in stages
            )
            and isinstance(validation_dag, Mapping)
            and validation_dag.get("passed") is True
            and isinstance(validation_protected, Mapping)
            and validation_protected.get("reason")
            == "implementation_protected_path_verification_lock_timeout"
            and validation_protected.get("verification_deferred") is True
            and validation_protected.get("shared_checkout_restored") is False
            and validation_protected.get("lock") == lock
        )
        event_ids = tuple(
            str(event.get("event_id") or "")
            for event in (started, timeout, finished, cleanup)
        )
        if (
            not exact_lock
            or not exact_mutations
            or not exact_validation
            or claim_cleared.get("task_id")
            != str(getattr(attempt, "task_alias", "") or "")
            or claim_cleared.get("branch") != ""
            or isinstance(claim_cleared.get("lock_owner_pid"), bool)
            or not isinstance(claim_cleared.get("lock_owner_pid"), int)
            or int(claim_cleared["lock_owner_pid"]) <= 0
            or not str(claim_cleared.get("lock_path") or "")
            or snapshot.get("task_id")
            != str(getattr(attempt, "task_alias", "") or "")
            or snapshot.get("attempt") != 1
            or snapshot.get("workspace_path") != started.get("worktree_path")
            or snapshot_paths != protected_paths
            or expected_outputs.get("task_id")
            != str(getattr(attempt, "task_alias", "") or "")
            or expected_outputs.get("passed") is not True
            or expected_outputs.get("issues") != []
            or expected_outputs.get("completion_authoritative") is not False
            or expected_outputs.get("proof_authoritative") is not False
            or not isinstance(expected_outputs.get("expected_paths"), list)
            or not expected_outputs.get("expected_paths")
            or expected_outputs.get("expected_paths")
            != proposal.get("changed_paths")
            or not isinstance(expected_outputs.get("staged_paths"), list)
            or not isinstance(expected_outputs.get("force_staged_paths"), list)
            or not str(expected_outputs.get("proposal_id") or "")
            or proposal.get("task_id")
            != str(getattr(attempt, "task_alias", "") or "")
            or proposal.get("attempted") is not True
            or proposal.get("accepted") is not True
            or proposal.get("reason_codes") != []
            or proposal.get("completion_authoritative") is not False
            or proposal.get("proof_authoritative") is not False
            or proposal.get("proposal_id")
            != expected_outputs.get("proposal_id")
            or proposal.get("repository_tree_id")
            != seed.get("baseline_commit")
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(proposal.get(field) or ""))
                is None
                for field in ("proposal_id", "policy_id", "receipt_id")
            )
            or not isinstance(proposal_gate, Mapping)
            or set(proposal_gate) != proposal_gate_fields
            or dict(proposal_gate)
            != {
                field: proposal.get(field) for field in proposal_gate_fields
            }
            or timeout.get("reason")
            != "implementation_protected_path_verification_lock_timeout"
            or timeout.get("verification_deferred") is not True
            or timeout.get("shared_checkout_restored") is not False
            or timeout.get("task_id")
            != str(getattr(attempt, "task_alias", "") or "")
            or timeout.get("attempt") != 1
            or timeout.get("workspace_path") != started.get("worktree_path")
            or not isinstance(protected, Mapping)
            or set(protected) != violation_fields
            or {field: protected.get(field) for field in violation_fields}
            != timeout_violation
            or not isinstance(validation_protected, Mapping)
            or set(validation_protected) != violation_fields
            or {
                field: validation_protected.get(field)
                for field in violation_fields
            }
            != timeout_violation
            or finished.get("task_id")
            != str(getattr(attempt, "task_alias", "") or "")
            or finished.get("task_cid") != str(attempt.task_cid)
            or finished.get("attempt") != 1
            or finished.get("returncode") != 1
            or finished.get("attempt_consumed") is not False
            or finished.get("provider_dispatched") is not False
            or finished.get("baseline_ref") != seed.get("baseline_commit")
            or finished.get("implementation_commit")
            != seed.get("preserved_commit")
            or finished.get("branch") != recovery_branch
            or finished.get("recovery_key") != recovery_key
            or finished.get("commit_result") != {"committed": False}
            or finished.get("merge_result")
            != {"merged": False, "reason": "not_attempted"}
            or started.get("task_id") != finished.get("task_id")
            or started.get("task_cid") != finished.get("task_cid")
            or started.get("attempt") != 1
            or started.get("attempt_consumed") is not False
            or started.get("provider_dispatched") is not False
            or started.get("baseline_ref") != seed.get("baseline_commit")
            or started.get("implementation_commit")
            != seed.get("preserved_commit")
            or started.get("branch") != recovery_branch
            or started.get("recovery_key") != recovery_key
            or started.get("worktree_path") != finished.get("worktree_path")
            or started.get("log_path") != finished.get("log_path")
            or not str(started.get("log_path") or "")
            or cleanup.get("cleaned") is not True
            or cleanup.get("removed_worktree") is not True
            or cleanup.get("deleted_branch") is not True
            or cleanup.get("branch") != recovery_branch
            or cleanup.get("worktree_path") != finished.get("worktree_path")
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", event_id) is None
                for event_id in event_ids
            )
        ):
            raise DatabasePortalBridgeError(
                "protected reconciliation self-lock evidence failed verification"
            )
        receipt: dict[str, Any] = {
            "schema": DATABASE_PORTAL_PROTECTED_RECONCILIATION_SELF_LOCK_SCHEMA,
            "disposition": "retry_exact_preserved_candidate",
            "reason": "protected_preservation_reconciliation_self_lock",
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "target_attempt_id": str(attempt.attempt_id),
            "target_claim_id": str(attempt.claim_id),
            "target_lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "target_attempt_number": int(attempt.attempt_number),
            "target_fencing_token": int(attempt.fencing_token),
            "target_fence_epoch": int(attempt.fence_epoch),
            "source_preservation_receipt_id": str(seed["receipt_id"]),
            "source_attempt_id": str(seed["attempt_id"]),
            "baseline_commit": str(seed["baseline_commit"]),
            "preserved_commit": str(seed["preserved_commit"]),
            "rescue_branch": str(seed["rescue_branch"]),
            "target_binding_id": str(binding.get("binding_id") or ""),
            "events_digest": _sha256_file(paths.events),
            "event_stream_id": str(finished.get("stream_id") or ""),
            "recovery_key": recovery_key,
            "recovery_branch": recovery_branch,
            "validation_started_event_id": event_ids[0],
            "verification_lock_timeout_event_id": event_ids[1],
            "validation_finished_event_id": event_ids[2],
            "cleanup_finished_event_id": event_ids[3],
            "lock_path": str(lock["lock_path"]),
            "lock_owner_pid": int(owner_pid),
            # Intent-repository control receipts prohibit JSON floats.  The
            # event stream and its digest retain the exact fractional wait;
            # the durable summary conservatively records whole seconds.
            "lock_waited_seconds": max(
                1,
                int(math.ceil(float(waited_seconds))),
            ),
            "provider_dispatched": False,
            "attempt_consumed": False,
            "validation_commands_passed": True,
            "verification_deferred": True,
            "merge_attempted": False,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def recover_consumed_attempt_retry(self, attempt: Any) -> Mapping[str, Any]:
        """Recover one consumed legacy Portal attempt without changing state.

        The resulting receipt explicitly classifies provider capacity as
        unproven.  A successor claim may use it only to carry forward Portal's
        independently durable ordinary-attempt counter.
        """

        paths, observed_binding = self._recovery_attempt_binding(
            attempt,
            recovery_name="consumed-attempt retry recovery",
        )
        receipt = self._consumed_attempt_retry_receipt(
            attempt=attempt,
            paths=paths,
            binding=observed_binding,
        )
        if receipt is None:
            raise DatabasePortalBridgeError(
                "attempt is not eligible for consumed-attempt retry recovery"
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
            or source_attempt_number <= 0
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

    def _protected_preservation_seed_from_record(
        self,
        *,
        attempt: Any,
        record: Any,
    ) -> dict[str, Any] | None:
        """Verify an exact preserved-candidate seed without consuming it."""

        body = dict(getattr(record, "body", {}) or {})
        status_receipt = body.get("completion_receipt")
        if not isinstance(status_receipt, Mapping):
            return None
        seed = status_receipt.get("protected_preservation_seed")
        if seed is None:
            return None
        source_attempt_id = str(
            status_receipt.get(
                "protected_preservation_source_attempt_id"
            )
            or ""
        )
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
            or not source_attempt_id
            or source_attempt_id != str(seed.get("attempt_id") or "")
            or source_attempt_id == str(attempt.attempt_id)
            or str(seed.get("claim_id") or "") == str(attempt.claim_id)
            or str(seed.get("lease_id") or "")
            == str(getattr(attempt, "lease_id", "") or "")
            or seed.get("task_cid") != str(attempt.task_cid)
            or seed.get("task_alias")
            != str(getattr(attempt, "task_alias", "") or "")
        ):
            raise DatabasePortalBridgeError(
                "database claim carries a malformed protected-preservation seed"
            )
        try:
            verified = DatabasePortalProtectedPathPreserved(seed).retry_receipt
        except ValueError as exc:
            raise DatabasePortalBridgeError(
                "database claim protected-preservation seed failed verification"
            ) from exc
        if (
            not self._preserved_commit_exists(
                commit=str(verified.get("preserved_commit") or ""),
                rescue_branch=str(verified.get("rescue_branch") or ""),
            )
            or not self._preserved_commit_descends_from(
                baseline_commit=str(verified.get("baseline_commit") or ""),
                preserved_commit=str(
                    verified.get("preserved_commit") or ""
                ),
            )
        ):
            raise DatabasePortalBridgeError(
                "database claim protected-preservation seed has no exact "
                "preserved candidate"
            )
        return dict(verified)

    def _reconcile_protected_preservation_seed(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        seed: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate and merge one preserved candidate without a provider pass.

        The source rescue ref remains immutable recovery authority.  A fresh,
        receipt-bound implementation worktree is the only object handed to
        Portal's ordinary reconciliation API, and every Git mutation is made
        while the Portal checkout-mutation lease is held.  If the merge train
        queues that work, this call retains the database attempt and polls in
        bounded 0.25-second intervals; the database daemon's provider-phase
        heartbeat continues renewing the attempt lease around this callback.
        """

        if (
            self.repository_root is None
            or self.merge_queue is None
            or not self.merge_target_branch
        ):
            raise DatabasePortalBridgeError(
                "protected preservation reconciliation authority is unavailable"
            )
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        baseline_commit = str(seed.get("baseline_commit") or "")
        preserved_commit = str(seed.get("preserved_commit") or "")
        rescue_branch = str(seed.get("rescue_branch") or "")
        _identity, recovery_key, recovery_branch = (
            self._protected_preservation_recovery_identity(
                attempt=attempt,
                binding=binding,
                seed=seed,
            )
        )
        recovery_digest = recovery_key.removeprefix("sha256:")
        safe_alias = re.sub(r"[^a-z0-9._-]+", "-", alias.lower()).strip("-")
        safe_alias = safe_alias or "protected-task"

        # Recheck the immutable source authority immediately before creating a
        # daemon or touching a checkout.  This also makes a replay after rescue
        # ref movement fail before any provider-shaped object can be reached.
        if (
            not self._preserved_commit_exists(
                commit=preserved_commit,
                rescue_branch=rescue_branch,
            )
            or not self._preserved_commit_descends_from(
                baseline_commit=baseline_commit,
                preserved_commit=preserved_commit,
            )
        ):
            raise DatabasePortalBridgeError(
                "protected preservation candidate changed before reconciliation"
            )

        daemon = self.portal_factory(paths, alias)
        if daemon is None:
            raise DatabasePortalBridgeError(
                "portal_factory did not return a Portal-compatible daemon"
            )
        close = getattr(daemon, "close_event_runtime", None) or getattr(
            daemon,
            "close",
            None,
        )
        claim_acquired = False
        claim_path: Any = None
        claim_metadata: Mapping[str, Any] | None = None
        try:
            from ..merge.checkout_lock import checkout_repository_id
            from .implementation_daemon import utc_now

            daemon_queue = getattr(daemon, "merge_queue", None)
            daemon_repo_root = getattr(daemon, "repo_root", None)
            daemon_target = str(
                getattr(daemon, "resolved_merge_target_branch", "") or ""
            )
            daemon_worktree_root = getattr(daemon, "worktree_root", None)
            load_tasks = getattr(daemon, "_load_tasks", None)
            run_mutation = getattr(
                daemon,
                "_run_checkout_mutation_transaction",
                None,
            )
            reconcile = getattr(
                daemon,
                "reconcile_validated_worktree_candidate",
                None,
            )
            consume_exact_merge = getattr(
                daemon,
                "_consume_exact_merge_candidate",
                None,
            )
            merge_callback = getattr(daemon, "_merge_train_callback", None)
            cancellation_requested = getattr(
                daemon,
                "_implementation_cancel_requested",
                None,
            )
            cleanup = getattr(daemon, "_cleanup_merged_worktree", None)
            claim_path_for = getattr(
                daemon,
                "_implementation_task_claim_path",
                None,
            )
            build_claim = getattr(
                daemon,
                "_build_implementation_task_claim_metadata",
                None,
            )
            acquire_claim = getattr(
                daemon,
                "_try_acquire_implementation_task_claim",
                None,
            )
            release_claim = getattr(
                daemon,
                "_release_implementation_task_claim",
                None,
            )
            queue_repository_id = str(
                getattr(self.merge_queue, "target_repository_id", "") or ""
            )
            queue_target = str(
                getattr(self.merge_queue, "target_branch", "") or ""
            )
            if (
                daemon_queue is not self.merge_queue
                or daemon_repo_root is None
                or Path(daemon_repo_root).absolute() != self.repository_root
                or daemon_target != self.merge_target_branch
                or queue_target != self.merge_target_branch
                or queue_repository_id
                != checkout_repository_id(self.repository_root)
                or getattr(self.merge_queue, "require_target_binding", False)
                is not True
                or daemon_worktree_root is None
                or not callable(load_tasks)
                or not callable(run_mutation)
                or not callable(reconcile)
                or (
                    not callable(consume_exact_merge)
                    and not callable(merge_callback)
                )
                or not callable(cleanup)
                or not callable(claim_path_for)
                or not callable(build_claim)
                or not callable(acquire_claim)
                or not callable(release_claim)
            ):
                raise DatabasePortalBridgeError(
                    "Portal protected-preservation reconciliation authority "
                    "does not match the target binding"
                )
            worktree_root = Path(daemon_worktree_root).absolute()
            if (
                worktree_root == self.repository_root
                or worktree_root == Path(worktree_root.anchor)
            ):
                raise DatabasePortalBridgeError(
                    "Portal protected-preservation worktree root is unsafe"
                )
            recovery_worktree = worktree_root / (
                f"protected-preservation-{safe_alias}-{recovery_digest[:20]}"
            )

            try:
                tasks = list(load_tasks())
            except Exception as exc:
                raise DatabasePortalBridgeError(
                    "Portal protected-preservation task projection is unreadable"
                ) from exc
            projection_already_terminal = bool(
                _projection_status(self._verify_projection(paths, binding))
                in _TERMINAL_STATUSES
            )
            task_status = (
                str(getattr(tasks[0], "status", "") or "").strip().lower()
                if len(tasks) == 1
                else ""
            )
            if (
                len(tasks) != 1
                or str(getattr(tasks[0], "task_id", "") or "") != alias
                or str(
                    getattr(tasks[0], "canonical_task_cid", "") or ""
                )
                != task_cid
                or (
                    task_status != "todo"
                    and not (
                        projection_already_terminal
                        and task_status in _TERMINAL_STATUSES
                    )
                )
            ):
                raise DatabasePortalBridgeError(
                    "Portal protected-preservation projection does not contain "
                    "the exact pending database task"
                )
            task = tasks[0]
            claim_path = claim_path_for(
                alias,
                canonical_task_cid=task_cid,
            )
            claim_metadata = build_claim(task, 1, utc_now())
            acquired, claim_reason, _existing_claim = acquire_claim(
                claim_path,
                claim_metadata,
            )
            claim_acquired = acquired is True
            if not claim_acquired:
                raise DatabasePortalBridgeError(
                    "protected preservation reconciliation task claim "
                    f"was unavailable: {claim_reason}"
                )

            def git(
                *arguments: str,
                cwd: Path | None = None,
                timeout: int = 30,
            ) -> subprocess.CompletedProcess[str]:
                try:
                    return subprocess.run(
                        ["git", *arguments],
                        cwd=cwd or self.repository_root,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=timeout,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise DatabasePortalBridgeError(
                        "protected preservation Git operation was unavailable"
                    ) from exc

            def branch_commit() -> str:
                observed = git(
                    "rev-parse",
                    "--verify",
                    f"refs/heads/{recovery_branch}^{{commit}}",
                    timeout=10,
                )
                return observed.stdout.strip() if observed.returncode == 0 else ""

            def exact_worktree(*, require_clean: bool = True) -> bool:
                if not recovery_worktree.exists():
                    return False
                head = git(
                    "rev-parse",
                    "--verify",
                    "HEAD^{commit}",
                    cwd=recovery_worktree,
                    timeout=10,
                )
                branch = git(
                    "symbolic-ref",
                    "--quiet",
                    "--short",
                    "HEAD",
                    cwd=recovery_worktree,
                    timeout=10,
                )
                status = git(
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    cwd=recovery_worktree,
                    timeout=10,
                )
                return bool(
                    head.returncode == 0
                    and head.stdout.strip() == preserved_commit
                    and branch.returncode == 0
                    and branch.stdout.strip() == recovery_branch
                    and branch_commit() == preserved_commit
                    and status.returncode == 0
                    and (not require_clean or not status.stdout)
                )

            def registered_recovery_worktree_is_exact() -> bool:
                listed = git("worktree", "list", "--porcelain", timeout=10)
                if listed.returncode != 0:
                    return False
                matching: list[dict[str, str]] = []
                for raw_record in listed.stdout.strip().split("\n\n"):
                    fields: dict[str, str] = {}
                    for line in raw_record.splitlines():
                        key, _separator, value = line.partition(" ")
                        fields[key] = value
                    if (
                        fields.get("worktree") == str(recovery_worktree)
                        or fields.get("branch")
                        == f"refs/heads/{recovery_branch}"
                    ):
                        matching.append(fields)
                return bool(
                    len(matching) == 1
                    and matching[0].get("worktree")
                    == str(recovery_worktree)
                    and matching[0].get("branch")
                    == f"refs/heads/{recovery_branch}"
                    and matching[0].get("HEAD") == preserved_commit
                    and exact_worktree()
                )

            def cleanup_owned_recovery() -> Mapping[str, Any]:
                current_branch_commit = branch_commit()
                if (
                    not recovery_worktree.exists()
                    and not current_branch_commit
                ):
                    return {"cleaned": True, "reason": "already_clean"}
                if (
                    current_branch_commit != preserved_commit
                    or not registered_recovery_worktree_is_exact()
                ):
                    raise DatabasePortalBridgeError(
                        "protected preservation recovery worktree lost its "
                        "exact cleanup identity"
                    )
                cleanup_result = cleanup(
                    recovery_worktree,
                    recovery_branch,
                    reusable=False,
                )
                if (
                    not isinstance(cleanup_result, Mapping)
                    or cleanup_result.get("cleaned") is not True
                ):
                    raise DatabasePortalBridgeError(
                        "protected preservation recovery worktree cleanup failed"
                    )
                return dict(cleanup_result)

            def queue_recovery_identity_is_exact(request_id: str) -> bool:
                try:
                    request = self.merge_queue.get(request_id)
                except Exception:
                    return False
                metadata = getattr(request, "metadata", None)
                return bool(
                    request is not None
                    and str(getattr(request, "branch_name", "") or "")
                    == recovery_branch
                    and str(getattr(request, "task_id", "") or "") == alias
                    and str(
                        getattr(request, "canonical_task_id", "") or ""
                    )
                    == task_cid
                    and str(getattr(request, "commit_sha", "") or "")
                    == preserved_commit
                    and isinstance(metadata, Mapping)
                    and str(metadata.get("worktree_path") or "")
                    == str(recovery_worktree)
                    and str(metadata.get("repo_root") or "")
                    == str(self.repository_root)
                    and str(metadata.get("todo_path") or "")
                    == str(paths.task_projection)
                    and str(metadata.get("state_path") or "")
                    == str(paths.state)
                    and str(metadata.get("events_path") or "")
                    == str(paths.events)
                    and str(metadata.get("target_repository_id") or "")
                    == queue_repository_id
                    and str(metadata.get("target_branch") or "")
                    == self.merge_target_branch
                )

            def queue_owns_recovery(request_id: str) -> bool:
                try:
                    request = self.merge_queue.get(request_id)
                except Exception:
                    return False
                return bool(
                    queue_recovery_identity_is_exact(request_id)
                    and str(getattr(request, "status", "") or "")
                    in {"pending", "processing"}
                )

            def consume_exact_queue_request(
                request_id: str,
            ) -> Mapping[str, Any] | None:
                """Advance only the receipt-bound request under the train lease."""

                if callable(consume_exact_merge):
                    result = consume_exact_merge(request_id)
                    return dict(result) if isinstance(result, Mapping) else None

                from ..merge.merge_train import MergeTrain

                train = MergeTrain(
                    repo_root=self.repository_root,
                    queue=self.merge_queue,
                    target_branch=self.merge_target_branch,
                    max_attempts=int(
                        getattr(self.merge_queue, "max_attempts", 3)
                    ),
                    merge_callback=merge_callback,
                    formal_verification_policy=getattr(
                        daemon,
                        "formal_verification_policy",
                        None,
                    ),
                    proof_gate=getattr(daemon, "proof_gate", None),
                    proof_cache_dir=getattr(
                        daemon,
                        "proof_cache_dir",
                        None,
                    ),
                    decision_runtime=getattr(
                        daemon,
                        "decision_runtime",
                        None,
                    ),
                    decision_runtime_cancellation=getattr(
                        daemon,
                        "implementation_cancelled",
                        None,
                    ),
                )
                claim_exact = getattr(
                    self.merge_queue,
                    "claim_pending_request",
                    None,
                )
                recover_abandoned = getattr(
                    train,
                    "_recover_abandoned_claims",
                    None,
                )
                process_claimed = getattr(train, "_process_claimed", None)
                if not all(
                    callable(operation)
                    for operation in (
                        claim_exact,
                        recover_abandoned,
                        process_claimed,
                    )
                ):
                    raise DatabasePortalBridgeError(
                        "merge train lacks exact protected-preservation "
                        "request authority"
                    )

                def process_exact() -> Mapping[str, Any] | None:
                    # The canonical train lease proves every old
                    # ``merge-train:*`` processing claim is abandoned.  It is
                    # safe to recover those claims, but only this exact request
                    # may be claimed or processed by this continuation.
                    recover_abandoned()
                    current = self.merge_queue.get(request_id)
                    status = str(getattr(current, "status", "") or "")
                    if status == "completed":
                        return {
                            "status": "completed",
                            "request_id": request_id,
                        }
                    if status != "pending":
                        return None
                    claimed = claim_exact(
                        request_id,
                        consumer_id=train.owner_id,
                    )
                    if claimed is None:
                        return None
                    if not queue_recovery_identity_is_exact(request_id):
                        raise DatabasePortalBridgeError(
                            "exact protected-preservation queue claim changed "
                            "identity"
                        )
                    raw = process_claimed(claimed)
                    if not isinstance(raw, Mapping):
                        raise DatabasePortalBridgeError(
                            "exact protected-preservation merge returned a "
                            "non-object result"
                        )
                    result = dict(raw)
                    if str(result.get("request_id") or "") != request_id:
                        raise DatabasePortalBridgeError(
                            "exact protected-preservation merge returned a "
                            "foreign request"
                        )
                    return result

                acquired, result = train.run_under_consumer_lease(
                    process_exact
                )
                return dict(result) if acquired and isinstance(result, Mapping) else None

            def target_contains_preserved_commit() -> bool:
                target_head = git(
                    "rev-parse",
                    "--verify",
                    f"refs/heads/{self.merge_target_branch}^{{commit}}",
                    timeout=10,
                )
                if target_head.returncode != 0:
                    return False
                target_commit = target_head.stdout.strip()
                if not re.fullmatch(r"[0-9a-f]{40}", target_commit):
                    return False
                ancestry = git(
                    "merge-base",
                    "--is-ancestor",
                    preserved_commit,
                    target_commit,
                    timeout=10,
                )
                return ancestry.returncode == 0

            def protected_queue_request_id() -> str:
                request_ids: list[str] = []
                for event in self._verified_event_chain(paths):
                    validation_result = event.get("validation_result")
                    merge_result = event.get("merge_result")
                    if (
                        event.get("type")
                        != "worktree_reconciliation_candidate_queued"
                        or str(event.get("task_id") or "") != alias
                        or str(event.get("canonical_task_cid") or "")
                        != task_cid
                        or str(event.get("baseline_ref") or "")
                        != baseline_commit
                        or str(event.get("implementation_commit") or "")
                        != preserved_commit
                        or event.get("attempt_consumed") is not False
                        or event.get("provider_dispatched") is not False
                        or not isinstance(validation_result, Mapping)
                        or validation_result.get("passed") is not True
                        or not isinstance(merge_result, Mapping)
                        or merge_result.get("queued") is not True
                    ):
                        continue
                    request_id = str(merge_result.get("request_id") or "")
                    if not request_id:
                        raise DatabasePortalBridgeError(
                            "protected preservation queue source has no exact "
                            "request identity"
                        )
                    request_ids.append(request_id)
                if len(request_ids) > 1:
                    raise DatabasePortalBridgeError(
                        "protected preservation queue source is ambiguous"
                    )
                return request_ids[0] if request_ids else ""

            def queue_completion_is_exact(request_id: str) -> bool:
                try:
                    request = self.merge_queue.get(request_id)
                except Exception:
                    return False
                metadata = getattr(request, "metadata", None)
                completion_metadata = (
                    metadata.get("completion")
                    if isinstance(metadata, Mapping)
                    else None
                )
                completion_semantics_exact = bool(
                    completion_metadata is None
                    or (
                        isinstance(completion_metadata, Mapping)
                        and completion_metadata.get("accepted") is not False
                        and completion_metadata.get("acceptance_pending")
                        is not True
                        and completion_metadata.get("completion_authoritative")
                        is not False
                        and completion_metadata.get("integrated") is not False
                        and str(completion_metadata.get("status") or "")
                        in {
                            "merged",
                            "already_merged",
                            "completed",
                            "deduplicated",
                        }
                    )
                )
                projection = self._verify_projection(paths, binding)
                exact_transport = bool(
                    request is not None
                    and str(getattr(request, "status", "") or "")
                    == "completed"
                    and str(getattr(request, "branch_name", "") or "")
                    == recovery_branch
                    and str(getattr(request, "task_id", "") or "") == alias
                    and str(
                        getattr(request, "canonical_task_id", "") or ""
                    )
                    == task_cid
                    and str(getattr(request, "commit_sha", "") or "")
                    == preserved_commit
                    and isinstance(metadata, Mapping)
                    and str(metadata.get("worktree_path") or "")
                    == str(recovery_worktree)
                    and str(metadata.get("repo_root") or "")
                    == str(self.repository_root)
                    and str(metadata.get("todo_path") or "")
                    == str(paths.task_projection)
                    and str(metadata.get("state_path") or "")
                    == str(paths.state)
                    and str(metadata.get("events_path") or "")
                    == str(paths.events)
                    and str(metadata.get("target_repository_id") or "")
                    == queue_repository_id
                    and str(metadata.get("target_branch") or "")
                    == self.merge_target_branch
                    and completion_semantics_exact
                    and target_contains_preserved_commit()
                    and _projection_status(projection) in _TERMINAL_STATUSES
                )
                if not exact_transport:
                    return False
                completion = self._ensure_protected_recovery_completion_event(
                    paths,
                    alias=alias,
                    task_cid=task_cid,
                    baseline_commit=baseline_commit,
                    implementation_commit=preserved_commit,
                    queue_reconciliation_proven=True,
                )
                return bool(
                    completion is not None
                    and str(completion.get("implementation_commit") or "")
                    == preserved_commit
                )

            if projection_already_terminal:
                # A process can die after the exact queue callback closes the
                # private projection but before this bridge removes its
                # deterministic recovery checkout or returns provider
                # acceptance.  Replays must enter the protected-seed path (not
                # the generic terminal fast path), prove the owned Git identity,
                # and finish cleanup without revalidating or re-enqueueing the
                # candidate.
                terminal_queue_request_id = protected_queue_request_id()
                if terminal_queue_request_id:
                    if not queue_completion_is_exact(
                        terminal_queue_request_id
                    ):
                        raise DatabasePortalBridgeError(
                            "protected preservation terminal replay lacks "
                            "exact completed queue and target ancestry proof"
                        )
                else:
                    if not target_contains_preserved_commit():
                        raise DatabasePortalBridgeError(
                            "protected preservation terminal replay commit "
                            "is not on the exact target branch"
                        )
                    self._ensure_protected_recovery_completion_event(
                        paths,
                        alias=alias,
                        task_cid=task_cid,
                        baseline_commit=baseline_commit,
                        implementation_commit=preserved_commit,
                    )
                cleanup_transaction = run_mutation(
                    task_id=alias,
                    branch=recovery_branch,
                    operation=(
                        "cleanup_replayed_protected_preservation_candidate"
                    ),
                    callback=lambda: dict(cleanup_owned_recovery()),
                    failure_fields={"cleaned": False},
                    extra={
                        "preserved_commit": preserved_commit,
                        "recovery_key": recovery_key,
                    },
                )
                if (
                    not isinstance(cleanup_transaction, Mapping)
                    or cleanup_transaction.get("cleaned") is not True
                ):
                    raise DatabasePortalBridgeError(
                        "protected preservation replay cleanup failed"
                    )
                return self._acceptance_receipt(
                    attempt=attempt,
                    paths=paths,
                    binding=binding,
                    summaries=(
                        {
                            "event": (
                                "protected_preservation_terminal_replayed"
                            ),
                            "task_id": alias,
                            "task_cid": task_cid,
                            "returncode": 0,
                            "reason": "terminal_projection_replayed",
                            "implementation_commit": preserved_commit,
                            "merged": True,
                            "recovery_key": recovery_key,
                        },
                    ),
                )

            def reconcile_under_checkout_authority() -> dict[str, Any]:
                if (
                    not self._preserved_commit_exists(
                        commit=preserved_commit,
                        rescue_branch=rescue_branch,
                    )
                    or not self._preserved_commit_descends_from(
                        baseline_commit=baseline_commit,
                        preserved_commit=preserved_commit,
                    )
                ):
                    raise DatabasePortalBridgeError(
                        "protected preservation candidate changed during "
                        "reconciliation"
                    )
                recovery_worktree.parent.mkdir(parents=True, exist_ok=True)
                existing_branch_commit = branch_commit()
                if recovery_worktree.exists() or existing_branch_commit:
                    if not registered_recovery_worktree_is_exact():
                        raise DatabasePortalBridgeError(
                            "protected preservation recovery worktree conflicts "
                            "with an existing checkout"
                        )
                else:
                    added = git(
                        "worktree",
                        "add",
                        "-b",
                        recovery_branch,
                        str(recovery_worktree),
                        preserved_commit,
                    )
                    if (
                        added.returncode != 0
                        or not registered_recovery_worktree_is_exact()
                    ):
                        raise DatabasePortalBridgeError(
                            "protected preservation recovery worktree could not "
                            "be created"
                        )

                retain_for_queue = False
                try:
                    raw_result = reconcile(
                        worktree_path=recovery_worktree,
                        branch_name=recovery_branch,
                        task=task,
                        baseline_ref=baseline_commit,
                        candidate_commit=preserved_commit,
                        changed_submodule_paths=None,
                        recovery_key=recovery_key,
                        preacquired_task_claim=claim_metadata,
                    )
                    if not isinstance(raw_result, Mapping):
                        raise DatabasePortalBridgeError(
                            "Portal protected-preservation reconciliation "
                            "returned a non-object result"
                        )
                    result = dict(raw_result)
                    merge_result = result.get("merge_result")
                    exact_result = bool(
                        result.get("task_id") == alias
                        and result.get("task_cid") == task_cid
                        and result.get("attempt_consumed") is False
                        and result.get("provider_dispatched") is False
                        and result.get("branch") == recovery_branch
                        and result.get("worktree_path")
                        == str(recovery_worktree)
                        and result.get("baseline_ref") == baseline_commit
                        and result.get("implementation_commit")
                        == preserved_commit
                        and result.get("recovery_key") == recovery_key
                        and isinstance(merge_result, Mapping)
                    )
                    if not exact_result:
                        raise DatabasePortalBridgeError(
                            "Portal protected-preservation reconciliation result "
                            "failed identity verification"
                        )
                    request_id = str(merge_result.get("request_id") or "")
                    queued = bool(
                        merge_result.get("queued") is True and request_id
                    )
                    if queued:
                        if queue_completion_is_exact(request_id):
                            result["returncode"] = 0
                            result["merge_result"] = {
                                **dict(merge_result),
                                "queued": False,
                                "merged": True,
                                "reason": "merged",
                            }
                        else:
                            retain_for_queue = (
                                queue_recovery_identity_is_exact(request_id)
                            )
                        if (
                            result["merge_result"].get("queued") is True
                            and not retain_for_queue
                        ):
                            raise DatabasePortalBridgeError(
                                "Portal protected-preservation queued candidate "
                                "lacks an exact durable queue owner"
                            )
                    result["protected_preservation_recovery_key"] = recovery_key
                    result["protected_preservation_queue_owned"] = (
                        retain_for_queue
                    )
                    return result
                finally:
                    if not retain_for_queue:
                        cleanup_owned_recovery()

            transaction = run_mutation(
                task_id=alias,
                branch=recovery_branch,
                operation="reconcile_protected_preservation_candidate",
                callback=reconcile_under_checkout_authority,
                failure_fields={
                    "attempt_consumed": False,
                    "provider_dispatched": False,
                },
                extra={
                    "baseline_commit": baseline_commit,
                    "preserved_commit": preserved_commit,
                    "recovery_key": recovery_key,
                    "rescue_branch": rescue_branch,
                },
            )
            if not isinstance(transaction, Mapping):
                raise DatabasePortalBridgeError(
                    "protected preservation checkout mutation returned no result"
                )
            merge_result = transaction.get("merge_result")
            if (
                isinstance(merge_result, Mapping)
                and merge_result.get("queued") is True
            ):
                raw_continuation_timeout = getattr(
                    daemon,
                    "implementation_timeout",
                    None,
                )
                if (
                    isinstance(raw_continuation_timeout, bool)
                    or not isinstance(raw_continuation_timeout, (int, float))
                    or not math.isfinite(float(raw_continuation_timeout))
                    or float(raw_continuation_timeout) <= 0
                ):
                    raise DatabasePortalBridgeError(
                        "Portal protected-preservation queue continuation "
                        "has no bounded implementation timeout"
                    )
                continuation_deadline = (
                    time.monotonic() + float(raw_continuation_timeout)
                )
                request_id = str(merge_result.get("request_id") or "")
                completed_before_wait = bool(
                    request_id and queue_completion_is_exact(request_id)
                )
                if (
                    not request_id
                    or (
                        not completed_before_wait
                        and not queue_recovery_identity_is_exact(request_id)
                    )
                ):
                    raise DatabasePortalBridgeError(
                        "protected preservation queued candidate lost its "
                        "durable owner"
                    )
                # Do not release this database attempt while a file-backed
                # Portal request is its sole completion authority.  A typed
                # outer deferral would discard the protected seed and let the
                # next database attempt dispatch again.  Instead, process one
                # same-target request at a time, or poll while another train
                # owns the exact request, until its completion callback has
                # durably closed this private projection.
                while not queue_completion_is_exact(request_id):
                    if time.monotonic() >= continuation_deadline:
                        # This is intentionally not a BridgeError.  Returning
                        # control lets the outer attempt-heartbeat wrapper
                        # surface a latched lease-renewal loss before this
                        # interruption; otherwise the same running attempt and
                        # exact queue/worktree authority remain restartable.
                        raise RuntimeError(
                            "protected preservation exact queue continuation "
                            "reached its implementation timeout"
                        )
                    request = self.merge_queue.get(request_id)
                    request_status = str(
                        getattr(request, "status", "") or ""
                    )
                    if (
                        callable(cancellation_requested)
                        and cancellation_requested()
                    ):
                        # A BridgeError would terminalize this database attempt
                        # and discard the only seed that authorizes the queued
                        # work.  Escape as a process-level interruption so the
                        # same attempt/root can resume without redispatch.
                        raise RuntimeError(
                            "protected preservation reconciliation cancelled "
                            "while exact queue work remains"
                        )
                    if not queue_recovery_identity_is_exact(request_id):
                        raise RuntimeError(
                            "protected preservation queue request changed "
                            "identity while completion remained pending"
                        )
                    if request is None or request_status not in {
                        "pending",
                        "processing",
                    }:
                        # Terminal queue transport state is not proof that the
                        # Portal completion callback failed.  Keep this exact
                        # database attempt resumable so an operator/train can
                        # revive and settle the same request; terminal
                        # projection replay above will then adopt and clean it.
                        raise RuntimeError(
                            "protected preservation queue request terminalized "
                            "without exact completion"
                        )
                    if not queue_owns_recovery(request_id):
                        raise DatabasePortalBridgeError(
                            "protected preservation active queue request changed "
                            "identity"
                        )
                    if request_status == "pending":
                        try:
                            continuation_result = consume_exact_queue_request(
                                request_id
                            )
                        except DatabasePortalBridgeError:
                            raise
                        except Exception:
                            # Another train can win its consumer lease after
                            # the row read above.  Preserve this running
                            # database attempt and re-read the exact durable
                            # request; a transient consumer exception grants
                            # no terminal authority.
                            if queue_completion_is_exact(request_id):
                                continuation_result = None
                            elif queue_owns_recovery(request_id):
                                continuation_result = None
                            else:
                                raise RuntimeError(
                                    "protected preservation merge continuation "
                                    "lost exact queue ownership"
                                )
                        if isinstance(continuation_result, Mapping):
                            continuation_request = continuation_result.get(
                                "request"
                            )
                            continued_request_id = str(
                                continuation_result.get("request_id")
                                or (
                                    continuation_request.get("request_id")
                                    if isinstance(
                                        continuation_request,
                                        Mapping,
                                    )
                                    else ""
                                )
                            )
                            if (
                                continued_request_id
                                and continued_request_id != request_id
                            ):
                                raise DatabasePortalBridgeError(
                                    "protected preservation exact continuation "
                                    "returned a foreign request"
                                )
                            if continued_request_id:
                                continued = self.merge_queue.get(
                                    continued_request_id
                                )
                                continued_metadata = getattr(
                                    continued,
                                    "metadata",
                                    None,
                                )
                                if (
                                    continued is None
                                    or not isinstance(
                                        continued_metadata,
                                        Mapping,
                                    )
                                    or str(
                                        continued_metadata.get(
                                            "target_repository_id"
                                        )
                                        or ""
                                    )
                                    != queue_repository_id
                                    or str(
                                        continued_metadata.get("target_branch")
                                        or ""
                                    )
                                    != self.merge_target_branch
                                ):
                                    raise DatabasePortalBridgeError(
                                        "protected preservation merge continuation "
                                        "left the bound queue target"
                                    )
                    if not queue_completion_is_exact(request_id):
                        remaining = continuation_deadline - time.monotonic()
                        if remaining > 0:
                            time.sleep(min(0.25, remaining))
                cleanup_transaction = run_mutation(
                    task_id=alias,
                    branch=recovery_branch,
                    operation=(
                        "cleanup_reconciled_protected_preservation_candidate"
                    ),
                    callback=lambda: dict(cleanup_owned_recovery()),
                    failure_fields={"cleaned": False},
                    extra={
                        "preserved_commit": preserved_commit,
                        "recovery_key": recovery_key,
                    },
                )
                if (
                    not isinstance(cleanup_transaction, Mapping)
                    or cleanup_transaction.get("cleaned") is not True
                ):
                    raise DatabasePortalBridgeError(
                        "protected preservation completed queue worktree cleanup "
                        "failed"
                    )
                transaction = {
                    **dict(transaction),
                    "returncode": 0,
                    "merge_result": {
                        **dict(merge_result),
                        "queued": False,
                        "merged": True,
                        "reason": "merged",
                    },
                }
                merge_result = transaction["merge_result"]
            merged = bool(
                transaction.get("returncode") == 0
                and isinstance(merge_result, Mapping)
                and merge_result.get("merged") is True
                and merge_result.get("queued") is not True
            )
            if not merged:
                reason = str(
                    transaction.get("reason")
                    or (
                        merge_result.get("reason")
                        if isinstance(merge_result, Mapping)
                        else ""
                    )
                    or "protected_preservation_reconciliation_incomplete"
                )
                raise DatabasePortalBridgeError(reason)
            if not target_contains_preserved_commit():
                raise DatabasePortalBridgeError(
                    "protected preservation merged result is not on the exact "
                    "target branch"
                )
            projection = self._verify_projection(paths, binding)
            if _projection_status(projection) in _TERMINAL_STATUSES:
                self._ensure_protected_recovery_completion_event(
                    paths,
                    alias=alias,
                    task_cid=task_cid,
                    baseline_commit=baseline_commit,
                    implementation_commit=preserved_commit,
                )
            if (
                _projection_status(projection) not in _TERMINAL_STATUSES
                or not self._has_completion_event(paths, alias)
            ):
                raise DatabasePortalBridgeError(
                    "protected preservation merge lacks durable projected "
                    "completion"
                )
            summary = {
                "event": "protected_preservation_reconciled",
                "task_id": alias,
                "task_cid": task_cid,
                "returncode": 0,
                "reason": str(merge_result.get("reason") or "merged"),
                "implementation_commit": preserved_commit,
                "merged": True,
                "recovery_key": recovery_key,
            }
            return self._acceptance_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                summaries=(summary,),
            )
        finally:
            if claim_acquired and claim_path is not None and claim_metadata is not None:
                release_claim = getattr(
                    daemon,
                    "_release_implementation_task_claim",
                    None,
                )
                if callable(release_claim):
                    release_claim(claim_path, claim_metadata)
            if callable(close):
                close()

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
        existing_events: list[dict[str, Any]] = []
        seed_event_index = -1
        if paths.events.exists():
            existing_events = self._verified_event_chain(paths)
            for index, event in enumerate(existing_events):
                if (
                    event.get("type")
                    == "database_portal_validation_retry_seeded"
                    and event.get("seed_id") == seed_body["seed_id"]
                ):
                    existing_seed_event = event
                    seed_event_index = index
                    break
            if existing_seed_event is None:
                raise DatabasePortalBridgeError(
                    "Portal attempt event stream predates its required retry seed"
                )
            if (
                seed_event_index != 0
                or sum(
                    event.get("type")
                    == "database_portal_validation_retry_seeded"
                    and event.get("target_database_attempt_id")
                    == str(attempt.attempt_id)
                    for event in existing_events
                )
                != 1
                or set(existing_seed_event)
                != _CONSUMED_ATTEMPT_SEED_EVENT_FIELDS[
                    "database_portal_validation_retry_seeded"
                ]
                or any(
                    existing_seed_event.get(key) != value
                    for key, value in seed_body.items()
                )
            ):
                raise DatabasePortalBridgeError(
                    "Portal validation retry seed event conflicts with its claim"
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
            exact_seed_state = bool(
                isinstance(current_state, Mapping)
                and len(existing_events) == seed_event_index + 1
                and all(
                    current_state.get(key) == value
                    for key, value in state_seed.items()
                )
            )
            target_portal_attempt = source_portal_attempt + 1
            progressed_events = existing_events[seed_event_index + 1 :]
            started = (
                progressed_events[0]
                if len(progressed_events) == 1
                else None
            )
            progressed_adoptable_state = bool(
                isinstance(current_state, Mapping)
                and isinstance(started, Mapping)
                and started.get("type") == "implementation_started"
                and started.get("task_id") == alias
                and started.get("canonical_task_cid") == task_cid
                and started.get("attempt") == target_portal_attempt
                and started.get("provider_dispatched") is False
                and current_state.get("implementation_attempts")
                == {alias: target_portal_attempt}
                and current_state.get("implementation_attempts_by_cid")
                == {task_cid: target_portal_attempt}
                and current_state.get("active_task_id") == alias
                and current_state.get("active_task_cid") == task_cid
                and current_state.get("active_attempt")
                == target_portal_attempt
                and current_state.get("implementation_in_progress") is True
                and current_state.get("last_implementation_task_id") == alias
                and current_state.get("last_implementation_task_key")
                == _canonical_task_key
                and current_state.get("last_implementation_task_cid")
                == task_cid
                and current_state.get("last_implementation_returncode") is None
                and current_state.get("last_implementation_finished_at") == ""
            )
            if not exact_seed_state and not progressed_adoptable_state:
                raise DatabasePortalBridgeError(
                    "Portal retry seed state conflicts with its source receipt"
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

    def _capacity_retry_seed_from_record(
        self,
        *,
        attempt: Any,
        record: Any,
    ) -> dict[str, Any] | None:
        """Verify the exact capacity receipt carried by a successor claim."""

        body = dict(getattr(record, "body", {}) or {})
        status_receipt = body.get("completion_receipt")
        if not isinstance(status_receipt, Mapping):
            return None
        seed = status_receipt.get("capacity_retry_seed")
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
                "database claim carries a malformed capacity retry seed"
            )
        value = dict(seed)
        receipt_id = value.pop("receipt_id", None)
        source_attempt_number = seed.get("attempt_number")
        target_attempt_number = getattr(attempt, "attempt_number", 0)
        source_portal_attempt = seed.get("portal_attempt")
        capacity = seed.get("codex_capacity_receipt")
        proof = seed.get("post_dispatch_capacity_proof")
        primary = seed.get("primary_receipt")
        outcome = seed.get("route_outcome")
        expected_seed_fields = {
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
            "ordinary_retry_generation",
            "max_task_attempts",
            "remaining_task_attempts",
            "attempt_consumed",
            "provider_dispatched",
            "backoff_seconds",
            "retry_not_before_ms",
            "binding_id",
            "events_digest",
            "event_stream_id",
            "implementation_event_id",
            "post_dispatch_capacity_proof",
            "primary_receipt",
            "route_outcome",
            "codex_capacity_receipt",
            "receipt_id",
        }
        if (
            set(seed) != expected_seed_fields
            or seed.get("schema") != DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA
            or seed.get("disposition") != "retry"
            or seed.get("reason") != "dual_provider_capacity_exhausted"
            or seed.get("task_cid") != str(attempt.task_cid)
            or seed.get("task_alias")
            != str(getattr(attempt, "task_alias", "") or "")
            or seed.get("attempt_consumed") is not True
            or seed.get("provider_dispatched") is not True
            or seed.get("max_task_attempts") != self.max_task_attempts
            or isinstance(source_attempt_number, bool)
            or not isinstance(source_attempt_number, int)
            or isinstance(target_attempt_number, bool)
            or not isinstance(target_attempt_number, int)
            or target_attempt_number <= 0
            or str(seed.get("attempt_id") or "")
            == str(attempt.attempt_id)
            or str(seed.get("claim_id") or "")
            == str(attempt.claim_id)
            or str(seed.get("lease_id") or "")
            == str(getattr(attempt, "lease_id", "") or "")
            or status_receipt.get("capacity_retry_source_attempt_id")
            != str(seed.get("attempt_id") or "")
            or isinstance(source_portal_attempt, bool)
            or not isinstance(source_portal_attempt, int)
            or not 1 <= source_portal_attempt < self.max_task_attempts
            or seed.get("ordinary_retry_generation")
            != source_portal_attempt
            or seed.get("remaining_task_attempts")
            != self.max_task_attempts - source_portal_attempt
            or not all(
                isinstance(item, Mapping)
                for item in (proof, primary, outcome, capacity)
            )
            or capacity.get("fallback_provider_id") != "codex"
            or capacity.get("fallback_model_id") != "gpt-5.6-terra"
            or capacity.get("attempt_consumed") is not True
            or capacity.get("provider_dispatched") is not True
            or proof.get("task_revision_cid") != str(attempt.task_cid)
            or proof.get("task_id")
            != str(getattr(attempt, "task_alias", "") or "")
            or proof.get("attempt") != source_portal_attempt
            or proof.get("proof_id")
            != _sha256_bytes(
                _canonical_json(
                    {
                        key: item
                        for key, item in proof.items()
                        if key != "proof_id"
                    }
                )
            )
            or proof.get("primary_receipt_id")
            != primary.get("receipt_id")
            or proof.get("route_outcome_id") != outcome.get("outcome_id")
            or proof.get("capacity_receipt_id")
            != capacity.get("receipt_id")
            or proof.get("invocation_binding_id")
            != capacity.get("invocation_binding_id")
            or proof.get("route_id") != capacity.get("route_id")
            or proof.get("decision_id") != capacity.get("decision_id")
            or outcome.get("fallback_capacity_receipt") != capacity
            or outcome.get("invocation_binding_id")
            != capacity.get("invocation_binding_id")
            or outcome.get("decision_id") != capacity.get("decision_id")
            or capacity.get("receipt_id")
            != _content_addressed_record(
                capacity, identity_field="receipt_id"
            )
            or outcome.get("outcome_id")
            != _content_addressed_record(outcome, identity_field="outcome_id")
            or primary.get("receipt_id")
            != _content_addressed_record(primary, identity_field="receipt_id")
            or receipt_id != _sha256_bytes(_canonical_json(value))
        ):
            raise DatabasePortalBridgeError(
                "database claim capacity retry seed failed verification"
            )
        return dict(seed)

    def _initialize_capacity_retry_seed(
        self,
        *,
        attempt: Any,
        record: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Seed the prior ordinary Portal attempt counter exactly once."""

        seed = self._capacity_retry_seed_from_record(
            attempt=attempt,
            record=record,
        )
        if seed is None:
            return None
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        source_portal_attempt = int(seed["portal_attempt"])
        seed_body = {
            "schema": DATABASE_PORTAL_CAPACITY_RETRY_SEED_SCHEMA,
            "task_id": alias,
            "canonical_task_cid": task_cid,
            "source_database_attempt_id": str(seed.get("attempt_id") or ""),
            "target_database_attempt_id": str(attempt.attempt_id),
            "target_claim_id": str(attempt.claim_id),
            "source_retry_receipt_id": str(seed.get("receipt_id") or ""),
            "portal_attempt": source_portal_attempt,
            "capacity_retry_receipt": dict(seed),
            "completion_authoritative": False,
        }
        seed_body["seed_id"] = _sha256_bytes(_canonical_json(seed_body))
        existing_seed_event: Mapping[str, Any] | None = None
        existing_events: list[dict[str, Any]] = []
        seed_event_index = -1
        if paths.events.exists():
            existing_events = self._verified_event_chain(paths)
            for index, event in enumerate(existing_events):
                if (
                    event.get("type")
                    == "database_portal_capacity_retry_seeded"
                    and event.get("seed_id") == seed_body["seed_id"]
                ):
                    existing_seed_event = event
                    seed_event_index = index
                    break
            if existing_seed_event is None:
                raise DatabasePortalBridgeError(
                    "Portal event stream predates its required capacity seed"
                )
            if (
                sum(
                    event.get("type")
                    == "database_portal_capacity_retry_seeded"
                    and event.get("target_database_attempt_id")
                    == str(attempt.attempt_id)
                    for event in existing_events
                )
                != 1
                or any(
                    existing_seed_event.get(key) != value
                    for key, value in seed_body.items()
                )
            ):
                raise DatabasePortalBridgeError(
                    "Portal capacity retry seed event conflicts with its claim"
                )
        else:
            existing_seed_event = append_jsonl_event(
                paths.events,
                "database_portal_capacity_retry_seeded",
                seed_body,
            )
        state_seed = {
            "implementation_attempts": {alias: source_portal_attempt},
            "implementation_attempts_by_cid": {
                task_cid: source_portal_attempt,
            },
            "last_implementation_task_id": alias,
            "last_implementation_task_cid": task_cid,
            "last_implementation_returncode": 1,
        }
        if paths.state.exists():
            try:
                current_state = json.loads(paths.state.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeError(
                    "Portal capacity retry seed state is unreadable"
                ) from exc
            exact_seed_state = bool(
                isinstance(current_state, Mapping)
                and all(
                    current_state.get(key) == value
                    for key, value in state_seed.items()
                )
            )
            target_portal_attempt = source_portal_attempt + 1
            started = [
                event
                for event in existing_events[seed_event_index + 1 :]
                if event.get("type") == "implementation_started"
                and event.get("task_id") == alias
                and event.get("canonical_task_cid") == task_cid
                and event.get("attempt") == target_portal_attempt
            ]
            progressed_adoptable_state = bool(
                isinstance(current_state, Mapping)
                and len(started) == 1
                and current_state.get("implementation_attempts")
                == {alias: target_portal_attempt}
                and current_state.get("implementation_attempts_by_cid")
                == {task_cid: target_portal_attempt}
                and current_state.get("active_task_id") == alias
                and current_state.get("active_task_cid") == task_cid
                and current_state.get("active_attempt")
                == target_portal_attempt
                and current_state.get("implementation_in_progress") is True
                and current_state.get("last_implementation_task_id") == alias
                and current_state.get("last_implementation_task_cid")
                == task_cid
                and current_state.get("last_implementation_returncode") is None
                and not current_state.get("last_implementation_finished_at")
            )
            if not exact_seed_state and not progressed_adoptable_state:
                raise DatabasePortalBridgeError(
                    "Portal capacity retry seed state conflicts with its receipt"
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
            "portal_attempt": source_portal_attempt,
        }

    def _consumed_attempt_retry_seed_from_record(
        self,
        *,
        attempt: Any,
        record: Any,
    ) -> dict[str, Any] | None:
        """Verify an exact consumed-attempt receipt on a successor claim."""

        body = dict(getattr(record, "body", {}) or {})
        status_receipt = body.get("completion_receipt")
        if not isinstance(status_receipt, Mapping):
            return None
        seed = status_receipt.get("consumed_attempt_retry_seed")
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
                "database claim carries a malformed consumed-attempt retry seed"
            )

        expected_seed_fields = {
            "schema",
            "disposition",
            "reason",
            "failure_class",
            "provider_capacity_classification",
            "capacity_retry_proven",
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "lease_id",
            "attempt_number",
            "fencing_token",
            "fence_epoch",
            "source_task_revision",
            "portal_attempt",
            "ordinary_retry_generation",
            "retry_budget_basis",
            "legacy_database_attempts_excluded",
            "max_task_attempts",
            "remaining_task_attempts",
            "attempt_consumed",
            "provider_dispatched",
            "backoff_seconds",
            "retry_not_before_ms",
            "binding_id",
            "events_digest",
            "event_stream_id",
            "implementation_started_event_id",
            "implementation_finished_event_id",
            "baseline_commit",
            "implementation_returncode",
            "receipt_id",
        }
        value = dict(seed)
        receipt_id = str(value.pop("receipt_id", "") or "")
        source_attempt_number = seed.get("attempt_number")
        target_attempt_number = getattr(attempt, "attempt_number", 0)
        source_revision = seed.get("source_task_revision")
        source_portal_attempt = seed.get("portal_attempt")
        implementation_returncode = seed.get("implementation_returncode")
        digest_fields = (
            "binding_id",
            "events_digest",
            "implementation_started_event_id",
            "implementation_finished_event_id",
        )
        if (
            set(seed) != expected_seed_fields
            or seed.get("schema")
            != DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA
            or seed.get("disposition") != "retry"
            or seed.get("reason") != "unclassified_post_dispatch_failure"
            or seed.get("failure_class")
            != "unclassified_post_dispatch_failure"
            or seed.get("provider_capacity_classification") != "unproven"
            or seed.get("capacity_retry_proven") is not False
            or seed.get("task_cid") != str(attempt.task_cid)
            or seed.get("task_alias")
            != str(getattr(attempt, "task_alias", "") or "")
            or seed.get("attempt_consumed") is not True
            or seed.get("provider_dispatched") is not True
            or seed.get("backoff_seconds") != 0
            or seed.get("retry_not_before_ms") != 0
            or seed.get("retry_budget_basis") != "portal_attempt"
            or seed.get("legacy_database_attempts_excluded") is not True
            or seed.get("max_task_attempts") != self.max_task_attempts
            or isinstance(source_attempt_number, bool)
            or not isinstance(source_attempt_number, int)
            or source_attempt_number < 1
            or isinstance(target_attempt_number, bool)
            or not isinstance(target_attempt_number, int)
            or target_attempt_number < 1
            or str(seed.get("attempt_id") or "") == str(attempt.attempt_id)
            or str(seed.get("claim_id") or "") == str(attempt.claim_id)
            or str(seed.get("lease_id") or "")
            == str(getattr(attempt, "lease_id", "") or "")
            or status_receipt.get("consumed_attempt_retry_source_attempt_id")
            != str(seed.get("attempt_id") or "")
            or isinstance(source_revision, bool)
            or not isinstance(source_revision, int)
            or source_revision < 1
            or isinstance(source_portal_attempt, bool)
            or not isinstance(source_portal_attempt, int)
            or not 1 <= source_portal_attempt < self.max_task_attempts
            or seed.get("ordinary_retry_generation")
            != source_portal_attempt
            or seed.get("remaining_task_attempts")
            != self.max_task_attempts - source_portal_attempt
            or isinstance(implementation_returncode, bool)
            or not isinstance(implementation_returncode, int)
            or implementation_returncode != 1
            or not str(seed.get("event_stream_id") or "")
            or re.fullmatch(
                r"[0-9a-f]{40}", str(seed.get("baseline_commit") or "")
            )
            is None
            or any(
                re.fullmatch(
                    r"sha256:[0-9a-f]{64}", str(seed.get(field) or "")
                )
                is None
                for field in digest_fields
            )
            or receipt_id != _sha256_bytes(_canonical_json(value))
        ):
            raise DatabasePortalBridgeError(
                "database claim consumed-attempt retry seed failed verification"
            )
        return dict(seed)

    def _initialize_consumed_attempt_retry_seed(
        self,
        *,
        attempt: Any,
        record: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Seed one verified legacy consumption into the successor lane."""

        seed = self._consumed_attempt_retry_seed_from_record(
            attempt=attempt,
            record=record,
        )
        if seed is None:
            return None
        alias = str(binding.get("task_alias") or "")
        task_cid = str(attempt.task_cid)
        source_portal_attempt = int(seed["portal_attempt"])
        seed_body = {
            "schema": DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SEED_SCHEMA,
            "task_id": alias,
            "canonical_task_cid": task_cid,
            "source_database_attempt_id": str(seed.get("attempt_id") or ""),
            "target_database_attempt_id": str(attempt.attempt_id),
            "target_claim_id": str(attempt.claim_id),
            "source_retry_receipt_id": str(seed.get("receipt_id") or ""),
            "portal_attempt": source_portal_attempt,
            "consumed_attempt_retry_receipt": dict(seed),
            "completion_authoritative": False,
        }
        seed_body["seed_id"] = _sha256_bytes(_canonical_json(seed_body))
        existing_seed_event: Mapping[str, Any] | None = None
        existing_events: list[dict[str, Any]] = []
        seed_event_index = -1
        if paths.events.exists():
            existing_events = self._verified_event_chain(paths)
            for index, event in enumerate(existing_events):
                if (
                    event.get("type")
                    == "database_portal_consumed_attempt_retry_seeded"
                    and event.get("seed_id") == seed_body["seed_id"]
                ):
                    existing_seed_event = event
                    seed_event_index = index
                    break
            if existing_seed_event is None:
                raise DatabasePortalBridgeError(
                    "Portal event stream predates its required consumed-attempt seed"
                )
            if (
                sum(
                    event.get("type")
                    == "database_portal_consumed_attempt_retry_seeded"
                    and event.get("target_database_attempt_id")
                    == str(attempt.attempt_id)
                    for event in existing_events
                )
                != 1
                or any(
                    existing_seed_event.get(key) != value
                    for key, value in seed_body.items()
                )
            ):
                raise DatabasePortalBridgeError(
                    "Portal consumed-attempt retry seed event conflicts with its claim"
                )
        else:
            existing_seed_event = append_jsonl_event(
                paths.events,
                "database_portal_consumed_attempt_retry_seeded",
                seed_body,
            )

        state_seed = {
            "implementation_attempts": {alias: source_portal_attempt},
            "implementation_attempts_by_cid": {
                task_cid: source_portal_attempt,
            },
            "last_implementation_task_id": alias,
            "last_implementation_task_cid": task_cid,
            "last_implementation_returncode": int(
                seed["implementation_returncode"]
            ),
        }
        if paths.state.exists():
            try:
                current_state = json.loads(paths.state.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeError(
                    "Portal consumed-attempt retry seed state is unreadable"
                ) from exc
            exact_seed_state = bool(
                isinstance(current_state, Mapping)
                and all(
                    current_state.get(key) == value
                    for key, value in state_seed.items()
                )
            )
            target_portal_attempt = source_portal_attempt + 1
            started = [
                event
                for event in existing_events[seed_event_index + 1 :]
                if event.get("type") == "implementation_started"
                and event.get("task_id") == alias
                and event.get("canonical_task_cid") == task_cid
                and event.get("attempt") == target_portal_attempt
            ]
            progressed_adoptable_state = bool(
                isinstance(current_state, Mapping)
                and len(started) == 1
                and current_state.get("implementation_attempts")
                == {alias: target_portal_attempt}
                and current_state.get("implementation_attempts_by_cid")
                == {task_cid: target_portal_attempt}
                and current_state.get("active_task_id") == alias
                and current_state.get("active_task_cid") == task_cid
                and current_state.get("active_attempt")
                == target_portal_attempt
                and current_state.get("implementation_in_progress") is True
                and current_state.get("last_implementation_task_id") == alias
                and current_state.get("last_implementation_task_cid")
                == task_cid
                and current_state.get("last_implementation_returncode") is None
                and not current_state.get("last_implementation_finished_at")
            )
            if not exact_seed_state and not progressed_adoptable_state:
                raise DatabasePortalBridgeError(
                    "Portal consumed-attempt retry seed state conflicts with its receipt"
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
            "portal_attempt": source_portal_attempt,
        }

    def _verify_vrif_benchmark_acceptance(
        self,
        *,
        record: Any,
        baseline_commit: str,
        baseline_tree: str,
        implementation_commit: str,
    ) -> None:
        """Independently reproduce the owner-exact VRIF-030 benchmark."""

        if self.repository_root is None:
            raise DatabasePortalBridgeError(
                "VRIF-030 acceptance has no repository authority"
            )
        body = getattr(record, "body", None)
        body = body if isinstance(body, Mapping) else {}
        outputs = {_safe_output_path(path) for path in _output_values(record, body)}
        validations = _validation_values(record, body)
        if outputs != set(_VRIF_BENCHMARK_OUTPUT_PATHS) or len(validations) != 1:
            raise DatabasePortalBridgeError(
                "VRIF-030 acceptance contract differs from the sealed task"
            )

        from ..residual_intelligence.benchmark import (
            MANIFEST_SCHEMA,
            build_frozen_benchmark_contract,
            sha256_identity,
        )
        from ..residual_intelligence.contracts import (
            PROGRAM_ID,
            ResidualIntelligenceError,
            ResidualTaskFamily,
        )
        from ..task_sources.control_plane_contracts import content_identity

        def blob(path: str) -> bytes:
            return _git_blob_at_commit(
                self.repository_root,
                commit=implementation_commit,
                path=path,
            )

        objective_paths = (
            "docs/architecture/agent_supervisor_residual_intelligence.objectives.md",
            "docs/architecture/agent_supervisor_residual_intelligence.todo.md",
        )
        operation_path = (
            "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"
        )
        provider_path = (
            "config/agent_supervisor_residual_intelligence_scheduler.json"
        )
        admission_path = (
            "benchmarks/agent_supervisor/residual_intelligence/"
            "synthetic_training_admission.json"
        )
        split_path = (
            "benchmarks/agent_supervisor/residual_intelligence/"
            "synthetic_split_manifest.json"
        )
        inventory_path = (
            "docs/architecture/residual_intelligence_inventory/"
            "residual_model_call_inventory.json"
        )
        objective_artifacts = {
            path: sha256_identity(blob(path)) for path in objective_paths
        }
        admission = _strict_json_bytes(
            blob(admission_path),
            noun="VRIF-030 training admission",
        )
        split = _strict_json_bytes(
            blob(split_path),
            noun="VRIF-030 split manifest",
        )
        if not isinstance(admission, Mapping) or not isinstance(split, Mapping):
            raise DatabasePortalBridgeError(
                "VRIF-030 owner inputs are not JSON objects"
            )
        admission_body = dict(admission)
        admission_id = str(admission_body.pop("admission_id", "") or "")
        if admission_id != content_identity(admission_body):
            raise DatabasePortalBridgeError(
                "VRIF-030 training admission identity does not verify"
            )
        base_bindings = {
            "repository_states": sha256_identity(
                {"commit": baseline_commit, "tree": baseline_tree}
            ),
            "objective_revisions": sha256_identity(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "residual-benchmark-objective-revisions@1"
                    ),
                    "artifacts": objective_artifacts,
                }
            ),
            "operation_catalog": sha256_identity(blob(operation_path)),
            "provider_policy": sha256_identity(blob(provider_path)),
            "tokenizer": sha256_identity(
                {
                    "admission_id": admission_id,
                    "disposition": "no_learned_tokenizer_admitted",
                }
            ),
            "model_versions": sha256_identity(
                {
                    "inventory_blob_identity": sha256_identity(
                        blob(inventory_path)
                    ),
                    "disposition": "training_unavailable",
                }
            ),
            "validation_policy": sha256_identity(
                {
                    "argv": [[command] for command in validations],
                    "test_blob_identity": sha256_identity(
                        blob(_VRIF_BENCHMARK_TEST_PATH)
                    ),
                }
            ),
        }
        task_families = [family.value for family in ResidualTaskFamily]
        try:
            expected = build_frozen_benchmark_contract(
                task_families=task_families,
                source_commit=baseline_commit,
                source_tree=baseline_tree,
                split_root=str(split.get("split_root") or ""),
                base_bindings=base_bindings,
            )
        except ResidualIntelligenceError as exc:
            raise DatabasePortalBridgeError(
                "VRIF-030 owner benchmark reconstruction failed"
            ) from exc

        manifest = _strict_json_bytes(
            blob(_VRIF_BENCHMARK_MANIFEST_PATH),
            noun="VRIF-030 benchmark manifest",
        )
        cases_blob = blob(_VRIF_BENCHMARK_CASES_PATH)
        try:
            case_lines = cases_blob.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise DatabasePortalBridgeError(
                "VRIF-030 benchmark cases are not UTF-8"
            ) from exc
        if not case_lines or any(not line.strip() for line in case_lines):
            raise DatabasePortalBridgeError(
                "VRIF-030 benchmark cases are empty or contain blank lines"
            )
        cases = [
            _strict_json_bytes(
                line.encode("utf-8"),
                noun=f"VRIF-030 benchmark case line {index}",
            )
            for index, line in enumerate(case_lines, start=1)
        ]
        expected_manifest = {
            "schema": MANIFEST_SCHEMA,
            "program_identifier": PROGRAM_ID,
            "status": "staged_not_qualified",
            "owner_task": _VRIF_BENCHMARK_TASK_ALIAS,
            "source_revision": baseline_commit,
            "partitions": expected["partitions"],
            "required_case_kinds": expected["case_kinds"],
            "task_families": task_families,
            "training_admission": "training_unavailable",
            "weights_committed": False,
            "large_corpus_committed": False,
            "promotion_evidence": False,
            "benchmark_freeze": expected["benchmark_freeze"],
        }
        if manifest != expected_manifest or cases != expected["cases"]:
            raise DatabasePortalBridgeError(
                "VRIF-030 artifacts differ from the owner-exact benchmark contract"
            )

    def _verify_vrif_terminal_acceptance(
        self,
        *,
        record: Any,
        baseline_commit: str,
        baseline_tree: str,
        implementation_commit: str,
    ) -> None:
        """Require owner-canonical VRIF-032 JSON/Markdown before completion."""

        if self.repository_root is None:
            raise DatabasePortalBridgeError(
                "VRIF-032 acceptance has no repository authority"
            )
        body = getattr(record, "body", None)
        body = body if isinstance(body, Mapping) else {}
        outputs = {_safe_output_path(path) for path in _output_values(record, body)}
        if outputs != set(_VRIF_TERMINAL_OUTPUT_PATHS):
            raise DatabasePortalBridgeError(
                "VRIF-032 acceptance contract differs from the sealed task"
            )
        report_blob = _git_blob_at_commit(
            self.repository_root,
            commit=implementation_commit,
            path=_VRIF_RELEASE_REPORT_JSON_PATH,
        )
        markdown_blob = _git_blob_at_commit(
            self.repository_root,
            commit=implementation_commit,
            path=_VRIF_RELEASE_REPORT_MARKDOWN_PATH,
        )
        report = _strict_json_bytes(
            report_blob,
            noun="VRIF-032 release report",
        )
        drift = report.get("drift") if isinstance(report, Mapping) else None
        if (
            not isinstance(report, Mapping)
            or set(report) != set(_VRIF_TERMINAL_REPORT_FIELDS)
            or report.get("end_tree") != baseline_tree
            or not isinstance(drift, Mapping)
            or drift.get("evaluated_tree") != baseline_tree
        ):
            raise DatabasePortalBridgeError(
                "VRIF-032 report does not bind the exact Portal baseline tree"
            )
        from ..residual_intelligence.contracts import ResidualIntelligenceError
        from ..residual_intelligence.release import (
            ResidualIntelligenceReleaseReport,
            render_vrif_release_report_markdown,
            validate_release_claims,
        )

        try:
            typed_report = validate_release_claims(
                ResidualIntelligenceReleaseReport.from_dict(report)
            )
        except ResidualIntelligenceError as exc:
            raise DatabasePortalBridgeError(
                "VRIF-032 report fails the trusted release contract"
            ) from exc
        if typed_report.to_dict() != dict(report):
            raise DatabasePortalBridgeError(
                "VRIF-032 report differs from its trusted typed projection"
            )

        expected_markdown = render_vrif_release_report_markdown(report).encode(
            "utf-8"
        )
        if markdown_blob != expected_markdown:
            raise DatabasePortalBridgeError(
                "VRIF-032 Markdown is not the owner-canonical report rendering"
            )
        for path, current_blob in (
            (_VRIF_RELEASE_REPORT_JSON_PATH, report_blob),
            (_VRIF_RELEASE_REPORT_MARKDOWN_PATH, markdown_blob),
        ):
            if _git_blob_at_commit(
                self.repository_root,
                commit=baseline_commit,
                path=path,
            ) == current_blob:
                raise DatabasePortalBridgeError(
                    "VRIF-032 report is unchanged from its Portal baseline"
                )

    def _verify_vrif_semantic_acceptance(
        self,
        *,
        attempt: Any,
        baseline_commit: str,
        baseline_tree: str,
        implementation_commit: str,
    ) -> None:
        alias = str(getattr(attempt, "task_alias", "") or "")
        if alias not in {_VRIF_BENCHMARK_TASK_ALIAS, _VRIF_TERMINAL_TASK_ALIAS}:
            return
        record = self._record_for_attempt(self.task_source, attempt)
        if alias == _VRIF_BENCHMARK_TASK_ALIAS:
            self._verify_vrif_benchmark_acceptance(
                record=record,
                baseline_commit=baseline_commit,
                baseline_tree=baseline_tree,
                implementation_commit=implementation_commit,
            )
        else:
            self._verify_vrif_terminal_acceptance(
                record=record,
                baseline_commit=baseline_commit,
                baseline_tree=baseline_tree,
                implementation_commit=implementation_commit,
            )

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
        completion = self._completion_event_evidence(
            paths,
            alias=alias,
            task_cid=str(attempt.task_cid),
        )
        if completion is None:
            raise DatabasePortalBridgeError(
                "Portal completion lacks a matching durable task_completed event"
            )
        implementation_commit = str(
            completion.get("implementation_commit") or ""
        )
        baseline_commit = str(completion.get("baseline_commit") or "")
        completion_event_id = str(
            completion.get("completion_event_id") or ""
        )
        completion_source_event_id = str(
            completion.get("completion_source_event_id") or ""
        )
        completion_source_event_type = str(
            completion.get("completion_source_event_type") or ""
        )
        completion_source_portal_attempt = completion.get(
            "completion_source_portal_attempt"
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
            "baseline_commit": baseline_commit,
            "implementation_commit": implementation_commit,
            "completion_event_id": completion_event_id,
            "completion_source_event_id": completion_source_event_id,
            "completion_source_event_type": completion_source_event_type,
            "completion_source_portal_attempt": (
                completion_source_portal_attempt
            ),
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
            "baseline_commit": baseline_commit,
            "implementation_commit": implementation_commit,
            "completion_event_id": completion_event_id,
            "completion_source_event_id": completion_source_event_id,
            "portal_evidence": evidence,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def _post_merge_completion_recovery_seed_from_record(
        self,
        *,
        attempt: Any,
        record: Any,
    ) -> dict[str, Any] | None:
        """Reproduce one owner-carried repaired-completion seed.

        The seed is proposal evidence, never completion authority.  It is
        admitted only from the exact successor ``database_claim`` and is then
        independently reproduced from the target-bound queue row, its sealed
        source projection, the content-addressed repair/requalification
        receipt, and the unchanged current target generation.
        """

        body = getattr(record, "body", None)
        status_receipt = (
            body.get("completion_receipt") if isinstance(body, Mapping) else None
        )
        seed = (
            status_receipt.get("post_merge_completion_recovery_seed")
            if isinstance(status_receipt, Mapping)
            else None
        )
        if seed is None:
            return None
        if not isinstance(seed, Mapping):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery seed is malformed"
            )
        value = dict(seed)
        seed_id = str(value.pop("seed_id", "") or "")
        integer_fields = (
            "attempt_number",
            "fencing_token",
            "fence_epoch",
            "source_task_revision",
            "queue_source_fencing_token",
            "queue_source_fence_epoch",
        )
        string_fields = (
            "task_cid",
            "task_alias",
            "attempt_id",
            "claim_id",
            "lease_id",
            "owner_session_id",
            "request_id",
            "candidate_commit",
            "qualified_target_commit",
            "qualification_kind",
            "qualification_receipt_id",
            "queue_source_attempt_id",
            "queue_source_claim_id",
            "queue_source_lease_id",
            "queue_source_binding_id",
            "queue_source_projection_immutable_digest",
            "recovery_evidence_id",
            "terminal_reason",
        )
        record_revision = getattr(record, "revision", None)
        schema = value.get("schema")
        expected_fields = (
            _POST_MERGE_COMPLETION_RECOVERY_SEED_V2_FIELDS
            if schema
            == DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2
            else _POST_MERGE_COMPLETION_RECOVERY_SEED_FIELDS
        )
        recovery_control_revision = value.get(
            "recovery_control_revision",
            value.get("source_task_revision"),
        )
        if schema == DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2:
            integer_fields = (*integer_fields, "recovery_control_revision")
        if (
            set(seed) != expected_fields
            or schema
            not in {
                DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA,
                DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2,
            }
            or seed_id != _sha256_bytes(_canonical_json(value))
            or any(
                not isinstance(value.get(field), str) or not value[field]
                for field in string_fields
            )
            or any(
                isinstance(value.get(field), bool)
                or not isinstance(value.get(field), int)
                or int(value[field]) < 1
                for field in integer_fields
            )
            or value.get("terminal_reason")
            not in _DATABASE_POST_MERGE_COMPLETION_RECOVERY_TERMINAL_REASONS
            or value.get("qualification_kind")
            not in {"repair", "requalification", "callback_integration"}
            or any(
                re.fullmatch(r"[0-9a-f]{40}", str(value.get(field) or ""))
                is None
                for field in ("candidate_commit", "qualified_target_commit")
            )
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get(field) or ""))
                is None
                for field in (
                    "queue_source_binding_id",
                    "queue_source_projection_immutable_digest",
                    "recovery_evidence_id",
                )
            )
            or not seed_id
            or not isinstance(status_receipt, Mapping)
            or status_receipt.get("operation") != "database_claim"
            or status_receipt.get("post_merge_completion_recovery_source_attempt_id")
            != value.get("attempt_id")
            or status_receipt.get("post_merge_completion_recovery_seed")
            != dict(seed)
            or status_receipt.get("attempt_id") != str(attempt.attempt_id)
            or status_receipt.get("claim_id") != str(attempt.claim_id)
            or status_receipt.get("lease_id") != str(attempt.lease_id)
            or status_receipt.get("owner_session_id")
            != str(attempt.owner_session_id)
            or status_receipt.get("fencing_token")
            != int(attempt.fencing_token)
            or status_receipt.get("fence_epoch") != int(attempt.fence_epoch)
            or value.get("task_cid") != str(attempt.task_cid)
            or value.get("task_alias") != str(attempt.task_alias)
            or value.get("attempt_id") == str(attempt.attempt_id)
            or value.get("claim_id") == str(attempt.claim_id)
            or value.get("lease_id") == str(attempt.lease_id)
            or isinstance(record_revision, bool)
            or not isinstance(record_revision, int)
            or isinstance(recovery_control_revision, bool)
            or not isinstance(recovery_control_revision, int)
            or int(recovery_control_revision) + 2 != record_revision
            or (
                schema
                == DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2
                and int(recovery_control_revision)
                <= int(value["source_task_revision"])
            )
        ):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery seed failed claim verification"
            )
        if self.merge_queue is None:
            raise DatabasePortalBridgeError(
                "post-merge completion recovery seed has no merge queue"
            )
        request = self.merge_queue.get(str(value["request_id"]))
        projection = (
            self._owned_post_merge_recovery_projection(
                request,
                allowed_task_statuses=frozenset({"in_progress"}),
                allow_shared_lane_source=True,
            )
            if request is not None
            else None
        )
        if projection is None:
            raise DatabasePortalBridgeError(
                "post-merge completion recovery seed lost its owned queue row"
            )
        binding = projection.binding
        if (
            str(getattr(request, "canonical_task_id", "") or "")
            != value["task_cid"]
            or str(getattr(request, "task_id", "") or "")
            != value["task_alias"]
            or str(getattr(request, "commit_sha", "") or "")
            != value["candidate_commit"]
            or value["queue_source_attempt_id"]
            != str(binding.get("attempt_id") or "")
            or value["queue_source_claim_id"]
            != str(binding.get("claim_id") or "")
            or value["queue_source_lease_id"]
            != str(binding.get("lease_id") or "")
            or value["queue_source_fencing_token"]
            != binding.get("fencing_token")
            or value["queue_source_fence_epoch"] != binding.get("fence_epoch")
            or value["queue_source_binding_id"]
            != str(binding.get("binding_id") or "")
            or value["queue_source_projection_immutable_digest"]
            != str(binding.get("projection_immutable_digest") or "")
        ):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery seed changed queue source"
            )
        metadata = getattr(request, "metadata", None)
        completion = metadata.get("completion") if isinstance(metadata, Mapping) else None
        repair_receipt = (
            completion.get("repair_receipt")
            if isinstance(completion, Mapping)
            else None
        )
        if value["qualification_kind"] != "callback_integration":
            if not isinstance(repair_receipt, Mapping):
                raise DatabasePortalBridgeError(
                    "post-merge completion recovery seed has no repair receipt"
                )
            target_identity = self._repair_receipt_current_target_identity(
                repair_receipt
            )
            if (
                target_identity is None
                or target_identity[0] != value["qualified_target_commit"]
                or (
                    value["qualification_kind"] == "repair"
                    and target_identity[2] is not True
                )
                or (
                    value["qualification_kind"] == "requalification"
                    and target_identity[2] is not False
                )
            ):
                raise DatabasePortalBridgeError(
                    "post-merge completion recovery seed target generation changed"
                )
            if value["qualification_kind"] == "requalification":
                receipt_path = self._post_merge_requalification_receipt_path(
                    projection,
                    source_receipt_id=str(
                        repair_receipt.get("receipt_id") or ""
                    ),
                    current_head=target_identity[0],
                    current_tree=target_identity[1],
                )
                if not receipt_path.is_file():
                    raise DatabasePortalBridgeError(
                        "post-merge completion recovery seed lacks durable requalification"
                    )
        evidence = self._post_merge_recovery_evidence(
            request,
            projection,
            evidence_digest=lambda item: _sha256_bytes(_canonical_json(item)),
        )
        qualification_field = {
            "repair": "repair_receipt",
            "requalification": "requalification_receipt",
            "callback_integration": "callback_requalification_receipt",
        }[str(value["qualification_kind"])]
        qualification = (
            evidence.get(qualification_field)
            if isinstance(evidence, Mapping)
            else None
        )
        expected_schema = {
            "repair": _DATABASE_POST_MERGE_RECOVERY_SCHEMA,
            "requalification": _DATABASE_POST_MERGE_REQUALIFICATION_RECOVERY_SCHEMA,
            "callback_integration": _DATABASE_POST_MERGE_CALLBACK_INTEGRATION_RECOVERY_SCHEMA,
        }[str(value["qualification_kind"])]
        expected_target_field = (
            "repair_commit"
            if value["qualification_kind"] == "repair"
            else "qualified_target_commit"
        )
        expected_receipt_field = {
            "repair": "repair_receipt_id",
            "requalification": "requalification_receipt_id",
            "callback_integration": "callback_requalification_receipt_id",
        }[str(value["qualification_kind"])]
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("schema") != expected_schema
            or evidence.get("evidence_id") != value["recovery_evidence_id"]
            or evidence.get("request_id") != value["request_id"]
            or evidence.get("candidate_commit") != value["candidate_commit"]
            or evidence.get(expected_target_field)
            != value["qualified_target_commit"]
            or evidence.get(expected_receipt_field)
            != value["qualification_receipt_id"]
            or not isinstance(qualification, Mapping)
        ):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery seed evidence changed"
            )
        source_repair = (
            qualification
            if value["qualification_kind"]
            in {"repair", "callback_integration"}
            else qualification.get("source_repair_receipt")
        )
        validations = qualification.get("validation")
        baseline_commit = (
            str(source_repair.get("baseline_commit") or "")
            if isinstance(source_repair, Mapping)
            else ""
        )
        validation = validations[0] if isinstance(validations, list) and len(validations) == 1 else None
        if (
            not isinstance(source_repair, Mapping)
            or not isinstance(validation, Mapping)
            or validation.get("task_id") != value["task_alias"]
            or validation.get("passed") is not True
            or validation.get("returncode") != 0
            or re.fullmatch(r"[0-9a-f]{40}", baseline_commit) is None
        ):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery seed has no exact evaluation"
            )
        if self.repository_root is None:
            raise DatabasePortalBridgeError(
                "post-merge completion recovery seed has no repository"
            )
        try:
            parents = subprocess.run(
                ["git", "rev-list", "--parents", "-n", "1", value["candidate_commit"]],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DatabasePortalBridgeError(
                "post-merge completion recovery candidate lineage is unavailable"
            ) from exc
        if (
            parents.returncode != 0
            or parents.stdout.split() != [value["candidate_commit"], baseline_commit]
        ):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery candidate is not one exact commit"
            )
        return {
            **dict(seed),
            "baseline_commit": baseline_commit,
            "request": request,
            "recovery_evidence": dict(evidence),
            "qualification_receipt": dict(qualification),
        }

    @classmethod
    def _ensure_post_merge_completion_recovery_events(
        cls,
        paths: DatabasePortalAttemptPaths,
        *,
        alias: str,
        task_cid: str,
        request_id: str,
        baseline_commit: str,
        implementation_commit: str,
        seed_id: str,
        recovery_evidence_id: str,
    ) -> Mapping[str, Any]:
        """Append one crash-idempotent zero-provider evaluated lineage."""

        reason = "post_merge_completion_recovery_seed"
        source_payload: dict[str, Any] = {
            "task_id": alias,
            "canonical_task_cid": task_cid,
            "attempt": 1,
            "attempt_consumed": False,
            "provider_dispatched": False,
            "returncode": 0,
            "baseline_ref": baseline_commit,
            "implementation_commit": implementation_commit,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "reason": reason,
                "recovery_evidence_id": recovery_evidence_id,
            },
            "merge_result": {
                "attempted": True,
                "merged": False,
                "queued": True,
                "request_id": request_id,
                "reason": reason,
            },
            "reason": reason,
            "post_merge_completion_recovery_seed_id": seed_id,
        }
        merge_payload: dict[str, Any] = {
            "task_id": alias,
            "canonical_task_cid": task_cid,
            "attempt": 1,
            "attempt_consumed": False,
            "provider_dispatched": False,
            "baseline_ref": baseline_commit,
            "implementation_commit": implementation_commit,
            "resolved": True,
            "reason": reason,
            "merge_result": {
                "attempted": True,
                "merged": True,
                "queued": False,
                "request_id": request_id,
                "reason": reason,
            },
            "post_merge_completion_recovery_seed_id": seed_id,
        }
        completion_payload: dict[str, Any] = {
            "task_id": alias,
            "canonical_task_cid": task_cid,
            "implementation_commit": implementation_commit,
            "baseline_commit": baseline_commit,
            "attempt": 1,
            "reason": reason,
            "post_merge_completion_recovery_seed_id": seed_id,
        }

        def exact_events(event_type: str, payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
            return [
                event
                for event in cls._verified_event_chain(paths)
                if event.get("type") == event_type
                and event.get("post_merge_completion_recovery_seed_id") == seed_id
                and all(event.get(key) == value for key, value in payload.items())
                and set(event) == set(payload) | _PORTAL_EVENT_ENVELOPE_FIELDS
            ]

        try:
            existing = cls._verified_event_chain(paths)
        except DatabasePortalBridgeError:
            if paths.events.exists():
                raise
            existing = []
        foreign_completion = [
            event
            for event in existing
            if event.get("type") == "task_completed"
            and str(event.get("task_id") or "") == alias
            and str(event.get("canonical_task_cid") or "") == task_cid
            and event.get("post_merge_completion_recovery_seed_id") != seed_id
        ]
        if foreign_completion:
            raise DatabasePortalBridgeError(
                "post-merge completion recovery cannot repair a bare completion"
            )
        same_seed = [
            event
            for event in existing
            if event.get("post_merge_completion_recovery_seed_id") == seed_id
        ]
        same_seed_sources = [
            event
            for event in same_seed
            if event.get("type")
            == "worktree_reconciliation_candidate_queued"
        ]
        sources = exact_events(
            "worktree_reconciliation_candidate_queued", source_payload
        ) if existing else []
        if len(sources) > 1 or len(sources) != len(same_seed_sources):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery source is conflicting"
            )
        if not sources:
            if any(
                event.get("type") in {"merge_reconciled", "task_completed"}
                for event in same_seed
            ):
                raise DatabasePortalBridgeError(
                    "post-merge completion recovery stages are out of order"
                )
            append_jsonl_event(
                paths.events,
                "worktree_reconciliation_candidate_queued",
                source_payload,
            )
            sources = exact_events(
                "worktree_reconciliation_candidate_queued",
                source_payload,
            )
        if len(sources) != 1 or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(sources[0].get("event_id") or ""),
        ) is None:
            raise DatabasePortalBridgeError(
                "post-merge completion recovery source identity is malformed"
            )
        source_event_id = str(sources[0]["event_id"])
        merge_payload["completion_source_event_id"] = source_event_id
        completion_payload["completion_source_event_id"] = source_event_id
        merges = exact_events("merge_reconciled", merge_payload)
        same_seed_merges = [
            event for event in same_seed if event.get("type") == "merge_reconciled"
        ]
        if len(merges) > 1 or len(merges) != len(same_seed_merges):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery reconciliation is conflicting"
            )
        if not merges:
            if any(
                event.get("type") == "task_completed" for event in same_seed
            ):
                raise DatabasePortalBridgeError(
                    "post-merge completion recovery stages are out of order"
                )
            append_jsonl_event(paths.events, "merge_reconciled", merge_payload)
        completions = exact_events("task_completed", completion_payload)
        same_seed_completions = [
            event for event in same_seed if event.get("type") == "task_completed"
        ]
        if (
            len(completions) > 1
            or len(completions) != len(same_seed_completions)
        ):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery completion is conflicting"
            )
        if not completions:
            append_jsonl_event(paths.events, "task_completed", completion_payload)
        completion = cls._completion_event_evidence(
            paths,
            alias=alias,
            task_cid=task_cid,
        )
        if (
            completion is None
            or completion.get("implementation_commit") != implementation_commit
            or completion.get("baseline_commit") != baseline_commit
        ):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery did not establish exact lineage"
            )
        return completion

    def _accept_post_merge_completion_recovery_seed(
        self,
        *,
        attempt: Any,
        record: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        seed: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Close a fresh successor projection without provider dispatch."""

        alias = str(binding.get("task_alias") or "")
        source = self._ensure_post_merge_completion_recovery_events(
            paths,
            alias=alias,
            task_cid=str(attempt.task_cid),
            request_id=str(seed["request_id"]),
            baseline_commit=str(seed["baseline_commit"]),
            implementation_commit=str(seed["candidate_commit"]),
            seed_id=str(seed["seed_id"]),
            recovery_evidence_id=str(seed["recovery_evidence_id"]),
        )
        projection = self._verify_projection(paths, binding)
        updated, count = _MUTABLE_PROJECTION_LINE.subn(
            "- Status: completed",
            projection,
        )
        if count != 1:
            raise DatabasePortalBridgeError(
                "post-merge completion recovery projection status is malformed"
            )
        if updated != projection:
            _atomic_write(paths.task_projection, updated.encode("utf-8"))
        if (
            _projection_status(self._verify_projection(paths, binding))
            not in _TERMINAL_STATUSES
        ):
            raise DatabasePortalBridgeError(
                "post-merge completion recovery did not close its projection"
            )
        # Reproduce target identity after every local append.  An advancing
        # target invalidates this seed; it never triggers a provider or a
        # generic retry under the old generation.
        reproduced = self._post_merge_completion_recovery_seed_from_record(
            attempt=attempt,
            record=record,
        )
        if (
            reproduced is None
            or reproduced.get("seed_id") != seed.get("seed_id")
            or reproduced.get("baseline_commit")
            != source.get("baseline_commit")
        ):
            raise DatabasePortalBridgeError(
                DATABASE_POST_MERGE_COMPLETION_TARGET_GENERATION_CHANGED_REASON
            )
        return self._acceptance_receipt(
            attempt=attempt,
            paths=paths,
            binding=binding,
            summaries=(),
        )

    def run_provider(self, attempt: Any) -> Mapping[str, Any]:
        """Run bounded real Portal passes and return only accepted evidence."""

        record = self._record_for_attempt(self.task_source, attempt)
        status_receipt = dict(getattr(record, "body", {}) or {}).get(
            "completion_receipt"
        )
        retry_seed_fields = (
            "validation_retry_seed",
            "capacity_retry_seed",
            "protected_preservation_seed",
            "consumed_attempt_retry_seed",
            "post_merge_completion_recovery_seed",
        )
        if isinstance(status_receipt, Mapping) and sum(
            status_receipt.get(field) is not None
            for field in retry_seed_fields
        ) > 1:
            raise DatabasePortalBridgeError(
                "database claim carries conflicting retry seeds"
            )
        protected_seed = self._protected_preservation_seed_from_record(
            attempt=attempt,
            record=record,
        )
        paths, binding = self._ensure_attempt_projection(attempt, record)
        post_merge_completion_seed = (
            self._post_merge_completion_recovery_seed_from_record(
                attempt=attempt,
                record=record,
            )
        )
        if post_merge_completion_seed is not None:
            return self._accept_post_merge_completion_recovery_seed(
                attempt=attempt,
                record=record,
                paths=paths,
                binding=binding,
                seed=post_merge_completion_seed,
            )
        projection = self._verify_projection(paths, binding)
        if protected_seed is not None:
            return self._reconcile_protected_preservation_seed(
                attempt=attempt,
                paths=paths,
                binding=binding,
                seed=protected_seed,
            )
        if (
            _projection_status(projection) in _TERMINAL_STATUSES
            and self._has_completion_event(
                paths,
                str(binding.get("task_alias") or ""),
            )
        ):
            return self._acceptance_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                summaries=(),
            )
        # A prior process can die after Portal durably appends its terminal
        # retry event but before the outer database attempt commits ``failed``.
        # Replay that exact event before seed initialization or provider work;
        # an advanced attempt-local state must never cause a second dispatch.
        if paths.events.is_file():
            recovered_preservation = (
                self._protected_path_preservation_receipt(
                    attempt=attempt,
                    paths=paths,
                    binding=binding,
                )
            )
            if recovered_preservation is not None:
                raise DatabasePortalProtectedPathPreserved(
                    recovered_preservation
                )
            recovered_capacity = self._capacity_retry_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                implementation={},
            )
            if recovered_capacity is not None:
                if recovered_capacity.get("remaining_task_attempts") == 0:
                    raise DatabasePortalBridgeError(
                        "portal_retry_budget_exhausted"
                    )
                raise DatabasePortalCapacityRetry(recovered_capacity)
            recovered_validation = self._validation_retry_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                implementation=None,
            )
            if recovered_validation is not None:
                raise DatabasePortalValidationRetry(recovered_validation)
            recovered_consumed = self._consumed_attempt_retry_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
            )
            if recovered_consumed is not None:
                raise DatabasePortalConsumedAttemptTerminal(
                    recovered_consumed
                )
        validation_seed = self._initialize_validation_retry_seed(
            attempt=attempt,
            record=record,
            paths=paths,
            binding=binding,
        )
        capacity_seed = self._initialize_capacity_retry_seed(
            attempt=attempt,
            record=record,
            paths=paths,
            binding=binding,
        )
        consumed_attempt_seed = self._initialize_consumed_attempt_retry_seed(
            attempt=attempt,
            record=record,
            paths=paths,
            binding=binding,
        )
        if sum(
            seed is not None
            for seed in (
                validation_seed,
                capacity_seed,
                consumed_attempt_seed,
            )
        ) > 1:
            raise DatabasePortalBridgeError(
                "database claim carries conflicting retry seeds"
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
        try:
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
                    )
                raw_result = daemon.run_once()
                if not isinstance(raw_result, Mapping):
                    raise DatabasePortalBridgeError("Portal daemon returned a non-object result")
                summary = _bounded_portal_result(raw_result)
                summaries.append(summary)
                self._verify_projection(paths, binding)
                implementation = raw_result.get("implementation_result")
                protected_violation = (
                    implementation.get("protected_path_violation")
                    if isinstance(implementation, Mapping)
                    else None
                )
                if (
                    paths.events.is_file()
                    and isinstance(implementation, Mapping)
                    and (
                        implementation.get("reason")
                        == "implementation_protected_path_mutated"
                        or (
                            isinstance(protected_violation, Mapping)
                            and protected_violation.get("reason")
                            == "implementation_protected_path_mutated"
                        )
                    )
                ):
                    protected_preservation = (
                        self._protected_path_preservation_receipt(
                            attempt=attempt,
                            paths=paths,
                            binding=binding,
                        )
                    )
                    if protected_preservation is not None:
                        raise DatabasePortalProtectedPathPreserved(
                            protected_preservation
                        )
                deferral = self._typed_deferral(raw_result)
                if deferral is not None:
                    reason, backoff_seconds = deferral
                    raise DatabasePortalBridgeDeferred(
                        reason,
                        backoff_seconds=backoff_seconds,
                    )
                capacity_retry_receipt = self._capacity_retry_receipt(
                    attempt=attempt,
                    paths=paths,
                    binding=binding,
                    implementation=(
                        implementation
                        if isinstance(implementation, Mapping)
                        else {}
                    ),
                )
                if capacity_retry_receipt is not None:
                    if capacity_retry_receipt.get("remaining_task_attempts") == 0:
                        raise DatabasePortalBridgeError(
                            "portal_retry_budget_exhausted"
                        )
                    raise DatabasePortalCapacityRetry(
                        capacity_retry_receipt
                    )
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
                if paths.events.is_file():
                    consumed_attempt_receipt = (
                        self._consumed_attempt_retry_receipt(
                            attempt=attempt,
                            paths=paths,
                            binding=binding,
                        )
                    )
                    if consumed_attempt_receipt is not None:
                        raise DatabasePortalConsumedAttemptTerminal(
                            consumed_attempt_receipt
                        )
                failure = self._terminal_failure(raw_result)
                if failure:
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
                before_request_id=cursor,
            )

        completion_page = completion_recovery_page(
            cursors["completed_requests"]
        )

        def reopen_under_checkout_transaction(
            completed: Any,
            projection: _DatabasePortalRecoveryProjection,
        ) -> dict[str, Any] | None:
            """Fence target identity through the terminal queue CAS."""

            # Most completed rows are ordinary repair completions.  Prove the
            # immutable historical-shortcut envelope before requiring the
            # additional checkout-mutation authority; the complete Git and
            # queue proof is deliberately repeated under that authority below.
            metadata = getattr(completed, "metadata", None)
            if (
                projection.task_status != "blocked"
                or str(getattr(completed, "status", "") or "")
                != "completed"
                or not isinstance(metadata, Mapping)
                or "completion" in metadata
            ):
                return None
            canonical = str(
                getattr(completed, "canonical_identity", "") or ""
            )
            candidate = str(
                getattr(completed, "commit_sha", "") or ""
            )
            dedupe_key = str(
                getattr(completed, "dedupe_key", "") or ""
            )
            make_key = getattr(train, "_dedupe_key", None)
            read_receipt = getattr(train, "_read_receipt", None)
            if not callable(make_key) or not callable(read_receipt):
                raise DatabasePortalBridgeError(
                    "merge train lacks false-completion recovery verification"
                )
            receipt_key = str(make_key(canonical, candidate) or "")
            receipt = read_receipt(receipt_key) if receipt_key else None
            if (
                receipt_key != dedupe_key
                or not isinstance(receipt, Mapping)
                or receipt.get("status") != "already_merged"
                or receipt.get("reason")
                != "declared_outputs_already_on_target"
                or receipt.get("mutation_short_circuited") is not True
            ):
                return None
            request_id = str(
                getattr(completed, "request_id", "") or ""
            )
            no_authority_reason = (
                "false_positive_completion_reopen_not_authorized"
            )

            def verify_and_reopen() -> dict[str, Any]:
                # Acquiring the checkout lease may wait behind another
                # mutation.  Re-read both authorities inside it, then keep the
                # exact captured target stable through the queue CAS.
                current = self.merge_queue.get(request_id)
                current_projection = (
                    self._owned_post_merge_recovery_projection(current)
                    if current is not None
                    else None
                )
                if current_projection is None:
                    return {
                        "schema": _DATABASE_POST_MERGE_RECOVERY_SCHEMA,
                        "attempted": False,
                        "recovered": False,
                        "reason": no_authority_reason,
                        "request_id": request_id,
                        "write_count": 0,
                    }
                self._preauthorize_post_merge_recovery(
                    current,
                    current_projection,
                    preauthorize=preauthorize,
                    evidence_digest=digest,
                )
                target = str(train._target_commit() or "")
                reopened = self._reopen_false_positive_completed_request(
                    current,
                    current_projection,
                    train=train,
                    target_commit=target,
                )
                if reopened is not None:
                    return reopened
                return {
                    "schema": _DATABASE_POST_MERGE_RECOVERY_SCHEMA,
                    "attempted": False,
                    "recovered": False,
                    "reason": no_authority_reason,
                    "request_id": request_id,
                    "write_count": 0,
                }

            portal = self.portal_factory(
                projection.paths,
                str(getattr(completed, "task_id", "") or ""),
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
                run_checkout_mutation = getattr(
                    portal,
                    "_run_checkout_mutation_transaction",
                    None,
                )
                if (
                    portal_queue is not self.merge_queue
                    or portal_repo_root is None
                    or self.repository_root is None
                    or Path(portal_repo_root).absolute()
                    != self.repository_root
                    or portal_target != self.merge_target_branch
                    or not callable(run_checkout_mutation)
                ):
                    raise DatabasePortalBridgeError(
                        "Portal recovery daemon lacks checkout mutation "
                        "authority for the selected target"
                    )
                guarded = run_checkout_mutation(
                    task_id=str(getattr(completed, "task_id", "") or ""),
                    attempt=int(getattr(completed, "attempt", 0) or 0),
                    branch=str(
                        getattr(completed, "branch_name", "") or ""
                    ),
                    operation="reopen_false_positive_merge_completion",
                    callback=verify_and_reopen,
                    failure_fields={
                        "schema": _DATABASE_POST_MERGE_RECOVERY_SCHEMA,
                        "attempted": False,
                        "recovered": False,
                        "request_id": request_id,
                        "write_count": 0,
                    },
                )
            except Exception as exc:
                if _is_implementation_conflict(exc):
                    return None
                raise
            finally:
                if callable(close):
                    close()
            if not isinstance(guarded, Mapping):
                raise DatabasePortalBridgeError(
                    "checkout mutation returned invalid false-completion "
                    "recovery evidence"
                )
            result = dict(guarded)
            if str(result.get("reason") or "") == no_authority_reason:
                return None
            if (
                result.get("schema") == _DATABASE_POST_MERGE_RECOVERY_SCHEMA
                and result.get("request_id") == request_id
                and result.get("write_count") == 1
            ):
                # A release-pending suffix may replace the callback reason
                # after the queue CAS.  The exact write still occurred and
                # must keep the daemon awake for idempotent reconciliation.
                return result
            return None

        def replay_completed_page(
            page: Sequence[Any],
            *,
            allow_shared_lane_source: bool = False,
        ) -> Mapping[str, Any] | None:
            for snapshot in page:
                snapshot_projection = (
                    self._owned_post_merge_recovery_projection(
                        snapshot,
                        allow_shared_lane_source=allow_shared_lane_source,
                    )
                )
                if snapshot_projection is None:
                    continue
                completed = self.merge_queue.get(
                    str(getattr(snapshot, "request_id", "") or "")
                )
                projection = (
                    self._owned_post_merge_recovery_projection(
                        completed,
                        allow_shared_lane_source=allow_shared_lane_source,
                    )
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
                    train=train,
                )
                if evidence is not None:
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
                reopened = reopen_under_checkout_transaction(
                    completed,
                    projection,
                )
                if reopened is not None:
                    return reopened
            return None

        priority_task_cids_fn = getattr(
            database_daemon,
            "post_merge_completion_recovery_task_cids",
            None,
        )
        raw_priority_task_cids = (
            priority_task_cids_fn()
            if callable(priority_task_cids_fn)
            else ()
        )
        if (
            not isinstance(raw_priority_task_cids, Sequence)
            or isinstance(
                raw_priority_task_cids,
                (str, bytes, bytearray, memoryview),
            )
        ):
            raise DatabasePortalBridgeError(
                "database completion-recovery task identities are malformed"
            )
        priority_task_cids, priority_next_cursor = (
            self._priority_recovery_task_cid_page(
                raw_priority_task_cids,
                after_task_cid=cursors["priority_task_cids"],
            )
        )
        priority_completed_page = self._priority_repaired_completion_requests(
            priority_task_cids
        )
        if priority_completed_page:
            acquired, priority_result = train.run_under_consumer_lease(
                lambda: replay_completed_page(
                    priority_completed_page,
                    allow_shared_lane_source=True,
                )
            )
            if not acquired:
                return None
        else:
            priority_result = None
        if cursors["priority_task_cids"] != priority_next_cursor:
            cursors["priority_task_cids"] = priority_next_cursor
            self._save_post_merge_recovery_cursors(cursors)
        if priority_result is not None:
            return dict(priority_result)

        if completion_page:
            acquired, replay_result = train.run_under_consumer_lease(
                lambda: replay_completed_page(completion_page)
            )
            if not acquired:
                return None
            self._advance_post_merge_recovery_cursor(
                cursors,
                "completed_requests",
                completion_page,
            )
            if replay_result is not None:
                if (
                    str(replay_result.get("reason") or "")
                    == "false_positive_completion_reopened"
                ):
                    # The completed row may have an older immutable request
                    # id than this lane's existing pending/processing scan
                    # cursors.  Reset every stage it can traverse before the
                    # next tick; otherwise a continuously non-empty tail can
                    # starve the exact queue-authored recovery indefinitely.
                    reset_changed = False
                    for stage in (
                        "pending_requests",
                        "processing_requests",
                        "quarantined_requests",
                    ):
                        if cursors.get(stage):
                            cursors[stage] = ""
                            reset_changed = True
                    if reset_changed:
                        self._save_post_merge_recovery_cursors(cursors)
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
            if selected is None:
                # A stale owned row must not pin the first bounded page
                # forever.  Advance past every inspected conflict and retry it
                # after the durable cursor wraps, preserving both fairness and
                # eventual re-evaluation when database authority changes.
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
            portal_shortcut = getattr(
                recovery_train,
                "_portal_projection_invalid_metadata_already_on_target",
                None,
            )
            shortcut_overridden = False
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
                if not callable(portal_shortcut):
                    raise DatabasePortalBridgeError(
                        "merge train lacks exact Portal recovery routing"
                    )

                def route_exact_repair_through_portal(request: Any) -> bool:
                    if (
                        str(getattr(request, "request_id", "") or "")
                        == selected_request_id
                    ):
                        # This exact row has already passed the immutable
                        # projection and database preauthorization checks.  A
                        # generic "declared outputs already on target"
                        # shortcut would settle it without the repair receipt
                        # needed for the post-queue database CAS and crash
                        # replay.  Force only this row through the configured
                        # Portal repair callback.
                        return False
                    return bool(portal_shortcut(request))

                setattr(
                    recovery_train,
                    "_portal_projection_invalid_metadata_already_on_target",
                    route_exact_repair_through_portal,
                )
                shortcut_overridden = True
                yield
            finally:
                if shortcut_overridden:
                    setattr(
                        recovery_train,
                        "_portal_projection_invalid_metadata_already_on_target",
                        portal_shortcut,
                    )
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
                    train=train,
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

    def _require_portal_commit_lineage(
        self,
        *,
        baseline_commit: str,
        implementation_commit: str,
    ) -> str:
        """Resolve one exact Portal baseline tree and descendant commit."""

        if self.repository_root is None:
            raise DatabasePortalBridgeError(
                "database effect has no repository for Portal lineage proof"
            )
        if (
            not re.fullmatch(r"[0-9a-f]{40}", baseline_commit)
            or not re.fullmatch(r"[0-9a-f]{40}", implementation_commit)
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected malformed Portal commit lineage"
            )
        try:
            resolved_baseline = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"{baseline_commit}^{{commit}}",
                ],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            baseline_tree_result = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"{baseline_commit}^{{tree}}",
                ],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            resolved_implementation = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"{implementation_commit}^{{commit}}",
                ],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            ancestry = subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    baseline_commit,
                    implementation_commit,
                ],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DatabasePortalBridgeError(
                "database effect Portal lineage proof is unavailable"
            ) from exc
        baseline_tree = baseline_tree_result.stdout.strip()
        if (
            resolved_baseline.returncode != 0
            or resolved_baseline.stdout.strip() != baseline_commit
            or resolved_implementation.returncode != 0
            or resolved_implementation.stdout.strip() != implementation_commit
            or ancestry.returncode != 0
            or baseline_tree_result.returncode != 0
            or not re.fullmatch(r"[0-9a-f]{40}", baseline_tree)
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected unproven Portal commit lineage"
            )
        return baseline_tree

    def _require_accepted_provider(
        self,
        attempt: Any,
        provider_result: Mapping[str, Any],
    ) -> Mapping[str, str]:
        receipt_body = dict(provider_result)
        receipt_id = str(receipt_body.pop("receipt_id", "") or "")
        evidence = provider_result.get("portal_evidence")
        if (
            set(provider_result)
            != {
                "schema",
                "interface",
                "status",
                "provider",
                "execution_mode",
                "accepted",
                "completion_authority",
                "task_cid",
                "task_alias",
                "attempt_id",
                "binding_id",
                "evidence_digest",
                "baseline_commit",
                "implementation_commit",
                "completion_event_id",
                "completion_source_event_id",
                "portal_evidence",
                "receipt_id",
            }
            or provider_result.get("schema")
            != DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
            or provider_result.get("interface") != DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
            or provider_result.get("accepted") is not True
            or provider_result.get("status") != "succeeded"
            or provider_result.get("provider") != "PortalImplementationDaemon"
            or provider_result.get("execution_mode")
            != "database-authoritative-portal-bridge"
            or provider_result.get("completion_authority")
            != "DatabaseImplementationDaemon"
            or str(provider_result.get("task_cid") or "") != str(attempt.task_cid)
            or str(provider_result.get("attempt_id") or "") != str(attempt.attempt_id)
            or str(provider_result.get("task_alias") or "")
            != str(attempt.task_alias)
            or receipt_id != _sha256_bytes(_canonical_json(receipt_body))
            or not isinstance(evidence, Mapping)
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected unaccepted Portal provider evidence"
            )
        digest = str(provider_result.get("evidence_digest") or "")
        binding_id = str(provider_result.get("binding_id") or "")
        baseline_commit = str(provider_result.get("baseline_commit") or "")
        implementation_commit = str(
            provider_result.get("implementation_commit") or ""
        )
        completion_event_id = str(
            provider_result.get("completion_event_id") or ""
        )
        completion_source_event_id = str(
            provider_result.get("completion_source_event_id") or ""
        )
        if (
            set(evidence)
            != {
                "binding_id",
                "task_cid",
                "task_alias",
                "attempt_id",
                "projection_digest",
                "projection_immutable_digest",
                "state_digest",
                "events_digest",
                "baseline_commit",
                "implementation_commit",
                "completion_event_id",
                "completion_source_event_id",
                "completion_source_event_type",
                "completion_source_portal_attempt",
                "portal_passes",
            }
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", binding_id)
            or not re.fullmatch(r"[0-9a-f]{40}", baseline_commit)
            or not re.fullmatch(r"[0-9a-f]{40}", implementation_commit)
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", completion_event_id
            )
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", completion_source_event_id
            )
            or digest != _sha256_bytes(_canonical_json(evidence))
            or str(evidence.get("binding_id") or "") != binding_id
            or str(evidence.get("task_cid") or "") != str(attempt.task_cid)
            or str(evidence.get("task_alias") or "")
            != str(attempt.task_alias)
            or str(evidence.get("attempt_id") or "")
            != str(attempt.attempt_id)
            or str(evidence.get("baseline_commit") or "")
            != baseline_commit
            or str(evidence.get("implementation_commit") or "")
            != implementation_commit
            or str(evidence.get("completion_event_id") or "")
            != completion_event_id
            or str(evidence.get("completion_source_event_id") or "")
            != completion_source_event_id
            or str(evidence.get("completion_source_event_type") or "")
            not in {
                "implementation_finished",
                "worktree_reconciliation_candidate_queued",
            }
            or isinstance(
                evidence.get("completion_source_portal_attempt"),
                bool,
            )
            or not isinstance(
                evidence.get("completion_source_portal_attempt"),
                int,
            )
            or int(evidence["completion_source_portal_attempt"]) < 1
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected malformed Portal evidence identity"
            )
        baseline_tree = self._require_portal_commit_lineage(
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        )
        self._verify_vrif_semantic_acceptance(
            attempt=attempt,
            baseline_commit=baseline_commit,
            baseline_tree=baseline_tree,
            implementation_commit=implementation_commit,
        )
        return {
            "receipt_id": receipt_id,
            "evidence_digest": digest,
            "binding_id": binding_id,
            "baseline_commit": baseline_commit,
            "baseline_tree": baseline_tree,
            "implementation_commit": implementation_commit,
            "completion_event_id": completion_event_id,
        }

    def apply_effect(self, attempt: Any, provider_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind the already-applied Portal effect to the database phase."""

        accepted = self._require_accepted_provider(attempt, provider_result)
        completion_binding = {
            "schema": DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA,
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "binding_id": str(accepted["binding_id"]),
            "portal_receipt_id": str(accepted["receipt_id"]),
            "evidence_digest": str(accepted["evidence_digest"]),
            "baseline_commit": str(accepted["baseline_commit"]),
            "baseline_tree": str(accepted["baseline_tree"]),
            "implementation_commit": str(accepted["implementation_commit"]),
            "completion_event_id": str(accepted["completion_event_id"]),
        }
        completion_binding["receipt_id"] = _sha256_bytes(
            _canonical_json(completion_binding)
        )
        return {
            "status": "applied",
            "effect": "portal-supervised-accepted-effect",
            "effect_key": f"portal:{attempt.task_cid}:{attempt.attempt_id}",
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "binding_id": str(accepted["binding_id"]),
            "portal_receipt_id": str(accepted["receipt_id"]),
            "evidence_digest": str(accepted["evidence_digest"]),
            "baseline_commit": str(accepted["baseline_commit"]),
            "baseline_tree": str(accepted["baseline_tree"]),
            "implementation_commit": str(accepted["implementation_commit"]),
            "completion_event_id": str(accepted["completion_event_id"]),
            "portal_completion_binding": completion_binding,
        }

    def validate_effect(self, attempt: Any, effect_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Admit only an exact effect derived from accepted Portal evidence."""

        digest = str(effect_result.get("evidence_digest") or "")
        binding = effect_result.get("portal_completion_binding")
        binding_body = dict(binding) if isinstance(binding, Mapping) else {}
        binding_receipt_id = str(binding_body.pop("receipt_id", "") or "")
        if (
            set(effect_result)
            != {
                "status",
                "effect",
                "effect_key",
                "task_cid",
                "attempt_id",
                "binding_id",
                "portal_receipt_id",
                "evidence_digest",
                "baseline_commit",
                "baseline_tree",
                "implementation_commit",
                "completion_event_id",
                "portal_completion_binding",
            }
            or effect_result.get("status") != "applied"
            or effect_result.get("effect") != "portal-supervised-accepted-effect"
            or str(effect_result.get("effect_key") or "")
            != f"portal:{attempt.task_cid}:{attempt.attempt_id}"
            or str(effect_result.get("task_cid") or "") != str(attempt.task_cid)
            or str(effect_result.get("attempt_id") or "") != str(attempt.attempt_id)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or not isinstance(binding, Mapping)
            or set(binding) != _DATABASE_PORTAL_COMPLETION_BINDING_FIELDS
            or binding.get("schema")
            != DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA
            or binding_receipt_id
            != _sha256_bytes(_canonical_json(binding_body))
            or str(binding.get("task_cid") or "") != str(attempt.task_cid)
            or str(binding.get("attempt_id") or "")
            != str(attempt.attempt_id)
            or str(binding.get("binding_id") or "")
            != str(effect_result.get("binding_id") or "")
            or str(binding.get("portal_receipt_id") or "")
            != str(effect_result.get("portal_receipt_id") or "")
            or str(binding.get("evidence_digest") or "") != digest
            or str(binding.get("baseline_commit") or "")
            != str(effect_result.get("baseline_commit") or "")
            or str(binding.get("baseline_tree") or "")
            != str(effect_result.get("baseline_tree") or "")
            or str(binding.get("implementation_commit") or "")
            != str(effect_result.get("implementation_commit") or "")
            or str(binding.get("completion_event_id") or "")
            != str(effect_result.get("completion_event_id") or "")
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(effect_result.get("binding_id") or ""),
            )
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(effect_result.get("portal_receipt_id") or ""),
            )
            or not re.fullmatch(
                r"[0-9a-f]{40}",
                str(effect_result.get("baseline_commit") or ""),
            )
            or not re.fullmatch(
                r"[0-9a-f]{40}",
                str(effect_result.get("baseline_tree") or ""),
            )
            or not re.fullmatch(
                r"[0-9a-f]{40}",
                str(effect_result.get("implementation_commit") or ""),
            )
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(effect_result.get("completion_event_id") or ""),
            )
        ):
            raise DatabasePortalBridgeError(
                "database validation rejected unbound Portal effect evidence"
            )
        # Effect receipts are durable and may predate the currently running
        # bridge process.  Re-run the owner-computed VRIF acceptance contract
        # at the validation boundary as well as at fresh effect admission so
        # a crash/restart cannot replay a structurally valid legacy receipt
        # around a newly introduced semantic guard.
        baseline_commit = str(effect_result.get("baseline_commit") or "")
        implementation_commit = str(
            effect_result.get("implementation_commit") or ""
        )
        resolved_baseline_tree = self._require_portal_commit_lineage(
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
        )
        if resolved_baseline_tree != str(
            effect_result.get("baseline_tree") or ""
        ):
            raise DatabasePortalBridgeError(
                "database validation rejected a stale Portal baseline tree"
            )
        self._verify_vrif_semantic_acceptance(
            attempt=attempt,
            baseline_commit=baseline_commit,
            baseline_tree=resolved_baseline_tree,
            implementation_commit=implementation_commit,
        )
        return {
            "outcome": "passed",
            "evidence_digest": digest,
            "argv": ["portal-supervisor-gates"],
            "validator": self.INTERFACE,
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(effect_result.get("portal_receipt_id") or ""),
            "portal_completion_binding": dict(binding),
        }


__all__ = (
    "DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA",
    "DATABASE_PORTAL_CAPACITY_RETRY_SCHEMA",
    "DATABASE_PORTAL_CAPACITY_RETRY_SEED_SCHEMA",
    "DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA",
    "DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SCHEMA",
    "DATABASE_PORTAL_CONSUMED_ATTEMPT_RETRY_SEED_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA",
    "DATABASE_PORTAL_PROTECTED_PATH_PRESERVATION_SCHEMA",
    "DATABASE_PORTAL_PROTECTED_RECONCILIATION_SELF_LOCK_SCHEMA",
    "DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA",
    "DATABASE_PORTAL_VALIDATION_RETRY_SEED_SCHEMA",
    "DatabasePortalAttemptPaths",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeError",
    "DatabasePortalCapacityRetry",
    "DatabasePortalConsumedAttemptTerminal",
    "DatabasePortalExecutionBridge",
    "DatabasePortalProtectedPathPreserved",
    "DatabasePortalValidationRetry",
    "PortalDaemonFactory",
    "verify_database_portal_attempt_projection",
)
