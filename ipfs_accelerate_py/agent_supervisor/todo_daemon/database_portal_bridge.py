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
import stat
import subprocess
import tempfile
import time
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

from ..merge.merge_queue import FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA
from ..merge.protected_recovery_fence import (
    FENCE_CONTENTION_BACKOFF_SECONDS,
    is_protected_recovery_fence_contention,
)
from ..runtime.event_log import append_jsonl_event, utc_now
from ..validation.validation_commands import validation_command_repository_root
from .implementation_timeout import DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS
from .landed_completion_recovery import (
    LandedCompletionRecoveryError,
    build_landed_completion_claim_seed,
    discover_landed_completion_recovery,
    revalidate_landed_completion_repository,
    verify_landed_completion_claim_seed,
    verify_landed_completion_recovery_receipt,
)

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
DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-consumed-no-progress@1"
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
DATABASE_PORTAL_VALIDATION_RETRY_ORDER_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-validation-retry-order-repair@1"
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
_MAX_DATABASE_PORTAL_CAPACITY_BACKOFF_SECONDS: Final[int] = 31 * 86_400
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
DATABASE_PORTAL_SKIP_CONTENTION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "inflight_process",
        "provider_capacity_backoff",
        "task_claim_lock_exists",
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
_POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V2_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "post-merge-callback-integration-requalification@2"
)
_POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V3_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "post-merge-callback-integration-requalification@3"
)
_POST_MERGE_CALLBACK_VALIDATION_WORKSPACE_HYGIENE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "post-merge-callback-validation-workspace-hygiene@1"
)
_POST_MERGE_SETTLED_CALLBACK_INTEGRATION_SOURCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "post-merge-settled-callback-integration-source@2"
)
_POST_MERGE_SETTLED_CALLBACK_INTEGRATION_SOURCE_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema",
            "source_shape",
            "settlement_receipt_id",
            "quarantine_receipt_id",
            "quarantine_receipt",
            "revival_id",
            "revival",
            "enqueue_event_id",
            "enqueue_event_digest",
            "projected_source_event_id",
            "projected_source_event_digest",
            "reconciliation_event_id",
            "reconciliation_event_digest",
            "terminal_event_id",
            "terminal_event_digest",
            "status_event_id",
            "status_event_digest",
            "completion_event_id",
            "completion_event_digest",
            "source_id",
        }
    )
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
_POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V2_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            *_POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_FIELDS,
            "settled_integration_source",
        }
    )
)
_POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V3_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        *_POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V2_FIELDS,
        "workspace_hygiene",
    }
)
_POST_MERGE_CALLBACK_VALIDATION_WORKSPACE_HYGIENE_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        "schema",
        "target_commit",
        "target_tree",
        "declared_entries",
        "pre_validation_identities",
        "generated_identities",
        "restored_identities",
        "generated_dirty_paths",
        "restoration_performed",
        "final_clean",
        "hygiene_id",
    }
)
_POST_MERGE_CALLBACK_VALIDATION_OUTPUT_IDENTITY_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        "path",
        "index_mode",
        "index_object_id",
        "worktree_mode",
        "worktree_object_id",
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
DATABASE_PORTAL_ATTEMPT_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "attempt_id",
        "claim_id",
        "attempt_number",
        "owner_session_id",
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
        "landed_completion_recovery_seed_id",
        "binding_id",
    }
)

# Backward-compatible private alias for the local verifier.  The public
# constant is also consumed by the implementation supervisor so the producer,
# verifier, and source-reload quiescence proof cannot silently drift onto
# different exact-field schemas.
_DATABASE_PORTAL_ATTEMPT_BINDING_FIELDS = (
    DATABASE_PORTAL_ATTEMPT_BINDING_FIELDS
)


def _is_implementation_conflict(exc: BaseException) -> bool:
    """Return whether ``exc`` is a database implementation conflict.

    Running ``python -m ...implementation_daemon`` binds daemon classes to
    ``__main__``.  Relative imports of ``DatabaseImplementationConflictError``
    then see a different type, so identity-based ``except`` misses live
    preauthorization conflicts and fails the maintenance tick.
    """

    return type(exc).__name__ == "DatabaseImplementationConflictError"


DATABASE_PORTAL_INFLIGHT_POLL_SECONDS: Final[float] = 15.0
_MAX_DATABASE_PORTAL_PASS_PREVIEW: Final[int] = 16
DATABASE_PORTAL_QUOTA_FALLBACK_AUTHORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.grok-quota-fallback-authority@2"
)


def _monotonic_seconds() -> float:
    """Return the local monotonic clock (a narrow seam for deterministic tests)."""

    return time.monotonic()


def _sleep_seconds(seconds: float) -> None:
    """Sleep without ever widening the closed in-flight polling interval."""

    time.sleep(seconds)


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
        "owner_session_id",
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
    if type(binding.get("attempt_number")) is not int or int(
        binding["attempt_number"]
    ) < 1:
        raise DatabasePortalBridgeError(
            "database Portal attempt binding attempt number is invalid"
        )
    landed_recovery_seed_id = binding.get(
        "landed_completion_recovery_seed_id"
    )
    if type(landed_recovery_seed_id) is not str or (
        landed_recovery_seed_id
        and re.fullmatch(r"sha256:[0-9a-f]{64}", landed_recovery_seed_id)
        is None
    ):
        raise DatabasePortalBridgeError(
            "database Portal attempt binding landed recovery seed is invalid"
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
        "Database attempt number": str(binding["attempt_number"]),
        "Database owner session ID": str(binding["owner_session_id"]),
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
        "attempt_number": int(binding["attempt_number"]),
        "owner_session_id": str(binding["owner_session_id"]),
        "lease_id": str(binding["lease_id"]),
        "task_alias": task_alias,
        "task_cid": task_cid,
        "canonical_task_key": canonical_task_key,
        "goal_cid": str(binding["goal_cid"]),
        "plan_cid": str(binding["plan_cid"]),
        "task_revision": int(binding["task_revision"]),
        "fencing_token": int(binding["fencing_token"]),
        "fence_epoch": int(binding["fence_epoch"]),
        "landed_completion_recovery_seed_id": str(
            binding["landed_completion_recovery_seed_id"]
        ),
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
        merge_target_ref: str = "HEAD",
        worktree_submodule_paths: Sequence[str] = (),
        task_header_prefix: str = "## ",
        max_passes: int = 4,
        max_task_attempts: int = 0,
        implementation_timeout: float = DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS,
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
        if (
            isinstance(implementation_timeout, bool)
            or not isinstance(implementation_timeout, (int, float))
            or not math.isfinite(float(implementation_timeout))
            or float(implementation_timeout) <= 0
        ):
            raise ValueError("implementation_timeout must be finite and positive")
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
        self.merge_target_ref = str(merge_target_ref or "HEAD").strip() or "HEAD"
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
        self.implementation_timeout = float(implementation_timeout)

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

    def _settled_callback_integration_source_evidence(
        self,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
        *,
        train: Any,
        receipt_key: str,
        settlement_receipt: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Verify the closed historical integrated-quarantine settlement.

        One merge-train producer durably recorded a successful callback
        reconciliation, then quarantined the request because the subsequent
        task-board completion write could not produce its member receipt.  A
        later train pass proved the candidate already integrated and rewrote
        the canonical receipt as an ``already_merged`` settlement.  Neither
        receipt is sufficient alone.  This alternate source class requires
        both immutable receipts, their two-hop Portal event linkage, the one
        exact queue revival, and fresh Git/blob identity at the current target.
        """

        if self.repository_root is None or self.merge_queue is None:
            return None
        from ..proof.formal_verification_contracts import content_identity

        metadata = getattr(request, "metadata", None)
        task_alias = str(getattr(request, "task_id", "") or "")
        task_cid = str(getattr(request, "canonical_task_id", "") or "")
        task_key = str(getattr(request, "canonical_task_key", "") or "")
        canonical = str(getattr(request, "canonical_identity", "") or "")
        request_id = str(getattr(request, "request_id", "") or "")
        candidate = str(getattr(request, "commit_sha", "") or "")
        task_payload = metadata.get("task") if isinstance(metadata, Mapping) else None
        outputs = (
            task_payload.get("outputs") if isinstance(task_payload, Mapping) else None
        )
        validation_proof = (
            metadata.get("validation_proof") if isinstance(metadata, Mapping) else None
        )
        baseline = (
            str(metadata.get("baseline_ref") or "")
            if isinstance(metadata, Mapping)
            else ""
        )
        candidate_tree = (
            str(metadata.get("candidate_tree") or "")
            if isinstance(metadata, Mapping)
            else ""
        )
        completion_task_cids = (
            metadata.get("completion_task_cids")
            if isinstance(metadata, Mapping)
            else None
        )
        if (
            str(getattr(request, "status", "") or "") != "completed"
            or not isinstance(metadata, Mapping)
            or metadata.get("schema") != _MERGE_CANDIDATE_SCHEMA
            or "completion" in metadata
            or not all((task_alias, task_cid, task_key, request_id))
            or canonical != task_key
            or re.fullmatch(r"[0-9a-f]{40}", candidate) is None
            or re.fullmatch(r"[0-9a-f]{40}", baseline) is None
            or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", candidate_tree) is None
            or metadata.get("implementation_commit") != candidate
            or completion_task_cids != {task_alias: task_cid}
            or not isinstance(task_payload, Mapping)
            or task_payload.get("task_id") != task_alias
            or task_payload.get("canonical_task_cid") != task_cid
            or task_payload.get("canonical_task_key") != task_key
            or not isinstance(outputs, list)
            or not outputs
            or len(outputs) > 4096
            or len(set(outputs)) != len(outputs)
            or any(not isinstance(item, str) or not item for item in outputs)
            or not isinstance(validation_proof, Mapping)
            or validation_proof.get("attempted") is not True
            or validation_proof.get("passed") is not True
            or validation_proof.get("returncode") != 0
            or validation_proof.get("target_commit") != candidate
        ):
            return None

        settlement_fields = {
            "already_merged",
            "canonical_task_id",
            "commit_sha",
            "distributed_publication_admission",
            "finished_at",
            "integrated",
            "merge_commit",
            "merged",
            "mutation_short_circuited",
            "reason",
            "request_id",
            "started_at",
            "status",
            "target_branch",
            "target_commit",
            "task_id",
        }
        settlement = dict(settlement_receipt)
        admission = settlement.get("distributed_publication_admission")
        settlement_started = settlement.get("started_at")
        settlement_finished = settlement.get("finished_at")
        integration = str(settlement.get("target_commit") or "")
        if (
            set(settlement) != settlement_fields
            or settlement.get("status") != "already_merged"
            or settlement.get("reason") != "declared_outputs_already_on_target"
            or settlement.get("already_merged") is not True
            or settlement.get("integrated") is not True
            or settlement.get("merged") is not False
            or settlement.get("mutation_short_circuited") is not True
            or settlement.get("request_id") != request_id
            or settlement.get("task_id") != task_alias
            or settlement.get("canonical_task_id") != canonical
            or settlement.get("commit_sha") != candidate
            or settlement.get("target_branch") != self.merge_target_branch
            or settlement.get("merge_commit") != integration
            or re.fullmatch(r"[0-9a-f]{40}", integration) is None
            or isinstance(settlement_started, bool)
            or not isinstance(settlement_started, (int, float))
            or isinstance(settlement_finished, bool)
            or not isinstance(settlement_finished, (int, float))
            or not math.isfinite(float(settlement_started))
            or not math.isfinite(float(settlement_finished))
            or float(settlement_started) > float(settlement_finished)
            or not isinstance(admission, Mapping)
            or set(admission)
            != {"schema", "admitted", "distributed", "request_id", "status"}
            or admission.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/distributed-lane-admission@1"
            or admission.get("admitted") is not True
            or admission.get("distributed") is not False
            or admission.get("request_id") != request_id
            or admission.get("status") != "local"
        ):
            return None

        revivals = metadata.get("revivals")
        revival = (
            revivals[0] if isinstance(revivals, list) and len(revivals) == 1 else None
        )
        revival_fields = {
            "at",
            "previous_enqueued_at",
            "previous_failure_count",
            "previous_failure_reason",
            "reason",
        }
        revival_at = revival.get("at") if isinstance(revival, Mapping) else None
        previous_enqueued_at = (
            revival.get("previous_enqueued_at")
            if isinstance(revival, Mapping)
            else None
        )
        if (
            not isinstance(revival, Mapping)
            or set(revival) != revival_fields
            or revival.get("previous_failure_count") != 1
            or revival.get("previous_failure_reason")
            != "merge_completion_receipt_invalid"
            or revival.get("reason")
            != (
                "merge train proved quarantined candidate already "
                "integrated into exact target"
            )
            or isinstance(revival_at, bool)
            or not isinstance(revival_at, (int, float))
            or isinstance(previous_enqueued_at, bool)
            or not isinstance(previous_enqueued_at, (int, float))
            or not math.isfinite(float(revival_at))
            or not math.isfinite(float(previous_enqueued_at))
            or float(previous_enqueued_at) > float(revival_at)
            or getattr(request, "attempt", None) != 1
            or getattr(request, "failure_count", None) != 0
            or str(getattr(request, "failure_reason", "") or "")
            or float(getattr(request, "enqueued_at", -1.0)) != float(revival_at)
        ):
            return None

        read_receipt = getattr(train, "_read_receipt", None)
        receipt_path = getattr(train, "_receipt_path", None)
        if not callable(read_receipt) or not callable(receipt_path):
            raise DatabasePortalBridgeError(
                "merge train lacks settled callback recovery receipts"
            )
        quarantine_key = f"quarantine-{request_id}"
        try:
            train_receipt_dir = getattr(train, "receipt_dir", None)
            if (
                not isinstance(train_receipt_dir, Path)
                or train_receipt_dir.is_symlink()
                or not train_receipt_dir.is_dir()
            ):
                return None
            train_receipt_dir_resolved = train_receipt_dir.resolve(strict=True)
            settlement_path = Path(receipt_path(receipt_key))
            quarantine_path = Path(receipt_path(quarantine_key))
            if (
                settlement_path.is_symlink()
                or quarantine_path.is_symlink()
                or not settlement_path.is_file()
                or not quarantine_path.is_file()
                or settlement_path.stat().st_size <= 0
                or quarantine_path.stat().st_size <= 0
                or settlement_path.stat().st_size
                > _MAX_DATABASE_PORTAL_PROJECTION_BYTES
                or quarantine_path.stat().st_size
                > _MAX_DATABASE_PORTAL_PROJECTION_BYTES
                or settlement_path.resolve(strict=True).parent
                != quarantine_path.resolve(strict=True).parent
                or settlement_path.resolve(strict=True).parent
                != train_receipt_dir_resolved
            ):
                return None
        except (OSError, RuntimeError, ValueError):
            return None
        quarantine_raw = read_receipt(quarantine_key)
        quarantine = dict(quarantine_raw) if isinstance(quarantine_raw, Mapping) else {}
        quarantine_fields = {
            "acceptance_pending",
            "accepted",
            "canonical_task_id",
            "commit_sha",
            "failure_count",
            "finished_at",
            "integrated",
            "max_attempts",
            "merge_result",
            "merged",
            "reason",
            "request_id",
            "retryable",
            "started_at",
            "status",
            "target_branch",
            "task_id",
        }
        quarantine_started = quarantine.get("started_at")
        quarantine_finished = quarantine.get("finished_at")
        quarantine_merge = quarantine.get("merge_result")
        if (
            not quarantine
            or set(quarantine) != quarantine_fields
            or metadata.get("quarantine") != quarantine
            or quarantine.get("status") != "quarantined"
            or quarantine.get("reason") != "merge_completion_receipt_invalid"
            or quarantine.get("accepted") is not False
            or quarantine.get("integrated") is not False
            or quarantine.get("merged") is not False
            or quarantine.get("acceptance_pending") is not False
            or quarantine.get("retryable") is not False
            or quarantine.get("failure_count") != 1
            or isinstance(quarantine.get("max_attempts"), bool)
            or not isinstance(quarantine.get("max_attempts"), int)
            or int(quarantine.get("max_attempts")) < 1
            or quarantine.get("request_id") != request_id
            or quarantine.get("task_id") != task_alias
            or quarantine.get("canonical_task_id") != canonical
            or quarantine.get("commit_sha") != candidate
            or quarantine.get("target_branch") != self.merge_target_branch
            or isinstance(quarantine_started, bool)
            or not isinstance(quarantine_started, (int, float))
            or isinstance(quarantine_finished, bool)
            or not isinstance(quarantine_finished, (int, float))
            or not all(
                math.isfinite(float(item))
                for item in (quarantine_started, quarantine_finished)
            )
            or not (
                float(previous_enqueued_at)
                <= float(quarantine_started)
                <= float(quarantine_finished)
                < float(revival_at)
                <= float(settlement_started)
                <= float(settlement_finished)
            )
            or not isinstance(quarantine_merge, Mapping)
        ):
            return None
        proof = quarantine_merge.get("integration_commit_proof")
        invariant = quarantine_merge.get("post_merge_declared_output_invariant")
        reconciliation_receipt = quarantine_merge.get("merge_reconciliation_receipt")
        completion_error = quarantine_merge.get("completion_receipt_error")
        todo_result = quarantine_merge.get("todo_update_result")
        checks = invariant.get("checks") if isinstance(invariant, Mapping) else None
        quarantine_merge_fields = {
            "already_merged",
            "attempted",
            "branch",
            "cleanup_result",
            "command",
            "completion_receipt_error",
            "deterministic_conflict_repair",
            "finished_at",
            "generated_submodule_reconciliation",
            "identical_untracked_paths",
            "integration_commit_proof",
            "integration_occurred",
            "main_worktree_path",
            "merge_commit",
            "merge_reconciliation_receipt",
            "merged",
            "merged_gitlink_recording",
            "post_merge_declared_output_invariant",
            "reason",
            "resolved_generated_conflicts",
            "restored_generated_dirty_overlap",
            "returncode",
            "shared_worktree_path_scrub",
            "started_at",
            "stderr",
            "stdout",
            "submodule_failure_rollback",
            "submodule_merge_results",
            "target_branch",
            "target_commit",
            "todo_update_result",
            "used_ephemeral_main_worktree",
        }
        todo_result_fields = {
            "completion_reason",
            "lock_owner_branch",
            "lock_owner_lease_id",
            "lock_owner_pid",
            "lock_owner_task_id",
            "lock_path",
            "reason",
            "task_id",
            "updated",
        }
        lock_owner_pid = (
            todo_result.get("lock_owner_pid")
            if isinstance(todo_result, Mapping)
            else None
        )
        lock_owner_lease_id = (
            str(todo_result.get("lock_owner_lease_id") or "")
            if isinstance(todo_result, Mapping)
            else ""
        )
        from ..merge.checkout_lock import checkout_mutation_lock_path

        expected_lock_path = checkout_mutation_lock_path(self.repository_root)
        if (
            set(quarantine_merge) != quarantine_merge_fields
            or quarantine_merge.get("attempted") is not True
            or quarantine_merge.get("integration_occurred") is not True
            or quarantine_merge.get("merged") is not False
            or quarantine_merge.get("already_merged") is not False
            or quarantine_merge.get("reason") != "merge_completion_receipt_invalid"
            or quarantine_merge.get("returncode") != 2
            or quarantine_merge.get("merge_commit") != integration
            or quarantine_merge.get("target_commit") != integration
            or quarantine_merge.get("target_branch") != self.merge_target_branch
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
            or invariant.get("reason") != "declared_outputs_tracked"
            or invariant.get("mode") != "repository_tree"
            or invariant.get("repository_ref") != integration
            or invariant.get("task_ids") != [task_alias]
            or invariant.get("missing_outputs") != []
            or invariant.get("unsafe_outputs") != []
            or invariant.get("untracked_outputs") != []
            or not isinstance(checks, list)
            or [
                str(item.get("path") or "")
                for item in checks
                if isinstance(item, Mapping)
            ]
            != outputs
            or not isinstance(reconciliation_receipt, Mapping)
            or set(reconciliation_receipt) != {"recorded", "replayed", "event_id"}
            or reconciliation_receipt.get("recorded") is not True
            or reconciliation_receipt.get("replayed") is not False
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(reconciliation_receipt.get("event_id") or ""),
            )
            is None
            or not isinstance(completion_error, Mapping)
            or set(completion_error)
            != {"reason", "expected_task_cids", "receipt_task_cids"}
            or completion_error.get("reason") != "completion_receipt_binding_mismatch"
            or completion_error.get("expected_task_cids") != {task_alias: task_cid}
            or completion_error.get("receipt_task_cids") != {}
            or not isinstance(todo_result, Mapping)
            or set(todo_result) != todo_result_fields
            or todo_result.get("updated") is not False
            or todo_result.get("task_id") != task_alias
            or todo_result.get("completion_reason") != "single_task"
            or todo_result.get("reason") != "checkout_mutation_lock_exists"
            or Path(str(todo_result.get("lock_path") or "")) != expected_lock_path
            or not expected_lock_path.is_absolute()
            or not lock_owner_lease_id
            or lock_owner_lease_id != lock_owner_lease_id.strip()
            or len(lock_owner_lease_id) > 512
            or isinstance(lock_owner_pid, bool)
            or not isinstance(lock_owner_pid, int)
            or lock_owner_pid <= 0
            or todo_result.get("lock_owner_branch") != ""
            or todo_result.get("lock_owner_task_id") != ""
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

        indexed = list(enumerate(events))
        enqueues = [
            (index, event)
            for index, event in indexed
            if event.get("type") == "merge_candidate_enqueued"
            and event_request_id(event) == request_id
        ]
        projected = [
            (index, event)
            for index, event in indexed
            if event.get("type") == "worktree_reconciliation_candidate_queued"
            and event_request_id(event) == request_id
        ]
        reconciliations = [
            (index, event)
            for index, event in indexed
            if event.get("type") == "merge_reconciled"
            and (
                event_request_id(event) == request_id
                or event.get("event_id") == reconciliation_receipt.get("event_id")
            )
        ]
        terminals = [
            (index, event)
            for index, event in indexed
            if event.get("type") == "implementation_finished"
            and event_request_id(event) == request_id
        ]
        status_events = [
            (index, event)
            for index, event in indexed
            if event.get("type") == "todo_status_updated"
            and event.get("task_id") == task_alias
            and event.get("completion_reason") == "merged_status_repair"
        ]
        completions = [
            (index, event)
            for index, event in indexed
            if event.get("type") == "task_completed"
            and event.get("task_id") == task_alias
            and event.get("canonical_task_cid") == task_cid
        ]
        if not all(
            len(items) == 1
            for items in (
                enqueues,
                projected,
                reconciliations,
                terminals,
                status_events,
                completions,
            )
        ):
            return None
        enqueue_index, enqueue = enqueues[0]
        projected_index, source_event = projected[0]
        reconciliation_index, reconciliation = reconciliations[0]
        terminal_index, terminal = terminals[0]
        status_index, status_event = status_events[0]
        completion_index, completion = completions[0]
        source_event_id = str(source_event.get("event_id") or "")
        reconciliation_event_id = str(reconciliation.get("event_id") or "")
        terminal_event_id = str(terminal.get("event_id") or "")
        status_event_id = str(status_event.get("event_id") or "")
        completion_event_id = str(completion.get("event_id") or "")
        if (
            not enqueue_index
            < projected_index
            < reconciliation_index
            < terminal_index
            < status_index
            < completion_index
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", event_id) is None
                for event_id in (
                    str(enqueue.get("event_id") or ""),
                    source_event_id,
                    reconciliation_event_id,
                    terminal_event_id,
                    status_event_id,
                    completion_event_id,
                )
            )
            or reconciliation_event_id != reconciliation_receipt.get("event_id")
            or not self._exact_callback_reconciliation_for_completion_source(
                reconciliation,
                source_event,
                alias=task_alias,
                task_cid=task_cid,
            )
        ):
            return None
        provenance = source_event.get("merge_queue_synchronous_source")
        provenance_body = dict(provenance) if isinstance(provenance, Mapping) else {}
        provenance_id = str(provenance_body.pop("source_projection_id", "") or "")
        source_merge = source_event.get("merge_result")
        terminal_merge = terminal.get("merge_result")
        terminal_validation = terminal.get("validation_result")
        receipt_evidence = reconciliation.get("completion_receipt_evidence")
        expected_member_receipts = (
            receipt_evidence.get("completion_receipts")
            if isinstance(receipt_evidence, Mapping)
            else None
        )
        if (
            not provenance_body
            or provenance_id != content_identity(provenance_body)
            or provenance_body.get("merge_candidate_enqueued_event_id")
            != enqueue.get("event_id")
            or enqueue.get("task_id") != task_alias
            or enqueue.get("canonical_task_cid") != task_cid
            or enqueue.get("canonical_task_key") != task_key
            or enqueue.get("attempt") != source_event.get("attempt")
            or enqueue.get("baseline_ref") != baseline
            or enqueue.get("implementation_commit") != candidate
            or enqueue.get("attempted") is not False
            or enqueue.get("queued") is not True
            or enqueue.get("merged") is not False
            or enqueue.get("reason") != "merge_queued"
            or enqueue.get("completion_task_cids") != {task_alias: task_cid}
            or not isinstance(source_merge, Mapping)
            or terminal.get("task_id") != task_alias
            or terminal.get("canonical_task_cid") != task_cid
            or terminal.get("canonical_task_key") != task_key
            or terminal.get("attempt") != source_event.get("attempt")
            or terminal.get("attempt_consumed") is not True
            or terminal.get("provider_dispatched") is not True
            or type(terminal.get("returncode")) is not int
            or terminal.get("returncode") != 0
            or terminal.get("branch") != source_event.get("branch")
            or terminal.get("baseline_ref") != baseline
            or terminal.get("implementation_commit") != candidate
            or terminal.get("board_completion")
            != {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            }
            or not isinstance(terminal_validation, Mapping)
            or terminal_validation.get("attempted") is not True
            or terminal_validation.get("passed") is not True
            or terminal_validation.get("returncode") != 0
            or not isinstance(terminal_merge, Mapping)
            or terminal_merge.get("attempted") is not False
            or terminal_merge.get("queued") is not True
            or terminal_merge.get("merged") is not False
            or terminal_merge.get("reason") != "merge_queued"
            or terminal_merge.get("request_id") != request_id
            or terminal_merge.get("branch") != source_merge.get("branch")
            or terminal_merge.get("implementation_commit") != candidate
            or terminal_merge.get("canonical_task_cid") != task_cid
            or terminal_merge.get("canonical_task_key") != task_key
            or terminal_merge.get("completion_task_cids") != {task_alias: task_cid}
            or terminal_merge.get("target_repository_id")
            != source_merge.get("target_repository_id")
            or terminal_merge.get("target_branch") != self.merge_target_branch
            or terminal_merge.get("train_result") != quarantine
            or reconciliation.get("integration_commit_proof") != proof
            or reconciliation.get("post_merge_declared_output_invariant") != invariant
            or status_event.get("updated") is not True
            or status_event.get("updated_task_ids") != [task_alias]
            or status_event.get("missing_task_ids") != []
            or status_event.get("missing_status_task_ids") != []
            or not isinstance(expected_member_receipts, list)
            or len(expected_member_receipts) != 1
            or status_event.get("completion_receipts") != expected_member_receipts
            or completion.get("reason") != "task_became_completed"
            or completion.get("completion_receipt_repair") is not False
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
            head_text = (
                head.stdout.decode().strip()
                if isinstance(head.stdout, bytes)
                else str(head.stdout).strip()
            )
            current_tree_result = git("rev-parse", "--verify", f"{head_text}^{{tree}}")
            current_tree = (
                current_tree_result.stdout.decode().strip()
                if isinstance(current_tree_result.stdout, bytes)
                else str(current_tree_result.stdout).strip()
            )
            candidate_tree_result = git(
                "rev-parse", "--verify", f"{candidate}^{{tree}}"
            )
            candidate_tree_text = (
                candidate_tree_result.stdout.decode().strip()
                if isinstance(candidate_tree_result.stdout, bytes)
                else str(candidate_tree_result.stdout).strip()
            )
            parents = git("rev-list", "--parents", "-n", "1", candidate)
            parent_text = (
                parents.stdout.decode().strip()
                if isinstance(parents.stdout, bytes)
                else str(parents.stdout).strip()
            )
            candidate_integration = git(
                "merge-base", "--is-ancestor", candidate, integration
            )
            integration_current = git(
                "merge-base", "--is-ancestor", integration, head_text
            )
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
        for output in outputs:
            try:
                safe_path = _safe_output_path(output)
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
            if not raw_entry or any(
                item.stdout != observed[0].stdout for item in observed[1:]
            ):
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

        canonical_settlement = _canonical_json(settlement)
        canonical_quarantine = _canonical_json(quarantine)
        canonical_revival = _canonical_json(revival)
        if any(
            len(item) > _MAX_DATABASE_PORTAL_PROJECTION_BYTES
            for item in (
                canonical_settlement,
                canonical_quarantine,
                canonical_revival,
            )
        ):
            return None
        settlement_id = _sha256_bytes(canonical_settlement)
        settled_source: dict[str, Any] = {
            "schema": _POST_MERGE_SETTLED_CALLBACK_INTEGRATION_SOURCE_SCHEMA,
            "source_shape": "settled_integrated_quarantine",
            "settlement_receipt_id": settlement_id,
            "quarantine_receipt_id": _sha256_bytes(canonical_quarantine),
            "quarantine_receipt": canonical_quarantine.decode("utf-8"),
            "revival_id": _sha256_bytes(canonical_revival),
            "revival": canonical_revival.decode("utf-8"),
            "enqueue_event_id": str(enqueue.get("event_id") or ""),
            "enqueue_event_digest": _sha256_bytes(_canonical_json(enqueue)),
            "projected_source_event_id": source_event_id,
            "projected_source_event_digest": _sha256_bytes(
                _canonical_json(source_event)
            ),
            "reconciliation_event_id": reconciliation_event_id,
            "reconciliation_event_digest": _sha256_bytes(
                _canonical_json(reconciliation)
            ),
            "terminal_event_id": terminal_event_id,
            "terminal_event_digest": _sha256_bytes(_canonical_json(terminal)),
            "status_event_id": status_event_id,
            "status_event_digest": _sha256_bytes(_canonical_json(status_event)),
            "completion_event_id": completion_event_id,
            "completion_event_digest": _sha256_bytes(_canonical_json(completion)),
        }
        settled_source["source_id"] = content_identity(settled_source)
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
                _canonical_json(source_event.get("validation_result"))
            ),
            "queue_validation_proof_digest": _sha256_bytes(
                _canonical_json(validation_proof)
            ),
            "train_dedupe_key": receipt_key,
            "train_receipt_id": settlement_id,
            "train_receipt": canonical_settlement.decode("utf-8"),
            "current_target_commit": head_text,
            "current_target_tree": current_tree,
            "entries": entries,
            "settled_integration_source": settled_source,
        }

    def _callback_integration_source_evidence(
        self,
        request: Any,
        projection: _DatabasePortalRecoveryProjection,
        *,
        train: Any,
    ) -> dict[str, Any] | None:
        """Verify one exact callback integration that missed database settlement.

        This is intentionally closed over two historical schema-v3 shapes: a
        bare completion that missed reconciliation, or a zero-provider callback
        confirmation whose terminal event omitted only the redundant target
        commit.  A completed row with a successful receipt is not enough: the
        queue row, full train receipt, exact Portal lineage, Git ancestry, and
        every declared output blob must all agree.
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
        if (
            isinstance(receipt, Mapping)
            and receipt.get("status") == "already_merged"
        ):
            return self._settled_callback_integration_source_evidence(
                request,
                projection,
                train=train,
                receipt_key=receipt_key,
                settlement_receipt=receipt,
            )
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
        common_source_valid = bool(
            source_event.get("task_id") == task_alias
            and source_event.get("canonical_task_cid") == task_cid
            and source_event.get("canonical_task_key") == task_key
            and type(source_attempt) is int
            and source_attempt >= 1
            and source_event.get("returncode") == 0
            and source_event.get("baseline_ref") == baseline
            and source_event.get("implementation_commit") == candidate
            and re.fullmatch(r"sha256:[0-9a-f]{64}", source_event_id)
            is not None
            and type(source_sequence) is int
            and isinstance(event_validation, Mapping)
            and event_validation.get("attempted") is True
            and event_validation.get("passed") is True
            and event_validation.get("returncode") == 0
            and isinstance(event_merge, Mapping)
            and event_merge.get("request_id") == request_id
            and event_merge.get("implementation_commit") == candidate
            and event_merge.get("completion_task_cids")
            == {task_alias: task_cid}
            and isinstance(event_board, Mapping)
            and len(completions) == 1
            and completions[0].get("reason") == "task_became_completed"
            and completions[0].get("completion_receipt_repair") is False
            and type(completion_sequence) is int
            and completion_sequence > source_sequence
        )
        legacy_bare_completion = bool(
            common_source_valid
            and source_event.get("attempt_consumed") is True
            and source_event.get("provider_dispatched") is True
            and event_merge.get("queued") is True
            and event_merge.get("merged") is False
            and event_merge.get("reason") == "merge_queued"
            and event_board
            == {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            }
            and not reconciliations
        )
        try:
            exact_completion = self._completion_event_evidence(
                projection.paths,
                alias=task_alias,
                task_cid=task_cid,
            )
        except DatabasePortalBridgeError:
            exact_completion = None
        # One pre-fix producer copied the immutable integration into
        # ``merge_commit`` after the exact callback reconciliation, but omitted
        # the synonymous ``target_commit`` key from its terminal confirmation.
        # Admit only that missing-key shape; an explicit conflicting target
        # remains terminally invalid.
        reconciliation = reconciliations[0] if len(reconciliations) == 1 else {}
        historical_zero_provider_confirmation = bool(
            common_source_valid
            and isinstance(exact_completion, Mapping)
            and exact_completion.get("completion_source_event_type")
            == "implementation_finished"
            and exact_completion.get("completion_source_event_id")
            == source_event_id
            and exact_completion.get("completion_event_id")
            == completions[0].get("event_id")
            and exact_completion.get("completion_source_portal_attempt")
            == source_attempt
            and exact_completion.get("baseline_commit") == baseline
            and exact_completion.get("implementation_commit") == candidate
            and source_event.get("attempt_consumed") is True
            and source_event.get("provider_dispatched") is False
            and event_merge.get("attempted") is True
            and event_merge.get("queued") is False
            and event_merge.get("merged") is True
            and event_merge.get("reason") == "merged"
            and event_merge.get("merge_commit") == integration
            and "target_commit" not in event_merge
            and event_merge.get("target_repository_id")
            == str(getattr(self.merge_queue, "target_repository_id", "") or "")
            and event_merge.get("target_branch") == self.merge_target_branch
            and event_board
            == {
                "complete": True,
                "pending_merge": False,
                "reason": "merged_into_target",
            }
            and len(reconciliations) == 1
            and reconciliation.get("request_id") == request_id
            and reconciliation.get("merge_commit") == integration
            and reconciliation.get("target_commit") == integration
        )
        if not legacy_bare_completion and not historical_zero_provider_confirmation:
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
    def _callback_validation_output_identities(
        worktree: Path,
        entries: Any,
    ) -> list[dict[str, str]] | None:
        """Capture exact index/worktree identities for declared blob outputs."""

        if not isinstance(entries, list) or not entries or len(entries) > 4096:
            return None
        identities: list[dict[str, str]] = []
        seen: set[str] = set()
        git_id = r"[0-9a-f]{40}(?:[0-9a-f]{24})?"
        for raw_entry in entries:
            path = str(raw_entry.get("path") if isinstance(raw_entry, Mapping) else "")
            try:
                safe_path = _safe_output_path(path)
            except DatabasePortalBridgeError:
                return None
            if (
                not isinstance(raw_entry, Mapping)
                or set(raw_entry) != {"path", "mode", "object_type", "object_id"}
                or safe_path != path
                or path in seen
                or raw_entry.get("mode") not in {"100644", "100755"}
                or raw_entry.get("object_type") != "blob"
                or re.fullmatch(git_id, str(raw_entry.get("object_id") or ""))
                is None
            ):
                return None
            seen.add(path)
            literal_pathspec = f":(top,literal){path}"
            try:
                indexed = subprocess.run(
                    ["git", "ls-files", "--stage", "-z", "--", literal_pathspec],
                    cwd=worktree,
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            index_mode = ""
            index_object_id = ""
            if indexed.returncode != 0:
                return None
            if indexed.stdout:
                records = indexed.stdout.split(b"\0")
                if len(records) != 2 or records[-1] != b"" or b"\t" not in records[0]:
                    return None
                prefix, raw_path = records[0].split(b"\t", 1)
                try:
                    observed_path = raw_path.decode("utf-8")
                    index_fields = prefix.decode("ascii").split()
                except UnicodeDecodeError:
                    return None
                if (
                    observed_path != path
                    or len(index_fields) != 3
                    or index_fields[0] not in {"100644", "100755"}
                    or re.fullmatch(git_id, index_fields[1]) is None
                    or index_fields[2] != "0"
                ):
                    return None
                index_mode, index_object_id = index_fields[:2]

            candidate = worktree.joinpath(*PurePosixPath(path).parts)
            parent = worktree
            for part in PurePosixPath(path).parts[:-1]:
                parent /= part
                try:
                    parent_stat = parent.lstat()
                except FileNotFoundError:
                    break
                except OSError:
                    return None
                if not stat.S_ISDIR(parent_stat.st_mode):
                    return None
            try:
                candidate_stat = candidate.lstat()
            except FileNotFoundError:
                worktree_mode = ""
                worktree_object_id = ""
            except OSError:
                return None
            else:
                if not stat.S_ISREG(candidate_stat.st_mode):
                    return None
                worktree_mode = (
                    "100755" if candidate_stat.st_mode & 0o111 else "100644"
                )
                try:
                    hashed = subprocess.run(
                        ["git", "hash-object", "--no-filters", "--", path],
                        cwd=worktree,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=10,
                    )
                except (OSError, subprocess.SubprocessError):
                    return None
                worktree_object_id = hashed.stdout.strip()
                if (
                    hashed.returncode != 0
                    or re.fullmatch(git_id, worktree_object_id) is None
                ):
                    return None
            identities.append(
                {
                    "path": path,
                    "index_mode": index_mode,
                    "index_object_id": index_object_id,
                    "worktree_mode": worktree_mode,
                    "worktree_object_id": worktree_object_id,
                }
            )
        return identities

    @staticmethod
    def _callback_validation_generated_dirty_paths(raw: Any) -> list[str] | None:
        """Admit only unique unstaged content changes to declared files."""

        if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\0"):
            return None
        paths: list[str] = []
        for record in raw[:-1].split(b"\0"):
            # Staging, untracked files, renames/copies, deletions, type/mode
            # changes, and submodule states are not validation hygiene.
            if len(record) < 4 or record[:3] != b" M ":
                return None
            try:
                path = record[3:].decode("utf-8")
                safe_path = _safe_output_path(path)
            except (UnicodeDecodeError, DatabasePortalBridgeError):
                return None
            if safe_path != path or path in paths:
                return None
            paths.append(path)
        return sorted(paths) if paths else None

    @staticmethod
    def _verified_callback_validation_workspace_hygiene(
        raw: Any,
        *,
        source: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Verify V3's content-addressed declared-output restoration proof."""

        from ..proof.formal_verification_contracts import content_identity

        if not isinstance(raw, Mapping):
            return None
        value = dict(raw)
        hygiene_id = str(value.pop("hygiene_id", "") or "")
        declared = value.get("declared_entries")
        dirty_paths = value.get("generated_dirty_paths")
        pre = value.get("pre_validation_identities")
        generated = value.get("generated_identities")
        restored = value.get("restored_identities")
        source_entries = source.get("entries")
        if (
            set(raw)
            != _POST_MERGE_CALLBACK_VALIDATION_WORKSPACE_HYGIENE_FIELDS
            or value.get("schema")
            != _POST_MERGE_CALLBACK_VALIDATION_WORKSPACE_HYGIENE_SCHEMA
            or value.get("target_commit") != source.get("current_target_commit")
            or value.get("target_tree") != source.get("current_target_tree")
            or declared != source_entries
            or source.get("task_ids") != [_VRIF_TERMINAL_TASK_ALIAS]
            or not isinstance(source_entries, list)
            or not isinstance(dirty_paths, list)
            or not dirty_paths
            or any(type(path) is not str for path in dirty_paths)
            or dirty_paths != sorted(set(dirty_paths))
            or not set(dirty_paths).issubset(
                {
                    _VRIF_RELEASE_REPORT_JSON_PATH,
                    _VRIF_RELEASE_REPORT_MARKDOWN_PATH,
                }
            )
            or value.get("restoration_performed") is not True
            or value.get("final_clean") is not True
            or hygiene_id != content_identity(value)
            or any(not isinstance(item, list) for item in (pre, generated, restored))
            or not (len(pre) == len(generated) == len(restored) == len(source_entries))
        ):
            return None
        source_by_path = {
            str(item.get("path") if isinstance(item, Mapping) else ""): item
            for item in source_entries
        }
        if (
            len(source_by_path) != len(source_entries)
            or any(path not in source_by_path for path in dirty_paths)
        ):
            return None
        observed_dirty: list[str] = []
        for index, source_entry in enumerate(source_entries):
            if not isinstance(source_entry, Mapping):
                return None
            path = str(source_entry.get("path") or "")
            expected_identity = {
                "path": path,
                "index_mode": str(source_entry.get("mode") or ""),
                "index_object_id": str(source_entry.get("object_id") or ""),
                "worktree_mode": str(source_entry.get("mode") or ""),
                "worktree_object_id": str(source_entry.get("object_id") or ""),
            }
            before = pre[index]
            during = generated[index]
            after = restored[index]
            if (
                not isinstance(before, Mapping)
                or not isinstance(during, Mapping)
                or not isinstance(after, Mapping)
                or set(before)
                != _POST_MERGE_CALLBACK_VALIDATION_OUTPUT_IDENTITY_FIELDS
                or set(during)
                != _POST_MERGE_CALLBACK_VALIDATION_OUTPUT_IDENTITY_FIELDS
                or set(after)
                != _POST_MERGE_CALLBACK_VALIDATION_OUTPUT_IDENTITY_FIELDS
                or dict(before) != expected_identity
                or dict(after) != expected_identity
                or during.get("path") != path
                or during.get("index_mode") != expected_identity["index_mode"]
                or during.get("index_object_id")
                != expected_identity["index_object_id"]
                or during.get("worktree_mode")
                != expected_identity["worktree_mode"]
            ):
                return None
            changed = during.get("worktree_object_id") != source_entry.get("object_id")
            if changed:
                observed_dirty.append(path)
                if re.fullmatch(
                    r"[0-9a-f]{40}(?:[0-9a-f]{24})?",
                    str(during.get("worktree_object_id") or ""),
                ) is None:
                    return None
        if sorted(observed_dirty) != dirty_paths:
            return None
        return {**value, "hygiene_id": hygiene_id}

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
        settled_source = value.get("settled_integration_source")
        schema = value.get("schema")
        is_settled = isinstance(settled_source, Mapping)
        is_v3 = bool(
            is_settled
            and schema
            == _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V3_SCHEMA
        )
        expected_fields = (
            _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V3_FIELDS
            if is_v3
            else (
                _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V2_FIELDS
                if is_settled
                else _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_FIELDS
            )
        )
        expected_schema = (
            _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V3_SCHEMA
            if is_v3
            else (
                _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V2_SCHEMA
                if is_settled
                else _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_SCHEMA
            )
        )
        expected_source_fields = {
            key
            for key in expected_fields
            if key
            not in {"schema", "validation", "workspace_hygiene", "receipt_id"}
        }
        if (
            set(raw) != expected_fields
            or value.get("schema") != expected_schema
            or any(value.get(key) != source.get(key) for key in expected_source_fields)
            or not isinstance(validation, list)
            or len(validation) != 1
            or receipt_id != content_identity(value)
        ):
            return None
        if is_settled:
            settled_value = dict(settled_source)
            source_id = str(settled_value.pop("source_id", "") or "")
            canonical_strings: list[tuple[str, str]] = [
                ("quarantine_receipt", "quarantine_receipt_id"),
                ("revival", "revival_id"),
            ]
            if (
                set(settled_source)
                != _POST_MERGE_SETTLED_CALLBACK_INTEGRATION_SOURCE_FIELDS
                or settled_value.get("schema")
                != _POST_MERGE_SETTLED_CALLBACK_INTEGRATION_SOURCE_SCHEMA
                or settled_value.get("source_shape") != "settled_integrated_quarantine"
                or settled_value.get("settlement_receipt_id")
                != value.get("train_receipt_id")
                or settled_value.get("projected_source_event_id")
                != value.get("source_event_id")
                or source_id != content_identity(settled_value)
                or any(
                    re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(settled_value.get(field) or ""),
                    )
                    is None
                    for field in (
                        "settlement_receipt_id",
                        "quarantine_receipt_id",
                        "revival_id",
                        "enqueue_event_id",
                        "enqueue_event_digest",
                        "projected_source_event_id",
                        "projected_source_event_digest",
                        "reconciliation_event_id",
                        "reconciliation_event_digest",
                        "terminal_event_id",
                        "terminal_event_digest",
                        "status_event_id",
                        "status_event_digest",
                        "completion_event_id",
                        "completion_event_digest",
                    )
                )
            ):
                return None
            for json_field, identity_field in canonical_strings:
                canonical = settled_value.get(json_field)
                if not isinstance(canonical, str):
                    return None
                try:
                    parsed = json.loads(canonical)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return None
                if (
                    not isinstance(parsed, Mapping)
                    or _canonical_json(parsed).decode("utf-8") != canonical
                    or _sha256_bytes(canonical.encode("utf-8"))
                    != settled_value.get(identity_field)
                ):
                    return None
        if is_v3 and (
            DatabasePortalExecutionBridge._verified_callback_validation_workspace_hygiene(
                value.get("workspace_hygiene"),
                source=source,
            )
            is None
        ):
            return None
        item = validation[0]
        digests = (
            item.get("validation_result_digests") if isinstance(item, Mapping) else None
        )
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
            or any(
                re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(digest)) is None
                for digest in digests
            )
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
        settled_source = isinstance(source.get("settled_integration_source"), Mapping)
        hygiene_eligible = (
            settled_source and task_alias == _VRIF_TERMINAL_TASK_ALIAS
        )
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
                    pre_identities: list[dict[str, str]] | None = None
                    if hygiene_eligible:
                        initial_head = subprocess.run(
                            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
                            cwd=temporary,
                            capture_output=True,
                            check=False,
                            text=True,
                            timeout=10,
                        )
                        initial_tree = subprocess.run(
                            ["git", "rev-parse", "--verify", "HEAD^{tree}"],
                            cwd=temporary,
                            capture_output=True,
                            check=False,
                            text=True,
                            timeout=10,
                        )
                        initial_status = subprocess.run(
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
                        pre_identities = self._callback_validation_output_identities(
                            temporary,
                            source.get("entries"),
                        )
                        source_entries = source.get("entries")
                        expected_identities = [
                            {
                                "path": str(entry.get("path") or ""),
                                "index_mode": str(entry.get("mode") or ""),
                                "index_object_id": str(entry.get("object_id") or ""),
                                "worktree_mode": str(entry.get("mode") or ""),
                                "worktree_object_id": str(entry.get("object_id") or ""),
                            }
                            for entry in (
                                source_entries
                                if isinstance(source_entries, list)
                                else ()
                            )
                            if isinstance(entry, Mapping)
                        ]
                        if (
                            initial_head.returncode != 0
                            or initial_head.stdout.strip() != current_head
                            or initial_tree.returncode != 0
                            or initial_tree.stdout.strip() != current_tree
                            or initial_status.returncode != 0
                            or initial_status.stdout
                            or pre_identities is None
                            or pre_identities != expected_identities
                        ):
                            return result
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
                        or after.returncode != 0
                        or after.stdout.strip() != current_head
                    ):
                        return result
                    workspace_hygiene: dict[str, Any] | None = None
                    settled_generated_identities = (
                        self._callback_validation_output_identities(
                            temporary,
                            source.get("entries"),
                        )
                        if hygiene_eligible
                        else None
                    )
                    if (
                        hygiene_eligible
                        and not status.stdout
                        and settled_generated_identities != pre_identities
                    ):
                        return result
                    if status.stdout:
                        if not hygiene_eligible or pre_identities is None:
                            return result
                        dirty_paths = self._callback_validation_generated_dirty_paths(
                            status.stdout
                        )
                        declared_entries = source.get("entries")
                        generated_identities = settled_generated_identities
                        if (
                            dirty_paths is None
                            or not set(dirty_paths).issubset(
                                {
                                    _VRIF_RELEASE_REPORT_JSON_PATH,
                                    _VRIF_RELEASE_REPORT_MARKDOWN_PATH,
                                }
                            )
                            or not isinstance(declared_entries, list)
                            or generated_identities is None
                            or any(
                                path
                                not in {
                                    str(entry.get("path") or "")
                                    for entry in declared_entries
                                    if isinstance(entry, Mapping)
                                }
                                for path in dirty_paths
                            )
                        ):
                            return result
                        dirty_set = set(dirty_paths)
                        for before_identity, generated_identity in zip(
                            pre_identities,
                            generated_identities,
                        ):
                            path_value = before_identity["path"]
                            if (
                                generated_identity["index_mode"]
                                != before_identity["index_mode"]
                                or generated_identity["index_object_id"]
                                != before_identity["index_object_id"]
                                or generated_identity["worktree_mode"]
                                != before_identity["worktree_mode"]
                                or (
                                    path_value in dirty_set
                                    and generated_identity["worktree_object_id"]
                                    == before_identity["worktree_object_id"]
                                )
                                or (
                                    path_value not in dirty_set
                                    and generated_identity != before_identity
                                )
                            ):
                                return result
                        restored = subprocess.run(
                            [
                                "git",
                                "restore",
                                f"--source={current_head}",
                                "--worktree",
                                "--",
                                *[
                                    f":(top,literal){path_value}"
                                    for path_value in dirty_paths
                                ],
                            ],
                            cwd=temporary,
                            capture_output=True,
                            check=False,
                            timeout=30,
                        )
                        restored_identities = (
                            self._callback_validation_output_identities(
                                temporary,
                                declared_entries,
                            )
                        )
                        final_head = subprocess.run(
                            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
                            cwd=temporary,
                            capture_output=True,
                            check=False,
                            text=True,
                            timeout=10,
                        )
                        final_tree = subprocess.run(
                            ["git", "rev-parse", "--verify", "HEAD^{tree}"],
                            cwd=temporary,
                            capture_output=True,
                            check=False,
                            text=True,
                            timeout=10,
                        )
                        final_status = subprocess.run(
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
                        final_target = subprocess.run(
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
                            restored.returncode != 0
                            or restored_identities != pre_identities
                            or final_head.returncode != 0
                            or final_head.stdout.strip() != current_head
                            or final_tree.returncode != 0
                            or final_tree.stdout.strip() != current_tree
                            or final_status.returncode != 0
                            or final_status.stdout
                            or final_target.returncode != 0
                            or final_target.stdout.strip() != current_head
                        ):
                            return result
                        from ..proof.formal_verification_contracts import (
                            content_identity,
                        )

                        workspace_hygiene = {
                            "schema": (
                                _POST_MERGE_CALLBACK_VALIDATION_WORKSPACE_HYGIENE_SCHEMA
                            ),
                            "target_commit": current_head,
                            "target_tree": current_tree,
                            "declared_entries": [
                                dict(entry) for entry in declared_entries
                            ],
                            "pre_validation_identities": [
                                dict(item) for item in pre_identities
                            ],
                            "generated_identities": [
                                dict(item) for item in generated_identities
                            ],
                            "restored_identities": [
                                dict(item) for item in restored_identities
                            ],
                            "generated_dirty_paths": dirty_paths,
                            "restoration_performed": True,
                            "final_clean": True,
                        }
                        workspace_hygiene["hygiene_id"] = content_identity(
                            workspace_hygiene
                        )
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
                    if workspace_hygiene is not None:
                        result["workspace_hygiene"] = workspace_hygiene
                    return result
                except (OSError, subprocess.SubprocessError):
                    return result
                finally:
                    if added:
                        cleanup = cleanup_workspace(temporary, ephemeral=True)
                        if (
                            not isinstance(cleanup, Mapping)
                            or cleanup.get("cleaned") is not True
                        ):
                            result["passed"] = False

            transaction = run_mutation(
                task_id=task_alias,
                branch=self.merge_target_branch,
                operation="requalify_post_merge_callback_integration",
                callback=validate,
                failure_fields={"passed": False},
                extra={
                    "current_target_commit": current_head,
                    "source_integration_commit": str(
                        source.get("integration_commit") or ""
                    ),
                },
            )
            validations = (
                transaction.get("validation")
                if isinstance(transaction, Mapping)
                else None
            )
            workspace_hygiene = (
                transaction.get("workspace_hygiene")
                if isinstance(transaction, Mapping)
                else None
            )
            if (
                not isinstance(transaction, Mapping)
                or transaction.get("passed") is not True
                or not isinstance(validations, list)
                or len(validations) != 1
                or (
                    workspace_hygiene is not None
                    and (
                        not settled_source
                        or self._verified_callback_validation_workspace_hygiene(
                            workspace_hygiene,
                            source=source,
                        )
                        is None
                    )
                )
            ):
                return None
        finally:
            if callable(close):
                close()
        from ..proof.formal_verification_contracts import content_identity

        qualified = {
            "schema": (
                _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V3_SCHEMA
                if workspace_hygiene is not None
                else _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_V2_SCHEMA
                if settled_source
                else _POST_MERGE_CALLBACK_INTEGRATION_REQUALIFICATION_SCHEMA
            ),
            **dict(source),
            "validation": [dict(item) for item in validations],
        }
        if workspace_hygiene is not None:
            qualified["workspace_hygiene"] = dict(workspace_hygiene)
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

    def recover_landed_completion(self, attempt: Any) -> Mapping[str, Any] | None:
        """Propose a landed candidate for a newer, freshly validated claim.

        This callback is read-only.  It neither rearms nor completes the task;
        the database daemon independently binds any returned receipt to the
        exact failed attempt before asking the Quack owner for a retry CAS.
        """

        if self.repository_root is None:
            return None
        record = self._record_for_attempt(self.task_source, attempt)
        body = dict(getattr(record, "body", {}) or {})
        control_receipt = body.get("completion_receipt")
        record_status = str(getattr(record, "status", "") or "").strip().lower()
        if (
            str(getattr(attempt, "status", "") or "").strip().lower()
            != "failed"
            or record_status != "blocked"
            or not isinstance(control_receipt, Mapping)
            or control_receipt.get("operation")
            != "database_portal_terminal_failure"
            or control_receipt.get("reason") != "portal_provider_failed"
            or control_receipt.get("retryable") is not False
            or control_receipt.get("attempt_id")
            != str(getattr(attempt, "attempt_id", "") or "")
            or control_receipt.get("claim_id")
            != str(getattr(attempt, "claim_id", "") or "")
            or control_receipt.get("lease_id")
            != str(getattr(attempt, "lease_id", "") or "")
            or control_receipt.get("owner_session_id")
            != str(getattr(attempt, "owner_session_id", "") or "")
            or control_receipt.get("attempt_number")
            != int(getattr(attempt, "attempt_number", 0) or 0)
            or control_receipt.get("fencing_token")
            != int(getattr(attempt, "fencing_token", 0) or 0)
            or control_receipt.get("fence_epoch")
            != int(getattr(attempt, "fence_epoch", 0) or 0)
            or control_receipt.get("execution_revision")
            != int(getattr(attempt, "revision", 0) or 0)
            or control_receipt.get("execution_finished_at_ms")
            != getattr(attempt, "finished_at_ms", None)
        ):
            return None
        repository = self._validation_repository_scope(body)
        outputs = self._scope_outputs(_output_values(record, body), repository)
        validations = self._scope_validations(
            _validation_values(record, body),
            repository,
        )
        revision = getattr(record, "revision", 0)
        if not outputs or not validations or type(revision) is not int or revision < 1:
            return None
        return discover_landed_completion_recovery(
            repo_root=self.repository_root,
            target_ref=self.merge_target_ref,
            task_cid=str(getattr(attempt, "task_cid", "") or ""),
            task_alias=str(
                getattr(record, "task_alias", "")
                or getattr(attempt, "task_alias", "")
                or ""
            ),
            declared_outputs=outputs,
            source_attempt_id=str(getattr(attempt, "attempt_id", "") or ""),
            source_claim_id=str(getattr(attempt, "claim_id", "") or ""),
            source_lease_id=str(getattr(attempt, "lease_id", "") or ""),
            source_owner_session_id=str(
                getattr(attempt, "owner_session_id", "") or ""
            ),
            source_attempt_number=int(
                getattr(attempt, "attempt_number", 0) or 0
            ),
            source_fencing_token=int(
                getattr(attempt, "fencing_token", 0) or 0
            ),
            source_fence_epoch=int(getattr(attempt, "fence_epoch", 0) or 0),
            source_execution_revision=int(
                getattr(attempt, "revision", 0) or 0
            ),
            source_execution_finished_at_ms=int(
                getattr(attempt, "finished_at_ms", 0) or 0
            ),
            source_control_revision=int(revision),
        )

    def _landed_completion_claim_seed_from_record(
        self,
        *,
        attempt: Any,
        record: Any,
    ) -> dict[str, Any] | None:
        """Verify and bind a source recovery receipt to this live claim."""

        body = dict(getattr(record, "body", {}) or {})
        status_receipt = body.get("completion_receipt")
        if not isinstance(status_receipt, Mapping):
            return None
        raw_recovery = status_receipt.get("landed_completion_recovery_seed")
        if raw_recovery is None:
            return None
        if (
            status_receipt.get("operation") != "database_claim"
            or status_receipt.get("attempt_id")
            != str(getattr(attempt, "attempt_id", "") or "")
            or status_receipt.get("claim_id")
            != str(getattr(attempt, "claim_id", "") or "")
            or status_receipt.get("owner_session_id")
            != str(getattr(attempt, "owner_session_id", "") or "")
            or status_receipt.get("attempt_number")
            != int(getattr(attempt, "attempt_number", 0) or 0)
            or status_receipt.get("fencing_token")
            != int(getattr(attempt, "fencing_token", 0) or 0)
            or status_receipt.get("fence_epoch")
            != int(getattr(attempt, "fence_epoch", 0) or 0)
            or status_receipt.get("lease_id")
            != str(getattr(attempt, "lease_id", "") or "")
            or not isinstance(raw_recovery, Mapping)
        ):
            raise DatabasePortalBridgeError(
                "database claim carries a malformed landed recovery seed"
            )
        task_cid = str(getattr(attempt, "task_cid", "") or "")
        task_alias = str(
            getattr(record, "task_alias", "")
            or getattr(attempt, "task_alias", "")
            or ""
        )
        try:
            recovery = verify_landed_completion_recovery_receipt(
                raw_recovery,
                task_cid=task_cid,
                task_alias=task_alias,
            )
            if self.repository_root is None:
                raise LandedCompletionRecoveryError(
                    "landed recovery has no repository authority"
                )
            repository_evidence = revalidate_landed_completion_repository(
                recovery,
                repo_root=self.repository_root,
                target_ref=self.merge_target_ref,
            )
            scoped_outputs = self._scope_outputs(
                _output_values(record, body),
                self._validation_repository_scope(body),
            )
            if recovery.get("declared_outputs") != scoped_outputs:
                raise LandedCompletionRecoveryError(
                    "landed recovery outputs changed before the target claim"
                )
            return build_landed_completion_claim_seed(
                recovery,
                target_task_cid=task_cid,
                target_task_alias=task_alias,
                target_attempt_id=str(getattr(attempt, "attempt_id", "") or ""),
                target_claim_id=str(getattr(attempt, "claim_id", "") or ""),
                target_owner_session_id=str(
                    getattr(attempt, "owner_session_id", "") or ""
                ),
                target_attempt_number=int(
                    getattr(attempt, "attempt_number", 0) or 0
                ),
                target_fencing_token=int(
                    getattr(attempt, "fencing_token", 0) or 0
                ),
                target_fence_epoch=int(
                    getattr(attempt, "fence_epoch", 0) or 0
                ),
                target_lease_id=str(getattr(attempt, "lease_id", "") or ""),
                validated_target_commit=repository_evidence[
                    "current_target_commit"
                ],
                validated_target_tree=repository_evidence["current_target_tree"],
            )
        except (LandedCompletionRecoveryError, OSError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "database claim landed recovery seed failed verification"
            ) from exc

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
        landed_recovery_seed = self._landed_completion_claim_seed_from_record(
            attempt=attempt,
            record=record,
        )
        payload = {
            "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": canonical_task_cid,
            "canonical_task_key": canonical_task_key,
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
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
            "landed_completion_recovery_seed_id": str(
                (landed_recovery_seed or {}).get("seed_id") or ""
            ),
        }
        payload["binding_id"] = _sha256_bytes(_canonical_json(payload))
        return payload

    def _verify_landed_completion_target_stable(
        self,
        attempt: Any,
        binding: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Refuse acceptance if the target moved after claim projection."""

        expected_seed_id = str(
            binding.get("landed_completion_recovery_seed_id") or ""
        )
        if not expected_seed_id:
            return None
        record = self._record_for_attempt(self.task_source, attempt)
        current = self._landed_completion_claim_seed_from_record(
            attempt=attempt,
            record=record,
        )
        if current is None or current.get("seed_id") != expected_seed_id:
            raise DatabasePortalBridgeError(
                "landed completion target changed during fresh validation"
            )
        return dict(current)

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
        landed_recovery_seed = self._landed_completion_claim_seed_from_record(
            attempt=attempt,
            record=record,
        )
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
            "task key",
            "task cid",
            "canonical task key",
            "canonical task cid",
            "canonical_task_key",
            "canonical_task_cid",
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
            f"- Canonical Task Key: {canonical_task_key}",
            f"- Canonical Task CID: {canonical_task_cid}",
            f"- Database task CID: {_line_value(attempt.task_cid)}",
            f"- Database attempt ID: {_line_value(attempt.attempt_id)}",
            f"- Database claim ID: {_line_value(attempt.claim_id)}",
            f"- Database attempt number: {_line_value(attempt.attempt_number)}",
            f"- Database owner session ID: {_line_value(attempt.owner_session_id)}",
            f"- Database fencing token: {_line_value(attempt.fencing_token)}",
            f"- Database fence epoch: {_line_value(attempt.fence_epoch)}",
            f"- Database lease ID: {_line_value(getattr(attempt, 'lease_id', '') or '')}",
            f"- Database dependency CIDs: {_line_value(getattr(record, 'dependencies', ()))}",
            f"- Canonical task key: {canonical_task_key}",
            f"- Canonical task CID: {canonical_task_cid}",
            # The canonical database revision owns only its typed output/effect
            # paths.  Validation imports are readable context, never an
            # implicit mutation grant for this private Portal projection.
            "- Scope expansion policy: exact",
            "- Projection authority: false",
        ]
        if landed_recovery_seed is not None:
            lines.append(
                "- Landed completion recovery: "
                + _line_value(landed_recovery_seed)
            )
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

    @classmethod
    def _has_completion_event(
        cls,
        paths: DatabasePortalAttemptPaths,
        alias: str,
        canonical_task_key: str,
        canonical_task_cid: str,
    ) -> bool:
        if not paths.events.is_file():
            return False
        for event in reversed(cls._verified_event_chain(paths)):
            if (
                event.get("type") == "task_completed"
                and str(event.get("task_id") or "") == alias
                and str(event.get("canonical_task_key") or "") == canonical_task_key
                and str(event.get("canonical_task_cid") or "") == canonical_task_cid
            ):
                return True
        return False

    def _portal_completion_event_identity(
        self,
        *,
        paths: DatabasePortalAttemptPaths,
        projection_text: str,
        binding: Mapping[str, Any],
    ) -> tuple[str, str]:
        """Derive the exact non-authoritative identity Portal records.

        A database attempt projection can carry extra path authority.  Portal
        therefore derives a projection-local task identity for its lifecycle
        events instead of treating the projected database CID as authority.
        The bridge may accept that local event only after the immutable
        single-task projection and its database binding have been verified.
        """

        # Import lazily because implementation_daemon imports its runner,
        # which in turn binds this bridge.
        from .implementation_daemon import (
            parse_task_text,
            portal_task_identity,
        )

        alias = str(binding.get("task_alias") or "")
        task_cid = str(binding.get("task_cid") or "")
        task_key = str(binding.get("canonical_task_key") or "")
        if not alias or not task_cid or not task_key:
            raise DatabasePortalBridgeError(
                "Portal completion projection differs from its database binding"
            )
        observed_binding = self._read_binding(paths.binding)
        if observed_binding != binding:
            raise DatabasePortalBridgeError(
                "Portal completion projection differs from its database binding"
            )
        verified_projection = verify_database_portal_attempt_projection(
            paths.task_projection,
            expected_task_alias=alias,
            expected_task_cid=task_cid,
            allowed_root=self.attempt_root,
        )
        if (
            verified_projection.get("binding_id") != binding.get("binding_id")
            or verified_projection.get("canonical_task_key") != task_key
        ):
            raise DatabasePortalBridgeError(
                "Portal completion projection differs from its database binding"
            )
        try:
            projected_tasks = parse_task_text(
                projection_text,
                path=paths.task_projection,
                # The bridge default (``## ``) normalizes to ``## ##`` and is
                # not a valid parser prefix.  This is a private, verified
                # single-task projection, so bind parsing to its exact alias
                # and confirm that alias again below.
                task_header_prefix=f"## {alias}",
            )
        except (TypeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "Portal completion projection identity is malformed"
            ) from exc
        if len(projected_tasks) != 1:
            raise DatabasePortalBridgeError(
                "Portal completion projection is not exactly one task"
            )
        task = projected_tasks[0]
        metadata = task.metadata
        if (
            task.task_id != alias
            or metadata.get("projection authority") != "false"
            or metadata.get("database task cid") != task_cid
            or metadata.get("canonical task cid") != task_cid
            or metadata.get("canonical task key") != task_key
        ):
            raise DatabasePortalBridgeError(
                "Portal completion projection differs from its database binding"
            )
        try:
            identity = portal_task_identity(
                task,
                todo_path=paths.task_projection,
            )
        except (TypeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "Portal completion event identity cannot be derived"
            ) from exc
        if not identity.canonical_task_key or not identity.canonical_task_cid:
            raise DatabasePortalBridgeError(
                "Portal completion event identity is absent"
            )
        return identity.canonical_task_key, identity.canonical_task_cid

    @staticmethod
    def _has_completion_event_candidate(
        paths: DatabasePortalAttemptPaths,
        alias: str,
        canonical_task_cid: str,
    ) -> bool:
        """Detect terminal evidence that must be accepted or rejected in place.

        A completion for the projected alias and task CID that is missing or
        contradicts the canonical task key is not authority.  It is still a
        terminal candidate, however, and must flow through the exact
        acceptance checks instead of allowing a second provider dispatch.
        """

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
                and str(event.get("canonical_task_cid") or "")
                == canonical_task_cid
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

    @staticmethod
    def _landed_no_change_completion_source(
        events: Sequence[Mapping[str, Any]],
        *,
        terminal_index: int,
        terminal: Mapping[str, Any],
        alias: str,
        task_cid: str,
        verified_claim_seed: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Admit one terminal no-change result bound to a live landed claim."""

        validation = terminal.get("validation_result")
        bypass = (
            validation.get("pre_dispatch_no_change")
            if isinstance(validation, Mapping)
            else None
        )
        if not isinstance(bypass, Mapping) or bypass.get("kind") != (
            "database_landed_completion_revalidation"
        ):
            return None
        if verified_claim_seed is None:
            raise DatabasePortalBridgeError(
                "landed no-change completion has no live claim authority"
            )
        try:
            verified_seed = verify_landed_completion_claim_seed(
                verified_claim_seed,
                task_cid=task_cid,
                task_alias=alias,
            )
        except LandedCompletionRecoveryError as exc:
            raise DatabasePortalBridgeError(
                "landed no-change completion claim seed is invalid"
            ) from exc
        if verified_seed != dict(verified_claim_seed):
            raise DatabasePortalBridgeError(
                "landed no-change completion claim seed changed"
            )

        expected_bypass_fields = {
            "kind",
            "claim_seed",
            "attempt",
            "eligible",
            "provider_dispatched",
            "reason",
            "receipt_id",
        }
        bypass_body = dict(bypass)
        bypass_receipt_id = str(bypass_body.pop("receipt_id", "") or "")
        from ..proof.formal_verification_contracts import content_identity

        portal_attempt = bypass.get("attempt")
        if (
            set(bypass) != expected_bypass_fields
            or bypass.get("claim_seed") != verified_seed
            or isinstance(portal_attempt, bool)
            or not isinstance(portal_attempt, int)
            or portal_attempt < 1
            or bypass.get("eligible") is not True
            or bypass.get("provider_dispatched") is not False
            or bypass.get("reason")
            != "declared_validation_proved_existing_contract"
            or bypass_receipt_id != content_identity(bypass_body)
        ):
            raise DatabasePortalBridgeError(
                "landed no-change completion bypass receipt is invalid"
            )

        bypass_events = [
            event
            for event in events[:terminal_index]
            if (
                event.get("type")
                == "implementation_provider_bypassed_already_satisfied"
                and str(event.get("task_id") or "") == alias
                and str(event.get("canonical_task_cid") or "") == task_cid
                and event.get("kind")
                == "database_landed_completion_revalidation"
            )
        ]
        if len(bypass_events) != 1:
            raise DatabasePortalBridgeError(
                "landed no-change completion bypass event is absent or ambiguous"
            )
        bypass_event = bypass_events[0]
        if any(
            bypass_event.get(field) != bypass.get(field)
            for field in expected_bypass_fields
        ):
            raise DatabasePortalBridgeError(
                "landed no-change completion bypass event conflicts with terminal"
            )

        target_commit = str(verified_seed["validated_target_commit"])
        commit_result = terminal.get("commit_result")
        no_change_guard = (
            commit_result.get("no_change_guard")
            if isinstance(commit_result, Mapping)
            else None
        )
        merge_result = terminal.get("merge_result")
        cleanup_result = terminal.get("cleanup_result")
        board_completion = terminal.get("board_completion")
        terminal_event_id = str(terminal.get("event_id") or "")
        if (
            str(terminal.get("task_id") or "") != alias
            or str(terminal.get("canonical_task_cid") or "") != task_cid
            or terminal.get("attempt") != portal_attempt
            or type(terminal.get("returncode")) is not int
            or terminal.get("returncode") != 0
            or terminal.get("provider_dispatched") is not False
            or terminal.get("attempt_consumed") is not True
            or str(terminal.get("baseline_ref") or "") != target_commit
            or terminal.get("implementation_commit") != ""
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", terminal_event_id)
            or validation.get("attempted") is not True
            or validation.get("passed") is not True
            or type(validation.get("returncode")) is not int
            or validation.get("returncode") != 0
            or not isinstance(no_change_guard, Mapping)
            or no_change_guard.get("allowed") is not True
            or no_change_guard.get("reasons") != []
            or str(no_change_guard.get("baseline_ref") or "") != target_commit
            or str(no_change_guard.get("current_head") or "") != target_commit
            or not isinstance(commit_result, Mapping)
            or commit_result.get("committed") is not False
            or commit_result.get("reason") != "no_changes"
            or not isinstance(merge_result, Mapping)
            or dict(merge_result)
            != {"merged": False, "reason": "not_attempted"}
            or not isinstance(cleanup_result, Mapping)
            or cleanup_result.get("cleaned") is not True
            or board_completion
            != {
                "complete": True,
                "pending_merge": False,
                "reason": "validated_no_change_completion",
            }
        ):
            raise DatabasePortalBridgeError(
                "landed no-change completion terminal authority is invalid"
            )
        return {
            "implementation_commit": target_commit,
            "baseline_commit": target_commit,
            "completion_source_event_id": terminal_event_id,
            "completion_source_event_type": "implementation_finished",
            "completion_source_portal_attempt": portal_attempt,
            "_source_event_index": terminal_index,
            "_source_event": terminal,
            "_source_merged": False,
            "_source_queued": False,
            "_source_landed_revalidated": True,
        }

    @classmethod
    def _completion_event_evidence(
        cls,
        paths: DatabasePortalAttemptPaths,
        *,
        alias: str,
        task_cid: str,
        completion_task_cid: str | None = None,
        verified_landed_completion_claim_seed: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        """Bind projected completion to one exact implementation commit.

        Portal's canonical ``task_completed`` event binds the task revision,
        but older producers do not copy the implementation commit into that
        event.  In that case the same verified event chain must contain a
        successful, task-bound implementation/reconciliation event before the
        completion event.  Conflicting commit evidence fails closed.
        """

        events = cls._verified_event_chain(paths)
        completion_cid = completion_task_cid or task_cid
        completions = [
            (index, event)
            for index, event in enumerate(events)
            if (
                event.get("type") == "task_completed"
                and str(event.get("task_id") or "") == alias
                and str(event.get("canonical_task_cid") or "")
                == completion_cid
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
                landed_source = cls._landed_no_change_completion_source(
                    events,
                    terminal_index=event_index,
                    terminal=event,
                    alias=alias,
                    task_cid=task_cid,
                    verified_claim_seed=(
                        verified_landed_completion_claim_seed
                    ),
                )
                if landed_source is not None:
                    sources.append(landed_source)
                    continue
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
                    # Historical terminal confirmations omitted this redundant
                    # key after the exact reconciliation had already sealed it.
                    # Fall back only when the key is absent, never when an
                    # explicit value conflicts.
                    and (
                        terminal_merge.get("target_commit")
                        if "target_commit" in terminal_merge
                        else terminal_merge.get("merge_commit")
                    )
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
                selected_source.get("_source_landed_revalidated") is True
                or selected_source.get("_source_merged") is True
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
        source.pop("_source_landed_revalidated", None)
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
        canonical_task_key: str,
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
                "canonical_task_key": canonical_task_key,
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
            if reason in DATABASE_PORTAL_SKIP_CONTENTION_REASONS:
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

        implementation = result.get("implementation_result")
        if (
            isinstance(implementation, Mapping)
            and implementation.get("deferred") is True
            and str(implementation.get("reason") or "")
            == "external_protected_recovery_owner_active"
        ):
            # A verified-live paired supervisor is a closed pre-provider
            # wait.  Do not collapse it into the outer unchanged
            # ``external_protected_checkout_recovery_required`` shortcut.
            raw_backoff = implementation.get("backoff_seconds")
            if (
                isinstance(raw_backoff, bool)
                or not isinstance(raw_backoff, int)
                or raw_backoff < 0
                or raw_backoff > _MAX_DATABASE_PORTAL_BACKOFF_SECONDS
            ):
                raise DatabasePortalBridgeError(
                    "Portal verified-live recovery deferral returned an "
                    "invalid backoff_seconds value"
                )
            return (
                "external_protected_recovery_owner_active",
                int(raw_backoff),
            )

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
        if not isinstance(implementation, Mapping):
            return None
        if (
            implementation.get("skipped") is True
            and str(implementation.get("reason") or "").strip()
            in DATABASE_PORTAL_SKIP_CONTENTION_REASONS
        ):
            skip_reason = str(implementation.get("reason") or "").strip()
            raw_backoff = implementation.get(
                "backoff_seconds",
                (
                    INFLIGHT_PROCESS_BACKOFF_SECONDS
                    if skip_reason == _INFLIGHT_PROCESS_SKIP_REASON
                    else DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS
                ),
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
            return (skip_reason, int(raw_backoff))
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
    def _same_claim_inflight_identity(
        result: Mapping[str, Any],
        *,
        binding: Mapping[str, Any],
    ) -> tuple[str, int, str] | None:
        """Return the exact claim-private lifecycle identity for a live provider.

        ``inflight_process`` is the only Portal wait result whose payload binds
        the selected task, Portal-local attempt, and worktree together. Other
        contention reasons may describe a sibling owner, so they deliberately
        remain ordinary database deferrals rather than keeping this callback
        alive.
        """

        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return None
        task_id = implementation.get("task_id")
        canonical_task_cid = implementation.get("canonical_task_cid")
        portal_attempt = implementation.get("attempt")
        worktree_path = implementation.get("worktree_path")
        if (
            implementation.get("skipped") is not True
            or implementation.get("reason") != "inflight_process"
            or task_id != binding.get("task_alias")
            or (
                canonical_task_cid not in (None, "")
                and canonical_task_cid != binding.get("task_cid")
            )
            or type(portal_attempt) is not int
            or portal_attempt < 1
            or not isinstance(worktree_path, str)
            or not worktree_path.strip()
            or len(worktree_path.encode("utf-8", errors="surrogatepass"))
            > _MAX_REPOSITORY_PATH_BYTES
        ):
            return None
        return task_id, portal_attempt, worktree_path

    def _same_claim_pending_merge_identity(
        self,
        result: Mapping[str, Any],
        *,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> tuple[str, str, int, str, str, str] | None:
        """Bind one queued candidate to this claim-private Portal projection.

        A successful provider pass can finish by handing its validated
        candidate to the asynchronous merge train. That is neither database
        completion nor a retryable provider result: the same database callback
        must remain alive while Portal reconciles the queue. Admit that wait
        only from the closed, identity-bound pending-merge result emitted by
        ``PortalImplementationDaemon``.
        """

        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return None
        board_completion = implementation.get("board_completion")
        merge_result = implementation.get("merge_result")
        if not isinstance(board_completion, Mapping) or not isinstance(
            merge_result, Mapping
        ):
            return None
        projection_text = self._verify_projection(paths, binding)
        _portal_task_key, portal_task_cid = self._portal_completion_event_identity(
            paths=paths,
            projection_text=projection_text,
            binding=binding,
        )
        task_id = implementation.get("task_id")
        task_cid = implementation.get("task_cid")
        canonical_task_cid = implementation.get("canonical_task_cid")
        portal_attempt = implementation.get("attempt")
        implementation_commit = str(
            implementation.get("implementation_commit") or ""
        )
        request_id = str(merge_result.get("request_id") or "")
        merge_task_cid = merge_result.get("canonical_task_cid")
        merge_commit = str(merge_result.get("implementation_commit") or "")
        if (
            type(implementation.get("returncode")) is not int
            or implementation.get("returncode") != 0
            or implementation.get("attempt_consumed") is not True
            or implementation.get("provider_dispatched") is not True
            or task_id != binding.get("task_alias")
            or task_cid != portal_task_cid
            or canonical_task_cid != portal_task_cid
            or type(portal_attempt) is not int
            or portal_attempt < 1
            or portal_attempt != binding.get("attempt_number")
            or board_completion.get("complete") is not False
            or board_completion.get("pending_merge") is not True
            or board_completion.get("reason")
            != "merge_queued_awaiting_integration"
            or merge_result.get("queued") is not True
            or merge_result.get("merged") is not False
            or merge_result.get("reason") != "merge_queued"
            or merge_task_cid != portal_task_cid
            or merge_commit != implementation_commit
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", implementation_commit)
            is None
            or not request_id
            or len(request_id.encode("utf-8", errors="surrogatepass")) > 512
            or any(character in request_id for character in "\x00\r\n")
        ):
            return None
        return (
            str(task_id),
            portal_task_cid,
            portal_attempt,
            implementation_commit,
            request_id,
            str(paths.task_projection),
        )

    @staticmethod
    def _claims_pending_merge(result: Mapping[str, Any]) -> bool:
        """Detect an attempted pending-merge handoff before admitting it."""

        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return False
        board_completion = implementation.get("board_completion")
        merge_result = implementation.get("merge_result")
        return bool(
            isinstance(board_completion, Mapping)
            and board_completion.get("pending_merge") is True
        ) or bool(
            isinstance(merge_result, Mapping)
            and merge_result.get("queued") is True
        )

    @staticmethod
    def _pending_merge_state_is_current(
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        pending_identity: tuple[str, str, int, str, str, str],
    ) -> bool:
        """Verify Portal still owns the exact candidate awaiting integration."""

        state = DatabasePortalExecutionBridge._read_json_object(
            paths.state,
            noun="Portal pending-merge state",
        )
        alias, portal_task_cid, portal_attempt, commit, _request_id, _path = (
            pending_identity
        )
        statuses = state.get("task_statuses")
        attempts = state.get("implementation_attempts")
        attempts_by_cid = state.get("implementation_attempts_by_cid")
        return bool(
            alias == binding.get("task_alias")
            and portal_attempt == binding.get("attempt_number")
            and isinstance(statuses, Mapping)
            and statuses.get(alias) == "merge-queued"
            and isinstance(attempts, Mapping)
            and attempts.get(alias) == portal_attempt
            and isinstance(attempts_by_cid, Mapping)
            and attempts_by_cid.get(portal_task_cid) == portal_attempt
            and state.get("last_implementation_task_id") == alias
            and state.get("last_implementation_task_cid") == portal_task_cid
            and state.get("last_implementation_commit") == commit
            and type(state.get("last_implementation_returncode")) is int
            and state.get("last_implementation_returncode") == 0
        )

    @staticmethod
    def _continues_verified_quota_fallback(
        result: Mapping[str, Any],
        *,
        attempt: Any,
        binding: Mapping[str, Any],
    ) -> bool:
        """Keep one verified Grok-quota handoff inside the same Portal claim.

        The fallback authority is bound to the attempt-local event chain.  If
        the bridge exported this result as an ordinary database deferral, the
        successor database claim would receive a new private Portal journal
        and could not replay that authority.  Continuing here grants no
        completion or provider effect: the next bounded Portal pass still has
        to independently validate the latch before it can select Codex.
        """

        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return False
        authority = implementation.get("quota_fallback_authority")
        if not isinstance(authority, Mapping):
            return False
        expected_authority_fields = {
            "schema",
            "primary_provider",
            "primary_model",
            "failure_class",
            "evidence_sha256",
            "task_id",
            "canonical_task_cid",
            "attempt",
            "primary_returncode",
            "start_event_id",
            "start_sequence",
            "command_sha256",
            "runner_receipt_id",
            "runner_receipt",
        }
        portal_attempt = implementation.get("attempt")
        start_sequence = authority.get("start_sequence")
        return bool(
            implementation.get("deferred") is True
            and implementation.get("reason") == "provider_capacity_exhausted"
            and implementation.get("attempt_consumed") is False
            and implementation.get("task_prompt_dispatched") is False
            and implementation.get("providers") == ["grok"]
            and implementation.get("failure_class") == "hard_quota_exhausted"
            and implementation.get("hard_quota_exhausted_providers") == ["grok"]
            and type(implementation.get("returncode")) is int
            and implementation.get("returncode") != 0
            and implementation.get("task_id") == binding.get("task_alias")
            and implementation.get("canonical_task_cid")
            == str(getattr(attempt, "task_cid", "") or "")
            and type(portal_attempt) is int
            and portal_attempt > 0
            and set(authority) == expected_authority_fields
            and authority.get("schema")
            == DATABASE_PORTAL_QUOTA_FALLBACK_AUTHORITY_SCHEMA
            and authority.get("primary_provider") == "grok"
            and authority.get("primary_model") == "grok-4.6"
            and authority.get("failure_class") == "hard_quota_exhausted"
            and authority.get("task_id") == implementation.get("task_id")
            and authority.get("task_id") == binding.get("task_alias")
            and authority.get("canonical_task_cid")
            == implementation.get("canonical_task_cid")
            and authority.get("canonical_task_cid")
            == str(getattr(attempt, "task_cid", "") or "")
            and authority.get("attempt") == implementation.get("attempt")
            and authority.get("primary_returncode")
            == implementation.get("returncode")
            and isinstance(authority.get("runner_receipt"), Mapping)
            and type(start_sequence) is int
            and start_sequence > 0
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(authority.get("start_event_id") or ""),
            )
            is not None
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(authority.get("command_sha256") or ""),
            )
            is not None
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(authority.get("evidence_sha256") or ""),
            )
            is not None
            and authority.get("runner_receipt_id")
            == authority["runner_receipt"].get("receipt_id")
            and authority.get("evidence_sha256")
            == authority["runner_receipt"].get("evidence_sha256")
            and authority["runner_receipt"].get("failure_class")
            == "hard_quota_exhausted"
            and authority["runner_receipt"].get("primary_dispatched") is False
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
        return self._validation_retry_seed_from_body(
            attempt=attempt,
            record=record,
            body=body,
        )

    def _validation_retry_seed_from_body(
        self,
        *,
        attempt: Any,
        record: Any,
        body: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Verify one exact claim body carrying a validation retry seed."""

        status_receipt = body.get("completion_receipt")
        if not isinstance(status_receipt, Mapping):
            return None
        seed = status_receipt.get("validation_retry_seed")
        if seed is None:
            return None
        if (
            status_receipt.get("operation")
            not in {"database_claim", "database_attempt_admitted"}
            or status_receipt.get("attempt_id") != str(attempt.attempt_id)
            or status_receipt.get("claim_id") != str(attempt.claim_id)
            or status_receipt.get("owner_session_id")
            != str(getattr(attempt, "owner_session_id", "") or "")
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
            or len(changed_paths) != len(scoped_outputs)
            or set(changed_paths) != set(scoped_outputs)
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

    def verify_validation_retry_successor_recovery(
        self,
        attempt: Any,
        record: Any,
        historical_claim_body: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Prove that a failed seed handoff hit only the old order predicate.

        This is a read-only admission check for the database coordinator.  It
        deliberately replays the historical ``database_claim`` body through
        the same repository-aware seed verifier used before Portal dispatch.
        Recovery is admitted only when the preserved candidate, nested
        repository scope, path safety, and exact output population all still
        verify and the sole legacy mismatch is list order.
        """

        if not isinstance(historical_claim_body, Mapping):
            raise DatabasePortalBridgeError(
                "validation retry successor has no historical claim body"
            )
        current_body = dict(getattr(record, "body", {}) or {})
        claim_body = dict(historical_claim_body)
        current_semantic_body = dict(current_body)
        historical_semantic_body = dict(claim_body)
        current_semantic_body.pop("completion_receipt", None)
        historical_semantic_body.pop("completion_receipt", None)
        if (
            str(getattr(record, "task_cid", "") or "")
            != str(getattr(attempt, "task_cid", "") or "")
            or str(
                getattr(record, "task_alias", "")
                or getattr(attempt, "task_alias", "")
                or ""
            )
            != str(getattr(attempt, "task_alias", "") or "")
            or _canonical_json(current_semantic_body)
            != _canonical_json(historical_semantic_body)
        ):
            raise DatabasePortalBridgeError(
                "validation retry successor task definition changed"
            )

        seed = self._validation_retry_seed_from_body(
            attempt=attempt,
            record=record,
            body=claim_body,
        )
        if seed is None:
            raise DatabasePortalBridgeError(
                "validation retry successor claim has no retry seed"
            )
        repository_scope = self._validation_repository_scope(claim_body)
        scoped_outputs = self._scope_outputs(
            _output_values(record, claim_body),
            repository_scope,
        )
        changed_paths = seed.get("changed_paths")
        if (
            not isinstance(changed_paths, list)
            or not changed_paths
            or len(set(changed_paths)) != len(changed_paths)
            or any(_safe_output_path(path) != path for path in changed_paths)
            or changed_paths == scoped_outputs
            or len(changed_paths) != len(scoped_outputs)
            or set(changed_paths) != set(scoped_outputs)
        ):
            raise DatabasePortalBridgeError(
                "validation retry successor was not caused solely by output order"
            )

        proof = {
            "schema": DATABASE_PORTAL_VALIDATION_RETRY_ORDER_REPAIR_SCHEMA,
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "target_attempt_id": str(attempt.attempt_id),
            "target_claim_id": str(attempt.claim_id),
            "target_lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "target_owner_session_id": str(
                getattr(attempt, "owner_session_id", "") or ""
            ),
            "target_attempt_number": int(attempt.attempt_number),
            "target_fencing_token": int(attempt.fencing_token),
            "target_fence_epoch": int(attempt.fence_epoch),
            "source_attempt_id": str(seed.get("attempt_id") or ""),
            "source_retry_receipt_id": str(seed.get("receipt_id") or ""),
            "historical_claim_body_digest": _sha256_bytes(
                _canonical_json(claim_body)
            ),
            "stable_task_body_digest": _sha256_bytes(
                _canonical_json(current_semantic_body)
            ),
            "repository_scope": repository_scope,
            "scoped_outputs": list(scoped_outputs),
            "changed_paths": list(changed_paths),
            "implementation_commit": str(
                seed.get("implementation_commit") or ""
            ),
            "rescue_branch": str(seed.get("rescue_branch") or ""),
            "preserved_commit_verified": True,
            "ordered_lists_differ": True,
            "exact_output_set_verified": True,
        }
        proof["proof_id"] = _sha256_bytes(_canonical_json(proof))
        return proof


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
                    canonical_task_key=str(
                        binding.get("canonical_task_key") or ""
                    ),
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
                        canonical_task_key=str(
                            binding.get("canonical_task_key") or ""
                        ),
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
                    canonical_task_key=str(
                        binding.get("canonical_task_key") or ""
                    ),
                    baseline_commit=baseline_commit,
                    implementation_commit=preserved_commit,
                )
            if (
                _projection_status(projection) not in _TERMINAL_STATUSES
                or not self._has_completion_event(
                    paths,
                    alias,
                    str(binding.get("canonical_task_key") or ""),
                    task_cid,
                )
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
            seed_fields_match = bool(
                isinstance(current_state, Mapping)
                and all(
                    current_state.get(key) == value
                    for key, value in state_seed.items()
                )
            )
            exact_seed_state = bool(
                seed_fields_match
                and len(existing_events) == seed_event_index + 1
            )
            compatible_progressed_state = bool(
                not seed_fields_match
                and self._validation_retry_seed_state_is_compatible(
                    current_state=current_state,
                    state_seed=state_seed,
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
            if not any(
                (
                    exact_seed_state,
                    progressed_adoptable_state,
                    compatible_progressed_state,
                )
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
        summary_count: int | None = None,
        summaries_digest: str | None = None,
    ) -> dict[str, Any]:
        verified_landed_completion_claim_seed = (
            self._verify_landed_completion_target_stable(attempt, binding)
        )
        alias = str(binding.get("task_alias") or "")
        projection_text = self._verify_projection(paths, binding)
        if _projection_status(projection_text) not in _TERMINAL_STATUSES:
            raise DatabasePortalBridgeDeferred("Portal task projection is not complete")
        completion_task_key, completion_task_cid = (
            self._portal_completion_event_identity(
                paths=paths,
                projection_text=projection_text,
                binding=binding,
            )
        )
        if not self._has_completion_event(
            paths,
            alias,
            completion_task_key,
            completion_task_cid,
        ):
            raise DatabasePortalBridgeError(
                "Portal completion lacks a matching durable task_completed event"
            )
        completion = self._completion_event_evidence(
            paths,
            alias=alias,
            task_cid=str(attempt.task_cid),
            completion_task_cid=completion_task_cid,
            verified_landed_completion_claim_seed=(
                verified_landed_completion_claim_seed
            ),
        )
        if completion is None:
            raise DatabasePortalBridgeError(
                "Portal completion lacks a matching durable task_completed event"
            )
        if (summary_count is None) != (summaries_digest is None):
            raise DatabasePortalBridgeError(
                "Portal pass count and digest must be supplied together"
            )
        if summary_count is None:
            summary_count = len(summaries)
            summaries_hasher = hashlib.sha256()
            for summary in summaries:
                encoded_summary = _canonical_json(dict(summary))
                summaries_hasher.update(
                    len(encoded_summary).to_bytes(8, "big")
                )
                summaries_hasher.update(encoded_summary)
            summaries_digest = "sha256:" + summaries_hasher.hexdigest()
        if (
            isinstance(summary_count, bool)
            or not isinstance(summary_count, int)
            or summary_count < len(summaries)
            or summary_count < 0
            or not isinstance(summaries_digest, str)
            or _SHA256_ID.fullmatch(summaries_digest) is None
        ):
            raise DatabasePortalBridgeError(
                "Portal pass summary authority is malformed"
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
            "portal_completion_event_identity": {
                "canonical_task_key": completion_task_key,
                "canonical_task_cid": completion_task_cid,
            },
            "baseline_commit": baseline_commit,
            "implementation_commit": implementation_commit,
            "completion_event_id": completion_event_id,
            "completion_source_event_id": completion_source_event_id,
            "completion_source_event_type": completion_source_event_type,
            "completion_source_portal_attempt": (
                completion_source_portal_attempt
            ),
            "portal_passes": [dict(item) for item in summaries],
            "portal_pass_count": int(summary_count),
            "portal_passes_truncated": summary_count > len(summaries),
            "portal_passes_digest": summaries_digest,
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
        canonical_task_key: str = "",
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
        if canonical_task_key:
            completion_payload["canonical_task_key"] = canonical_task_key

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
            canonical_task_key=str(
                binding.get("canonical_task_key") or ""
            ),
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

        inflight_deadline = _monotonic_seconds() + self.implementation_timeout
        record = self._record_for_attempt(self.task_source, attempt)
        execution_route_binding = self._execution_route_binding(
            attempt=attempt,
            record=record,
        )
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
            and (
                self._has_completion_event(
                    paths,
                    str(binding.get("task_alias") or ""),
                    str(binding.get("canonical_task_key") or ""),
                    str(binding.get("task_cid") or ""),
                )
                or self._has_completion_event_candidate(
                    paths,
                    str(binding.get("task_alias") or ""),
                    str(binding.get("task_cid") or ""),
                )
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
        summary_count = 0
        summaries_hasher = hashlib.sha256()
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
            quota_fallback_continued = False
            ordinary_passes = 0
            inflight_identity: tuple[str, int, str] | None = None
            pending_merge_identity: tuple[str, str, int, str, str, str] | None = None
            while ordinary_passes < self.max_passes:
                projection = self._verify_projection(paths, binding)
                if _projection_status(projection) in _TERMINAL_STATUSES:
                    completion_task_key, completion_task_cid = (
                        self._portal_completion_event_identity(
                            paths=paths,
                            projection_text=projection,
                            binding=binding,
                        )
                    )
                    alias = str(binding.get("task_alias") or "")
                    if (
                        self._has_completion_event(
                            paths,
                            alias,
                            completion_task_key,
                            completion_task_cid,
                        )
                        or self._has_completion_event_candidate(
                            paths,
                            alias,
                            str(binding.get("task_cid") or ""),
                        )
                    ):
                        return self._acceptance_receipt(
                            attempt=attempt,
                            paths=paths,
                            binding=binding,
                            summaries=summaries,
                            summary_count=summary_count,
                            summaries_digest=(
                                "sha256:" + summaries_hasher.copy().hexdigest()
                            ),
                        )
                # Once Portal has proved that this exact claim-private
                # lifecycle is still running, do not launch another pass at
                # or beyond the callback's wall-clock deadline.  The prior
                # implementation checked only after that extra pass returned,
                # which could overshoot the configured timeout and admit more
                # work after the database callback should have deferred.
                if (
                    (
                        inflight_identity is not None
                        or pending_merge_identity is not None
                    )
                    and _monotonic_seconds() >= inflight_deadline
                ):
                    if pending_merge_identity is not None:
                        # A queued candidate has already consumed and
                        # dispatched this attempt. Do not misreport it as the
                        # pre-dispatch typed deferral contract.
                        raise DatabasePortalBridgeError("pending_merge_timeout")
                    raise DatabasePortalBridgeDeferred(
                        "inflight_process",
                        backoff_seconds=(
                            DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS
                        ),
                    )
                raw_result = daemon.run_once()
                if not isinstance(raw_result, Mapping):
                    raise DatabasePortalBridgeError("Portal daemon returned a non-object result")
                summary = _bounded_portal_result(raw_result)
                encoded_summary = _canonical_json(summary)
                summaries_hasher.update(len(encoded_summary).to_bytes(8, "big"))
                summaries_hasher.update(encoded_summary)
                summary_count += 1
                if len(summaries) == _MAX_DATABASE_PORTAL_PASS_PREVIEW:
                    del summaries[0]
                summaries.append(summary)
                projection = self._verify_projection(paths, binding)
                if _projection_status(projection) in _TERMINAL_STATUSES:
                    completion_task_key, completion_task_cid = (
                        self._portal_completion_event_identity(
                            paths=paths,
                            projection_text=projection,
                            binding=binding,
                        )
                    )
                    if self._has_completion_event(
                        paths,
                        str(binding.get("task_alias") or ""),
                        completion_task_key,
                        completion_task_cid,
                    ):
                        return self._acceptance_receipt(
                            attempt=attempt,
                            paths=paths,
                            binding=binding,
                            summaries=summaries,
                            summary_count=summary_count,
                            summaries_digest=(
                                "sha256:" + summaries_hasher.copy().hexdigest()
                            ),
                        )
                current_inflight_identity = self._same_claim_inflight_identity(
                    raw_result,
                    binding=binding,
                )
                if current_inflight_identity is not None:
                    if pending_merge_identity is not None:
                        raise DatabasePortalBridgeError(
                            "Portal pending-merge lifecycle changed into an "
                            "inflight provider lifecycle"
                        )
                    if inflight_identity is None:
                        inflight_identity = current_inflight_identity
                    elif current_inflight_identity != inflight_identity:
                        raise DatabasePortalBridgeError(
                            "Portal inflight lifecycle identity changed while "
                            "the database claim was waiting"
                        )
                    remaining = inflight_deadline - _monotonic_seconds()
                    if remaining <= 0:
                        raise DatabasePortalBridgeDeferred(
                            "inflight_process",
                            backoff_seconds=(
                                DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS
                            ),
                        )
                    _sleep_seconds(
                        min(DATABASE_PORTAL_INFLIGHT_POLL_SECONDS, remaining)
                    )
                    continue
                claims_pending_merge = self._claims_pending_merge(raw_result)
                current_pending_merge_identity = (
                    self._same_claim_pending_merge_identity(
                        raw_result,
                        paths=paths,
                        binding=binding,
                    )
                    if claims_pending_merge
                    else None
                )
                if claims_pending_merge:
                    if current_pending_merge_identity is None:
                        raise DatabasePortalBridgeError(
                            "Portal pending-merge result does not match the "
                            "database claim"
                        )
                    if pending_merge_identity is None:
                        pending_merge_identity = current_pending_merge_identity
                    elif current_pending_merge_identity != pending_merge_identity:
                        raise DatabasePortalBridgeError(
                            "Portal pending-merge candidate identity changed "
                            "while the database claim was waiting"
                        )
                if pending_merge_identity is None:
                    ordinary_passes += 1
                if (
                    not quota_fallback_continued
                    and self._continues_verified_quota_fallback(
                        raw_result,
                        attempt=attempt,
                        binding=binding,
                    )
                ):
                    quota_fallback_continued = True
                    continue
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
                if pending_merge_identity is not None:
                    if not self._pending_merge_state_is_current(
                        paths,
                        binding,
                        pending_merge_identity,
                    ):
                        raise DatabasePortalBridgeError(
                            "Portal pending-merge state no longer matches the "
                            "queued candidate"
                        )
                    remaining = inflight_deadline - _monotonic_seconds()
                    if remaining <= 0:
                        raise DatabasePortalBridgeError("pending_merge_timeout")
                    _sleep_seconds(
                        min(DATABASE_PORTAL_INFLIGHT_POLL_SECONDS, remaining)
                    )
                    continue
            return self._acceptance_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                summaries=summaries,
                summary_count=summary_count,
                summaries_digest=(
                    "sha256:" + summaries_hasher.copy().hexdigest()
                ),
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
        portal_passes = evidence.get("portal_passes")
        portal_pass_count = evidence.get("portal_pass_count")
        portal_passes_truncated = evidence.get("portal_passes_truncated")
        portal_passes_digest = evidence.get("portal_passes_digest")
        completion_event_identity = evidence.get(
            "portal_completion_event_identity"
        )
        visible_passes_digest = ""
        if isinstance(portal_passes, list):
            visible_passes_hasher = hashlib.sha256()
            for summary in portal_passes:
                if not isinstance(summary, Mapping):
                    break
                encoded_summary = _canonical_json(dict(summary))
                visible_passes_hasher.update(
                    len(encoded_summary).to_bytes(8, "big")
                )
                visible_passes_hasher.update(encoded_summary)
            else:
                visible_passes_digest = (
                    "sha256:" + visible_passes_hasher.hexdigest()
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
                "portal_completion_event_identity",
                "baseline_commit",
                "implementation_commit",
                "completion_event_id",
                "completion_source_event_id",
                "completion_source_event_type",
                "completion_source_portal_attempt",
                "portal_passes",
                "portal_pass_count",
                "portal_passes_truncated",
                "portal_passes_digest",
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
            or not isinstance(completion_event_identity, Mapping)
            or set(completion_event_identity)
            != {"canonical_task_key", "canonical_task_cid"}
            or not str(
                completion_event_identity.get("canonical_task_key") or ""
            )
            or not str(
                completion_event_identity.get("canonical_task_cid") or ""
            )
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
            or not isinstance(portal_passes, list)
            or isinstance(portal_pass_count, bool)
            or not isinstance(portal_pass_count, int)
            or portal_pass_count < len(portal_passes)
            or portal_passes_truncated
            is not (portal_pass_count > len(portal_passes))
            or not isinstance(portal_passes_digest, str)
            or _SHA256_ID.fullmatch(portal_passes_digest) is None
            or not visible_passes_digest
            or (
                portal_passes_truncated is False
                and portal_passes_digest != visible_passes_digest
            )
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
    "DATABASE_PORTAL_ATTEMPT_BINDING_FIELDS",
    "DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA",
    "DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA",
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
    "DATABASE_PORTAL_CANDIDATE_RETRY_REASONS",
    "DATABASE_PORTAL_CHECKOUT_CONTENTION_BACKOFF_SECONDS",
    "DATABASE_PORTAL_CHECKOUT_CONTENTION_REASONS",
    "DATABASE_PORTAL_SKIP_CONTENTION_REASONS",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeConsumedNoProgressError",
    "DatabasePortalBridgeError",
    "DatabasePortalCandidateRetry",
    "DatabasePortalCapacityRetry",
    "DatabasePortalConsumedAttemptTerminal",
    "DatabasePortalExecutionBridge",
    "DatabasePortalProtectedPathPreserved",
    "DatabasePortalValidationRetry",
    "PortalDaemonFactory",
    "database_portal_authoritative_repository_tree_id",
    "database_portal_consumed_no_progress_fingerprint",
    "is_protected_checkout_setup_block",
    "database_portal_task_contract_digest",
    "verify_database_portal_attempt_projection",
)
