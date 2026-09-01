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

import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ...llm_router import resolve_agent_implementation_route_binding
from ..proof.formal_verification_contracts import content_identity
from ..runtime.event_log import EVENT_LOG_MANIFEST_SCHEMA
from ..runtime.provider_failure_policy import (
    GROK_FAILURE_RECEIPT_PREFIX,
    GROK_ROUTE_OUTCOME_PREFIX,
    extract_grok_failure_receipts,
    extract_grok_route_outcomes,
    valid_grok_failure_receipt,
    valid_grok_route_outcome,
)
from ..task_sources.intent_repository import (
    VALIDATION_ARGV_REPRESENTATION,
    VALIDATION_REPRESENTATION_POLICY_KEY,
    VALIDATION_SHELL_TEXT_REPRESENTATION,
)

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
DATABASE_PORTAL_ATTEMPT_RECONCILIATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-attempt-reconciliation@1"
)
DATABASE_PORTAL_NO_PROVIDER_REARM_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-no-provider-rearm-evidence@1"
)
DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-deferred-provider-rearm-evidence@1"
)
DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-terminal-no-effect-route-rearm-evidence@1"
)
DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_REASON: Final[str] = (
    "authenticated_Grok_4.5_primary_is_unavailable;_"
    "Codex_requires_typed_hard-quota_exhaustion_authority"
)
DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_BACKOFF_SECONDS: Final[int] = 300
DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-interrupted-implementation-rearm-evidence@1"
)
DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-terminal-quiescent-deferred-rearm-evidence@1"
)
DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_AUTHORIZATION_SCHEMA: Final[
    str
] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-interrupted-implementation-rearm-authorization@1"
)
DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-stale-dispatch-migration-rearm-evidence@1"
)
DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_AUTHORIZATION_SCHEMA: Final[
    str
] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-stale-dispatch-migration-rearm-authorization@1"
)
_STALE_DISPATCH_MIGRATION_REPLAY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "reconciled",
        "blocked",
        "reason",
        "task_id",
        "canonical_task_cid",
        "attempt",
        "migration_id",
        "preparation_event_id",
        "state_recovery_event_id",
        "migration_terminal_event_id",
        "migration_receipt_id",
        "legacy_claim_release_receipt_id",
        "legacy_claim_release_event_id",
        "pre_state_digest",
        "post_state_digest",
        "provider_dispatched",
        "implementation_dispatched",
        "acceptance_inferred",
        "retained_candidate_disposition",
        "stale_lock_cleared",
        "stale_lock_clear_event_id",
    }
)
DATABASE_PORTAL_TERMINAL_RECONCILIATION_LINK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-terminal-reconciliation-link@1"
)
DATABASE_PORTAL_TERMINAL_RECONCILIATION_LINK_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema",
            "attempt_id",
            "claim_id",
            "task_cid",
            "attempt_number",
            "owner_session_id",
            "lease_id",
            "fencing_token",
            "fence_epoch",
            "binding_id",
            "nested_state_digest",
            "nested_reason",
            "nested_reconciled",
            "trigger",
            "intended_database_disposition",
            "prepared_reconciliation_receipt_id",
            "commit_barrier_receipt_id",
            "evidence_id",
        }
    )
)
DATABASE_PORTAL_NO_PROVIDER_REARM_EVIDENCE_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema",
            "attempt_id",
            "claim_id",
            "task_cid",
            "task_alias",
            "attempt_number",
            "owner_session_id",
            "lease_id",
            "fencing_token",
            "fence_epoch",
            "attempt_root_key",
            "attempt_authority_root_digest",
            "attempt_root_digest",
            "binding_id",
            "binding_admission_id",
            "binding_admission_digest",
            "projection_immutable_digest",
            "nested_task_cid",
            "nested_attempt",
            "event_stream_id",
            "event_snapshot_id",
            "event_manifest_digest",
            "event_count",
            "event_head_sequence",
            "event_head_id",
            "task_selected_event_id",
            "setup_event_count",
            "setup_event_ids_digest",
            "cleanup_event_id",
            "exception_event_id",
            "finished_event_id",
            "state_digest",
            "outer_block_receipt_digest",
            "provider_dispatched",
            "validation_attempted",
            "commit_created",
            "merge_attempted",
            "cleanup_terminal",
            "evidence_id",
        }
    )
)
DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        "schema",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "attempt_number",
        "owner_session_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        "attempt_root_key",
        "attempt_authority_root_digest",
        "attempt_root_digest",
        "binding_id",
        "binding_admission_id",
        "binding_admission_digest",
        "projection_immutable_digest",
        "nested_task_cid",
        "nested_attempt",
        "event_stream_id",
        "event_snapshot_id",
        "event_manifest_digest",
        "event_count",
        "event_head_sequence",
        "event_head_id",
        "task_selected_event_id",
        "retry_deferred_event_id",
        "daemon_pass_event_id",
        "diagnostic_event_count",
        "diagnostic_event_ids_digest",
        "deferred_reason",
        "deferred_backoff_seconds",
        "diagnostic_receipt_id",
        "state_digest",
        "outer_block_receipt_digest",
        "provider_dispatched",
        "attempt_consumed",
        "validation_attempted",
        "commit_created",
        "merge_attempted",
        "acceptance_inferred",
        "route_deferred",
        "nested_state_quiescent",
        "evidence_id",
    }
)
DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        "schema",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "attempt_number",
        "owner_session_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        "attempt_root_key",
        "attempt_authority_root_digest",
        "attempt_root_digest",
        "binding_id",
        "binding_admission_id",
        "binding_admission_digest",
        "projection_immutable_digest",
        "task_revision",
        "board_namespace",
        "nested_task_cid",
        "nested_attempt",
        "event_stream_id",
        "event_snapshot_id",
        "event_manifest_digest",
        "event_count",
        "event_head_sequence",
        "event_head_id",
        "prelude_event_count",
        "prelude_event_ids_digest",
        "task_selected_event_id",
        "diagnostic_event_count",
        "diagnostic_event_ids_digest",
        "protected_snapshot_recorded_event_id",
        "implementation_started_event_id",
        "pre_implementation_event_id",
        "pre_implementation_receipt_cid",
        "protected_snapshot_cleared_event_id",
        "worktree_release_event_id",
        "implementation_finished_event_id",
        "daemon_pass_event_id",
        "state_digest",
        "outer_block_receipt_digest",
        "command_sha256",
        "route_plan_sha256",
        "route_id",
        "primary_provider",
        "primary_model",
        "fallback_provider",
        "fallback_model",
        "fallback_reasoning_effort",
        "log_relative_path",
        "log_sha256",
        "log_size",
        "log_identity_digest",
        "quota_probe_receipt_id",
        "quota_probe_receipt_digest",
        "route_outcome_id",
        "route_outcome_digest",
        "failure_class",
        "verifier_status",
        "runner_returncode",
        "provider_dispatched",
        "wrapper_process_dispatched",
        "quota_probe_dispatched",
        "primary_model_dispatched",
        "fallback_model_dispatched",
        "implementation_dispatched",
        "provider_effect_committed",
        "implementation_effect_committed",
        "legacy_nested_attempt_consumed",
        "rearm_attempt_consumed",
        "attempt_consumed",
        "validation_attempted",
        "commit_created",
        "merge_attempted",
        "acceptance_inferred",
        "protected_snapshot_unchanged",
        "workspace_unchanged",
        "cleanup_terminal",
        "route_denied",
        "historical_receipt_only",
        "fresh_fallback_authority",
        "nested_state_quiescent",
        "evidence_id",
    }
)
DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        "schema",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "attempt_number",
        "owner_session_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        "attempt_root_key",
        "attempt_authority_root_digest",
        "attempt_root_digest",
        "binding_id",
        "binding_admission_id",
        "binding_admission_digest",
        "projection_immutable_digest",
        "nested_task_cid",
        "nested_attempt",
        "terminal_reconciliation_evidence_id",
        "first_clear_receipt_id",
        "interrupted_retry_evidence_id",
        "interrupted_retry_id",
        "state_recovery_event_id",
        "claim_release_receipt_id",
        "prepared_reconciliation_receipt_id",
        "commit_barrier_receipt_id",
        "state_digest",
        "outer_block_receipt_digest",
        "rearm_authorization_id",
        "provider_dispatched",
        "implementation_dispatched",
        "validation_attempted",
        "commit_created",
        "merge_attempted",
        "acceptance_inferred",
        "recovery_terminal",
        "retained_candidate_disposition",
        "evidence_id",
    }
)
DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        "schema",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "attempt_number",
        "owner_session_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        "attempt_root_key",
        "attempt_authority_root_digest",
        "attempt_root_digest",
        "attempt_directory_names_digest",
        "binding_id",
        "binding_admission_id",
        "binding_admission_digest",
        "projection_immutable_digest",
        "task_revision",
        "board_namespace",
        "nested_task_cid",
        "nested_attempt",
        "terminal_reconciliation_evidence_id",
        "prepared_reconciliation_receipt_id",
        "commit_barrier_receipt_id",
        "event_stream_id",
        "event_snapshot_id",
        "event_manifest_digest",
        "event_count",
        "event_head_sequence",
        "event_head_id",
        "resource_deferred_daemon_pass_event_id",
        "diagnostic_event_count",
        "diagnostic_event_ids_digest",
        "shutdown_reconciliation_event_count",
        "shutdown_reconciliation_event_ids_digest",
        "state_digest",
        "outer_block_receipt_digest",
        "selection_idle_reason",
        "nested_reconciliation_reason",
        "provider_runner_fence_reason",
        "provider_dispatched",
        "implementation_dispatched",
        "task_never_selected",
        "attempt_consumed",
        "validation_attempted",
        "commit_created",
        "merge_attempted",
        "acceptance_inferred",
        "route_deferred",
        "nested_state_quiescent",
        "evidence_id",
    }
)
DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_FIELDS: Final[
    frozenset[str]
] = frozenset(
    {
        "schema",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "attempt_number",
        "owner_session_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        "attempt_root_key",
        "attempt_authority_root_digest",
        "attempt_root_digest",
        "binding_id",
        "binding_admission_id",
        "binding_admission_digest",
        "projection_immutable_digest",
        "nested_task_cid",
        "nested_attempt",
        "terminal_reconciliation_evidence_id",
        "first_clear_receipt_id",
        "migration_retry_evidence_id",
        "migration_id",
        "migration_preparation_event_id",
        "state_recovery_event_id",
        "migration_terminal_event_id",
        "migration_receipt_id",
        "legacy_claim_release_receipt_id",
        "legacy_claim_release_event_id",
        "stale_lock_cleared",
        "stale_lock_clear_event_id",
        "prepared_reconciliation_receipt_id",
        "commit_barrier_receipt_id",
        "pre_state_digest",
        "state_digest",
        "outer_block_receipt_digest",
        "rearm_authorization_id",
        "provider_dispatched",
        "implementation_dispatched",
        "validation_attempted",
        "commit_created",
        "merge_attempted",
        "acceptance_inferred",
        "recovery_terminal",
        "retained_candidate_disposition",
        "evidence_id",
    }
)
_EVENT_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "generation",
        "updated_at",
        "active_path",
        "stream_id",
        "snapshot_id",
        "earliest_sequence",
        "latest_sequence",
        "last_event_id",
        "active_indexed_bytes",
        "files",
        "manifest_digest",
    }
)
_EVENT_MANIFEST_FILE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "path",
        "size_bytes",
        "event_count",
        "sha256",
        "first_sequence",
        "last_sequence",
        "start_previous_event_id",
        "offset_index",
        "canonical_events",
        "device",
        "inode",
        "mtime_ns",
    }
)
_EVENT_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "type",
        "timestamp",
        "stream_id",
        "snapshot_id",
        "sequence",
        "previous_event_id",
        "event_id",
    }
)
_SUBMODULE_CLEANUP_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "path",
        "branch",
        "removed_worktree",
        "deleted_branch",
        "cleaned",
        "errors",
        "nested_submodule_cleanup",
        "independent_checkout",
    }
)
_NO_PROVIDER_EVENT_FIELDS: Final[dict[str, frozenset[str]]] = {
    "dirty_submodule_reset_deferred": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "attempted",
            "dirty_count",
            "reset",
            "generated_artifact_preservation",
        }
    ),
    "task_selected": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "title",
            "track",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    "local_submodule_source_discovered": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "repo_root",
            "source_root",
            "source_key",
            "source",
            "expected_ref",
        }
    ),
    "implementation_protected_path_snapshot_recorded": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "attempt",
            "workspace_path",
            "protected_paths",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    "implementation_started": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "attempt",
            "outputs",
            "command",
            "log_path",
            "worktree_path",
            "branch",
            "baseline_ref",
            "workspace_setup",
            "cache_hit",
            "setup_duration_seconds",
            "saved_duration_seconds",
            "checkpoint_directory",
            "timeout_policy",
            "execution_mode",
            "provider_dispatched",
            "worktree_lifecycle",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    # This is the exact predecessor event.  In particular, the later
    # ``effective_*`` audit fields are deliberately not accepted here.
    "pre_implementation_kernel_evaluated": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "event",
            "task_id",
            "attempt",
            "disposition",
            "provider_authorized",
            "provider_hook_count",
            "skip_provider",
            "reason_code",
            "receipt_cid",
            "residual_packet_cid",
            "analytical_candidate_count",
            "kernel_receipt",
            "interface",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    "implementation_protected_path_snapshot_cleared": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "attempt",
            "reason",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    "worktree_pool_lease_released": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "attempted",
            "handoff_reason",
            "worktree_path",
            "released",
            "reason",
            "pooled",
            "entry_id",
            "cache_key",
            "base_ref",
            "base_commit",
            "branch",
            "dependency_paths",
            "reused",
            "cache_hit",
            "setup_seconds",
            "estimated_seconds_saved",
            "setup_time_saved_seconds",
            "invalidation_reason",
            "invalidation_reasons",
            "lifecycle_finalize",
        }
    ),
    "worktree_cleanup_fenced": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "worktree_path",
            "branch",
            "allowed",
            "disposition",
            "reason",
            "failure_kind",
            "attempt_consumed",
            "provider_call_allowed",
            "record",
        }
    ),
    "cleanup_finished": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "started_at",
            "finished_at",
            "worktree_path",
            "branch",
            "removed_worktree",
            "deleted_branch",
            "cleaned",
            "submodule_cleanup",
            "lifecycle_finalize",
        }
    ),
    "stale_worktree_cleanup": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "attempted",
            "max_age_seconds",
            "removed_count",
            "skipped_count",
            "removed",
            "skipped",
        }
    ),
    "implementation_shutdown_reconciled": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "reconciled",
            "blocked",
            "reason",
            "reconciled_at",
            "task_id",
            "attempt",
            "attempt_recovery",
            "task_claim_reconciliation",
            "protected_path_reconciliation",
            "worktree_lifecycle_reconciliation",
            "stale_lock_cleared",
        }
    ),
    "failed_setup_worktree_cleanup": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "attempt",
            "worktree_path",
            "requested_worktree_path",
            "branch",
            "cleanup_result",
            "exception_result",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    "implementation_exception": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "attempt",
            "phase",
            "exception_type",
            "message",
            "worktree_path",
            "branch",
            "cleanup_result",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    "implementation_finished": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "task_cid",
            "attempt",
            "attempt_consumed",
            "returncode",
            "implementation_commit",
            "worktree_path",
            "branch",
            "log_path",
            "baseline_ref",
            "provider_dispatched",
            "validation_result",
            "commit_result",
            "merge_result",
            "board_completion",
            "cleanup_result",
            "lifecycle_finalize",
            "exception_result",
            "failed_preservation_result",
            "workspace_setup",
            "cache_hit",
            "setup_duration_seconds",
            "saved_duration_seconds",
            "diagnostic_receipt_id",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    "implementation_resource_claim_lock_cleared": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "lock_path",
            "branch",
            "lock_owner_pid",
        }
    ),
    "implementation_retry_deferred": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "task_id",
            "attempt",
            "skipped",
            "reason",
            "backoff_seconds",
            "attempt_consumed",
            "provider_dispatched",
            "diagnostic_receipt_id",
            "active_task_cleared",
            "canonical_task_key",
            "canonical_task_cid",
            "board_namespace",
        }
    ),
    "daemon_pass": _EVENT_ENVELOPE_FIELDS
    | frozenset(
        {
            "completed_count",
            "ready_count",
            "selectable_ready_count",
            "eligible_ready_count",
            "strict_deprioritized_ready_count",
            "waiting_count",
            "blocked_count",
            "active_task_id",
            "selection_idle_reason",
            "max_task_attempts",
            "retry_budget_reset_task_ids",
            "retry_budget_rearmed_task_ids",
            "retry_budget_reset_deferred_task_ids",
            "released_retry_budget_strategy_block_task_ids",
            "attempt_limited_task_ids",
            "ordinary_provider_dispatch_allowed",
            "protected_path_conflicts",
            "projection_delta_keys",
            "shared_completed_task_ids",
            "shared_active_merge_task_ids",
            "quarantined_manual_completion_status_task_ids",
            "manual_completion_authority_task_ids",
            "manual_completion_authority_required_task_ids",
            "manual_completion_authority_revalidation_only",
            "manual_completion_revalidation_task_ids",
            "manual_completion_revalidation_only_task_ids",
            "manual_completion_authority_dependency_task_ids",
            "manual_completion_authority_affected_goal_ids",
            "manual_completion_renewal_quarantined_task_ids",
            "completion_receipt_task_ids",
            "execution_slice_task_statuses",
            "execution_slice_task_cids_by_id",
            "virgin_task_transfer",
        }
    ),
}
_SETUP_EVENT_FIELD_VARIANTS: Final[
    dict[str, frozenset[frozenset[str]]]
] = {
    # The protected quota/medium predecessor predates ``exception_result``.
    # Keep its exact historical shape as a separate closed variant instead of
    # weakening either version with an optional authority-bearing field.
    "implementation_finished": frozenset(
        {
            _NO_PROVIDER_EVENT_FIELDS["implementation_finished"]
            - frozenset({"exception_result"}),
        }
    ),
    "pre_implementation_kernel_evaluated": frozenset(
        {
            _NO_PROVIDER_EVENT_FIELDS[
                "pre_implementation_kernel_evaluated"
            ]
            | frozenset(
                {
                    "effective_provider_authorized",
                    "effective_skip_provider",
                    "effective_reason_code",
                }
            ),
        }
    ),
    "implementation_resource_claim_lock_cleared": frozenset(
        {
            _NO_PROVIDER_EVENT_FIELDS[
                "implementation_resource_claim_lock_cleared"
            ]
            | frozenset(
                {
                    "canonical_task_key",
                    "canonical_task_cid",
                    "board_namespace",
                }
            ),
        }
    ),
    "nested_submodule_initialization_guarded": frozenset(
        {
            _EVENT_ENVELOPE_FIELDS
            | frozenset({"reason", "source_key", "fallback_returncode"}),
            _EVENT_ENVELOPE_FIELDS
            | frozenset(
                {
                    "path",
                    "relative",
                    "parent_relative",
                    "reason",
                    "depth",
                    "max_depth",
                    "path_parts",
                    "max_path_parts",
                    "path_bytes",
                    "max_path_bytes",
                    "path_sha256",
                    "expected_gitlink_ref_available",
                    "expected_gitlink_ref_sha256",
                    "matched_identity_sha256",
                }
            ),
        }
    ),
    "submodule_worktree_base_ref_retried": frozenset(
        {
            _EVENT_ENVELOPE_FIELDS
            | frozenset({"reason", "source_key", "fallback_returncode"}),
            _EVENT_ENVELOPE_FIELDS
            | frozenset(
                {
                    "worktree_path",
                    "source",
                    "source_key",
                    "bad_ref",
                    "fallback_ref",
                    "fallback_returncode",
                    "fallback_error",
                }
            ),
        }
    ),
}
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
_ATTEMPT_INTEGER_FIELDS: Final[tuple[str, str, str]] = (
    "attempt_number",
    "fencing_token",
    "fence_epoch",
)


def _strict_attempt_integer_fields(record: Mapping[str, Any]) -> bool:
    """Reject bool/float aliases for JSON attempt authority integers."""

    return bool(
        type(record.get("attempt_number")) is int
        and int(record["attempt_number"]) >= 1
        and type(record.get("fencing_token")) is int
        and int(record["fencing_token"]) >= 0
        and type(record.get("fence_epoch")) is int
        and int(record["fence_epoch"]) >= 0
    )


def _closed_typed_record_matches(
    observed: Any,
    expected: Mapping[str, Any],
) -> bool:
    """Compare a closed JSON object without Python numeric type coercion."""

    return bool(
        isinstance(observed, Mapping)
        and set(observed) == set(expected)
        and all(
            type(observed.get(name)) is type(value)
            and observed.get(name) == value
            for name, value in expected.items()
        )
    )


_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_VALIDATION_REPRESENTATIONS: Final[frozenset[str]] = frozenset(
    {
        VALIDATION_SHELL_TEXT_REPRESENTATION,
        VALIDATION_ARGV_REPRESENTATION,
    }
)
_ATTEMPT_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
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
_ATTEMPT_RECONCILIATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "attempt_number",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "attempt_root",
        "stage",
        "trigger",
        "reconciled_at",
        "reconciled",
        "blocked",
        "reason",
        "binding_id",
        "binding_path",
        "historical_binding",
        "nested_state",
        "provider_runner_fence",
        "provider_runner_reconciliation_authority",
        "portal_reconciliation",
        "terminal_provider_evidence",
        "terminal_provider_receipt_id",
        "error_type",
        "error",
        "intended_database_disposition",
        "database_disposition",
        "database_attempt_status",
        "database_attempt_phase",
        "database_task_status",
        "terminal_reconciliation_evidence_id",
        "retry_receipt",
        "prepared_reconciliation_receipt_id",
        "receipt_id",
    }
)
_PROVIDER_RUNNER_FENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "applicable",
        "safe_to_restart",
        "fenced",
        "reason",
        "pid",
        "pid_reused",
        "parent_pid_before_fence",
        "host_fenced",
        "container_fence",
    }
)
_ORDINARY_GROK_CONTAINER_FENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_id",
        "attempt",
        "task_revision_cid",
        "workspace_path",
        "runner_pid",
        "runner_receipt_id",
        "safe_to_restart",
        "removed",
        "reason",
        "detail",
        "receipt_id",
    }
)
_ORDINARY_GROK_CONTAINER_FENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "ordinary-grok-orphan-container-fence@1"
)


class DatabasePortalBridgeError(RuntimeError):
    """A database claim could not obtain trustworthy Portal evidence."""


class DatabasePortalBridgeDeferred(DatabasePortalBridgeError):
    """Portal execution made bounded progress but is not yet acceptable."""


class DatabasePortalProviderRouteDeferred(DatabasePortalBridgeDeferred):
    """The sealed Portal route deferred before provider dispatch."""

    def __init__(
        self,
        message: str,
        *,
        backoff_seconds: int,
        retry_not_before_ms: int | None = None,
    ) -> None:
        if (
            isinstance(backoff_seconds, bool)
            or not isinstance(backoff_seconds, int)
            or backoff_seconds <= 0
            or backoff_seconds
            > DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_BACKOFF_SECONDS
        ):
            raise DatabasePortalBridgeError(
                "Portal provider deferral backoff is malformed or future-unbounded"
            )
        if retry_not_before_ms is not None and (
            isinstance(retry_not_before_ms, bool)
            or not isinstance(retry_not_before_ms, int)
            or retry_not_before_ms <= 0
        ):
            raise DatabasePortalBridgeError(
                "Portal provider deferral deadline is malformed"
            )
        self.backoff_seconds = backoff_seconds
        self.retry_not_before_ms = retry_not_before_ms
        super().__init__(message)


class DatabasePortalPreEntryPublicationDeferred(DatabasePortalBridgeDeferred):
    """A transient immutable publication fault occurred before Portal entry."""


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
    reconciliation: Path


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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    """Create *path* and durably publish every new directory entry."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise DatabasePortalBridgeError(
                "database Portal attempt directory has no existing authority root"
            )
        cursor = parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise DatabasePortalBridgeError(
            "database Portal attempt directory authority is not a directory"
        )
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if directory.is_symlink() or not directory.is_dir():
            raise DatabasePortalBridgeError(
                "database Portal attempt directory changed during publication"
            )
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    # A previous interrupted publication can leave the directory visible in
    # this process without proving its directory entry durable.  Re-fsync the
    # exact attempt directory and parent on every prepared-stage resume.
    _fsync_directory(path)
    _fsync_directory(path.parent)


@contextmanager
def _immutable_publication_lock(path: Path) -> Any:
    """Serialize bounded cooperating publication/recovery in one directory."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence directory is unavailable"
        ) from exc
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence directory authority changed"
            )
        deadline = time.monotonic() + 5.0
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise DatabasePortalBridgeError(
                        "database Portal immutable evidence publication lock timed out"
                    ) from exc
                time.sleep(0.01)
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _cleanup_immutable_publication_stages_unlocked(path: Path) -> None:
    """Remove only bounded, exact, non-authoritative crashed stage files."""

    exact_stage = re.compile(
        rf"^\.{re.escape(path.name)}\.[A-Za-z0-9_-]+\.stage$"
    )
    try:
        stages = sorted(
            child
            for child in path.parent.iterdir()
            if exact_stage.fullmatch(child.name)
        )
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence stage set is unreadable"
        ) from exc
    if len(stages) > 8:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence has too many stages"
        )
    stage_records: list[tuple[Path, Path, os.stat_result]] = []
    for stage_path in stages:
        try:
            metadata = stage_path.lstat()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence stage is unreadable"
            ) from exc
        if (
            stage_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or int(metadata.st_nlink) not in {1, 2}
            or int(metadata.st_size) > 262_144
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence stage is unsafe"
            )
        ready_path = stage_path.with_name(
            stage_path.name[: -len(".stage")] + ".tmp"
        )
        if int(metadata.st_nlink) == 2:
            try:
                ready_metadata = ready_path.lstat()
            except OSError as exc:
                raise DatabasePortalBridgeError(
                    "database Portal immutable evidence linked stage is incomplete"
                ) from exc
            if (
                ready_path.is_symlink()
                or not stat.S_ISREG(ready_metadata.st_mode)
                or ready_metadata.st_uid != os.geteuid()
                or int(ready_metadata.st_nlink) != 2
                or ready_metadata.st_dev != metadata.st_dev
                or ready_metadata.st_ino != metadata.st_ino
            ):
                raise DatabasePortalBridgeError(
                    "database Portal immutable evidence linked stage is not exact"
                )
        stage_records.append((stage_path, ready_path, metadata))
    removed = False
    for stage_path, _ready_path, _metadata in stage_records:
        try:
            stage_path.unlink()
            removed = True
        except FileNotFoundError:
            # The advisory lock excludes cooperating writers.  A vanished
            # stage has no authority; re-enumeration below still rejects any
            # replacement or unresolved exact stage.
            continue
    if removed:
        _fsync_directory(path.parent)
    try:
        remaining = [
            child
            for child in path.parent.iterdir()
            if exact_stage.fullmatch(child.name)
        ]
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence stage set changed unreadably"
        ) from exc
    if remaining:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence stage cleanup did not converge"
        )


def _atomic_write_if_absent(path: Path, payload: bytes) -> None:
    """Publish immutable evidence without replacing an existing object."""

    _ensure_durable_directory(path.parent)
    with _immutable_publication_lock(path):
        _cleanup_immutable_publication_stages_unlocked(path)
        # An earlier process can have stopped at any durable publication
        # boundary.  Repair only an exact ready temporary while no cooperating
        # writer can still be mutating it.
        _recover_immutable_link_publication_unlocked(
            path,
            expected_payload=payload,
        )
        if path.exists() or path.is_symlink():
            _durably_revalidate_immutable_final_unlocked(
                path,
                expected_payload=payload,
            )
            return

        descriptor, stage_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".stage",
            dir=path.parent,
        )
        stage_path = Path(stage_name)
        ready_path = stage_path.with_name(
            stage_path.name[: -len(".stage")] + ".tmp"
        )
        stage_removed = False
        ready_owned = False
        ready_removed = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(stage_path, ready_path)
            except FileExistsError as exc:
                raise DatabasePortalBridgeError(
                    "database Portal immutable evidence ready name already exists"
                ) from exc
            ready_owned = True
            _fsync_directory(path.parent)
            stage_path.unlink()
            stage_removed = True
            _fsync_directory(path.parent)
            try:
                os.link(ready_path, path)
            except FileExistsError:
                _recover_immutable_link_publication_unlocked(
                    path,
                    expected_payload=payload,
                    require_final=True,
                )
            except FileNotFoundError:
                _recover_immutable_link_publication_unlocked(
                    path,
                    expected_payload=payload,
                    require_final=True,
                )
            try:
                ready_path.unlink()
            except FileNotFoundError:
                pass
            ready_removed = True
            _fsync_directory(path.parent)
            _fsync_directory(path.parent.parent)
            _durably_revalidate_immutable_final_unlocked(
                path,
                expected_payload=payload,
            )
        finally:
            cleanup_changed = False
            if not stage_removed:
                try:
                    stage_path.unlink()
                    cleanup_changed = True
                except FileNotFoundError:
                    pass
            if ready_owned and not ready_removed:
                try:
                    ready_path.unlink()
                    cleanup_changed = True
                except FileNotFoundError:
                    pass
            if cleanup_changed:
                _fsync_directory(path.parent)


def _immutable_receipt_payload_is_exact(path: Path, payload: bytes) -> bool:
    """Return whether *payload* is the canonical self-CID receipt at *path*."""

    if len(payload) > 262_144:
        return False

    def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        receipt = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite value")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(receipt, Mapping):
        return False
    receipt = dict(receipt)
    expected_id = f"sha256:{path.stem}"
    if receipt.get("receipt_id") != expected_id:
        return False
    unsigned = dict(receipt)
    unsigned.pop("receipt_id", None)
    if _sha256_bytes(_canonical_json(unsigned)) != expected_id:
        return False
    expected_storage = (
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return payload == expected_storage


def _recover_immutable_link_publication_unlocked(
    path: Path,
    *,
    expected_payload: bytes | None = None,
    require_final: bool = False,
) -> None:
    """Finish one bounded, exact immutable-publication crash prefix."""

    exact_temp = re.compile(
        rf"^\.{re.escape(path.name)}\.[A-Za-z0-9_-]+\.tmp$"
    )
    def publication_candidates() -> list[Path]:
        try:
            selected = sorted(
                child
                for child in path.parent.iterdir()
                if exact_temp.fullmatch(child.name)
            )
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence publication is unreadable"
            ) from exc
        if len(selected) > 8:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence has too many temporaries"
            )
        return selected

    def exact_regular(selected: Path) -> tuple[os.stat_result, bytes]:
        try:
            selected_stat = selected.lstat()
            selected_payload = selected.read_bytes()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence object is unreadable"
            ) from exc
        if (
            selected.is_symlink()
            or not stat.S_ISREG(selected_stat.st_mode)
            or selected_stat.st_uid != os.geteuid()
            or int(selected_stat.st_size) > 262_144
            or (
                expected_payload is not None
                and selected_payload != expected_payload
            )
            or (
                expected_payload is None
                and not _immutable_receipt_payload_is_exact(
                    path,
                    selected_payload,
                )
            )
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence object is not exact"
            )
        return selected_stat, selected_payload

    def settled_final() -> tuple[os.stat_result, bytes] | None:
        """Return only a twice-observed exact final with no temp authority."""

        try:
            selected_stat, selected_payload = exact_regular(path)
        except DatabasePortalBridgeError:
            if not (path.exists() or path.is_symlink()):
                return None
            # The name may have appeared between the failed observation and
            # the existence check.  It still must pass the ordinary strict
            # object validation; malformed visible state never disappears
            # into a benign concurrent-writer classification.
            selected_stat, selected_payload = exact_regular(path)
        if int(selected_stat.st_nlink) != 1 or publication_candidates():
            return None
        observed_stat, observed_payload = exact_regular(path)
        if (
            int(observed_stat.st_nlink) != 1
            or observed_payload != selected_payload
            or observed_stat.st_dev != selected_stat.st_dev
            or observed_stat.st_ino != selected_stat.st_ino
            or publication_candidates()
        ):
            return None
        return observed_stat, observed_payload

    def exact_candidate(selected: Path) -> tuple[os.stat_result, bytes] | None:
        """Validate a listed temp, or accept only a fully settled takeover."""

        try:
            return exact_regular(selected)
        except DatabasePortalBridgeError:
            if not (selected.exists() or selected.is_symlink()):
                if settled_final() is not None:
                    return None
            raise

    candidates = publication_candidates()

    path_visible = path.exists() or path.is_symlink()
    if not path_visible:
        if not candidates:
            if require_final and settled_final() is None:
                raise DatabasePortalBridgeError(
                    "database Portal immutable evidence publication did not "
                    "produce a settled final object"
                )
            return
        candidate_records: list[tuple[Path, os.stat_result, bytes]] = []
        for candidate in candidates:
            candidate_record = exact_candidate(candidate)
            if candidate_record is None:
                # Another exact recovery consumed the listed temporary and
                # left the sole exact final.  Nothing from the stale listing
                # remains authoritative to promote or remove.
                return
            candidate_records.append((candidate, *candidate_record))
        first_payload = candidate_records[0][2]
        if any(
            int(candidate_stat.st_nlink) != 1
            or candidate_payload != first_payload
            for _candidate, candidate_stat, candidate_payload in candidate_records
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence temporaries are ambiguous"
            )
        try:
            try:
                os.link(candidate_records[0][0], path)
            except FileExistsError:
                # An identical concurrent writer won publication after this
                # process observed the final absent.  Continue only through
                # the same strict final/temp revalidation below.
                pass
            except FileNotFoundError:
                # A concurrent exact recovery may already have linked the
                # candidate and removed its temporary name.  That race is
                # admissible only if the strict convergence check below sees
                # one exact final and no unresolved publication temporary.
                pass
            _fsync_directory(path.parent)
            for candidate, _candidate_stat, _candidate_payload in candidate_records:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    # A concurrent cleaner may have won this unlink.  Do not
                    # infer success from ENOENT itself; the final strict
                    # convergence check below is the only success authority.
                    continue
            _fsync_directory(path.parent)
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence promotion failed"
            ) from exc

    final_stat, final_payload = exact_regular(path)
    candidate_records = []
    for candidate in publication_candidates():
        candidate_record = exact_candidate(candidate)
        if candidate_record is None:
            return
        candidate_records.append((candidate, *candidate_record))
    final_links = int(final_stat.st_nlink)
    if final_links == 1:
        if any(
            int(candidate_stat.st_nlink) != 1
            or candidate_payload != final_payload
            for _candidate, candidate_stat, candidate_payload in candidate_records
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence has an unrelated temporary"
            )
    elif final_links == 2:
        linked_candidates = [
            (temporary_stat, temporary_payload)
            for _temporary, temporary_stat, temporary_payload in candidate_records
            if temporary_stat.st_dev == final_stat.st_dev
            and temporary_stat.st_ino == final_stat.st_ino
        ]
        if len(linked_candidates) != 1 or any(
            temporary_payload != final_payload
            or (
                int(temporary_stat.st_nlink) != 2
                if (
                    temporary_stat.st_dev == final_stat.st_dev
                    and temporary_stat.st_ino == final_stat.st_ino
                )
                else int(temporary_stat.st_nlink) != 1
            )
            for _temporary, temporary_stat, temporary_payload in candidate_records
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence temporary is not exact"
            )
    else:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence link count is ambiguous"
        )

    if candidate_records:
        try:
            for temporary, _temporary_stat, _temporary_payload in candidate_records:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    # As above, concurrent disappearance is acceptable only
                    # when the final/no-temp state converges exactly below.
                    continue
            _fsync_directory(path.parent)
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence temporary recovery failed"
            ) from exc
    settled = settled_final()
    if settled is None:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence recovery did not converge"
        )
    repaired_stat, repaired_payload = settled
    if (
        int(repaired_stat.st_nlink) != 1
        or repaired_payload != final_payload
        or repaired_stat.st_dev != final_stat.st_dev
        or repaired_stat.st_ino != final_stat.st_ino
    ):
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence recovery changed authority"
        )


def _durably_revalidate_immutable_final_unlocked(
    path: Path,
    *,
    expected_payload: bytes | None,
) -> None:
    """Persist and reobserve the sole exact final before authority advances."""

    _recover_immutable_link_publication_unlocked(
        path,
        expected_payload=expected_payload,
        require_final=True,
    )
    _fsync_directory(path.parent)
    _fsync_directory(path.parent.parent)
    _recover_immutable_link_publication_unlocked(
        path,
        expected_payload=expected_payload,
        require_final=True,
    )


def _recover_immutable_link_publication(
    path: Path,
    *,
    expected_payload: bytes | None = None,
    require_final: bool = False,
) -> None:
    """Recover an exact immutable publication under its crash-released lock."""

    with _immutable_publication_lock(path):
        _cleanup_immutable_publication_stages_unlocked(path)
        _recover_immutable_link_publication_unlocked(
            path,
            expected_payload=expected_payload,
            require_final=require_final,
        )
        if path.exists() or path.is_symlink():
            _durably_revalidate_immutable_final_unlocked(
                path,
                expected_payload=expected_payload,
            )


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
    return _line_value(
        value.get("path")
        or value.get("output")
        or value.get("artifact_id")
        or value.get("fluent_id")
        or value
    )


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
            raw_policy = item.get("policy")
            if raw_policy is None:
                policy: Mapping[str, Any] = {}
            elif isinstance(raw_policy, Mapping):
                policy = raw_policy
            else:
                raise DatabasePortalBridgeError(
                    "database validation policy is malformed"
                )
            representation = str(
                policy.get(VALIDATION_REPRESENTATION_POLICY_KEY)
                or item.get(VALIDATION_REPRESENTATION_POLICY_KEY)
                or ""
            ).strip()
            if representation and representation not in _VALIDATION_REPRESENTATIONS:
                raise DatabasePortalBridgeError(
                    "database validation representation is unknown"
                )
            if isinstance(argv, Sequence) and not isinstance(
                argv, (str, bytes, bytearray, memoryview)
            ):
                parts = tuple(argv)
                if not parts or any(
                    not isinstance(part, str) or not part.strip()
                    for part in parts
                ):
                    raise DatabasePortalBridgeError(
                        "database validation argv is malformed"
                    )
                if representation == VALIDATION_SHELL_TEXT_REPRESENTATION:
                    if len(parts) != 1:
                        raise DatabasePortalBridgeError(
                            "shell_text validation must contain exactly one command"
                        )
                    # Shell-text is an already reviewed command program.  It
                    # must survive the database projection byte-for-text;
                    # shlex.join([program]) would quote the whole program as
                    # one missing executable.
                    value = _line_value(parts[0])
                else:
                    # Explicit argv, and legacy untyped rows, retain their
                    # shell-safe projection behavior.
                    value = shlex.join(parts)
            else:
                command = item.get("command") or item.get("value")
                if representation == VALIDATION_ARGV_REPRESENTATION:
                    raise DatabasePortalBridgeError(
                        "argv validation must provide an argv sequence"
                    )
                if (
                    representation == VALIDATION_SHELL_TEXT_REPRESENTATION
                    and not isinstance(
                    command, str
                    )
                ):
                    raise DatabasePortalBridgeError(
                        "shell_text validation must provide command text"
                    )
                value = _line_value(command or item)
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
        task_header_prefix: str = "## ",
        max_passes: int = 4,
        worktree_submodule_paths: Sequence[str] = (),
    ) -> None:
        if not callable(portal_factory):
            raise TypeError("portal_factory must be callable")
        if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        self.task_source = task_source
        self.attempt_root = Path(attempt_root).absolute()
        self.portal_factory = portal_factory
        self.task_header_prefix = str(task_header_prefix or "## ")
        self.max_passes = max_passes
        if isinstance(worktree_submodule_paths, (str, bytes, bytearray)):
            raise TypeError("worktree_submodule_paths must be a sequence of paths")
        declared_submodules = tuple(worktree_submodule_paths)
        if (
            len(set(declared_submodules)) != len(declared_submodules)
            or any(
                not isinstance(path, str)
                or not path
                or path != PurePosixPath(path).as_posix()
                or PurePosixPath(path).is_absolute()
                or any(
                    part in {"", ".", ".."}
                    for part in PurePosixPath(path).parts
                )
                or "\\" in path
                or "\x00" in path
                for path in declared_submodules
            )
        ):
            raise ValueError("worktree_submodule_paths must be canonical and unique")
        self.worktree_submodule_paths = declared_submodules
        self._binding_recorder: (
            Callable[[Any, Mapping[str, Any], str], Any] | None
        ) = None
        self._reconciliation_binding_recorder: (
            Callable[[Any, Mapping[str, Any], str], Any] | None
        ) = None
        self._binding_lookup: Callable[[Any], Mapping[str, Any] | None] | None = None

    def bind_attempt_binding_authority(
        self,
        *,
        recorder: Callable[[Any, Mapping[str, Any], str], Any],
        reconciliation_recorder: Callable[
            [Any, Mapping[str, Any], str], Any
        ],
        lookup: Callable[[Any], Mapping[str, Any] | None],
    ) -> None:
        if (
            not callable(recorder)
            or not callable(reconciliation_recorder)
            or not callable(lookup)
        ):
            raise TypeError("database Portal binding authority must be callable")
        if (
            self._binding_recorder is not None
            or self._reconciliation_binding_recorder is not None
            or self._binding_lookup is not None
        ):
            raise DatabasePortalBridgeError(
                "database Portal binding authority is already bound"
            )
        self._binding_recorder = recorder
        self._reconciliation_binding_recorder = reconciliation_recorder
        self._binding_lookup = lookup

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
            reconciliation=root / "database-attempt-reconciliations",
        )

    @staticmethod
    def _binding_without_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = dict(binding)
        unsigned.pop("binding_id", None)
        return unsigned

    @classmethod
    def _verify_binding_identity(cls, binding: Mapping[str, Any]) -> None:
        """Verify the closed binding before it can nominate nested state."""

        if frozenset(str(key) for key in binding) != _ATTEMPT_BINDING_FIELDS:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding has unknown or missing fields"
            )
        if (
            binding.get("schema") != DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
            or binding.get("interface") != DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
            or binding.get("authoritative_task_store") != "duckdb"
            or binding.get("projection_authority") is not False
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt binding has an unknown authority"
            )
        for field in (
            "attempt_id",
            "claim_id",
            "task_cid",
            "task_alias",
            "binding_id",
        ):
            if not isinstance(binding.get(field), str) or not str(
                binding.get(field) or ""
            ).strip():
                raise DatabasePortalBridgeError(
                    f"database Portal attempt binding lacks {field}"
                )
        for field in ("fencing_token", "fence_epoch", "task_revision"):
            value = binding.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DatabasePortalBridgeError(
                    f"database Portal attempt binding has malformed {field}"
                )
        expected_id = _sha256_bytes(
            _canonical_json(cls._binding_without_identity(binding))
        )
        if str(binding.get("binding_id") or "") != expected_id:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding identity does not verify"
            )

    @staticmethod
    def _strict_state_record(
        path: Path,
    ) -> tuple[Mapping[str, Any] | None, str]:
        if not path.exists():
            return None, ""
        if path.is_symlink() or not path.is_file():
            raise DatabasePortalBridgeError(
                "database Portal nested task state is not a regular file"
            )

        def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise DatabasePortalBridgeError(
                        "database Portal nested task state contains duplicate keys"
                    )
                result[key] = value
            return result

        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            payload = json.loads(
                text,
                object_pairs_hook=closed_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    DatabasePortalBridgeError(
                        "database Portal nested task state contains a nonfinite value"
                    )
                ),
            )
        except (
            OSError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise DatabasePortalBridgeError(
                "database Portal nested task state is unreadable"
            ) from exc
        if not isinstance(payload, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal nested task state is malformed"
            )
        from .implementation_daemon import PortalTaskState

        allowed_fields = {item.name for item in fields(PortalTaskState)}
        unknown = set(payload) - allowed_fields
        if unknown:
            raise DatabasePortalBridgeError(
                "database Portal nested task state has unknown fields"
            )
        defaults = PortalTaskState()
        nullable_integer_fields = {
            "last_implementation_returncode",
            "last_merge_returncode",
        }
        for state_field in fields(PortalTaskState):
            name = state_field.name
            if name not in payload:
                continue
            observed = payload[name]
            expected = getattr(defaults, name)
            valid = False
            if name in nullable_integer_fields:
                valid = observed is None or (
                    isinstance(observed, int) and not isinstance(observed, bool)
                )
            elif isinstance(expected, bool):
                valid = isinstance(observed, bool)
            elif isinstance(expected, int):
                valid = isinstance(observed, int) and not isinstance(
                    observed, bool
                )
            elif isinstance(expected, str):
                valid = isinstance(observed, str)
            elif isinstance(expected, list):
                valid = isinstance(observed, list) and all(
                    isinstance(item, str) for item in observed
                )
            elif isinstance(expected, dict):
                valid = isinstance(observed, dict)
            if not valid:
                raise DatabasePortalBridgeError(
                    "database Portal nested task state has malformed " + name
                )
        return payload, _sha256_bytes(raw)

    @staticmethod
    def _strict_state_payload(path: Path) -> Mapping[str, Any] | None:
        payload, _digest = DatabasePortalExecutionBridge._strict_state_record(
            path
        )
        return payload

    def _verify_nested_state_identity(
        self,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        identity: Mapping[str, str],
        *,
        payload: Mapping[str, Any] | None,
        state_digest: str,
    ) -> dict[str, Any]:
        """Bind a nonempty nested Portal state to its exact DB attempt."""

        if payload is None:
            return {
                "present": False,
                "state_path": str(paths.state),
                "state_digest": "",
                "active": False,
            }
        alias = str(binding.get("task_alias") or "")
        active_task_id = str(payload.get("active_task_id") or "")
        active_task_key = str(payload.get("active_task_key") or "")
        active_task_cid = str(payload.get("active_task_cid") or "")
        if active_task_id and active_task_id != alias:
            raise DatabasePortalBridgeError(
                "database Portal nested state names a different task"
            )
        if active_task_key and active_task_key != identity["canonical_task_key"]:
            raise DatabasePortalBridgeError(
                "database Portal nested state task key changed"
            )
        if active_task_cid and active_task_cid != identity["canonical_task_cid"]:
            raise DatabasePortalBridgeError(
                "database Portal nested state task CID changed"
            )
        task_identities = payload.get("task_identities") or {}
        if not isinstance(task_identities, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal nested state task identities are malformed"
            )
        nested_identity = task_identities.get(alias)
        if nested_identity is not None:
            if not isinstance(nested_identity, Mapping) or any(
                str(nested_identity.get(field) or "") != expected
                for field, expected in identity.items()
                if field != "task_id"
            ):
                raise DatabasePortalBridgeError(
                    "database Portal nested state identity is not exact"
                )
        named_task_ids: set[str] = set()
        for field in (
            "completed_task_ids",
            "ready_task_ids",
            "selectable_ready_task_ids",
            "eligible_ready_task_ids",
            "waiting_task_ids",
            "blocked_task_ids",
        ):
            values = payload.get(field) or []
            if not isinstance(values, list):
                raise DatabasePortalBridgeError(
                    "database Portal nested task population is malformed"
                )
            named_task_ids.update(str(item) for item in values if str(item))
        statuses = payload.get("task_statuses") or {}
        if not isinstance(statuses, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal nested task statuses are malformed"
            )
        named_task_ids.update(str(item) for item in statuses)
        named_task_ids.update(str(item) for item in task_identities)
        if named_task_ids - {alias}:
            raise DatabasePortalBridgeError(
                "database Portal nested state contains an unrelated task"
            )
        raw_attempt = payload.get("active_attempt", 0)
        if isinstance(raw_attempt, bool):
            raise DatabasePortalBridgeError(
                "database Portal nested active attempt is malformed"
            )
        try:
            active_attempt = int(raw_attempt or 0)
        except (TypeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal nested active attempt is malformed"
            ) from exc
        if active_attempt < 0:
            raise DatabasePortalBridgeError(
                "database Portal nested active attempt is malformed"
            )
        active = bool(
            payload.get("implementation_in_progress")
            or active_task_id
            or active_attempt
            or payload.get("active_worktree_path")
            or payload.get("active_branch")
            or payload.get("active_provider_runner")
        )
        if active and (
            active_task_id != alias
            or active_task_key != identity["canonical_task_key"]
            or active_task_cid != identity["canonical_task_cid"]
            or not isinstance(nested_identity, Mapping)
        ):
            raise DatabasePortalBridgeError(
                "database Portal active nested state lacks its exact task identity"
            )
        return {
            "present": True,
            "state_path": str(paths.state),
            "state_digest": state_digest,
            "active": active,
            "active_task_id": active_task_id,
            "active_attempt": active_attempt,
            "active_phase": str(payload.get("active_phase") or ""),
            "active_phase_detail": str(
                payload.get("active_phase_detail") or ""
            ),
            "active_worktree_path": str(
                payload.get("active_worktree_path") or ""
            ),
            "active_branch": str(payload.get("active_branch") or ""),
        }

    @staticmethod
    def _validated_provider_runner_fence(
        raw: Any,
        *,
        expected_runner_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal provider fence returned a non-object"
            )
        result = dict(raw)
        if set(result) - _PROVIDER_RUNNER_FENCE_FIELDS:
            raise DatabasePortalBridgeError(
                "database Portal provider fence returned unknown fields"
            )
        for field in ("applicable", "safe_to_restart", "fenced"):
            if not isinstance(result.get(field), bool):
                raise DatabasePortalBridgeError(
                    "database Portal provider fence has malformed " + field
                )
        reason = result.get("reason")
        if (
            not isinstance(reason, str)
            or not reason
            or len(reason) > 256
            or any(character in reason for character in "\0\n\r")
        ):
            raise DatabasePortalBridgeError(
                "database Portal provider fence has malformed reason"
            )
        for field in ("pid", "parent_pid_before_fence"):
            if field not in result:
                continue
            value = result[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DatabasePortalBridgeError(
                    "database Portal provider fence has malformed " + field
                )
        if "pid_reused" in result and not isinstance(
            result["pid_reused"], bool
        ):
            raise DatabasePortalBridgeError(
                "database Portal provider fence has malformed pid_reused"
            )
        if "host_fenced" in result and not isinstance(
            result["host_fenced"], bool
        ):
            raise DatabasePortalBridgeError(
                "database Portal provider fence has malformed host_fenced"
            )
        container_fence = result.get("container_fence")
        if "host_fenced" in result and container_fence is None:
            raise DatabasePortalBridgeError(
                "database Portal host fence lacks its container receipt"
            )
        if container_fence is not None:
            if (
                not isinstance(container_fence, Mapping)
                or set(container_fence)
                != _ORDINARY_GROK_CONTAINER_FENCE_FIELDS
            ):
                raise DatabasePortalBridgeError(
                    "database Portal container fence has malformed shape"
                )
            container_reason = container_fence.get("reason")
            container_detail = container_fence.get("detail")
            unsigned_container_fence = {
                key: value
                for key, value in container_fence.items()
                if key != "receipt_id"
            }
            if not isinstance(expected_runner_receipt, Mapping):
                raise DatabasePortalBridgeError(
                    "database Portal container fence lacks its active runner binding"
                )
            expected_container_binding = {
                "task_id": expected_runner_receipt.get("task_id"),
                "attempt": expected_runner_receipt.get("attempt"),
                "task_revision_cid": expected_runner_receipt.get(
                    "task_revision_cid"
                ),
                "workspace_path": expected_runner_receipt.get("workspace_path"),
                "runner_pid": expected_runner_receipt.get("pid"),
                "runner_receipt_id": expected_runner_receipt.get("receipt_id"),
            }
            if (
                container_fence.get("schema")
                != _ORDINARY_GROK_CONTAINER_FENCE_SCHEMA
                or content_identity(unsigned_container_fence)
                != container_fence.get("receipt_id")
                or not isinstance(
                    container_fence.get("safe_to_restart"), bool
                )
                or not isinstance(container_fence.get("removed"), bool)
                or isinstance(container_fence.get("attempt"), bool)
                or not isinstance(container_fence.get("attempt"), int)
                or int(container_fence.get("attempt")) < 1
                or isinstance(container_fence.get("runner_pid"), bool)
                or not isinstance(container_fence.get("runner_pid"), int)
                or int(container_fence.get("runner_pid")) <= 1
                or not isinstance(container_detail, Mapping)
                or any(
                    not isinstance(container_fence.get(field), str)
                    or not str(container_fence.get(field))
                    for field in (
                        "task_id",
                        "task_revision_cid",
                        "workspace_path",
                        "runner_receipt_id",
                        "reason",
                    )
                )
                or (
                    container_fence.get("runner_pid") != result.get("pid")
                )
                or any(
                    container_fence.get(field) != expected
                    for field, expected in expected_container_binding.items()
                )
                or (
                    container_fence.get("safe_to_restart")
                    is not result.get("safe_to_restart")
                )
                or (
                    container_reason
                    not in {
                        "ordinary_grok_orphan_private_lease_absent",
                        "ordinary_grok_orphan_container_absent",
                        "ordinary_grok_orphan_container_removed",
                        "ordinary_grok_orphan_container_fence_unproven",
                    }
                )
                or (
                    container_fence.get("removed") is True
                    and (
                        container_fence.get("safe_to_restart") is not True
                        or container_reason
                        != "ordinary_grok_orphan_container_removed"
                    )
                )
                or (
                    container_reason
                    == "ordinary_grok_orphan_container_removed"
                    and container_fence.get("removed") is not True
                )
                or (
                    container_fence.get("safe_to_restart") is False
                    and container_reason
                    != "ordinary_grok_orphan_container_fence_unproven"
                )
                or (
                    container_fence.get("safe_to_restart") is True
                    and container_reason
                    == "ordinary_grok_orphan_container_fence_unproven"
                )
                or (
                    container_fence.get("safe_to_restart") is True
                    and (
                        result.get("applicable") is not True
                        or result.get("fenced") is not True
                        or result.get("reason")
                        != "ordinary_provider_runner_exact_birth_fenced"
                        or "host_fenced" in result
                    )
                )
                or (
                    container_fence.get("safe_to_restart") is False
                    and (
                        result.get("applicable") is not True
                        or result.get("host_fenced") is not True
                        or result.get("fenced") is not False
                        or result.get("reason")
                        != "ordinary_grok_orphan_container_fence_unproven"
                    )
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal container fence receipt is invalid"
                )
        if result["fenced"] is True and (
            result["applicable"] is not True
            or result["safe_to_restart"] is not True
        ):
            raise DatabasePortalBridgeError(
                "database Portal provider fence contradicts its authority"
            )
        return result

    def validate_active_attempt_roots(
        self,
        attempts: Sequence[Any],
    ) -> dict[str, str]:
        """Validate only DB-nominated roots for the exact running attempts.

        The attempt-id hash is the one path-selection authority.  Historical,
        sibling, and merely similar directories are never scanned and cannot
        nominate a process, lifecycle record, or database mutation.
        """

        if self.attempt_root.is_symlink() or (
            self.attempt_root.exists() and not self.attempt_root.is_dir()
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt authority is not a regular directory"
            )
        selected: dict[str, str] = {}
        for attempt in attempts:
            attempt_id = str(attempt.attempt_id)
            paths = self._paths(attempt)
            has_any = any(
                path.exists()
                for path in (paths.binding, paths.task_projection, paths.state)
            )
            if not has_any:
                continue
            if not paths.binding.exists():
                raise DatabasePortalBridgeError(
                    "database Portal active attempt lacks its exact binding"
                )
            payload = self._read_binding(paths.binding)
            self._verify_binding_identity(payload)
            if str(payload.get("attempt_id") or "") != attempt_id:
                raise DatabasePortalBridgeError(
                    "database Portal active attempt binding changed identity"
                )
            selected[attempt_id] = str(paths.binding)
        return selected

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
        def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise DatabasePortalBridgeError(
                        "database Portal attempt binding contains duplicate keys"
                    )
                result[key] = value
            return result

        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=closed_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    DatabasePortalBridgeError(
                        "database Portal attempt binding contains a nonfinite value"
                    )
                ),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is unreadable"
            ) from exc
        if not isinstance(value, Mapping):
            raise DatabasePortalBridgeError("database Portal attempt binding is malformed")
        return value

    def _ensure_attempt_projection(
        self,
        attempt: Any,
        record: Any,
        *,
        admit_before_publish: bool = False,
    ) -> tuple[DatabasePortalAttemptPaths, Mapping[str, Any]]:
        paths = self._paths(attempt)
        seed = self._render_projection(attempt, record)
        expected = self._binding(attempt, record, seed)
        durable_binding = (
            self._binding_lookup(attempt)
            if admit_before_publish and self._binding_lookup is not None
            else None
        )
        durable_stage = str((durable_binding or {}).get("stage") or "")
        if isinstance(durable_binding, Mapping):
            if (
                durable_binding.get("binding_id") != expected.get("binding_id")
                or durable_binding.get("projection_immutable_digest")
                != expected.get("projection_immutable_digest")
                or durable_stage not in {"prepared", "published", "portal_entered"}
            ):
                raise DatabasePortalBridgeError(
                    "database Portal durable binding changed across resume"
                )
        binding_present = paths.binding.exists()
        projection_present = paths.task_projection.exists()
        if paths.root.exists() and (
            paths.root.is_symlink() or not paths.root.is_dir()
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt directory is not an exact directory"
            )
        if paths.binding.is_symlink() or paths.task_projection.is_symlink():
            raise DatabasePortalBridgeError(
                "database Portal attempt projection contains a symlink"
            )
        if binding_present:
            observed = self._read_binding(paths.binding)
            if observed != expected:
                raise DatabasePortalBridgeError(
                    "database Portal attempt binding changed across resume"
                )
        if projection_present:
            self._verify_projection(paths, expected)
        if isinstance(durable_binding, Mapping) and durable_stage in {
            "published",
            "portal_entered",
        }:
            if not binding_present or not projection_present:
                raise DatabasePortalBridgeError(
                    "database Portal published artifacts disappeared before resume"
                )
            return paths, expected
        if isinstance(durable_binding, Mapping) and durable_stage == "prepared":
            if projection_present and not binding_present:
                raise DatabasePortalBridgeError(
                    "database Portal prepared projection lacks its binding"
                )
        elif binding_present or projection_present:
            if not binding_present or not projection_present:
                raise DatabasePortalBridgeError(
                    "database Portal unadmitted projection is partial"
                )
            if admit_before_publish and self._binding_recorder is not None:
                # A released predecessor may already have entered Portal.  A
                # complete exact legacy pair is therefore admitted directly
                # as entered, never retrospectively called pre-Portal.
                self._binding_recorder(attempt, expected, "portal_entered")
            return paths, expected
        if (
            admit_before_publish
            and self._binding_recorder is not None
            and not isinstance(durable_binding, Mapping)
        ):
            # The exact fenced database admission is durable before either
            # filesystem artifact is published.  A crash can therefore never
            # leave an unadmitted self-hashed projection as historical cleanup
            # authority, and Portal construction remains strictly later.
            self._binding_recorder(attempt, expected, "prepared")
        _ensure_durable_directory(paths.root)
        if not paths.binding.exists():
            _atomic_write(
                paths.binding,
                json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
        if not paths.task_projection.exists():
            _atomic_write(paths.task_projection, seed.encode("utf-8"))
        self._verify_projection(paths, expected)
        if (
            admit_before_publish
            and self._binding_recorder is not None
            and durable_stage != "published"
        ):
            self._binding_recorder(attempt, expected, "published")
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

    def _projection_task(
        self,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        projection_text: str | None = None,
    ) -> Any:
        """Parse the sealed task through the current Portal task authority.

        Importing the parser locally avoids a module-import cycle: the Portal
        daemon imports this bridge only while constructing a database-backed
        execution route.  Parsing the attempt-local projection is important;
        reproducing the identity here would create a second task-identity
        authority and could drift from the completion event producer.
        """

        from .implementation_daemon import parse_task_text

        text = (
            projection_text
            if projection_text is not None
            else self._verify_projection(paths, binding)
        )
        alias = str(binding.get("task_alias") or "")
        try:
            tasks = parse_task_text(
                text,
                path=paths.task_projection,
                # The database claim already chose one exact alias.  Parsing
                # by that alias is stricter than a board-wide family prefix
                # and also works for direct bridge users whose outer parser
                # prefix is unrelated to the claimed task family.
                task_header_prefix=f"## {alias}",
            )
        except (TypeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "Portal task projection identity is malformed"
            ) from exc
        if len(tasks) != 1 or tasks[0].task_id != alias:
            raise DatabasePortalBridgeError(
                "Portal task projection identity does not match the claimed task"
            )
        return tasks[0]

    def _projection_task_identity(
        self,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        projection_text: str | None = None,
    ) -> dict[str, str]:
        """Return the current Portal authority's identity for the sealed task."""

        task = self._projection_task(paths, binding, projection_text)
        identity = {
            "task_id": task.task_id,
            "canonical_task_key": str(task.canonical_task_key or ""),
            "canonical_task_cid": str(task.canonical_task_cid or ""),
            "board_namespace": str(task.board_namespace or ""),
        }
        if any(not value for value in identity.values()):
            raise DatabasePortalBridgeError(
                "Portal task projection lacks a complete canonical identity"
            )
        return identity

    @staticmethod
    def _has_completion_event(
        paths: DatabasePortalAttemptPaths,
        identity: Mapping[str, str],
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
                and all(
                    str(event.get(field) or "") == expected
                    for field, expected in identity.items()
                )
            ):
                return True
        return False

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
        identity = self._projection_task_identity(
            paths,
            binding,
            projection_text,
        )
        if _projection_status(projection_text) not in _TERMINAL_STATUSES:
            raise DatabasePortalBridgeDeferred("Portal task projection is not complete")
        if not self._has_completion_event(paths, identity):
            raise DatabasePortalBridgeError(
                "Portal completion lacks an exact canonical task_completed event"
            )
        evidence = {
            "binding_id": str(binding.get("binding_id") or ""),
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "canonical_task_key": identity["canonical_task_key"],
            "canonical_task_cid": identity["canonical_task_cid"],
            "board_namespace": identity["board_namespace"],
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
            "canonical_task_key": identity["canonical_task_key"],
            "canonical_task_cid": identity["canonical_task_cid"],
            "board_namespace": identity["board_namespace"],
            "attempt_id": str(attempt.attempt_id),
            "binding_id": str(binding.get("binding_id") or ""),
            "evidence_digest": evidence_digest,
            "portal_evidence": evidence,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    @staticmethod
    def _prepare_private_event_log(paths: DatabasePortalAttemptPaths) -> None:
        """Create the attempt-local active journal with owner-only write mode."""

        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = -1
        directory_fd = -1
        try:
            directory_fd = os.open(paths.root, directory_flags)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(
                    paths.events.name,
                    flags,
                    0o600,
                    dir_fd=directory_fd,
                )
                os.fsync(descriptor)
                os.fsync(directory_fd)
            except FileExistsError as exc:
                observed = os.stat(
                    paths.events.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or observed.st_uid != os.geteuid()
                    or observed.st_gid != os.getegid()
                    or int(observed.st_nlink) != 1
                    or stat.S_IMODE(observed.st_mode) != 0o600
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal event journal is not owner-private"
                    ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if directory_fd >= 0:
                os.close(directory_fd)

    def run_provider(self, attempt: Any) -> Mapping[str, Any]:
        """Run bounded real Portal passes and return only accepted evidence."""

        record = self._record_for_attempt(self.task_source, attempt)
        try:
            paths, binding = self._ensure_attempt_projection(
                attempt,
                record,
                admit_before_publish=True,
            )
        except DatabasePortalPreEntryPublicationDeferred:
            raise
        except (OSError, TimeoutError, ConnectionError) as exc:
            durable = (
                self._binding_lookup(attempt)
                if self._binding_lookup is not None
                else None
            )
            stage = str((durable or {}).get("stage") or "")
            if stage in {"prepared", "published"}:
                raise DatabasePortalPreEntryPublicationDeferred(
                    "database_portal_preentry_" + stage
                ) from exc
            raise
        if self._binding_recorder is not None:
            # This durable monotonic transition precedes Portal construction;
            # any later absence or partial artifact set is therefore unknown,
            # never evidence of a safe pre-provider retry.
            try:
                self._binding_recorder(attempt, binding, "portal_entered")
            except DatabasePortalPreEntryPublicationDeferred:
                raise
            except (OSError, TimeoutError, ConnectionError) as exc:
                durable = (
                    self._binding_lookup(attempt)
                    if self._binding_lookup is not None
                    else None
                )
                if str((durable or {}).get("stage") or "") == "published":
                    raise DatabasePortalPreEntryPublicationDeferred(
                        "database_portal_preentry_published"
                    ) from exc
                raise
        self._prepare_private_event_log(paths)
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
                identity = self._projection_task_identity(
                    paths,
                    binding,
                    projection,
                )
                if _projection_status(
                    projection
                ) in _TERMINAL_STATUSES and self._has_completion_event(
                    paths, identity
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
                failure = self._terminal_failure(raw_result)
                if failure:
                    implementation = raw_result.get("implementation_result")
                    if (
                        isinstance(implementation, Mapping)
                        and implementation.get("deferred") is True
                    ):
                        expected_deferral_fields = {
                            "deferred",
                            "skipped",
                            "reason",
                            "task_id",
                            "attempt",
                            "backoff_seconds",
                            "attempt_consumed",
                            "provider_dispatched",
                            "diagnostic_receipt_id",
                            "active_task_cleared",
                        }
                        if not (
                            set(implementation) == expected_deferral_fields
                            and implementation.get("skipped") is True
                            and implementation.get("task_id")
                            == str(binding.get("task_alias") or "")
                            and type(implementation.get("attempt")) is int
                            and int(implementation["attempt"]) >= 1
                            and isinstance(implementation.get("reason"), str)
                            and bool(str(implementation["reason"]))
                            and isinstance(
                                implementation.get("diagnostic_receipt_id"),
                                str,
                            )
                            and implementation.get("attempt_consumed") is False
                            and implementation.get("provider_dispatched") is False
                            and implementation.get("active_task_cleared") is True
                            and type(implementation.get("backoff_seconds")) is int
                            and int(implementation["backoff_seconds"]) > 0
                            and int(implementation["backoff_seconds"])
                            <= DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_BACKOFF_SECONDS
                        ):
                            raise DatabasePortalBridgeError(
                                "Portal provider deferral lacks exact pre-dispatch "
                                "authority"
                            )
                        raise DatabasePortalProviderRouteDeferred(
                            failure,
                            backoff_seconds=int(
                                implementation["backoff_seconds"]
                            ),
                        )
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
            )
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(daemon, "close", None)
            if callable(close):
                close()

    def recover_provider_result(self, attempt: Any) -> Mapping[str, Any] | None:
        """Recover an exact terminal Portal receipt without dispatching a model.

        The outer database daemon calls this only after it finds a durable
        provider-dispatch marker with no matching provider phase.  Recovery is
        deliberately read-only: absent or incomplete Portal evidence returns
        ``None`` and the outer attempt fails closed rather than reimplementing
        work whose outcome is unknown.
        """

        record = self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        if not paths.binding.is_file() or not paths.task_projection.is_file():
            return None
        seed = self._render_projection(attempt, record)
        expected = self._binding(attempt, record, seed)
        observed = self._read_binding(paths.binding)
        if observed != expected:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding changed before recovery"
            )
        projection = self._verify_projection(paths, expected)
        identity = self._projection_task_identity(paths, expected, projection)
        if (
            _projection_status(projection) not in _TERMINAL_STATUSES
            or not self._has_completion_event(paths, identity)
        ):
            return None
        return self._acceptance_receipt(
            attempt=attempt,
            paths=paths,
            binding=expected,
            summaries=(),
        )

    @staticmethod
    def _authority_fingerprint(item: os.stat_result) -> tuple[
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
    ]:
        return (
            int(item.st_dev),
            int(item.st_ino),
            int(item.st_mode),
            int(item.st_uid),
            int(item.st_gid),
            int(item.st_nlink),
            int(item.st_size),
            int(item.st_mtime_ns),
            int(item.st_ctime_ns),
        )

    @classmethod
    def _validate_private_directory(
        cls,
        item: os.stat_result,
        *,
        authority: str,
    ) -> None:
        if (
            not stat.S_ISDIR(item.st_mode)
            or item.st_uid != os.geteuid()
            or item.st_gid != os.getegid()
            or stat.S_IMODE(item.st_mode) != 0o700
        ):
            raise DatabasePortalBridgeError(
                f"database Portal {authority} is not an owner-private directory"
            )

    @classmethod
    def _open_pinned_private_attempt_directory(
        cls,
        *,
        authority_root: Path,
        attempt_key: str,
    ) -> tuple[int, int, int, tuple[Any, ...]]:
        """Pin parent, authority root, and exact attempt child without links."""

        if (
            not re.fullmatch(r"[0-9a-f]{24}", attempt_key)
            or authority_root.name in {"", ".", ".."}
        ):
            raise DatabasePortalBridgeError(
                "database Portal no-provider attempt root key is malformed"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptors: list[int] = []
        try:
            parent_fd = os.open(authority_root.parent, flags)
            descriptors.append(parent_fd)
            authority_fd = os.open(
                authority_root.name,
                flags,
                dir_fd=parent_fd,
            )
            descriptors.append(authority_fd)
            attempt_fd = os.open(attempt_key, flags, dir_fd=authority_fd)
            descriptors.append(attempt_fd)
            parent_stat = os.fstat(parent_fd)
            authority_stat = os.fstat(authority_fd)
            attempt_stat = os.fstat(attempt_fd)
            cls._validate_private_directory(
                parent_stat,
                authority="attempt parent",
            )
            cls._validate_private_directory(
                authority_stat,
                authority="attempt authority root",
            )
            cls._validate_private_directory(
                attempt_stat,
                authority="attempt root",
            )
            published_authority = os.stat(
                authority_root.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            published_attempt = os.stat(
                attempt_key,
                dir_fd=authority_fd,
                follow_symlinks=False,
            )
            if (
                cls._authority_fingerprint(authority_stat)
                != cls._authority_fingerprint(published_authority)
                or cls._authority_fingerprint(attempt_stat)
                != cls._authority_fingerprint(published_attempt)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal no-provider attempt directory was retargeted"
                )
            snapshot = (
                cls._authority_fingerprint(parent_stat),
                cls._authority_fingerprint(authority_stat),
                cls._authority_fingerprint(attempt_stat),
                authority_root.name,
                attempt_key,
            )
            return parent_fd, authority_fd, attempt_fd, snapshot
        except BaseException:
            for descriptor in reversed(descriptors):
                with suppress(OSError):
                    os.close(descriptor)
            raise

    @classmethod
    def _verify_pinned_private_attempt_directory(
        cls,
        *,
        parent_fd: int,
        authority_fd: int,
        attempt_fd: int,
        snapshot: tuple[Any, ...],
    ) -> None:
        parent_expected, authority_expected, attempt_expected, root_name, attempt_key = (
            snapshot
        )
        observed = (
            cls._authority_fingerprint(os.fstat(parent_fd)),
            cls._authority_fingerprint(os.fstat(authority_fd)),
            cls._authority_fingerprint(os.fstat(attempt_fd)),
            cls._authority_fingerprint(
                os.stat(root_name, dir_fd=parent_fd, follow_symlinks=False)
            ),
            cls._authority_fingerprint(
                os.stat(attempt_key, dir_fd=authority_fd, follow_symlinks=False)
            ),
        )
        if observed != (
            parent_expected,
            authority_expected,
            attempt_expected,
            authority_expected,
            attempt_expected,
        ):
            raise DatabasePortalBridgeError(
                "database Portal no-provider attempt authority changed during replay"
            )

    @classmethod
    def _read_exact_private_child(
        cls,
        directory_fd: int,
        name: str,
        *,
        maximum_bytes: int,
    ) -> tuple[bytes, tuple[int, ...]]:
        """Read one exact child through a pinned directory descriptor."""

        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\x00" in name
            or len(os.fsencode(name)) > 255
        ):
            raise DatabasePortalBridgeError(
                "database Portal recovery child name is malformed"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal recovery authority is unreadable"
            ) from exc
        try:
            before = os.fstat(descriptor)
            published_before = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_gid != os.getegid()
                or int(before.st_nlink) != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or int(before.st_size) > maximum_bytes
                or cls._authority_fingerprint(before)
                != cls._authority_fingerprint(published_before)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal recovery authority is not an exact private file"
                )
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(128 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            published_after = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(descriptor)
        fingerprint = cls._authority_fingerprint(before)
        if (
            len(raw) > maximum_bytes
            or len(raw) != int(before.st_size)
            or fingerprint != cls._authority_fingerprint(after)
            or fingerprint != cls._authority_fingerprint(published_after)
        ):
            raise DatabasePortalBridgeError(
                "database Portal recovery authority changed during read"
            )
        return raw, fingerprint

    @classmethod
    def _read_exact_terminal_no_effect_log(
        cls,
        attempt_fd: int,
        name: str,
        *,
        maximum_bytes: int = 256 * 1024,
    ) -> tuple[bytes, tuple[int, ...], str]:
        """Read one legacy runner log through the already-pinned attempt.

        The four predecessor attempts used an owner-private ``0700`` attempt
        directory containing an owner/group ``0775`` log directory, while the
        authoritative attempt log itself was owner-private ``0600``.  That
        exact historical mode profile is part of this versioned verifier; it
        is not a general relaxation of the private-file reader.
        """

        if (
            re.fullmatch(r"pctdd-[0-9]{3}-attempt-[1-9][0-9]*[.]log", name)
            is None
        ):
            raise DatabasePortalBridgeError(
                "database Portal terminal no-effect log name is malformed"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = -1
        try:
            descriptor = os.open(
                "implementation-logs",
                flags,
                dir_fd=attempt_fd,
            )
            directory_before = os.fstat(descriptor)
            published_before = os.stat(
                "implementation-logs",
                dir_fd=attempt_fd,
                follow_symlinks=False,
            )
            directory_fingerprint = cls._authority_fingerprint(
                directory_before
            )
            if (
                not stat.S_ISDIR(directory_before.st_mode)
                or directory_before.st_uid != os.geteuid()
                or directory_before.st_gid != os.getegid()
                or int(directory_before.st_nlink) < 2
                or stat.S_IMODE(directory_before.st_mode) != 0o775
                or directory_fingerprint
                != cls._authority_fingerprint(published_before)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal legacy log directory identity is invalid"
                )
            attempt_prefix = name[: -len(".log")]
            alias_prefix = attempt_prefix.rsplit("-attempt-", 1)[0]
            expected_names = sorted(
                {
                    f"{attempt_prefix}-context-receipt.json",
                    name,
                    f"{alias_prefix}-base-context-capsule.json",
                    f"{alias_prefix}-base-context-receipt.json",
                    f"{alias_prefix}-diagnostic-receipt.json",
                    f"{alias_prefix}-diagnostic-state.json",
                }
            )
            names_before = sorted(os.listdir(descriptor))
            if names_before != expected_names:
                raise DatabasePortalBridgeError(
                    "database Portal legacy log population is not exact"
                )
            entry_fingerprints: dict[str, tuple[int, ...]] = {}
            for entry_name in names_before:
                entry = os.stat(
                    entry_name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                entry_fingerprint = cls._authority_fingerprint(entry)
                expected_mode = 0o600 if entry_name == name else 0o664
                if (
                    not stat.S_ISREG(entry.st_mode)
                    or entry.st_uid != os.geteuid()
                    or entry.st_gid != os.getegid()
                    or int(entry.st_nlink) != 1
                    or stat.S_IMODE(entry.st_mode) != expected_mode
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal legacy log entry identity is invalid"
                    )
                entry_fingerprints[entry_name] = entry_fingerprint
            raw, log_fingerprint = cls._read_exact_private_child(
                descriptor,
                name,
                maximum_bytes=maximum_bytes,
            )
            names_after = sorted(os.listdir(descriptor))
            entries_after = {
                entry_name: cls._authority_fingerprint(
                    os.stat(
                        entry_name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                )
                for entry_name in names_after
            }
            directory_after = os.fstat(descriptor)
            published_after = os.stat(
                "implementation-logs",
                dir_fd=attempt_fd,
                follow_symlinks=False,
            )
            if (
                names_after != names_before
                or entries_after != entry_fingerprints
                or cls._authority_fingerprint(directory_after)
                != directory_fingerprint
                or cls._authority_fingerprint(published_after)
                != directory_fingerprint
            ):
                raise DatabasePortalBridgeError(
                    "database Portal legacy log directory changed during read"
                )
            identity = {
                "directory_name": "implementation-logs",
                "directory_fingerprint": list(directory_fingerprint),
                "entry_names_digest": _sha256_bytes(
                    _canonical_json(names_before)
                ),
                "entry_fingerprints": {
                    entry_name: list(entry_fingerprints[entry_name])
                    for entry_name in names_before
                },
                "log_name": name,
                "log_fingerprint": list(log_fingerprint),
            }
            return raw, log_fingerprint, _sha256_bytes(
                _canonical_json(identity)
            )
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal terminal no-effect log authority is unreadable"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @classmethod
    def _revalidate_exact_terminal_no_effect_log(
        cls,
        attempt_fd: int,
        name: str,
        *,
        expected_raw: bytes,
        expected_fingerprint: tuple[int, ...],
        expected_identity_digest: str,
    ) -> None:
        raw, fingerprint, identity_digest = (
            cls._read_exact_terminal_no_effect_log(
                attempt_fd,
                name,
            )
        )
        if (
            raw != expected_raw
            or fingerprint != expected_fingerprint
            or identity_digest != expected_identity_digest
        ):
            raise DatabasePortalBridgeError(
                "database Portal legacy log changed during snapshot replay"
            )

    @classmethod
    def _revalidate_private_children(
        cls,
        directory_fd: int,
        snapshots: Mapping[str, tuple[int, ...]],
    ) -> None:
        for name, expected in snapshots.items():
            try:
                observed = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise DatabasePortalBridgeError(
                    "database Portal recovery authority changed during replay"
                ) from exc
            if cls._authority_fingerprint(observed) != expected:
                raise DatabasePortalBridgeError(
                    "database Portal recovery authority changed during replay"
                )

    @staticmethod
    def _strict_json_object_bytes(
        raw: bytes,
        *,
        authority: str,
    ) -> dict[str, Any]:
        def closed_object(
            pairs: Sequence[tuple[str, Any]],
        ) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise DatabasePortalBridgeError(
                        f"database Portal {authority} contains duplicate keys"
                    )
                value[key] = item
            return value

        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=closed_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    DatabasePortalBridgeError(
                        f"database Portal {authority} contains a nonfinite value"
                    )
                ),
            )
        except DatabasePortalBridgeError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                f"database Portal {authority} is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise DatabasePortalBridgeError(
                f"database Portal {authority} is not an object"
            )
        return dict(value)

    @classmethod
    def _strict_portal_state_bytes(
        cls,
        raw: bytes,
    ) -> tuple[dict[str, Any], str]:
        payload = cls._strict_json_object_bytes(
            raw,
            authority="nested task state",
        )
        from .implementation_daemon import PortalTaskState

        defaults = PortalTaskState()
        expected_fields = {item.name for item in fields(PortalTaskState)}
        if set(payload) != expected_fields:
            raise DatabasePortalBridgeError(
                "database Portal nested task state population is not exact"
            )
        nullable_integer_fields = {
            "last_implementation_returncode",
            "last_merge_returncode",
        }
        for state_field in fields(PortalTaskState):
            name = state_field.name
            observed = payload[name]
            expected = getattr(defaults, name)
            valid = False
            if name in nullable_integer_fields:
                valid = observed is None or (
                    isinstance(observed, int) and not isinstance(observed, bool)
                )
            elif isinstance(expected, bool):
                valid = isinstance(observed, bool)
            elif isinstance(expected, int):
                valid = isinstance(observed, int) and not isinstance(observed, bool)
            elif isinstance(expected, str):
                valid = isinstance(observed, str)
            elif isinstance(expected, list):
                valid = isinstance(observed, list) and all(
                    isinstance(item, str) for item in observed
                )
            elif isinstance(expected, dict):
                valid = isinstance(observed, dict)
            if not valid:
                raise DatabasePortalBridgeError(
                    "database Portal nested task state has malformed " + name
                )
        for name in (
            "implementation_attempts",
            "implementation_attempts_by_cid",
            "protected_implementation_attempts",
        ):
            population = payload[name]
            if any(
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in population.items()
            ):
                raise DatabasePortalBridgeError(
                    "database Portal nested task state has malformed " + name
                )
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in payload["task_statuses"].items()
        ):
            raise DatabasePortalBridgeError(
                "database Portal nested task state has malformed task_statuses"
            )
        for name in ("task_artifacts", "task_validation"):
            population = payload[name]
            if any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, list)
                or any(not isinstance(item, str) for item in value)
                for key, value in population.items()
            ):
                raise DatabasePortalBridgeError(
                    "database Portal nested task state has malformed " + name
                )
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, Mapping)
            for key, value in payload["task_identities"].items()
        ):
            raise DatabasePortalBridgeError(
                "database Portal nested task state has malformed task_identities"
            )
        return payload, _sha256_bytes(raw)

    @staticmethod
    def _event_manifest_digest(value: Mapping[str, Any]) -> str:
        unsigned = dict(value)
        unsigned.pop("manifest_digest", None)
        return _sha256_bytes(_canonical_json(unsigned))

    @classmethod
    def _strict_sealed_event_manifest(
        cls,
        raw: bytes,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        manifest = cls._strict_json_object_bytes(
            raw,
            authority="event manifest",
        )
        if (
            set(manifest) != _EVENT_MANIFEST_FIELDS
            or manifest.get("schema") != EVENT_LOG_MANIFEST_SCHEMA
            or manifest.get("active_path") != "portal-events.jsonl"
            or manifest.get("manifest_digest")
            != cls._event_manifest_digest(manifest)
            or not isinstance(manifest.get("generation"), int)
            or isinstance(manifest.get("generation"), bool)
            or int(manifest["generation"]) < 0
            or not isinstance(manifest.get("updated_at"), str)
            or not str(manifest["updated_at"])
            or not isinstance(manifest.get("stream_id"), str)
            or not str(manifest["stream_id"])
            or not isinstance(manifest.get("snapshot_id"), str)
            or not str(manifest["snapshot_id"])
            or not isinstance(manifest.get("files"), list)
        ):
            raise DatabasePortalBridgeError(
                "database Portal event manifest is not exact"
            )
        integer_fields = (
            "earliest_sequence",
            "latest_sequence",
            "active_indexed_bytes",
        )
        if any(
            isinstance(manifest.get(name), bool)
            or not isinstance(manifest.get(name), int)
            or int(manifest[name]) < 0
            for name in integer_fields
        ):
            raise DatabasePortalBridgeError(
                "database Portal event manifest counters are malformed"
            )
        records: list[dict[str, Any]] = []
        names: list[str] = []
        active_name = str(manifest["active_path"])
        rotated_pattern = re.compile(
            re.escape(active_name) + r"\.rotated-[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
        for raw_record in manifest["files"]:
            if not isinstance(raw_record, Mapping):
                raise DatabasePortalBridgeError(
                    "database Portal event manifest file record is malformed"
                )
            record = dict(raw_record)
            name = str(record.get("path") or "")
            if (
                set(record) != _EVENT_MANIFEST_FILE_FIELDS
                or (name != active_name and rotated_pattern.fullmatch(name) is None)
                or record.get("canonical_events") is not True
            ):
                raise DatabasePortalBridgeError(
                    "database Portal event manifest file record is not exact"
                )
            for field_name in (
                "size_bytes",
                "event_count",
                "first_sequence",
                "last_sequence",
                "device",
                "inode",
                "mtime_ns",
            ):
                observed = record.get(field_name)
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, int)
                    or int(observed) < 0
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal event manifest file counter is malformed"
                    )
            if not isinstance(record.get("start_previous_event_id"), str):
                raise DatabasePortalBridgeError(
                    "database Portal event manifest chain anchor is malformed"
                )
            offsets = record.get("offset_index")
            if not isinstance(offsets, list) or any(
                not isinstance(item, list)
                or len(item) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in item
                )
                for item in offsets
            ):
                raise DatabasePortalBridgeError(
                    "database Portal event manifest offset index is malformed"
                )
            physical_digest = str(record.get("sha256") or "")
            if physical_digest and not re.fullmatch(
                r"[0-9a-f]{64}", physical_digest
            ):
                raise DatabasePortalBridgeError(
                    "database Portal event source digest is malformed"
                )
            if name != active_name and not physical_digest:
                raise DatabasePortalBridgeError(
                    "database Portal archived event source lacks its digest"
                )
            names.append(name)
            records.append(record)
        expected_names = sorted(name for name in names if name != active_name) + [
            active_name
        ]
        if (
            not records
            or len(records) > 64
            or names != expected_names
            or len(set(names)) != len(names)
            or int(manifest["earliest_sequence"]) != 1
            or int(manifest["latest_sequence"]) < 1
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(manifest.get("last_event_id") or ""),
            )
        ):
            raise DatabasePortalBridgeError(
                "database Portal event manifest population is incomplete"
            )
        return manifest, records

    @staticmethod
    def _successful_submodule_cleanup_shape(value: Any) -> bool:
        """Validate the producer's closed, recursively successful schema.

        This validator is intentionally suitable only for negative provider
        evidence.  It accepts no cleanup claim with an error, an incomplete
        child, an unknown field, or a malformed scalar.  The modest depth and
        population bounds keep replay of an owner-private but corrupt journal
        deterministic; the cleanup producer normally caps nesting at ten.
        """

        if not isinstance(value, list):
            return False
        pending: list[tuple[list[Any], int, str]] = [(value, 0, "")]
        population = 0
        paths: set[str] = set()
        while pending:
            records, depth, parent_path = pending.pop()
            if depth > 32:
                return False
            population += len(records)
            if population > 4096:
                return False
            for record in records:
                if (
                    not isinstance(record, Mapping)
                    or set(record) != _SUBMODULE_CLEANUP_RECORD_FIELDS
                ):
                    return False
                path = str(record.get("path") or "")
                branch = str(record.get("branch") or "")
                canonical_path = PurePosixPath(path).as_posix()
                if (
                    not isinstance(record.get("path"), str)
                    or not path
                    or path != canonical_path
                    or PurePosixPath(path).is_absolute()
                    or any(
                        part in {"", ".", ".."}
                        for part in PurePosixPath(path).parts
                    )
                    or "\\" in path
                    or "\x00" in path
                    or (parent_path and not path.startswith(parent_path + "/"))
                    or path in paths
                    or not isinstance(record.get("branch"), str)
                    or not branch
                    or any(
                        not isinstance(record.get(field), bool)
                        for field in (
                            "removed_worktree",
                            "deleted_branch",
                            "cleaned",
                            "independent_checkout",
                        )
                    )
                    or record.get("cleaned") is not True
                    or record.get("errors") != []
                    or not isinstance(
                        record.get("nested_submodule_cleanup"), list
                    )
                ):
                    return False
                paths.add(path)
                pending.append(
                    (
                        record["nested_submodule_cleanup"],
                        depth + 1,
                        path,
                    )
                )
        return True

    def _successful_submodule_cleanup(
        self,
        value: Any,
        *,
        outer_branch: str,
    ) -> bool:
        """Bind successful cleanup records to the exact Portal producer.

        A closed record with an arbitrary path or branch is not proof that the
        configured Portal cleanup route produced it.  Root entries therefore
        match the launch-time submodule authority exactly, while every nested
        branch is derived from the outer implementation branch and full path
        by the producer's current deterministic helper.  Branch collisions
        remain valid because distinct paths can sanitize to the same ref.
        """

        if not self._successful_submodule_cleanup_shape(value):
            return False
        if not value:
            return True
        if not self.worktree_submodule_paths:
            return False
        if tuple(record["path"] for record in value) != self.worktree_submodule_paths:
            return False
        from .implementation_daemon import PortalImplementationDaemon

        pending = list(value)
        while pending:
            record = pending.pop()
            path = str(record["path"])
            if record["branch"] != PortalImplementationDaemon._submodule_worktree_branch_name(
                outer_branch,
                path,
            ):
                return False
            pending.extend(record["nested_submodule_cleanup"])
        return True

    @staticmethod
    def _validate_no_provider_event_shape(event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        expected = _NO_PROVIDER_EVENT_FIELDS.get(event_type)
        variants = _SETUP_EVENT_FIELD_VARIANTS.get(event_type)
        keys = frozenset(str(key) for key in event)
        if (expected is None or keys != expected) and (
            variants is None or keys not in variants
        ):
            raise DatabasePortalBridgeError(
                "database Portal no-provider event schema is not closed"
            )
        if (
            not isinstance(event.get("type"), str)
            or not isinstance(event.get("timestamp"), str)
            or not str(event.get("timestamp") or "")
            or not isinstance(event.get("stream_id"), str)
            or not isinstance(event.get("snapshot_id"), str)
            or isinstance(event.get("sequence"), bool)
            or not isinstance(event.get("sequence"), int)
            or not isinstance(event.get("previous_event_id"), str)
            or not isinstance(event.get("event_id"), str)
        ):
            raise DatabasePortalBridgeError(
                "database Portal no-provider event envelope is malformed"
            )
        if event_type == "dirty_submodule_reset_deferred":
            reset_fields = {
                "path",
                "reset_ok",
                "update_ok",
                "preserved",
                "reason",
                "dirty_paths",
            }
            reset = event.get("reset")
            preservation = event.get("generated_artifact_preservation")
            if (
                not isinstance(event.get("attempted"), bool)
                or isinstance(event.get("dirty_count"), bool)
                or not isinstance(event.get("dirty_count"), int)
                or int(event["dirty_count"]) < 0
                or not isinstance(reset, list)
                or len(reset) != int(event["dirty_count"])
                or not isinstance(preservation, list)
                or any(not isinstance(item, Mapping) for item in preservation)
                or any(
                    not isinstance(item, Mapping)
                    or set(item) != reset_fields
                    or not isinstance(item.get("path"), str)
                    or not str(item.get("path") or "")
                    or item.get("reset_ok") is not False
                    or item.get("update_ok") is not False
                    or item.get("preserved") is not True
                    or item.get("reason") != "non_destructive_reconciliation"
                    or not isinstance(item.get("dirty_paths"), list)
                    or any(
                        not isinstance(path, str) or not path
                        for path in item["dirty_paths"]
                    )
                    for item in reset
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal dirty-submodule diagnostic is malformed"
                )
        elif event_type == "task_selected":
            if any(
                not isinstance(event.get(name), str)
                or not str(event.get(name) or "")
                for name in (
                    "task_id",
                    "title",
                    "track",
                    "canonical_task_key",
                    "canonical_task_cid",
                    "board_namespace",
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal task-selected event is malformed"
                )
        elif event_type == "nested_submodule_initialization_guarded":
            if "depth" in event:
                string_fields = (
                    "path",
                    "relative",
                    "parent_relative",
                    "reason",
                    "path_sha256",
                    "expected_gitlink_ref_sha256",
                    "matched_identity_sha256",
                )
                integer_fields = (
                    "depth",
                    "max_depth",
                    "path_parts",
                    "max_path_parts",
                    "path_bytes",
                    "max_path_bytes",
                )
                if (
                    any(
                        not isinstance(event.get(name), str)
                        or not str(event.get(name) or "")
                        for name in string_fields
                    )
                    or any(
                        isinstance(event.get(name), bool)
                        or not isinstance(event.get(name), int)
                        or int(event[name]) < 0
                        for name in integer_fields
                    )
                    or not isinstance(
                        event.get("expected_gitlink_ref_available"), bool
                    )
                    or any(
                        re.fullmatch(r"[0-9a-f]{64}", str(event[name]))
                        is None
                        for name in (
                            "path_sha256",
                            "expected_gitlink_ref_sha256",
                            "matched_identity_sha256",
                        )
                    )
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal nested-submodule diagnostic is malformed"
                    )
            elif (
                not isinstance(event.get("reason"), str)
                or not str(event.get("reason") or "")
                or not isinstance(event.get("source_key"), str)
                or not str(event.get("source_key") or "")
                or isinstance(event.get("fallback_returncode"), bool)
                or not isinstance(event.get("fallback_returncode"), int)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal nested-submodule diagnostic is malformed"
                )
        elif event_type == "local_submodule_source_discovered":
            if any(
                not isinstance(event.get(name), str)
                or not str(event.get(name) or "")
                for name in (
                    "repo_root",
                    "source_root",
                    "source_key",
                    "source",
                    "expected_ref",
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal local-source diagnostic is malformed"
                )
        elif event_type == "implementation_protected_path_snapshot_recorded":
            protected_paths = event.get("protected_paths")
            if (
                any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "task_id",
                        "workspace_path",
                        "canonical_task_key",
                        "canonical_task_cid",
                        "board_namespace",
                    )
                )
                or type(event.get("attempt")) is not int
                or int(event["attempt"]) < 1
                or not isinstance(protected_paths, list)
                or not protected_paths
                or any(
                    not isinstance(path, str)
                    or not path
                    or PurePosixPath(path).is_absolute()
                    or ".." in PurePosixPath(path).parts
                    for path in protected_paths
                )
                or len(set(protected_paths)) != len(protected_paths)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal protected snapshot event is malformed"
                )
        elif event_type == "implementation_started":
            command = event.get("command")
            outputs = event.get("outputs")
            if (
                any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "task_id",
                        "log_path",
                        "worktree_path",
                        "branch",
                        "baseline_ref",
                        "checkpoint_directory",
                        "execution_mode",
                        "canonical_task_key",
                        "canonical_task_cid",
                        "board_namespace",
                    )
                )
                or type(event.get("attempt")) is not int
                or int(event["attempt"]) < 1
                or not isinstance(command, list)
                or not command
                or any(not isinstance(item, str) or not item for item in command)
                or not isinstance(outputs, list)
                or any(not isinstance(item, str) or not item for item in outputs)
                or not isinstance(event.get("workspace_setup"), Mapping)
                or not isinstance(event.get("timeout_policy"), Mapping)
                or not isinstance(event.get("worktree_lifecycle"), Mapping)
                or not isinstance(event.get("cache_hit"), bool)
                or not isinstance(event.get("provider_dispatched"), bool)
                or any(
                    isinstance(event.get(name), bool)
                    or not isinstance(event.get(name), (int, float))
                    or float(event[name]) < 0.0
                    for name in (
                        "setup_duration_seconds",
                        "saved_duration_seconds",
                    )
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal implementation-start event is malformed"
                )
        elif event_type == "pre_implementation_kernel_evaluated":
            if (
                any(
                    not isinstance(event.get(name), str)
                    for name in (
                        "residual_packet_cid",
                        "receipt_cid",
                    )
                )
                or any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "event",
                        "task_id",
                        "disposition",
                        "reason_code",
                        "interface",
                        "canonical_task_key",
                        "canonical_task_cid",
                        "board_namespace",
                    )
                )
                or type(event.get("attempt")) is not int
                or int(event["attempt"]) < 1
                or type(event.get("provider_hook_count")) is not int
                or int(event["provider_hook_count"]) < 0
                or type(event.get("analytical_candidate_count")) is not int
                or int(event["analytical_candidate_count"]) < 0
                or not isinstance(event.get("provider_authorized"), bool)
                or not isinstance(event.get("skip_provider"), bool)
                or not isinstance(event.get("kernel_receipt"), Mapping)
                or (
                    "effective_provider_authorized" in event
                    and not isinstance(
                        event.get("effective_provider_authorized"), bool
                    )
                )
                or (
                    "effective_skip_provider" in event
                    and not isinstance(event.get("effective_skip_provider"), bool)
                )
                or (
                    "effective_reason_code" in event
                    and (
                        not isinstance(event.get("effective_reason_code"), str)
                        or not str(event.get("effective_reason_code") or "")
                    )
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal pre-implementation event is malformed"
                )
        elif event_type == "implementation_protected_path_snapshot_cleared":
            if (
                any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "task_id",
                        "reason",
                        "canonical_task_key",
                        "canonical_task_cid",
                        "board_namespace",
                    )
                )
                or type(event.get("attempt")) is not int
                or int(event["attempt"]) < 1
            ):
                raise DatabasePortalBridgeError(
                    "database Portal protected snapshot clear is malformed"
                )
        elif event_type == "worktree_pool_lease_released":
            if (
                any(
                    not isinstance(event.get(name), str)
                    for name in (
                        "invalidation_reason",
                        "handoff_reason",
                    )
                )
                or any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "worktree_path",
                        "reason",
                        "entry_id",
                        "cache_key",
                        "base_ref",
                        "base_commit",
                        "branch",
                    )
                )
                or any(
                    not isinstance(event.get(name), bool)
                    for name in (
                        "attempted",
                        "released",
                        "pooled",
                        "reused",
                        "cache_hit",
                    )
                )
                or any(
                    isinstance(event.get(name), bool)
                    or not isinstance(event.get(name), (int, float))
                    or float(event[name]) < 0.0
                    for name in (
                        "setup_seconds",
                        "estimated_seconds_saved",
                        "setup_time_saved_seconds",
                    )
                )
                or not isinstance(event.get("dependency_paths"), list)
                or not isinstance(event.get("invalidation_reasons"), list)
                or not isinstance(event.get("lifecycle_finalize"), Mapping)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal pool-release event is malformed"
                )
        elif event_type == "worktree_cleanup_fenced":
            if (
                any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "worktree_path",
                        "branch",
                        "disposition",
                        "reason",
                        "failure_kind",
                    )
                )
                or any(
                    not isinstance(event.get(name), bool)
                    for name in (
                        "allowed",
                        "attempt_consumed",
                        "provider_call_allowed",
                    )
                )
                or not isinstance(event.get("record"), Mapping)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal fenced-cleanup event is malformed"
                )
        elif event_type == "submodule_worktree_base_ref_retried":
            required_strings = (
                ("worktree_path", "source", "source_key", "bad_ref", "fallback_ref")
                if "bad_ref" in event
                else ("reason", "source_key")
            )
            if (
                any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in required_strings
                )
                or (
                    "fallback_error" in event
                    and not isinstance(event.get("fallback_error"), str)
                )
                or isinstance(event.get("fallback_returncode"), bool)
                or not isinstance(event.get("fallback_returncode"), int)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal base-ref diagnostic is malformed"
                )
        elif event_type == "cleanup_finished":
            if (
                any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "started_at",
                        "finished_at",
                        "worktree_path",
                        "branch",
                    )
                )
                or any(
                    not isinstance(event.get(name), bool)
                    for name in ("removed_worktree", "deleted_branch", "cleaned")
                )
                or not DatabasePortalExecutionBridge._successful_submodule_cleanup_shape(
                    event.get("submodule_cleanup")
                )
                or not isinstance(event.get("lifecycle_finalize"), Mapping)
            ):
                raise DatabasePortalBridgeError(
                    "database Portal cleanup event is malformed"
                )
        elif event_type == "stale_worktree_cleanup":
            removed = event.get("removed")
            skipped = event.get("skipped")
            removed_fields = {
                "age_seconds",
                "branch",
                "cleanup_result",
                "worktree_path",
            }
            skipped_fields = {"branch", "reason", "worktree_path"}
            cleanup_fields = {
                "branch",
                "cleaned",
                "deleted_branch",
                "finished_at",
                "lifecycle_finalize",
                "removed_worktree",
                "started_at",
                "submodule_cleanup",
                "worktree_path",
            }
            if (
                event.get("attempted") is not True
                or isinstance(event.get("max_age_seconds"), bool)
                or not isinstance(event.get("max_age_seconds"), (int, float))
                or float(event["max_age_seconds"]) <= 0.0
                or type(event.get("removed_count")) is not int
                or type(event.get("skipped_count")) is not int
                or not isinstance(removed, list)
                or not isinstance(skipped, list)
                or int(event["removed_count"]) != len(removed)
                or int(event["skipped_count"]) != len(skipped)
                or any(
                    not isinstance(item, Mapping)
                    or set(item) != skipped_fields
                    or any(
                        not isinstance(item.get(name), str)
                        or (name != "branch" and not str(item.get(name) or ""))
                        for name in skipped_fields
                    )
                    for item in skipped
                )
                or any(
                    not isinstance(item, Mapping)
                    or set(item) != removed_fields
                    or isinstance(item.get("age_seconds"), bool)
                    or not isinstance(item.get("age_seconds"), (int, float))
                    or float(item["age_seconds"]) < 0.0
                    or not isinstance(item.get("branch"), str)
                    or not str(item.get("branch") or "")
                    or not isinstance(item.get("worktree_path"), str)
                    or not str(item.get("worktree_path") or "")
                    or not isinstance(item.get("cleanup_result"), Mapping)
                    or set(item["cleanup_result"]) != cleanup_fields
                    or item["cleanup_result"].get("cleaned") is not True
                    or any(
                        not isinstance(
                            item["cleanup_result"].get(name), bool
                        )
                        for name in ("removed_worktree", "deleted_branch")
                    )
                    or any(
                        not isinstance(
                            item["cleanup_result"].get(name), str
                        )
                        or not str(
                            item["cleanup_result"].get(name) or ""
                        )
                        for name in (
                            "branch",
                            "finished_at",
                            "started_at",
                            "worktree_path",
                        )
                    )
                    or not DatabasePortalExecutionBridge._successful_submodule_cleanup_shape(
                        item["cleanup_result"].get("submodule_cleanup")
                    )
                    or not _closed_typed_record_matches(
                        item["cleanup_result"].get("lifecycle_finalize"),
                        {
                            "finalized": False,
                            "reason": "no_lifecycle_record",
                        },
                    )
                    for item in removed
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal stale-worktree diagnostic is malformed"
                )
        elif event_type == "implementation_shutdown_reconciled":
            if (
                event.get("reconciled") is not True
                or event.get("blocked") is not False
                or event.get("reason") != "already_quiesced"
                or not isinstance(event.get("reconciled_at"), str)
                or not str(event.get("reconciled_at") or "")
                or event.get("task_id") != ""
                or type(event.get("attempt")) is not int
                or event.get("attempt") != 0
                or event.get("attempt_recovery") != {}
                or not _closed_typed_record_matches(
                    event.get("task_claim_reconciliation"),
                    {
                        "blocked": False,
                        "reason": "no_task_claim_identity",
                        "reconciled": False,
                    },
                )
                or not _closed_typed_record_matches(
                    event.get("protected_path_reconciliation"),
                    {
                        "blocked": False,
                        "critical_section_entered": False,
                        "reason": "no_active_snapshot",
                        "scan_outside_lease": True,
                    },
                )
                or not _closed_typed_record_matches(
                    event.get("worktree_lifecycle_reconciliation"),
                    {
                        "blocked": False,
                        "reason": "no_active_worktree",
                        "reconciled": False,
                    },
                )
                or event.get("stale_lock_cleared") is not False
            ):
                raise DatabasePortalBridgeError(
                    "database Portal quiescent-shutdown event is malformed"
                )
        elif event_type in {
            "failed_setup_worktree_cleanup",
            "implementation_exception",
            "implementation_finished",
        }:
            if (
                any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "task_id",
                        "worktree_path",
                        "branch",
                        "canonical_task_key",
                        "canonical_task_cid",
                        "board_namespace",
                    )
                )
                or isinstance(event.get("attempt"), bool)
                or not isinstance(event.get("attempt"), int)
                or int(event["attempt"]) < 1
            ):
                raise DatabasePortalBridgeError(
                    "database Portal implementation terminal event is malformed"
                )
        elif event_type == "implementation_resource_claim_lock_cleared":
            if (
                not isinstance(event.get("task_id"), str)
                or not str(event.get("task_id") or "")
                or not isinstance(event.get("lock_path"), str)
                or not str(event.get("lock_path") or "")
                or not isinstance(event.get("branch"), str)
                or isinstance(event.get("lock_owner_pid"), bool)
                or not isinstance(event.get("lock_owner_pid"), int)
                or int(event["lock_owner_pid"]) <= 0
            ):
                raise DatabasePortalBridgeError(
                    "database Portal resource-lock diagnostic is malformed"
                )
        elif event_type == "implementation_retry_deferred":
            if (
                any(
                    not isinstance(event.get(name), str)
                    or not str(event.get(name) or "")
                    for name in (
                        "task_id",
                        "reason",
                        "canonical_task_key",
                        "canonical_task_cid",
                        "board_namespace",
                    )
                )
                or not isinstance(event.get("diagnostic_receipt_id"), str)
                or any(
                    isinstance(event.get(name), bool)
                    or not isinstance(event.get(name), int)
                    or int(event[name]) < minimum
                    for name, minimum in (
                        ("attempt", 1),
                        ("backoff_seconds", 0),
                    )
                )
                or any(
                    not isinstance(event.get(name), bool)
                    for name in (
                        "skipped",
                        "attempt_consumed",
                        "provider_dispatched",
                        "active_task_cleared",
                    )
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal provider-deferral event is malformed"
                )
        elif event_type == "daemon_pass":
            integer_fields = (
                "completed_count",
                "ready_count",
                "selectable_ready_count",
                "eligible_ready_count",
                "strict_deprioritized_ready_count",
                "waiting_count",
                "blocked_count",
                "max_task_attempts",
            )
            if (
                any(
                    isinstance(event.get(name), bool)
                    or not isinstance(event.get(name), int)
                    or int(event[name]) < 0
                    for name in integer_fields
                )
                or not isinstance(event.get("active_task_id"), str)
                or not isinstance(event.get("selection_idle_reason"), str)
                or not isinstance(
                    event.get("ordinary_provider_dispatch_allowed"), bool
                )
                or not isinstance(
                    event.get("execution_slice_task_statuses"), Mapping
                )
                or not isinstance(
                    event.get("execution_slice_task_cids_by_id"), Mapping
                )
            ):
                raise DatabasePortalBridgeError(
                    "database Portal daemon-pass event is malformed"
                )

    @classmethod
    @contextmanager
    def _shared_private_event_lock(
        cls,
        directory_fd: int,
        directory_names: set[str],
    ) -> Any:
        """Share the writer's lock without creating or repairing anything."""

        lock_name = ".portal-events.jsonl.lock"
        descriptor = -1
        if lock_name in directory_names:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                descriptor = os.open(
                    lock_name,
                    flags,
                    dir_fd=directory_fd,
                )
                identity = os.fstat(descriptor)
                published = os.stat(
                    lock_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(identity.st_mode)
                    or identity.st_uid != os.geteuid()
                    or identity.st_gid != os.getegid()
                    or int(identity.st_nlink) != 1
                    or stat.S_IMODE(identity.st_mode) != 0o600
                    or cls._authority_fingerprint(identity)
                    != cls._authority_fingerprint(published)
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal event lock is not an exact private file"
                    )
                try:
                    fcntl.flock(
                        descriptor,
                        fcntl.LOCK_SH | fcntl.LOCK_NB,
                    )
                except (BlockingIOError, PermissionError) as exc:
                    raise DatabasePortalBridgeError(
                        "database Portal event lock is currently unavailable"
                    ) from exc
            except BaseException:
                if descriptor >= 0:
                    with suppress(OSError):
                        os.close(descriptor)
                raise
        try:
            yield
        finally:
            if descriptor >= 0:
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _pinned_no_provider_snapshot(
        self,
        paths: DatabasePortalAttemptPaths,
    ) -> dict[str, Any]:
        """Read a closed terminal snapshot without a repairing scanner."""
        parent_fd = authority_fd = attempt_fd = -1
        snapshot: tuple[Any, ...] = ()
        try:
            parent_fd, authority_fd, attempt_fd, snapshot = (
                self._open_pinned_private_attempt_directory(
                    authority_root=self.attempt_root,
                    attempt_key=paths.root.name,
                )
            )
            names_before = set(os.listdir(attempt_fd))
            directory_before = self._authority_fingerprint(
                os.fstat(attempt_fd)
            )
            with self._shared_private_event_lock(
                attempt_fd,
                names_before,
            ):
                child_snapshots: dict[str, tuple[int, ...]] = {}

                def read_child(name: str, maximum: int) -> bytes:
                    raw, fingerprint = self._read_exact_private_child(
                        attempt_fd,
                        name,
                        maximum_bytes=maximum,
                    )
                    child_snapshots[name] = fingerprint
                    return raw

                manifest_name = "portal-events.jsonl.manifest.json"
                manifest_raw = read_child(manifest_name, 2 * 1024 * 1024)
                manifest, records = self._strict_sealed_event_manifest(
                    manifest_raw
                )
                source_names = [str(record["path"]) for record in records]
                physical_event_names = {
                    name
                    for name in names_before
                    if name == "portal-events.jsonl"
                    or name.startswith("portal-events.jsonl.rotated-")
                }
                if physical_event_names != set(source_names):
                    raise DatabasePortalBridgeError(
                        "database Portal event source population is not exact"
                    )

                binding_raw = read_child(
                    "database-attempt-binding.json",
                    256 * 1024,
                )
                projection_raw = read_child(
                    "task-projection.md",
                    2 * 1024 * 1024,
                )
                state_raw = read_child(
                    "portal-task-state.json",
                    2 * 1024 * 1024,
                )
                binding = self._strict_json_object_bytes(
                    binding_raw,
                    authority="attempt binding",
                )
                try:
                    projection = projection_raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise DatabasePortalBridgeError(
                        "database Portal task projection is malformed"
                    ) from exc
                state, state_digest = self._strict_portal_state_bytes(
                    state_raw
                )
                source_payloads: dict[str, bytes] = {}
                total_bytes = 0
                for record in records:
                    name = str(record["path"])
                    source = read_child(name, 16 * 1024 * 1024)
                    total_bytes += len(source)
                    if total_bytes > 16 * 1024 * 1024:
                        raise DatabasePortalBridgeError(
                            "database Portal event history is oversized"
                        )
                    fingerprint = child_snapshots[name]
                    physical_digest = str(record.get("sha256") or "")
                    if (
                        len(source) != int(record["size_bytes"])
                        or fingerprint[0] != int(record["device"])
                        or fingerprint[1] != int(record["inode"])
                        or fingerprint[7] != int(record["mtime_ns"])
                        or (
                            physical_digest
                            and hashlib.sha256(source).hexdigest()
                            != physical_digest
                        )
                    ):
                        raise DatabasePortalBridgeError(
                            "database Portal event source disagrees with manifest"
                        )
                    source_payloads[name] = source

                events_by_sequence: dict[int, dict[str, Any]] = {}
                latest_sequence = 0
                latest_event_id = ""
                for record in records:
                    name = str(record["path"])
                    source_count = 0
                    source_first = 0
                    source_last = 0
                    if str(record["start_previous_event_id"]) != latest_event_id:
                        raise DatabasePortalBridgeError(
                            "database Portal event segment anchor is stale"
                        )
                    for raw_line in source_payloads[name].splitlines():
                        if not raw_line.strip():
                            continue
                        if len(raw_line) > 2 * 1024 * 1024:
                            raise DatabasePortalBridgeError(
                                "database Portal event is oversized"
                            )
                        source_count += 1
                        event = self._strict_json_object_bytes(
                            raw_line,
                            authority="event",
                        )
                        self._validate_no_provider_event_shape(event)
                        sequence = int(event["sequence"])
                        if (
                            sequence < 1
                            or event["stream_id"] != manifest["stream_id"]
                            or event["snapshot_id"] != manifest["snapshot_id"]
                        ):
                            raise DatabasePortalBridgeError(
                                "database Portal event identity is malformed"
                            )
                        unsigned = dict(event)
                        event_id = str(unsigned.pop("event_id"))
                        if event_id != _sha256_bytes(_canonical_json(unsigned)):
                            raise DatabasePortalBridgeError(
                                "database Portal event identity does not verify"
                            )
                        source_first = source_first or sequence
                        source_last = sequence
                        known = events_by_sequence.get(sequence)
                        if known is not None:
                            if known != event:
                                raise DatabasePortalBridgeError(
                                    "database Portal event sequence conflicts"
                                )
                            continue
                        if (
                            sequence != latest_sequence + 1
                            or event["previous_event_id"] != latest_event_id
                        ):
                            raise DatabasePortalBridgeError(
                                "database Portal event chain is broken"
                            )
                        events_by_sequence[sequence] = event
                        latest_sequence = sequence
                        latest_event_id = event_id
                        if latest_sequence > 4096:
                            raise DatabasePortalBridgeError(
                                "database Portal event history is oversized"
                            )
                    if (
                        source_count != int(record["event_count"])
                        or source_first != int(record["first_sequence"])
                        or source_last != int(record["last_sequence"])
                    ):
                        raise DatabasePortalBridgeError(
                            "database Portal event segment population is stale"
                        )
                expected_latest = int(manifest["latest_sequence"])
                events = [
                    events_by_sequence[index]
                    for index in range(1, expected_latest + 1)
                    if index in events_by_sequence
                ]
                active_record = records[-1]
                if (
                    str(active_record["path"]) != manifest["active_path"]
                    or int(active_record["size_bytes"])
                    != int(manifest["active_indexed_bytes"])
                    or len(events) != expected_latest
                    or latest_sequence != expected_latest
                    or latest_event_id != manifest["last_event_id"]
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal event population is incomplete"
                    )
                terminal_log: dict[str, Any] = {}
                started = [
                    event
                    for event in events
                    if event.get("type") == "implementation_started"
                ]
                legacy_pre_kernel = [
                    event
                    for event in events
                    if event.get("type")
                    == "pre_implementation_kernel_evaluated"
                    and frozenset(event)
                    == _NO_PROVIDER_EVENT_FIELDS[
                        "pre_implementation_kernel_evaluated"
                    ]
                ]
                legacy_suffix = [
                    "implementation_protected_path_snapshot_recorded",
                    "implementation_started",
                    "pre_implementation_kernel_evaluated",
                    "implementation_protected_path_snapshot_cleared",
                    "worktree_pool_lease_released",
                    "implementation_finished",
                    "daemon_pass",
                ]
                read_legacy_log = bool(
                    len(events) >= len(legacy_suffix)
                    and [
                        str(event.get("type") or "")
                        for event in events[-len(legacy_suffix) :]
                    ]
                    == legacy_suffix
                    and len(legacy_pre_kernel) == 1
                )
                if read_legacy_log:
                    finished = [
                        event
                        for event in events
                        if event.get("type") == "implementation_finished"
                    ]
                    if len(started) != 1 or len(finished) != 1:
                        raise DatabasePortalBridgeError(
                            "database Portal terminal no-effect attempt population "
                            "is ambiguous"
                        )
                    task_id = str(started[0].get("task_id") or "")
                    nested_attempt = started[0].get("attempt")
                    if (
                        re.fullmatch(r"PCTDD-[0-9]{3}", task_id) is None
                        or type(nested_attempt) is not int
                        or nested_attempt < 1
                        or finished[0].get("task_id") != task_id
                        or finished[0].get("attempt") != nested_attempt
                    ):
                        raise DatabasePortalBridgeError(
                            "database Portal terminal no-effect attempt identity "
                            "is malformed"
                        )
                    log_name = (
                        f"{task_id.lower()}-attempt-{nested_attempt}.log"
                    )
                    log_raw, log_fingerprint, log_identity_digest = (
                        self._read_exact_terminal_no_effect_log(
                            attempt_fd,
                            log_name,
                        )
                    )
                    terminal_log = {
                        "terminal_no_effect_log_raw": log_raw,
                        "terminal_no_effect_log_relative_path": (
                            f"implementation-logs/{log_name}"
                        ),
                        "terminal_no_effect_log_fingerprint": log_fingerprint,
                        "terminal_no_effect_log_identity_digest": (
                            log_identity_digest
                        ),
                    }
                if terminal_log:
                    self._revalidate_exact_terminal_no_effect_log(
                        attempt_fd,
                        Path(
                            str(
                                terminal_log[
                                    "terminal_no_effect_log_relative_path"
                                ]
                            )
                        ).name,
                        expected_raw=terminal_log[
                            "terminal_no_effect_log_raw"
                        ],
                        expected_fingerprint=terminal_log[
                            "terminal_no_effect_log_fingerprint"
                        ],
                        expected_identity_digest=terminal_log[
                            "terminal_no_effect_log_identity_digest"
                        ],
                    )
                self._revalidate_private_children(
                    attempt_fd,
                    child_snapshots,
                )
                if (
                    set(os.listdir(attempt_fd)) != names_before
                    or self._authority_fingerprint(os.fstat(attempt_fd))
                    != directory_before
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal attempt snapshot changed during replay"
                    )
                return {
                    "binding": binding,
                    "projection": projection,
                    "state": state,
                    "state_digest": state_digest,
                    "events": events,
                    "manifest": manifest,
                    "directory_names": sorted(names_before),
                    **terminal_log,
                }
        finally:
            try:
                if snapshot:
                    self._verify_pinned_private_attempt_directory(
                        parent_fd=parent_fd,
                        authority_fd=authority_fd,
                        attempt_fd=attempt_fd,
                        snapshot=snapshot,
                    )
            finally:
                for descriptor in (attempt_fd, authority_fd, parent_fd):
                    if descriptor >= 0:
                        with suppress(OSError):
                            os.close(descriptor)

    def _interrupted_implementation_rearm_evidence(
        self,
        attempt: Any,
        receipt: Mapping[str, Any],
        *,
        expected_evidence_schema: str | None = None,
    ) -> dict[str, Any] | None:
        """Prove exact implementing-crash recovery without redispatch/completion.

        This successor proof is intentionally distinct from the setup-failure
        evidence below.  It binds the immutable first-clear receipt, the
        callback-free interrupted-retry adapter, its exact claim-release
        receipt, and the outer prepared reconciliation object.  Replaying the
        adapter is idempotent and may only finish that already-authorized
        control recovery; it cannot invoke a provider, validation, commit, or
        merge callback.
        """

        accepted_expected_schemas = {
            DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA,
            DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA,
        }
        if (
            expected_evidence_schema is not None
            and expected_evidence_schema not in accepted_expected_schemas
        ):
            return None

        link = receipt.get("terminal_reconciliation")
        if not isinstance(link, Mapping):
            return None
        link = dict(link)
        link_fields = {
            "schema",
            "attempt_id",
            "claim_id",
            "task_cid",
            "attempt_number",
            "owner_session_id",
            "lease_id",
            "fencing_token",
            "fence_epoch",
            "binding_id",
            "nested_state_digest",
            "nested_reason",
            "nested_reconciled",
            "trigger",
            "intended_database_disposition",
            "prepared_reconciliation_receipt_id",
            "commit_barrier_receipt_id",
            "evidence_id",
        }
        exact_attempt = {
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
        }
        unsigned_link = dict(link)
        link_evidence_id = str(unsigned_link.pop("evidence_id", "") or "")
        if (
            set(link) != link_fields
            or not _strict_attempt_integer_fields(link)
            or link.get("schema")
            != (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-terminal-reconciliation-link@1"
            )
            or any(
                type(link.get(name)) is not type(value)
                or link.get(name) != value
                for name, value in exact_attempt.items()
            )
            or link.get("nested_reason") != "nested_portal_attempt_reconciled"
            or link.get("nested_reconciled") is not True
            or link.get("intended_database_disposition")
            != "blocked_unknown_outcome"
            or not re.fullmatch(r"baguqeera[a-z2-7]{52}", link_evidence_id)
            or content_identity(unsigned_link) != link_evidence_id
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(link.get("prepared_reconciliation_receipt_id") or ""),
            )
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(link.get("commit_barrier_receipt_id") or ""),
            )
        ):
            return None

        paths = self._paths(attempt)
        try:
            binding = self._read_binding(paths.binding)
            self._verify_binding_identity(binding)
            durable_binding = (
                self._binding_lookup(attempt)
                if self._binding_lookup is not None
                else None
            )
            prepared = self.load_reconciliation_receipt(
                attempt,
                str(link["prepared_reconciliation_receipt_id"]),
                required_stage="prepared",
            )
            commit_barrier = self.load_reconciliation_receipt(
                attempt,
                str(link["commit_barrier_receipt_id"]),
                required_stage="commit_barrier",
            )
            source = (
                None
                if expected_evidence_schema
                == DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA
                else self._interrupted_implementation_retry_evidence(
                    attempt,
                    binding,
                )
            )
            migration_source = None
            if source is None and expected_evidence_schema != (
                DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
            ):
                migration_source = (
                    self._stale_dispatch_migration_retry_evidence(
                        attempt,
                        binding,
                    )
                )
                source = migration_source
            state_before, state_digest_before = self._strict_state_record(
                paths.state
            )
        except (DatabasePortalBridgeError, OSError, TypeError, ValueError):
            return None
        if source is None or not isinstance(durable_binding, Mapping):
            return None
        stale_dispatch_migration = migration_source is not None
        durable_expected = {
            **exact_attempt,
            "binding_id": str(binding.get("binding_id") or ""),
            "projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
            "stage": "portal_entered",
        }
        current_nested = prepared.get("nested_state")
        recovery = prepared.get("portal_reconciliation")
        provider_fence = prepared.get("provider_runner_fence")
        prepared_barrier_core = {
            name: value
            for name, value in prepared.items()
            if name not in {"stage", "receipt_id"}
        }
        commit_barrier_core = {
            name: value
            for name, value in commit_barrier.items()
            if name
            not in {
                "stage",
                "receipt_id",
                "prepared_reconciliation_receipt_id",
            }
        }
        if (
            any(
                type(durable_binding.get(name)) is not type(value)
                or durable_binding.get(name) != value
                for name, value in durable_expected.items()
            )
            or prepared.get("binding_id") != binding.get("binding_id")
            or prepared.get("intended_database_disposition")
            != "blocked_unknown_outcome"
            or prepared.get("reason") != "nested_portal_attempt_reconciled"
            or prepared.get("reconciled") is not True
            or prepared.get("blocked") is not False
            or prepared.get("receipt_id")
            != link.get("prepared_reconciliation_receipt_id")
            or prepared.get("trigger") != link.get("trigger")
            or link.get("binding_id") != prepared.get("binding_id")
            or link.get("nested_reason") != prepared.get("reason")
            or link.get("nested_reconciled")
            is not prepared.get("reconciled")
            or not isinstance(current_nested, Mapping)
            or current_nested.get("present") is not True
            or current_nested.get("active") is not False
            or current_nested.get("active_task_id") != ""
            or type(current_nested.get("active_attempt")) is not int
            or current_nested.get("active_attempt") != 0
            or current_nested.get("active_phase") != ""
            or current_nested.get("state_path") != str(paths.state)
            or (
                not stale_dispatch_migration
                and current_nested.get("state_digest") != state_digest_before
            )
            or link.get("nested_state_digest")
            != current_nested.get("state_digest")
            or commit_barrier.get("receipt_id")
            != link.get("commit_barrier_receipt_id")
            or commit_barrier.get("prepared_reconciliation_receipt_id")
            != prepared.get("receipt_id")
            or _canonical_json(commit_barrier_core)
            != _canonical_json(prepared_barrier_core)
            or commit_barrier.get("intended_database_disposition")
            != "blocked_unknown_outcome"
            or commit_barrier.get("trigger") != link.get("trigger")
            or not isinstance(state_before, Mapping)
            or state_before.get("implementation_in_progress") is not False
            or state_before.get("active_task_id") != ""
            or type(state_before.get("active_attempt")) is not int
            or state_before.get("active_attempt") != 0
            or state_before.get("active_phase") != ""
            or not isinstance(provider_fence, Mapping)
            or provider_fence.get("safe_to_restart") is not True
            or provider_fence.get("applicable") is not False
            or provider_fence.get("fenced") is not False
            or provider_fence.get("reason")
            != "ordinary_provider_runner_receipt_absent"
            or not isinstance(recovery, Mapping)
        ):
            return None

        source_receipt = source.get("reconciliation_receipt")
        source_evidence_id = str(source.get("evidence_id") or "")
        if (
            not isinstance(source_receipt, Mapping)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_evidence_id)
            or source_receipt.get("binding_id") != binding.get("binding_id")
        ):
            return None

        daemon = self.portal_factory(
            paths,
            str(binding.get("task_alias") or attempt.task_cid),
        )
        replay: dict[str, Any] = {}
        try:
            reconcile = getattr(
                daemon,
                (
                    "reconcile_stale_dispatch_release_migration"
                    if stale_dispatch_migration
                    else "reconcile_interrupted_database_implementation_attempt"
                ),
                None,
            )
            if not callable(reconcile):
                return None
            raw_replay = (
                reconcile(
                    source,
                    expected_pre_state_digest=str(
                        current_nested.get("state_digest") or ""
                    ),
                )
                if stale_dispatch_migration
                else reconcile(source)
            )
            if not isinstance(raw_replay, Mapping):
                return None
            replay = dict(raw_replay)
        except (DatabasePortalBridgeError, OSError, TypeError, ValueError):
            return None
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(
                daemon,
                "close",
                None,
            )
            if callable(close):
                close()
        try:
            state_after, state_digest_after = self._strict_state_record(paths.state)
        except (DatabasePortalBridgeError, OSError, TypeError, ValueError):
            return None
        if (
            not stale_dispatch_migration
            and (
                state_after != state_before
                or state_digest_after != state_digest_before
            )
        ):
            return None

        if stale_dispatch_migration:
            claim_release = recovery.get("task_claim_reconciliation")
            forbidden_terminal = recovery.get(
                "provider_forbidden_terminal_recovery"
            )
            source_nested = source_receipt.get("nested_state")
            source_portal = source_receipt.get("portal_reconciliation")
            source_claim = (
                source_portal.get("task_claim_reconciliation")
                if isinstance(source_portal, Mapping)
                else None
            )
            replay_lock_cleared = replay.get("stale_lock_cleared")
            replay_lock_event_id = replay.get("stale_lock_clear_event_id")
            if (
                set(replay) != _STALE_DISPATCH_MIGRATION_REPLAY_FIELDS
                or replay.get("reconciled") is not True
                or replay.get("blocked") is not False
                or replay.get("reason")
                != "stale_dispatch_release_migrated_for_retry"
                or replay.get("task_id") != binding.get("task_alias")
                or not isinstance(source_nested, Mapping)
                or not isinstance(source_claim, Mapping)
                or replay.get("canonical_task_cid")
                != source_claim.get("canonical_task_cid")
                or not re.fullmatch(
                    r"baguqeera[a-z2-7]{52}",
                    str(replay.get("canonical_task_cid") or ""),
                )
                or type(replay.get("attempt")) is not int
                or replay.get("attempt") != source_nested.get("active_attempt")
                or int(replay.get("attempt") or 0) < 1
                or any(
                    not re.fullmatch(
                        r"baguqeera[a-z2-7]{52}",
                        str(replay.get(name) or ""),
                    )
                    for name in (
                        "migration_id",
                        "migration_receipt_id",
                        "legacy_claim_release_receipt_id",
                    )
                )
                or any(
                    not re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(replay.get(name) or ""),
                    )
                    for name in (
                        "preparation_event_id",
                        "state_recovery_event_id",
                        "migration_terminal_event_id",
                        "legacy_claim_release_event_id",
                        "pre_state_digest",
                        "post_state_digest",
                    )
                )
                or type(replay_lock_cleared) is not bool
                or (
                    replay_lock_cleared is True
                    and not re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(replay_lock_event_id or ""),
                    )
                )
                or (
                    replay_lock_cleared is False
                    and replay_lock_event_id != ""
                )
                or replay.get("provider_dispatched") is not False
                or replay.get("implementation_dispatched") is not False
                or replay.get("acceptance_inferred") is not False
                or replay.get("retained_candidate_disposition")
                != "preserved_unvalidated"
                or replay.get("pre_state_digest")
                != current_nested.get("state_digest")
                or replay.get("post_state_digest") != state_digest_after
                or not isinstance(claim_release, Mapping)
                or claim_release.get("reconciled") is not True
                or claim_release.get("blocked") is not False
                or claim_release.get("reason")
                != "quiesced_task_claim_released"
                or claim_release.get("task_id") != binding.get("task_alias")
                or claim_release.get("task_status") != "todo"
                or claim_release.get(
                    "stale_dispatch_intent_released_for_retry"
                )
                is not True
                or claim_release.get("receipt_id")
                != replay.get("legacy_claim_release_receipt_id")
                or recovery.get("reconciled") is not True
                or recovery.get("blocked") is not False
                or recovery.get("reason") != "already_quiesced"
                or recovery.get("task_id") != binding.get("task_alias")
                or not _closed_typed_record_matches(
                    forbidden_terminal,
                    {
                        "applicable": False,
                        "blocked": False,
                        "implementation_dispatched": False,
                        "provider_dispatched": False,
                        "reason": (
                            "provider_forbidden_terminal_recovery_not_applicable"
                        ),
                        "reconciled": False,
                    },
                )
            ):
                return None

            authorization = {
                "schema": (
                    DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_AUTHORIZATION_SCHEMA
                ),
                **exact_attempt,
                "binding_id": str(binding.get("binding_id") or ""),
                "binding_admission_id": str(
                    durable_binding.get("record_id") or ""
                ),
                "binding_admission_digest": _sha256_bytes(
                    _canonical_json(dict(durable_binding))
                ),
                "projection_immutable_digest": str(
                    binding.get("projection_immutable_digest") or ""
                ),
                "nested_task_cid": str(
                    replay.get("canonical_task_cid") or ""
                ),
                "nested_attempt": replay.get("attempt"),
                "terminal_reconciliation_evidence_id": link_evidence_id,
                "first_clear_receipt_id": str(
                    source_receipt.get("receipt_id") or ""
                ),
                "migration_retry_evidence_id": source_evidence_id,
                "migration_id": str(replay.get("migration_id") or ""),
                "migration_preparation_event_id": str(
                    replay.get("preparation_event_id") or ""
                ),
                "state_recovery_event_id": str(
                    replay.get("state_recovery_event_id") or ""
                ),
                "migration_terminal_event_id": str(
                    replay.get("migration_terminal_event_id") or ""
                ),
                "migration_receipt_id": str(
                    replay.get("migration_receipt_id") or ""
                ),
                "legacy_claim_release_receipt_id": str(
                    replay.get("legacy_claim_release_receipt_id") or ""
                ),
                "legacy_claim_release_event_id": str(
                    replay.get("legacy_claim_release_event_id") or ""
                ),
                "stale_lock_cleared": replay_lock_cleared,
                "stale_lock_clear_event_id": str(
                    replay_lock_event_id or ""
                ),
                "prepared_reconciliation_receipt_id": str(
                    prepared.get("receipt_id") or ""
                ),
                "commit_barrier_receipt_id": str(
                    commit_barrier.get("receipt_id") or ""
                ),
                "pre_state_digest": str(
                    replay.get("pre_state_digest") or ""
                ),
                "state_digest": state_digest_after,
                "outer_block_receipt_digest": _sha256_bytes(
                    _canonical_json(dict(receipt))
                ),
            }
            authorization_id = _sha256_bytes(_canonical_json(authorization))
            migration_evidence: dict[str, Any] = {
                "schema": (
                    DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA
                ),
                **exact_attempt,
                "task_alias": str(binding.get("task_alias") or ""),
                "attempt_root_key": paths.root.name,
                "attempt_authority_root_digest": _sha256_bytes(
                    str(self.attempt_root).encode("utf-8")
                ),
                "attempt_root_digest": _sha256_bytes(
                    str(paths.root).encode("utf-8")
                ),
                **{
                    name: value
                    for name, value in authorization.items()
                    if name
                    not in {
                        "schema",
                        *exact_attempt,
                    }
                },
                "rearm_authorization_id": authorization_id,
                "provider_dispatched": False,
                "implementation_dispatched": False,
                "validation_attempted": False,
                "commit_created": False,
                "merge_attempted": False,
                "acceptance_inferred": False,
                "recovery_terminal": True,
                "retained_candidate_disposition": "preserved_unvalidated",
            }
            migration_evidence["evidence_id"] = _sha256_bytes(
                _canonical_json(migration_evidence)
            )
            return migration_evidence

        recovery_core = dict(recovery)
        forbidden_terminal = recovery_core.pop(
            "provider_forbidden_terminal_recovery",
            None,
        )
        original_lock_clear = recovery_core.pop("stale_lock_cleared", None)
        replay_core = dict(replay)
        replay_lock_clear = replay_core.pop("stale_lock_cleared", None)
        claim_release = replay.get("task_claim_reconciliation")
        released_attempt = (
            claim_release.get("released_unfinished_attempt")
            if isinstance(claim_release, Mapping)
            else None
        )
        replay_attempt = replay.get("attempt")
        released_from = (
            released_attempt.get("released_from")
            if isinstance(released_attempt, Mapping)
            else None
        )
        released_to = (
            released_attempt.get("released_to")
            if isinstance(released_attempt, Mapping)
            else None
        )
        if (
            _canonical_json(recovery_core) != _canonical_json(replay_core)
            or not isinstance(original_lock_clear, bool)
            or not isinstance(replay_lock_clear, bool)
            or not _closed_typed_record_matches(
                forbidden_terminal,
                {
                "applicable": False,
                "blocked": False,
                "implementation_dispatched": False,
                "provider_dispatched": False,
                "reason": "provider_forbidden_terminal_recovery_not_applicable",
                "reconciled": False,
                },
            )
            or replay.get("reconciled") is not True
            or replay.get("blocked") is not False
            or replay.get("reason")
            != "interrupted_implementation_recovered_for_retry"
            or replay.get("provider_dispatched") is not False
            or replay.get("implementation_dispatched") is not False
            or replay.get("acceptance_inferred") is not False
            or replay.get("retained_candidate_disposition")
            != "preserved_unvalidated"
            or not isinstance(claim_release, Mapping)
            or claim_release.get("reconciled") is not True
            or claim_release.get("blocked") is not False
            or claim_release.get("reason") != "quiesced_task_claim_released"
            or claim_release.get("task_id") != binding.get("task_alias")
            or claim_release.get("task_status") != "todo"
            or not isinstance(released_attempt, Mapping)
            or type(replay_attempt) is not int
            or int(replay_attempt) < 1
            or type(released_from) is not int
            or released_from != replay_attempt
            or type(released_to) is not int
            or released_to != replay_attempt - 1
        ):
            return None

        authorization = {
            "schema": (
                DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_AUTHORIZATION_SCHEMA
            ),
            **exact_attempt,
            "binding_id": str(binding.get("binding_id") or ""),
            "binding_admission_id": str(
                durable_binding.get("record_id") or ""
            ),
            "binding_admission_digest": _sha256_bytes(
                _canonical_json(dict(durable_binding))
            ),
            "projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
            "nested_task_cid": str(replay.get("canonical_task_cid") or ""),
            "nested_attempt": replay_attempt,
            "terminal_reconciliation_evidence_id": link_evidence_id,
            "first_clear_receipt_id": str(
                source_receipt.get("receipt_id") or ""
            ),
            "interrupted_retry_evidence_id": source_evidence_id,
            "interrupted_retry_id": str(
                claim_release.get("released_unfinished_retry_id") or ""
            ),
            "state_recovery_event_id": str(
                released_attempt.get("event_id") or ""
            ),
            "claim_release_receipt_id": str(
                claim_release.get("receipt_id") or ""
            ),
            "prepared_reconciliation_receipt_id": str(
                prepared.get("receipt_id") or ""
            ),
            "commit_barrier_receipt_id": str(
                commit_barrier.get("receipt_id") or ""
            ),
            "state_digest": state_digest_after,
            "outer_block_receipt_digest": _sha256_bytes(
                _canonical_json(dict(receipt))
            ),
        }
        rearm_authorization_id = _sha256_bytes(_canonical_json(authorization))

        evidence: dict[str, Any] = {
            "schema": (
                DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
            ),
            **exact_attempt,
            "task_alias": str(binding.get("task_alias") or ""),
            "attempt_root_key": paths.root.name,
            "attempt_authority_root_digest": _sha256_bytes(
                str(self.attempt_root).encode("utf-8")
            ),
            "attempt_root_digest": _sha256_bytes(
                str(paths.root).encode("utf-8")
            ),
            "binding_id": str(binding.get("binding_id") or ""),
            "binding_admission_id": str(
                durable_binding.get("record_id") or ""
            ),
            "binding_admission_digest": _sha256_bytes(
                _canonical_json(dict(durable_binding))
            ),
            "projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
            "nested_task_cid": str(replay.get("canonical_task_cid") or ""),
            "nested_attempt": replay_attempt,
            "terminal_reconciliation_evidence_id": link_evidence_id,
            "first_clear_receipt_id": str(
                source_receipt.get("receipt_id") or ""
            ),
            "interrupted_retry_evidence_id": source_evidence_id,
            "interrupted_retry_id": str(
                claim_release.get("released_unfinished_retry_id") or ""
            ),
            "state_recovery_event_id": str(
                released_attempt.get("event_id") or ""
            ),
            "claim_release_receipt_id": str(
                claim_release.get("receipt_id") or ""
            ),
            "prepared_reconciliation_receipt_id": str(
                prepared.get("receipt_id") or ""
            ),
            "commit_barrier_receipt_id": str(
                commit_barrier.get("receipt_id") or ""
            ),
            "state_digest": state_digest_after,
            "outer_block_receipt_digest": _sha256_bytes(
                _canonical_json(dict(receipt))
            ),
            "rearm_authorization_id": rearm_authorization_id,
            "provider_dispatched": False,
            "implementation_dispatched": False,
            "validation_attempted": False,
            "commit_created": False,
            "merge_attempted": False,
            "acceptance_inferred": False,
            "recovery_terminal": True,
            "retained_candidate_disposition": "preserved_unvalidated",
        }
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

    def _terminal_no_effect_route_log_admission(
        self,
        *,
        raw_log: bytes,
        command: Sequence[Any],
        task_alias: str,
        task_title: str,
        expected_worktree: str,
        expected_branch: str,
        expected_baseline: str,
        runner_returncode: int,
    ) -> dict[str, Any] | None:
        """Admit the one historical Grok quota/medium denied-route log.

        This proves only that the wrapper ran a typed quota preflight and
        denied the unauthorized fallback before either implementation model
        was dispatched.  In particular, it creates no fresh Codex authority.
        """

        expected_route = {
            "authorization": None,
            "fallback_implementer_identity": "codex",
            "fallback_model_id": "gpt-5.6-terra",
            "fallback_provider_id": "codex",
            "fallback_reasoning_effort": "medium",
            "fallback_trigger": "primary_quota_exhausted",
            "invocation_binding": None,
            "primary_model_id": "grok-4.6",
            "primary_provider_id": "grok_cli",
            "route_id": (
                "agent-supervisor-grok45-terra56-medium-hard-quota-v1"
            ),
        }
        try:
            text = raw_log.decode("utf-8", errors="strict")
            command_items = list(command)
            if (
                len(command_items) != 21
                or any(
                    not isinstance(item, str) or not item
                    for item in command_items
                )
                or not Path(command_items[0]).is_absolute()
                or re.fullmatch(r"python(?:3(?:[.][0-9]+)?)?", Path(command_items[0]).name)
                is None
                or command_items[1:3]
                != [
                    "-m",
                    "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
                ]
                or command_items[3::2]
                != [
                    "--workspace",
                    "--model",
                    "--max-turns",
                    "--mode",
                    "--codex-fallback-reasoning-effort",
                    "--codex-fallback-command-json",
                    "--grok-bin",
                    "--grok-failure-receipt-nonce",
                    "--agent-implementation-route-json",
                ]
            ):
                return None
            values = {
                command_items[index]: command_items[index + 1]
                for index in range(3, len(command_items), 2)
            }
            nonce = values["--grok-failure-receipt-nonce"]
            if (
                values["--workspace"] != expected_worktree
                or values["--model"] != "grok-4.6"
                or values["--max-turns"] != "100000"
                or values["--mode"] != "agent"
                or values["--codex-fallback-reasoning-effort"] != "medium"
                or not Path(values["--grok-bin"]).is_absolute()
                or Path(values["--grok-bin"]).name != "grok"
                or re.fullmatch(r"[0-9a-f]{64}", nonce) is None
            ):
                return None
            route_binding = self._strict_json_object_bytes(
                values["--agent-implementation-route-json"].encode("utf-8"),
                authority="historical implementation route",
            )
            if not _closed_typed_record_matches(
                route_binding,
                expected_route,
            ):
                return None
            route_plan = resolve_agent_implementation_route_binding(
                route_binding,
                # This version admits no authorization artifact or invocation
                # binding.  The resolver still receives an explicit local root
                # and independently reconstructs the canonical route identity.
                repo_root=expected_worktree,
            )
            if (
                route_plan.authorization is not None
                or route_plan.invocation_binding is not None
                or route_plan.as_binding_dict() != expected_route
            ):
                return None
            fallback = json.loads(
                values["--codex-fallback-command-json"],
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("nonfinite fallback command value")
                ),
            )
            if (
                not isinstance(fallback, list)
                or len(fallback) != 14
                or any(not isinstance(item, str) or not item for item in fallback)
                or not Path(fallback[0]).is_absolute()
                or Path(fallback[0]).name != "codex"
                or fallback[1:]
                != [
                    "exec",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--ephemeral",
                    "-s",
                    "workspace-write",
                    "-C",
                    expected_worktree,
                    "-m",
                    "gpt-5.6-terra",
                    "-c",
                    'model_reasoning_effort="medium"',
                    "-",
                ]
            ):
                return None
            receipt_lines = [
                line
                for line in text.split("\n")
                if line.startswith(GROK_FAILURE_RECEIPT_PREFIX)
            ]
            outcome_lines = [
                line
                for line in text.split("\n")
                if line.startswith(GROK_ROUTE_OUTCOME_PREFIX)
            ]
            receipts = extract_grok_failure_receipts(text)
            outcomes = extract_grok_route_outcomes(text)
            if (
                len(receipt_lines) != 1
                or len(outcome_lines) != 1
                or len(receipts) != 1
                or len(outcomes) != 1
            ):
                return None
            receipt = receipts[0]
            outcome = outcomes[0]
            probe_returncode = receipt.get("probe_returncode")
            if (
                type(probe_returncode) is not int
                or probe_returncode != 1
                or runner_returncode != 1
                or not valid_grok_failure_receipt(
                    receipt,
                    nonce=nonce,
                    model="grok-4.6",
                    returncode=probe_returncode,
                )
                or receipt.get("failure_class") != "hard_quota_exhausted"
                or receipt.get("primary_dispatched") is not False
                or not valid_grok_route_outcome(
                    outcome,
                    receipt=receipt,
                    route_plan=route_plan.as_outcome_dict(),
                    runner_returncode=runner_returncode,
                )
                or outcome.get("decision") != "denied"
                or outcome.get("verifier_status") != "not_run"
                or outcome.get("quota_evidence_id") != ""
                or outcome.get("fallback_dispatched") is not False
                or outcome.get("fallback_returncode") is not None
            ):
                return None
            lines = text.split("\n")
            if (
                "\r" in text
                or len(lines) != 11
                or lines[-1] != ""
                or lines[0] != f"Task: {task_alias} {task_title}"
                or not lines[1].startswith("Started: ")
                or lines[2] != f"Workspace: {expected_worktree}"
                or lines[3] != f"Branch: {expected_branch}"
                or lines[4] != f"Baseline: {expected_baseline}"
                or lines[5] != f"Command: {shlex.join(command_items)}"
                or lines[6] != ""
                or lines[7]
                != GROK_FAILURE_RECEIPT_PREFIX
                + json.dumps(
                    dict(receipt),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                or lines[8]
                != (
                    "Typed Grok preflight did not authorize fallback; "
                    "Codex fallback is forbidden"
                )
                or lines[9]
                != GROK_ROUTE_OUTCOME_PREFIX
                + json.dumps(
                    dict(outcome),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ):
                return None
            started_at = datetime.fromisoformat(lines[1][len("Started: ") :])
            if started_at.tzinfo is None:
                return None
        except (
            DatabasePortalBridgeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            OSError,
            TypeError,
            ValueError,
        ):
            return None
        return {
            "command_sha256": _sha256_bytes(
                _canonical_json(command_items)
            ),
            "route_plan_sha256": _sha256_bytes(
                _canonical_json(route_plan.as_binding_dict())
            ),
            "route_id": route_plan.route_id,
            "primary_provider": route_plan.primary_provider_id,
            "primary_model": route_plan.primary_model_id,
            "fallback_provider": route_plan.fallback_provider_id,
            "fallback_model": route_plan.fallback_model_id,
            "fallback_reasoning_effort": (
                route_plan.fallback_reasoning_effort
            ),
            "quota_probe_receipt_id": str(receipt["receipt_id"]),
            "quota_probe_receipt_digest": _sha256_bytes(
                _canonical_json(dict(receipt))
            ),
            "route_outcome_id": str(outcome["outcome_id"]),
            "route_outcome_digest": _sha256_bytes(
                _canonical_json(dict(outcome))
            ),
            "failure_class": str(receipt["failure_class"]),
            "verifier_status": str(outcome["verifier_status"]),
            "runner_returncode": runner_returncode,
        }

    def _terminal_no_effect_route_rearm_evidence(
        self,
        attempt: Any,
        *,
        receipt: Mapping[str, Any],
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        durable_binding: Mapping[str, Any],
        identity: Mapping[str, str],
        projection_track: str,
        projection_status: str,
        directory_names: Sequence[str],
        state: Mapping[str, Any],
        state_digest: str,
        events: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        log_raw: bytes | None,
        log_relative_path: str,
        log_identity_digest: str,
    ) -> Mapping[str, Any] | None:
        """Prove one closed predecessor attempt consumed no model effect."""

        base_names = {
            ".implementation.lock.update.lock",
            ".portal-events.jsonl.lock",
            "database-attempt-binding.json",
            "implementation-logs",
            "implementation_checkpoints",
            "portal-events.jsonl",
            "portal-events.jsonl.manifest.json",
            "portal-strategy.json",
            "portal-task-state.json",
            "task-projection.md",
            "task_queue.json",
        }
        checkpoint_names = {
            ".portal-task-state.event-driven-checkpoint.json.lock",
            "portal-task-state.event-driven-checkpoint.json",
        }
        observed_names = frozenset(directory_names)
        if (
            projection_status != "ready"
            or observed_names not in {frozenset(base_names), frozenset(base_names | checkpoint_names)}
            or log_raw is None
            or not log_relative_path
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", log_identity_digest)
        ):
            return None

        task_alias = str(binding.get("task_alias") or "")
        nested_task_cid = str(identity.get("canonical_task_cid") or "")
        canonical_task_key = str(identity.get("canonical_task_key") or "")
        board_namespace = str(identity.get("board_namespace") or "")
        task_title = str(identity.get("title") or "")
        if (
            task_alias not in {"PCTDD-005", "PCTDD-006", "PCTDD-007", "PCTDD-034"}
            or board_namespace
            != "parallel-content-sealing-proof-carrying-tdd-v1"
            or not nested_task_cid
            or not canonical_task_key
            or not task_title
            or type(binding.get("task_revision")) is not int
            or int(binding["task_revision"]) < 0
        ):
            return None

        selected_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("type") == "task_selected"
            and event.get("task_id") == task_alias
            and event.get("canonical_task_cid") == nested_task_cid
        ]
        suffix_types = [
            "implementation_protected_path_snapshot_recorded",
            "implementation_started",
            "pre_implementation_kernel_evaluated",
            "implementation_protected_path_snapshot_cleared",
            "worktree_pool_lease_released",
            "implementation_finished",
            "daemon_pass",
        ]
        if (
            len(selected_indexes) != 1
            or len(events) < len(suffix_types) + 1
            or [str(event.get("type") or "") for event in events[-7:]]
            != suffix_types
            or sum(event.get("type") == "task_selected" for event in events) != 1
        ):
            return None
        selected_index = selected_indexes[0]
        selected_event = events[selected_index]
        recorded, started, pre_kernel, cleared, released, finished, daemon_pass = (
            events[-7:]
        )
        prelude = list(events[:selected_index])
        diagnostic_events = list(events[selected_index + 1 : -7])
        expected_diagnostic_types = [
            "nested_submodule_initialization_guarded",
        ] * 5 + ["local_submodule_source_discovered"] * 7 + [
            "nested_submodule_initialization_guarded"
        ]
        if task_alias in {"PCTDD-005", "PCTDD-034"}:
            expected_diagnostic_types.insert(
                0, "implementation_resource_claim_lock_cleared"
            )
        if [str(event.get("type") or "") for event in diagnostic_events] != (
            expected_diagnostic_types
        ):
            return None

        if prelude:
            expected_prelude_types = [
                "worktree_cleanup_fenced",
                "worktree_cleanup_fenced",
                "daemon_pass",
                "worktree_cleanup_fenced",
                "worktree_cleanup_fenced",
                "worktree_cleanup_fenced",
                "worktree_cleanup_fenced",
                "worktree_cleanup_fenced",
                "worktree_cleanup_fenced",
            ]
            fenced = [
                event
                for event in prelude
                if event.get("type") == "worktree_cleanup_fenced"
            ]
            lifecycle_fields = {
                "schema",
                "task_id",
                "canonical_task_cid",
                "attempt",
                "workspace_path",
                "branch",
                "merge_target",
                "repo_root",
                "state_dir",
                "lease_id",
                "lane_id",
                "owner",
                "created_at",
                "updated_at",
                "expires_at",
                "fence",
                "state",
                "terminal_reason",
                "record_id",
            }
            if (
                task_alias != "PCTDD-034"
                or [str(event.get("type") or "") for event in prelude]
                != expected_prelude_types
                or len(fenced) != 8
                or any(
                    event.get("allowed") is not False
                    or event.get("attempt_consumed") is not False
                    or event.get("provider_call_allowed") is not False
                    or event.get("disposition") != "deny"
                    or event.get("failure_kind") != "lifecycle_race"
                    or event.get("reason")
                    != "nonterminal_preparing_owner_alive"
                    or not isinstance(event.get("record"), Mapping)
                    or set(event["record"]) != lifecycle_fields
                    or event["record"].get("schema")
                    != (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "worktree-lifecycle-record@1"
                    )
                    or event["record"].get("state") != "preparing"
                    or event["record"].get("terminal_reason") != ""
                    or type(event["record"].get("attempt")) is not int
                    or int(event["record"]["attempt"]) != 1
                    or not isinstance(event["record"].get("owner"), Mapping)
                    or set(event["record"]["owner"])
                    != {"boot_id", "parent_pid", "pid", "start_time_ticks"}
                    for event in fenced
                )
                or any(
                    dict(event["record"]) != dict(fenced[0]["record"])
                    or event.get("worktree_path")
                    != fenced[0].get("worktree_path")
                    or event.get("branch") != fenced[0].get("branch")
                    for event in fenced[1:]
                )
            ):
                return None
        elif task_alias == "PCTDD-034":
            return None

        exact_identity = {
            "task_id": task_alias,
            "canonical_task_key": canonical_task_key,
            "canonical_task_cid": nested_task_cid,
            "board_namespace": board_namespace,
        }
        nested_attempt = finished.get("attempt")
        if (
            selected_event.get("title") != task_title
            or selected_event.get("track") != projection_track
            or any(
                selected_event.get(name) != value
                for name, value in exact_identity.items()
            )
            or type(nested_attempt) is not int
            or nested_attempt != 1
            or any(
                event.get("attempt") != nested_attempt
                or any(event.get(name) != value for name, value in exact_identity.items())
                for event in (recorded, started, pre_kernel, cleared, finished)
            )
            or finished.get("task_cid") != nested_task_cid
        ):
            return None
        if frozenset(pre_kernel) != _NO_PROVIDER_EVENT_FIELDS[
            "pre_implementation_kernel_evaluated"
        ]:
            return None
        try:
            from .implementation_disposition import (
                implementation_disposition_cid,
                verify_pre_implementation_kernel_receipt,
            )

            verified_kernel = verify_pre_implementation_kernel_receipt(
                pre_kernel["kernel_receipt"],
                expected_task_cid=nested_task_cid,
            )
        except (ImportError, TypeError, ValueError):
            return None
        if (
            verified_kernel.attempt != nested_attempt
            or verified_kernel.disposition.value != "abstain_review"
            or verified_kernel.reason_code != "no_analytical_close"
            or verified_kernel.residual_packet_cid != ""
            or pre_kernel.get("event") != "pre_implementation_kernel_evaluated"
            or pre_kernel.get("disposition") != "abstain_review"
            or pre_kernel.get("provider_authorized") is not False
            or pre_kernel.get("provider_hook_count") != 0
            or pre_kernel.get("skip_provider") is not True
            or pre_kernel.get("reason_code") != "no_analytical_close"
            or pre_kernel.get("residual_packet_cid") != ""
            or pre_kernel.get("analytical_candidate_count") != 0
            or pre_kernel.get("interface")
            != "ImplementationDaemon@pre_implementation_kernel"
            or pre_kernel.get("receipt_cid")
            != implementation_disposition_cid(dict(pre_kernel["kernel_receipt"]))
        ):
            return None

        expected_log_path = str(paths.root / log_relative_path)
        worktree_path = str(started.get("worktree_path") or "")
        branch = str(started.get("branch") or "")
        baseline_ref = str(started.get("baseline_ref") or "")
        if (
            log_relative_path
            != f"implementation-logs/{task_alias.lower()}-attempt-1.log"
            or started.get("log_path") != expected_log_path
            or finished.get("log_path") != expected_log_path
            or recorded.get("workspace_path") != worktree_path
            or finished.get("worktree_path") != worktree_path
            or released.get("worktree_path") != worktree_path
            or finished.get("branch") != branch
            or released.get("branch") != branch
            or finished.get("baseline_ref") != baseline_ref
            or started.get("execution_mode") != "model-assisted"
            or started.get("provider_dispatched") is not False
            or cleared.get("reason") != "failed_agent_terminal_check_unchanged"
        ):
            return None
        log_admission = self._terminal_no_effect_route_log_admission(
            raw_log=log_raw,
            command=started.get("command") or (),
            task_alias=task_alias,
            task_title=task_title,
            expected_worktree=worktree_path,
            expected_branch=branch,
            expected_baseline=baseline_ref,
            runner_returncode=int(finished.get("returncode") or 0),
        )
        if log_admission is None:
            return None

        release_body = {
            key: value
            for key, value in released.items()
            if key not in _EVENT_ENVELOPE_FIELDS
        }
        cleanup_result = finished.get("cleanup_result")
        started_setup = started.get("workspace_setup")
        finished_setup = finished.get("workspace_setup")
        expected_finished_setup: dict[str, Any] | None = None
        if isinstance(started_setup, Mapping):
            expected_finished_setup = dict(started_setup)
            raw_preflight = started_setup.get(
                "validation_project_dependency_preflight"
            )
            if isinstance(raw_preflight, Mapping):
                def bounded_count(field: str) -> int:
                    value = raw_preflight.get(field)
                    return (
                        len(value)
                        if isinstance(value, Sequence)
                        and not isinstance(value, (str, bytes, bytearray))
                        else 0
                    )

                def bounded_int(field: str) -> int:
                    value = raw_preflight.get(field)
                    return (
                        max(0, value)
                        if type(value) is int
                        else 0
                    )

                expected_finished_setup[
                    "validation_project_dependency_preflight"
                ] = {
                    "schema": str(raw_preflight.get("schema") or "")[:512],
                    "receipt_id": str(raw_preflight.get("receipt_id") or "")[:1024],
                    "retry_fingerprint": str(raw_preflight.get("retry_fingerprint") or "")[:1024],
                    "passed": raw_preflight.get("passed") is True,
                    "applicable": raw_preflight.get("applicable") is True,
                    "reason": str(raw_preflight.get("reason") or "")[:1000],
                    "automatic_install_attempted": raw_preflight.get("automatic_install_attempted") is True,
                    "probe_scope": str(raw_preflight.get("probe_scope") or "")[:512],
                    "validation_command_count": bounded_int("validation_command_count"),
                    "project_count": bounded_count("projects"),
                    "project_root_count": bounded_count("project_roots"),
                    "missing_count": bounded_count("missing_requirements"),
                    "incompatible_count": bounded_count("incompatible_requirements"),
                    "invalid_requirement_count": bounded_count("invalid_requirements"),
                    "invalid_command_count": bounded_count("invalid_commands"),
                    "event_projection_compacted": True,
                    "full_receipt_event": "implementation_started",
                }
        if (
            released.get("attempted") is not True
            or released.get("released") is not True
            or released.get("pooled") is not True
            or released.get("reason") != "clean_prepared_workspace"
            or released.get("handoff_reason") != "implementation_command_failed"
            or released.get("base_ref") != started.get("workspace_setup", {}).get("base_ref")
            or released.get("base_commit") != started.get("workspace_setup", {}).get("base_commit")
            or released.get("lifecycle_finalize")
            != {
                "fence": released.get("lifecycle_finalize", {}).get("fence"),
                "finalized": True,
                "reason": "pool_release_implementation_command_failed",
                "state": "terminal",
            }
            or type(released.get("lifecycle_finalize", {}).get("fence")) is not int
            or not isinstance(cleanup_result, Mapping)
            or set(cleanup_result)
            != {"cleaned", "lifecycle_finalize", "pool_release", "pooled", "reason"}
            or cleanup_result.get("cleaned") is not True
            or cleanup_result.get("pooled") is not True
            or cleanup_result.get("reason")
            != "failed_implementation_pool_lease_released"
            or cleanup_result.get("pool_release") != release_body
            or cleanup_result.get("lifecycle_finalize")
            != {"finalized": False, "reason": "no_lifecycle_record"}
            or finished.get("lifecycle_finalize")
            != {"finalized": False, "reason": "no_lifecycle_record"}
            or finished.get("returncode") != 1
            or finished.get("attempt_consumed") is not True
            or finished.get("provider_dispatched") is not True
            or finished.get("implementation_commit") != ""
            or finished.get("validation_result")
            != {"attempted": False, "passed": True, "reason": "not_run", "results": [], "returncode": 0}
            or finished.get("commit_result") != {"committed": False}
            or finished.get("merge_result")
            != {"merged": False, "reason": "not_attempted"}
            or finished.get("board_completion")
            != {"complete": False, "pending_merge": False, "reason": "implementation_or_validation_failed"}
            or finished.get("failed_preservation_result") != {}
            or expected_finished_setup is None
            or finished_setup != expected_finished_setup
            or finished.get("cache_hit") != started.get("cache_hit")
            or finished.get("setup_duration_seconds")
            != started.get("setup_duration_seconds")
            or finished.get("saved_duration_seconds")
            != started.get("saved_duration_seconds")
        ):
            return None

        forbidden_types = {
            "provider_invocation_committed",
            "validation_started",
            "validation_finished",
            "commit_created",
            "merge_started",
            "merge_finished",
            "task_completed",
            "proof_accepted",
        }
        if any(event.get("type") in forbidden_types for event in events):
            return None

        state_identity = state.get("task_identities")
        identity_record = (
            state_identity.get(task_alias)
            if isinstance(state_identity, Mapping)
            else None
        )
        semantic_fingerprint = canonical_task_key.rsplit("/", 1)[-1]
        identity_fields = {
            "board_namespace",
            "canonical_task_cid",
            "canonical_task_key",
            "display_task_id",
            "identity_version",
            "semantic_fingerprint",
            "source_path",
        }
        active_string_fields = (
            "active_task_id",
            "active_task_key",
            "active_task_cid",
            "active_task_title",
            "active_task_track",
            "active_task_started_at",
            "active_phase",
            "active_phase_started_at",
            "active_phase_detail",
            "active_log_path",
            "active_worktree_path",
            "active_branch",
        )
        if not (
            state.get("implementation_in_progress") is False
            and all(state.get(name) == "" for name in active_string_fields)
            and state.get("active_attempt") == 0
            and state.get("active_provider_runner") == {}
            and state.get("last_implementation_task_id") == task_alias
            and state.get("last_implementation_task_key") == canonical_task_key
            and state.get("last_implementation_task_cid") == nested_task_cid
            and state.get("last_implementation_returncode") == 1
            and state.get("last_implementation_log_path") == expected_log_path
            and state.get("last_implementation_worktree_path") == worktree_path
            and state.get("last_implementation_branch") == branch
            and state.get("last_implementation_commit") == ""
            and state.get("last_merge_returncode") is None
            and state.get("last_merge_error") == "not_attempted"
            and all(
                state.get(name) == ""
                for name in (
                    "last_merge_started_at",
                    "last_merge_finished_at",
                    "last_merge_branch",
                    "last_merge_commit",
                )
            )
            and state.get("last_proof_workflow") == {}
            and isinstance(state_identity, Mapping)
            and set(state_identity) == {task_alias}
            and isinstance(identity_record, Mapping)
            and set(identity_record) == identity_fields
            and identity_record.get("display_task_id") == task_alias
            and identity_record.get("canonical_task_key") == canonical_task_key
            and identity_record.get("canonical_task_cid") == nested_task_cid
            and identity_record.get("board_namespace") == board_namespace
            and identity_record.get("identity_version") == 1
            and identity_record.get("semantic_fingerprint") == semantic_fingerprint
            and identity_record.get("source_path") == str(paths.task_projection)
            and state.get("implementation_attempts") == {task_alias: 1}
            and state.get("implementation_attempts_by_cid") == {nested_task_cid: 1}
            and state.get("task_statuses") == {task_alias: "ready"}
            and state.get("ready_task_ids") == [task_alias]
            and state.get("selectable_ready_task_ids") == [task_alias]
            and state.get("eligible_ready_task_ids") == [task_alias]
            and all(
                state.get(name) == []
                for name in (
                    "completed_task_ids",
                    "external_reserved_task_ids",
                    "assumed_completed_task_ids",
                    "strict_deprioritized_ready_task_ids",
                    "waiting_task_ids",
                    "blocked_task_ids",
                )
            )
            and state.get("task_count") == 1
            and state.get("ready_count") == 1
            and state.get("selectable_ready_count") == 1
            and state.get("eligible_ready_count") == 1
            and all(
                state.get(name) == 0
                for name in (
                    "completed_count",
                    "external_reserved_count",
                    "assumed_completed_count",
                    "strict_deprioritized_ready_count",
                    "waiting_count",
                    "blocked_count",
                )
            )
            and state.get("recommended_task_id") == ""
            and state.get("recommended_actions") == []
            and state.get("task_artifacts") == {task_alias: []}
            and isinstance(state.get("task_validation"), Mapping)
            and set(state["task_validation"]) == {task_alias}
            and isinstance(state["task_validation"][task_alias], list)
            and bool(state["task_validation"][task_alias])
            and all(
                isinstance(command_item, str) and bool(command_item)
                for command_item in state["task_validation"][task_alias]
            )
            and state.get("protected_implementation_attempts") == {}
            and state.get("retry_budget_repair_receipts") == {}
            and state.get("retry_budget_repair_rearm_receipts") == {}
            and state.get("stale_proposal_replay_rearm_receipts") == {}
            and state.get("validation_obsolescence_rearm_receipts") == {}
            and state.get("strategy_generation") == 0
            and state.get("selection_idle_reason") == ""
        ):
            return None

        empty_pass_lists = (
            "attempt_limited_task_ids",
            "completion_receipt_task_ids",
            "manual_completion_authority_affected_goal_ids",
            "manual_completion_authority_dependency_task_ids",
            "manual_completion_authority_required_task_ids",
            "manual_completion_authority_task_ids",
            "manual_completion_renewal_quarantined_task_ids",
            "manual_completion_revalidation_only_task_ids",
            "manual_completion_revalidation_task_ids",
            "quarantined_manual_completion_status_task_ids",
            "released_retry_budget_strategy_block_task_ids",
            "retry_budget_rearmed_task_ids",
            "retry_budget_reset_deferred_task_ids",
            "retry_budget_reset_task_ids",
            "shared_active_merge_task_ids",
            "shared_completed_task_ids",
        )
        expected_projection_delta = [
            "active_task_cid",
            "active_task_id",
            "active_task_key",
            "active_task_started_at",
            "active_task_title",
            "active_task_track",
            "eligible_ready_count",
            "eligible_ready_task_ids",
            "heartbeat_at",
            "last_progress_at",
        ]
        if task_alias != "PCTDD-034":
            expected_projection_delta.extend(
                [
                    "ready_count",
                    "ready_task_ids",
                ]
            )
        expected_projection_delta.extend(
            [
                "recommended_actions",
                "recommended_task_id",
                "selectable_ready_count",
                "selectable_ready_task_ids",
            ]
        )
        if task_alias != "PCTDD-034":
            expected_projection_delta.extend(
                [
                    "task_artifacts",
                    "task_count",
                    "task_identities",
                    "task_statuses",
                    "task_validation",
                ]
            )
        else:
            expected_projection_delta.append("selection_idle_reason")
        if not (
            daemon_pass.get("previous_event_id") == finished.get("event_id")
            and daemon_pass.get("completed_count") == 0
            and daemon_pass.get("ready_count") == 1
            and daemon_pass.get("selectable_ready_count") == 1
            and daemon_pass.get("eligible_ready_count") == 1
            and daemon_pass.get("strict_deprioritized_ready_count") == 0
            and daemon_pass.get("waiting_count") == 0
            and daemon_pass.get("blocked_count") == 0
            and daemon_pass.get("active_task_id") == ""
            and daemon_pass.get("selection_idle_reason") == ""
            and daemon_pass.get("max_task_attempts") == 1
            and daemon_pass.get("ordinary_provider_dispatch_allowed") is True
            and daemon_pass.get("execution_slice_task_statuses")
            == {task_alias: "ready"}
            and daemon_pass.get("execution_slice_task_cids_by_id")
            == {task_alias: nested_task_cid}
            and all(daemon_pass.get(name) == [] for name in empty_pass_lists)
            and daemon_pass.get("manual_completion_authority_revalidation_only")
            is False
            and daemon_pass.get("protected_path_conflicts") == {}
            and daemon_pass.get("virgin_task_transfer")
            == {"granted_away_task_ids": [], "granted_to_lane_task_ids": [], "mode": "", "request_task_id": ""}
            and daemon_pass.get("projection_delta_keys")
            == expected_projection_delta
        ):
            return None

        evidence: dict[str, Any] = {
            "schema": DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": task_alias,
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "attempt_root_key": paths.root.name,
            "attempt_authority_root_digest": _sha256_bytes(str(self.attempt_root).encode("utf-8")),
            "attempt_root_digest": _sha256_bytes(str(paths.root).encode("utf-8")),
            "binding_id": str(binding["binding_id"]),
            "binding_admission_id": str(durable_binding.get("record_id") or ""),
            "binding_admission_digest": _sha256_bytes(_canonical_json(dict(durable_binding))),
            "projection_immutable_digest": str(binding["projection_immutable_digest"]),
            "task_revision": int(binding["task_revision"]),
            "board_namespace": board_namespace,
            "nested_task_cid": nested_task_cid,
            "nested_attempt": nested_attempt,
            "event_stream_id": str(manifest["stream_id"]),
            "event_snapshot_id": str(manifest["snapshot_id"]),
            "event_manifest_digest": str(manifest.get("manifest_digest") or ""),
            "event_count": len(events),
            "event_head_sequence": int(manifest["latest_sequence"]),
            "event_head_id": str(manifest["last_event_id"]),
            "prelude_event_count": len(prelude),
            "prelude_event_ids_digest": _sha256_bytes(_canonical_json([str(event["event_id"]) for event in prelude])),
            "task_selected_event_id": str(selected_event["event_id"]),
            "diagnostic_event_count": len(diagnostic_events),
            "diagnostic_event_ids_digest": _sha256_bytes(_canonical_json([str(event["event_id"]) for event in diagnostic_events])),
            "protected_snapshot_recorded_event_id": str(recorded["event_id"]),
            "implementation_started_event_id": str(started["event_id"]),
            "pre_implementation_event_id": str(pre_kernel["event_id"]),
            "pre_implementation_receipt_cid": str(pre_kernel["receipt_cid"]),
            "protected_snapshot_cleared_event_id": str(cleared["event_id"]),
            "worktree_release_event_id": str(released["event_id"]),
            "implementation_finished_event_id": str(finished["event_id"]),
            "daemon_pass_event_id": str(daemon_pass["event_id"]),
            "state_digest": state_digest,
            "outer_block_receipt_digest": _sha256_bytes(_canonical_json(dict(receipt))),
            **log_admission,
            "log_relative_path": log_relative_path,
            "log_sha256": _sha256_bytes(log_raw),
            "log_size": len(log_raw),
            "log_identity_digest": log_identity_digest,
            "provider_dispatched": False,
            "wrapper_process_dispatched": True,
            "quota_probe_dispatched": True,
            "primary_model_dispatched": False,
            "fallback_model_dispatched": False,
            "implementation_dispatched": False,
            "provider_effect_committed": False,
            "implementation_effect_committed": False,
            "legacy_nested_attempt_consumed": True,
            "rearm_attempt_consumed": False,
            "attempt_consumed": False,
            "validation_attempted": False,
            "commit_created": False,
            "merge_attempted": False,
            "acceptance_inferred": False,
            "protected_snapshot_unchanged": True,
            "workspace_unchanged": True,
            "cleanup_terminal": True,
            "route_denied": True,
            "historical_receipt_only": True,
            "fresh_fallback_authority": False,
            "nested_state_quiescent": True,
        }
        if set(evidence) != (
            DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_FIELDS
            - {"evidence_id"}
        ):
            raise DatabasePortalBridgeError(
                "terminal no-effect evidence field construction drifted"
            )
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

    def _deferred_provider_rearm_evidence(
        self,
        attempt: Any,
        *,
        receipt: Mapping[str, Any],
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        durable_binding: Mapping[str, Any],
        identity: Mapping[str, str],
        projection_track: str,
        projection_status: str,
        directory_names: Sequence[str],
        state: Mapping[str, Any],
        state_digest: str,
        events: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Prove an exact Portal route deferral before provider dispatch.

        This is intentionally a separate evidence version from the historical
        worktree-setup failure proof.  No provider attempt completed here: the
        selected route emitted a closed ``implementation_retry_deferred``
        event with ``attempt_consumed=false`` and returned to a quiescent
        ready projection.  Sealed maintenance diagnostics may be interleaved,
        but no implementation, validation, commit, merge, or acceptance event
        is admitted.
        """

        if projection_status != "ready" or list(directory_names) != [
            ".portal-events.jsonl.lock",
            "database-attempt-binding.json",
            "portal-events.jsonl",
            "portal-events.jsonl.manifest.json",
            "portal-strategy.json",
            "portal-task-state.json",
            "task-projection.md",
            "task_queue.json",
        ]:
            return None

        task_alias = str(binding.get("task_alias") or "")
        nested_task_cid = str(identity.get("canonical_task_cid") or "")
        selected = [
            event
            for event in events
            if event.get("type") == "task_selected"
            and event.get("task_id") == task_alias
            and event.get("canonical_task_cid") == nested_task_cid
        ]
        deferred = [
            event
            for event in events
            if event.get("type") == "implementation_retry_deferred"
            and event.get("task_id") == task_alias
            and event.get("canonical_task_cid") == nested_task_cid
        ]
        passes = [event for event in events if event.get("type") == "daemon_pass"]
        if not (len(selected) == len(deferred) == len(passes) == 1):
            return None
        selected_event = selected[0]
        deferred_event = deferred[0]
        daemon_pass = passes[0]
        diagnostic_events = [
            event
            for event in events
            if event.get("type")
            in {
                "dirty_submodule_reset_deferred",
                "implementation_resource_claim_lock_cleared",
            }
        ]
        if len(events) != 3 + len(diagnostic_events):
            return None
        dirty_diagnostics = [
            event
            for event in diagnostic_events
            if event.get("type") == "dirty_submodule_reset_deferred"
        ]
        lock_diagnostics = [
            event
            for event in diagnostic_events
            if event.get("type") == "implementation_resource_claim_lock_cleared"
        ]
        if len(dirty_diagnostics) != 1 or len(lock_diagnostics) > 1:
            return None
        try:
            selected_sequence = int(selected_event["sequence"])
            deferred_sequence = int(deferred_event["sequence"])
            daemon_sequence = int(daemon_pass["sequence"])
        except (KeyError, TypeError, ValueError):
            return None
        before_selected = list(events[: selected_sequence - 1])
        between_selected_and_deferred = list(
            events[selected_sequence: deferred_sequence - 1]
        )
        deferred_reason = str(deferred_event.get("reason") or "")
        diagnostic_receipt_id = str(
            deferred_event.get("diagnostic_receipt_id") or ""
        )
        nested_attempt = deferred_event.get("attempt")
        if not (
            selected_sequence == 2
            and before_selected == dirty_diagnostics
            and all(
                event.get("type")
                == "implementation_resource_claim_lock_cleared"
                for event in between_selected_and_deferred
            )
            and between_selected_and_deferred == lock_diagnostics
            and all(
                event.get("task_id") != task_alias
                and isinstance(event.get("lock_path"), str)
                and bool(event.get("lock_path"))
                for event in lock_diagnostics
            )
            and deferred_sequence == selected_sequence + len(lock_diagnostics) + 1
            and daemon_sequence == deferred_sequence + 1
            and daemon_sequence == len(events)
            and daemon_pass.get("event_id") == manifest.get("last_event_id")
            and selected_event.get("track") == projection_track
            and selected_event.get("canonical_task_key")
            == identity.get("canonical_task_key")
            and selected_event.get("board_namespace")
            == identity.get("board_namespace")
            and deferred_event.get("canonical_task_key")
            == identity.get("canonical_task_key")
            and deferred_event.get("board_namespace")
            == identity.get("board_namespace")
            and type(nested_attempt) is int
            and nested_attempt == 1
            and deferred_event.get("skipped") is True
            and deferred_event.get("attempt_consumed") is False
            and deferred_event.get("provider_dispatched") is False
            and deferred_event.get("active_task_cleared") is True
            and deferred_reason
            == DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_REASON
            and type(deferred_event.get("backoff_seconds")) is int
            and int(deferred_event["backoff_seconds"])
            == DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_BACKOFF_SECONDS
            and diagnostic_receipt_id == ""
            and daemon_pass.get("previous_event_id")
            == deferred_event.get("event_id")
        ):
            return None
        if dirty_diagnostics:
            dirty = dirty_diagnostics[0]
            if not (
                dirty.get("attempted") is True
                and type(dirty.get("dirty_count")) is int
                and dirty.get("dirty_count") == len(dirty.get("reset") or [])
                and dirty.get("generated_artifact_preservation") == []
            ):
                return None
        if any(
            event.get("provider_dispatched") is True
            or event.get("attempt_consumed") is True
            or bool(str(event.get("implementation_commit") or ""))
            or event.get("type")
            in {
                "implementation_started",
                "implementation_finished",
                "provider_invocation_committed",
                "validation_finished",
                "merge_finished",
                "task_completed",
            }
            or (
                isinstance(event.get("validation_result"), Mapping)
                and event["validation_result"].get("attempted") is True
            )
            or (
                isinstance(event.get("commit_result"), Mapping)
                and event["commit_result"].get("committed") is True
            )
            or (
                isinstance(event.get("merge_result"), Mapping)
                and any(
                    event["merge_result"].get(field) is True
                    for field in ("attempted", "queued", "merged")
                )
            )
            for event in events
        ):
            return None

        state_identity = state.get("task_identities")
        identity_record = (
            state_identity.get(task_alias)
            if isinstance(state_identity, Mapping)
            else None
        )
        canonical_task_key = str(identity.get("canonical_task_key") or "")
        semantic_fingerprint = canonical_task_key.rsplit("/", 1)[-1]
        identity_fields = {
            "board_namespace",
            "canonical_task_cid",
            "canonical_task_key",
            "display_task_id",
            "identity_version",
            "semantic_fingerprint",
            "source_path",
        }
        expected_idle_reason = f"implementation_retry_deferred:{deferred_reason}"
        if not (
            state.get("implementation_in_progress") is False
            and all(
                state.get(name) == ""
                for name in (
                    "active_task_id",
                    "active_task_key",
                    "active_task_cid",
                    "active_task_title",
                    "active_task_track",
                    "active_task_started_at",
                    "active_phase",
                    "active_phase_started_at",
                    "active_phase_detail",
                    "active_log_path",
                    "active_worktree_path",
                    "active_branch",
                    "last_implementation_task_id",
                    "last_implementation_task_key",
                    "last_implementation_task_cid",
                    "last_implementation_started_at",
                    "last_implementation_finished_at",
                    "last_implementation_log_path",
                    "last_implementation_worktree_path",
                    "last_implementation_branch",
                    "last_implementation_commit",
                    "last_merge_started_at",
                    "last_merge_finished_at",
                    "last_merge_branch",
                    "last_merge_commit",
                )
            )
            and state.get("active_attempt") == 0
            and state.get("active_provider_runner") == {}
            and state.get("last_implementation_returncode") is None
            and state.get("last_proof_workflow") == {}
            and state.get("last_merge_returncode") is None
            and state.get("last_merge_error") == ""
            and isinstance(state_identity, Mapping)
            and set(state_identity) == {task_alias}
            and isinstance(identity_record, Mapping)
            and set(identity_record) == identity_fields
            and identity_record.get("display_task_id") == task_alias
            and identity_record.get("canonical_task_key") == canonical_task_key
            and identity_record.get("canonical_task_cid") == nested_task_cid
            and identity_record.get("board_namespace")
            == identity.get("board_namespace")
            and identity_record.get("identity_version") == 1
            and identity_record.get("semantic_fingerprint")
            == semantic_fingerprint
            and identity_record.get("source_path") == str(paths.task_projection)
            and state.get("implementation_attempts") == {}
            and state.get("implementation_attempts_by_cid") == {}
            and state.get("task_statuses") == {task_alias: "ready"}
            and state.get("ready_task_ids") == [task_alias]
            and state.get("selectable_ready_task_ids") == []
            and state.get("eligible_ready_task_ids") == []
            and state.get("completed_task_ids") == []
            and state.get("external_reserved_task_ids") == []
            and state.get("assumed_completed_task_ids") == []
            and state.get("strict_deprioritized_ready_task_ids") == []
            and state.get("waiting_task_ids") == []
            and state.get("blocked_task_ids") == []
            and state.get("task_count") == 1
            and state.get("ready_count") == 1
            and state.get("selectable_ready_count") == 0
            and state.get("eligible_ready_count") == 0
            and state.get("completed_count") == 0
            and state.get("external_reserved_count") == 0
            and state.get("assumed_completed_count") == 0
            and state.get("strict_deprioritized_ready_count") == 0
            and state.get("waiting_count") == 0
            and state.get("blocked_count") == 0
            and state.get("recommended_task_id") == ""
            and state.get("recommended_actions") == []
            and state.get("task_artifacts") == {task_alias: []}
            and isinstance(state.get("task_validation"), Mapping)
            and set(state["task_validation"]) == {task_alias}
            and isinstance(state["task_validation"][task_alias], list)
            and bool(state["task_validation"][task_alias])
            and all(
                isinstance(command, str) and bool(command)
                for command in state["task_validation"][task_alias]
            )
            and state.get("protected_implementation_attempts") == {}
            and state.get("retry_budget_repair_receipts") == {}
            and state.get("retry_budget_repair_rearm_receipts") == {}
            and state.get("stale_proposal_replay_rearm_receipts") == {}
            and state.get("validation_obsolescence_rearm_receipts") == {}
            and state.get("strategy_generation") == 0
            and state.get("selection_idle_reason") == expected_idle_reason
            and daemon_pass.get("completed_count") == 0
            and daemon_pass.get("ready_count") == 1
            and daemon_pass.get("selectable_ready_count") == 0
            and daemon_pass.get("eligible_ready_count") == 0
            and daemon_pass.get("strict_deprioritized_ready_count") == 0
            and daemon_pass.get("waiting_count") == 0
            and daemon_pass.get("blocked_count") == 0
            and daemon_pass.get("active_task_id") == ""
            and daemon_pass.get("selection_idle_reason") == expected_idle_reason
            and daemon_pass.get("max_task_attempts") == 1
            and daemon_pass.get("ordinary_provider_dispatch_allowed") is True
            and daemon_pass.get("execution_slice_task_statuses")
            == {task_alias: "ready"}
            and daemon_pass.get("execution_slice_task_cids_by_id")
            == {task_alias: nested_task_cid}
            and daemon_pass.get("shared_completed_task_ids") == []
            and all(
                daemon_pass.get(name) == []
                for name in (
                    "attempt_limited_task_ids",
                    "completion_receipt_task_ids",
                    "manual_completion_authority_affected_goal_ids",
                    "manual_completion_authority_dependency_task_ids",
                    "manual_completion_authority_required_task_ids",
                    "manual_completion_authority_task_ids",
                    "manual_completion_renewal_quarantined_task_ids",
                    "manual_completion_revalidation_only_task_ids",
                    "manual_completion_revalidation_task_ids",
                    "quarantined_manual_completion_status_task_ids",
                    "released_retry_budget_strategy_block_task_ids",
                    "retry_budget_rearmed_task_ids",
                    "retry_budget_reset_deferred_task_ids",
                    "retry_budget_reset_task_ids",
                    "shared_active_merge_task_ids",
                )
            )
            and daemon_pass.get("manual_completion_authority_revalidation_only")
            is False
            and daemon_pass.get("virgin_task_transfer")
            == {
                "granted_away_task_ids": [],
                "granted_to_lane_task_ids": [],
                "mode": "",
                "request_task_id": "",
            }
            and daemon_pass.get("projection_delta_keys")
            == [
                "active_task_cid",
                "active_task_id",
                "active_task_key",
                "active_task_started_at",
                "active_task_title",
                "active_task_track",
                "eligible_ready_count",
                "eligible_ready_task_ids",
                "heartbeat_at",
                "last_progress_at",
                "ready_count",
                "ready_task_ids",
                "recommended_actions",
                "recommended_task_id",
                "selectable_ready_count",
                "selectable_ready_task_ids",
                "task_artifacts",
                "task_count",
                "task_identities",
                "task_statuses",
                "task_validation",
            ]
            and daemon_pass.get("protected_path_conflicts") == {}
        ):
            return None

        evidence: dict[str, Any] = {
            "schema": DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": task_alias,
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "attempt_root_key": paths.root.name,
            "attempt_authority_root_digest": _sha256_bytes(
                str(self.attempt_root).encode("utf-8")
            ),
            "attempt_root_digest": _sha256_bytes(str(paths.root).encode("utf-8")),
            "binding_id": str(binding["binding_id"]),
            "binding_admission_id": str(durable_binding.get("record_id") or ""),
            "binding_admission_digest": _sha256_bytes(
                _canonical_json(dict(durable_binding))
            ),
            "projection_immutable_digest": str(
                binding["projection_immutable_digest"]
            ),
            "nested_task_cid": nested_task_cid,
            "nested_attempt": int(nested_attempt),
            "event_stream_id": str(manifest["stream_id"]),
            "event_snapshot_id": str(manifest["snapshot_id"]),
            "event_manifest_digest": str(manifest.get("manifest_digest") or ""),
            "event_count": len(events),
            "event_head_sequence": int(manifest["latest_sequence"]),
            "event_head_id": str(manifest["last_event_id"]),
            "task_selected_event_id": str(selected_event["event_id"]),
            "retry_deferred_event_id": str(deferred_event["event_id"]),
            "daemon_pass_event_id": str(daemon_pass["event_id"]),
            "diagnostic_event_count": len(diagnostic_events),
            "diagnostic_event_ids_digest": _sha256_bytes(
                _canonical_json(
                    [str(event["event_id"]) for event in diagnostic_events]
                )
            ),
            "deferred_reason": deferred_reason,
            "deferred_backoff_seconds": int(
                deferred_event["backoff_seconds"]
            ),
            "diagnostic_receipt_id": diagnostic_receipt_id,
            "state_digest": state_digest,
            "outer_block_receipt_digest": _sha256_bytes(
                _canonical_json(dict(receipt))
            ),
            "provider_dispatched": False,
            "attempt_consumed": False,
            "validation_attempted": False,
            "commit_created": False,
            "merge_attempted": False,
            "acceptance_inferred": False,
            "route_deferred": True,
            "nested_state_quiescent": True,
        }
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

    def _terminal_quiescent_deferred_rearm_evidence(
        self,
        attempt: Any,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Prove terminal-linked deferral before nested task selection.

        This evidence version covers a database attempt that crossed the
        Portal entry boundary, but whose single projected task was deferred by
        a resource claim before ``task_selected``.  The outer terminal saga is
        already durable, so the proof binds its prepared and commit-barrier
        receipts as well as the post-probe event head.  Repeated exact
        ``already_quiesced`` shutdown events are diagnostics, not attempts.
        """

        link = receipt.get("terminal_reconciliation")
        if type(link) is not dict:
            return None
        link = dict(link)
        exact_attempt = {
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
        }
        unsigned_link = dict(link)
        link_evidence_id = str(unsigned_link.pop("evidence_id", "") or "")
        if (
            set(link) != set(DATABASE_PORTAL_TERMINAL_RECONCILIATION_LINK_FIELDS)
            or link.get("schema")
            != DATABASE_PORTAL_TERMINAL_RECONCILIATION_LINK_SCHEMA
            or not _strict_attempt_integer_fields(link)
            or any(
                type(link.get(name)) is not type(value)
                or link.get(name) != value
                for name, value in exact_attempt.items()
            )
            or link.get("nested_reason") != "nested_portal_attempt_reconciled"
            or link.get("nested_reconciled") is not True
            or link.get("intended_database_disposition")
            != "blocked_unknown_outcome"
            or not re.fullmatch(r"baguqeera[a-z2-7]{52}", link_evidence_id)
            or content_identity(unsigned_link) != link_evidence_id
            or any(
                not re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(link.get(name) or ""),
                )
                for name in (
                    "prepared_reconciliation_receipt_id",
                    "commit_barrier_receipt_id",
                )
            )
        ):
            return None

        paths = self._paths(attempt)
        expected_root = self.attempt_root / hashlib.sha256(
            str(attempt.attempt_id).encode("utf-8")
        ).hexdigest()[:24]
        if paths.root != expected_root:
            return None
        try:
            sealed = self._pinned_no_provider_snapshot(paths)
            binding = dict(sealed["binding"])
            self._verify_binding_identity(binding)
            durable_binding = (
                self._binding_lookup(attempt)
                if self._binding_lookup is not None
                else None
            )
            projection = str(sealed["projection"])
            if (
                _projection_immutable_digest(projection)
                != str(binding.get("projection_immutable_digest") or "")
                or _HEADER.findall(projection)
                != [str(binding.get("task_alias") or "")]
            ):
                raise DatabasePortalBridgeError(
                    "database Portal task projection identity changed"
                )
            projection_task = self._projection_task(paths, binding, projection)
            identity = {
                "task_id": projection_task.task_id,
                "canonical_task_key": str(
                    projection_task.canonical_task_key or ""
                ),
                "canonical_task_cid": str(
                    projection_task.canonical_task_cid or ""
                ),
                "board_namespace": str(
                    projection_task.board_namespace or ""
                ),
            }
            if any(not value for value in identity.values()):
                raise DatabasePortalBridgeError(
                    "Portal task projection lacks a complete canonical identity"
                )
            projection_status = _projection_status(projection)
            state = dict(sealed["state"])
            state_digest = str(sealed["state_digest"])
            events = list(sealed["events"])
            manifest = dict(sealed["manifest"])
            directory_names = list(sealed["directory_names"])
            prepared = self.load_reconciliation_receipt(
                attempt,
                str(link["prepared_reconciliation_receipt_id"]),
                required_stage="prepared",
            )
            commit_barrier = self.load_reconciliation_receipt(
                attempt,
                str(link["commit_barrier_receipt_id"]),
                required_stage="commit_barrier",
            )
        except (DatabasePortalBridgeError, OSError, TypeError, ValueError):
            return None
        if not isinstance(durable_binding, Mapping):
            return None

        binding_expected = {
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
        }
        durable_expected = {
            **exact_attempt,
            "binding_id": str(binding.get("binding_id") or ""),
            "projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
            "stage": "portal_entered",
        }
        if (
            any(
                type(binding.get(name)) is not type(value)
                or binding.get(name) != value
                for name, value in binding_expected.items()
            )
            or any(
                type(durable_binding.get(name)) is not type(value)
                or durable_binding.get(name) != value
                for name, value in durable_expected.items()
            )
            or not str(durable_binding.get("record_id") or "")
            or link.get("binding_id") != binding.get("binding_id")
            or projection_status != "ready"
        ):
            return None

        required_directory_names = {
            ".implementation.lock.update.lock",
            ".portal-events.jsonl.lock",
            ".portal-task-state.event-driven-checkpoint.json.lock",
            "database-attempt-binding.json",
            "database-attempt-reconciliations",
            "portal-events.jsonl",
            "portal-events.jsonl.manifest.json",
            "portal-strategy.json",
            "portal-task-state.event-driven-checkpoint.json",
            "portal-task-state.json",
            "task-projection.md",
            "task_queue.json",
        }
        observed_directory_names = set(directory_names)
        unexpected_directory_names = observed_directory_names - (
            required_directory_names | {"portal-events.jsonl"}
        )
        if (
            not required_directory_names.issubset(observed_directory_names)
            or any(
                re.fullmatch(r"portal-events\.jsonl\.rotated-[0-9]+", name)
                is None
                for name in unexpected_directory_names
            )
        ):
            return None

        prepared_fields = {
            "schema",
            "interface",
            *(set(exact_attempt) - {"lease_id"}),
            "attempt_root",
            "task_alias",
            "stage",
            "trigger",
            "reconciled_at",
            "reconciled",
            "blocked",
            "reason",
            "binding_id",
            "historical_binding",
            "nested_state",
            "provider_runner_fence",
            "provider_runner_reconciliation_authority",
            "portal_reconciliation",
            "terminal_provider_evidence",
            "terminal_provider_receipt_id",
            "intended_database_disposition",
            "receipt_id",
        }
        commit_fields = prepared_fields | {"prepared_reconciliation_receipt_id"}
        prepared_barrier_core = {
            name: value
            for name, value in prepared.items()
            if name not in {"stage", "receipt_id"}
        }
        commit_barrier_core = {
            name: value
            for name, value in commit_barrier.items()
            if name
            not in {
                "stage",
                "receipt_id",
                "prepared_reconciliation_receipt_id",
            }
        }
        nested_state = prepared.get("nested_state")
        provider_fence = prepared.get("provider_runner_fence")
        reconciliation = prepared.get("portal_reconciliation")
        if not isinstance(reconciliation, Mapping):
            return None
        reconciliation = dict(reconciliation)
        reconciliation_timestamp = reconciliation.get("reconciled_at")
        expected_reconciliation = {
            "attempt": 0,
            "attempt_recovery": {},
            "blocked": False,
            "protected_path_reconciliation": {
                "blocked": False,
                "critical_section_entered": False,
                "reason": "no_active_snapshot",
                "scan_outside_lease": True,
            },
            "provider_forbidden_terminal_recovery": {
                "applicable": False,
                "blocked": False,
                "implementation_dispatched": False,
                "provider_dispatched": False,
                "reason": "provider_forbidden_terminal_recovery_not_applicable",
                "reconciled": False,
            },
            "reason": "already_quiesced",
            "reconciled": True,
            "reconciled_at": reconciliation_timestamp,
            "stale_lock_cleared": False,
            "task_claim_reconciliation": {
                "blocked": False,
                "reason": "no_task_claim_identity",
                "reconciled": False,
            },
            "task_id": "",
            "worktree_lifecycle_reconciliation": {
                "blocked": False,
                "reason": "no_active_worktree",
                "reconciled": False,
            },
        }
        expected_nested_state = {
            "active": False,
            "active_attempt": 0,
            "active_branch": "",
            "active_phase": "",
            "active_phase_detail": "",
            "active_task_id": "",
            "active_worktree_path": "",
            "present": True,
            "state_digest": state_digest,
            "state_path": str(paths.state),
        }
        if (
            set(prepared) != prepared_fields
            or set(commit_barrier) != commit_fields
            or prepared.get("receipt_id")
            != link.get("prepared_reconciliation_receipt_id")
            or commit_barrier.get("receipt_id")
            != link.get("commit_barrier_receipt_id")
            or commit_barrier.get("prepared_reconciliation_receipt_id")
            != prepared.get("receipt_id")
            or _canonical_json(commit_barrier_core)
            != _canonical_json(prepared_barrier_core)
            or prepared.get("binding_id") != binding.get("binding_id")
            or prepared.get("historical_binding") is not False
            or prepared.get("reason") != "nested_portal_attempt_reconciled"
            or prepared.get("reconciled") is not True
            or prepared.get("blocked") is not False
            or prepared.get("intended_database_disposition")
            != "blocked_unknown_outcome"
            or prepared.get("trigger") != link.get("trigger")
            or prepared.get("terminal_provider_evidence") is not False
            or prepared.get("terminal_provider_receipt_id") != ""
            or prepared.get("provider_runner_reconciliation_authority")
            != "not_applicable"
            or not isinstance(prepared.get("reconciled_at"), str)
            or not str(prepared.get("reconciled_at") or "")
            or not _closed_typed_record_matches(
                nested_state,
                expected_nested_state,
            )
            or link.get("nested_state_digest") != state_digest
            or not _closed_typed_record_matches(
                provider_fence,
                {
                    "applicable": False,
                    "fenced": False,
                    "reason": "ordinary_provider_runner_receipt_absent",
                    "safe_to_restart": True,
                },
            )
            or not isinstance(reconciliation_timestamp, str)
            or not str(reconciliation_timestamp or "")
            or not _closed_typed_record_matches(
                reconciliation,
                expected_reconciliation,
            )
        ):
            return None

        task_alias = str(binding.get("task_alias") or "")
        nested_task_cid = str(identity.get("canonical_task_cid") or "")
        canonical_task_key = str(identity.get("canonical_task_key") or "")
        semantic_fingerprint = canonical_task_key.rsplit("/", 1)[-1]
        identity_record = (
            state.get("task_identities", {}).get(task_alias)
            if isinstance(state.get("task_identities"), Mapping)
            else None
        )
        empty_state_strings = (
            "active_task_id",
            "active_task_key",
            "active_task_cid",
            "active_task_title",
            "active_task_track",
            "active_task_started_at",
            "active_phase",
            "active_phase_started_at",
            "active_phase_detail",
            "active_log_path",
            "active_worktree_path",
            "active_branch",
            "last_implementation_task_id",
            "last_implementation_task_key",
            "last_implementation_task_cid",
            "last_implementation_started_at",
            "last_implementation_finished_at",
            "last_implementation_log_path",
            "last_implementation_worktree_path",
            "last_implementation_branch",
            "last_implementation_commit",
            "last_merge_started_at",
            "last_merge_finished_at",
            "last_merge_branch",
            "last_merge_commit",
        )
        selection_idle_reason = (
            "all_selectable_ready_tasks_deferred_by_resource_claim"
        )
        if not (
            state.get("implementation_in_progress") is False
            and all(state.get(name) == "" for name in empty_state_strings)
            and type(state.get("active_attempt")) is int
            and state.get("active_attempt") == 0
            and state.get("active_provider_runner") == {}
            and state.get("last_implementation_returncode") is None
            and state.get("last_proof_workflow") == {}
            and state.get("last_merge_returncode") is None
            and state.get("last_merge_error") == ""
            and isinstance(identity_record, Mapping)
            and set(state.get("task_identities", {})) == {task_alias}
            and set(identity_record)
            == {
                "board_namespace",
                "canonical_task_cid",
                "canonical_task_key",
                "display_task_id",
                "identity_version",
                "semantic_fingerprint",
                "source_path",
            }
            and identity_record.get("display_task_id") == task_alias
            and identity_record.get("canonical_task_key") == canonical_task_key
            and identity_record.get("canonical_task_cid") == nested_task_cid
            and identity_record.get("board_namespace")
            == identity.get("board_namespace")
            and type(identity_record.get("identity_version")) is int
            and identity_record.get("identity_version") == 1
            and identity_record.get("semantic_fingerprint")
            == semantic_fingerprint
            and identity_record.get("source_path") == str(paths.task_projection)
            and state.get("implementation_attempts") == {}
            and state.get("implementation_attempts_by_cid") == {}
            and state.get("task_statuses") == {task_alias: "ready"}
            and state.get("ready_task_ids") == [task_alias]
            and state.get("selectable_ready_task_ids") == []
            and state.get("eligible_ready_task_ids") == []
            and state.get("completed_task_ids") == []
            and state.get("external_reserved_task_ids") == []
            and state.get("assumed_completed_task_ids") == []
            and state.get("strict_deprioritized_ready_task_ids") == []
            and state.get("waiting_task_ids") == []
            and state.get("blocked_task_ids") == []
            and state.get("task_count") == 1
            and state.get("ready_count") == 1
            and state.get("selectable_ready_count") == 0
            and state.get("eligible_ready_count") == 0
            and state.get("completed_count") == 0
            and state.get("external_reserved_count") == 0
            and state.get("assumed_completed_count") == 0
            and state.get("strict_deprioritized_ready_count") == 0
            and state.get("waiting_count") == 0
            and state.get("blocked_count") == 0
            and state.get("recommended_task_id") == ""
            and state.get("recommended_actions") == []
            and state.get("task_artifacts") == {task_alias: []}
            and isinstance(state.get("task_validation"), Mapping)
            and set(state["task_validation"]) == {task_alias}
            and isinstance(state["task_validation"][task_alias], list)
            and bool(state["task_validation"][task_alias])
            and all(
                isinstance(command, str) and bool(command)
                for command in state["task_validation"][task_alias]
            )
            and state.get("protected_implementation_attempts") == {}
            and state.get("retry_budget_repair_receipts") == {}
            and state.get("retry_budget_repair_rearm_receipts") == {}
            and state.get("stale_proposal_replay_rearm_receipts") == {}
            and state.get("validation_obsolescence_rearm_receipts") == {}
            and state.get("strategy_generation") == 0
            and state.get("selection_idle_reason") == selection_idle_reason
        ):
            return None

        selected_events = [
            event for event in events if event.get("type") == "task_selected"
        ]
        resource_deferred_events = [
            event
            for event in events
            if event.get("type") == "daemon_pass"
            and event.get("selection_idle_reason") == selection_idle_reason
        ]
        shutdown_events = [
            event
            for event in events
            if event.get("type") == "implementation_shutdown_reconciled"
        ]
        allowed_diagnostic_types = {
            "cleanup_finished",
            "dirty_submodule_reset_deferred",
            "stale_worktree_cleanup",
            "worktree_cleanup_fenced",
        }
        diagnostic_events = [
            event
            for event in events
            if event.get("type") in allowed_diagnostic_types
        ]
        forbidden_effect_types = {
            "implementation_started",
            "implementation_finished",
            "implementation_exception",
            "implementation_retry_deferred",
            "provider_invocation_committed",
            "validation_finished",
            "merge_finished",
            "task_completed",
        }
        if (
            selected_events
            or len(resource_deferred_events) != 1
            or not shutdown_events
            or len(events)
            != 1 + len(diagnostic_events) + len(shutdown_events)
            or any(
                event.get("type") in forbidden_effect_types
                or event.get("provider_dispatched") is True
                or event.get("attempt_consumed") is True
                or bool(str(event.get("implementation_commit") or ""))
                or isinstance(event.get("validation_result"), Mapping)
                or isinstance(event.get("commit_result"), Mapping)
                or isinstance(event.get("merge_result"), Mapping)
                for event in events
            )
            or any(
                int(event.get("sequence") or 0)
                <= int(resource_deferred_events[0].get("sequence") or 0)
                for event in shutdown_events
            )
            or any(
                event.get("reconciled_at") != reconciliation_timestamp
                for event in shutdown_events
            )
        ):
            return None
        for event in diagnostic_events:
            if event.get("type") == "worktree_cleanup_fenced" and not (
                event.get("allowed") is False
                and event.get("attempt_consumed") is False
                and event.get("provider_call_allowed") is False
                and event.get("disposition") == "deny"
            ):
                return None
            if event.get("type") == "cleanup_finished" and (
                event.get("cleaned") is not True
            ):
                return None

        daemon_pass = resource_deferred_events[0]
        if not (
            daemon_pass.get("completed_count") == state.get("completed_count")
            and daemon_pass.get("ready_count") == state.get("ready_count")
            and daemon_pass.get("selectable_ready_count")
            == state.get("selectable_ready_count")
            and daemon_pass.get("eligible_ready_count")
            == state.get("eligible_ready_count")
            and daemon_pass.get("strict_deprioritized_ready_count")
            == state.get("strict_deprioritized_ready_count")
            and daemon_pass.get("waiting_count") == state.get("waiting_count")
            and daemon_pass.get("blocked_count") == state.get("blocked_count")
            and daemon_pass.get("active_task_id") == ""
            and daemon_pass.get("selection_idle_reason")
            == selection_idle_reason
            and daemon_pass.get("max_task_attempts") == 1
            and daemon_pass.get("ordinary_provider_dispatch_allowed") is True
            and daemon_pass.get("execution_slice_task_statuses")
            == {task_alias: "ready"}
            and daemon_pass.get("execution_slice_task_cids_by_id")
            == {task_alias: nested_task_cid}
            and daemon_pass.get("shared_completed_task_ids") == []
            and daemon_pass.get("protected_path_conflicts") == {}
            and daemon_pass.get("manual_completion_authority_revalidation_only")
            is False
            and daemon_pass.get("virgin_task_transfer")
            == {
                "granted_away_task_ids": [],
                "granted_to_lane_task_ids": [],
                "mode": "",
                "request_task_id": "",
            }
            and daemon_pass.get("projection_delta_keys")
            == [
                "heartbeat_at",
                "last_progress_at",
                "ready_count",
                "ready_task_ids",
                "selection_idle_reason",
                "task_artifacts",
                "task_count",
                "task_identities",
                "task_statuses",
                "task_validation",
            ]
            and all(
                daemon_pass.get(name) == []
                for name in (
                    "attempt_limited_task_ids",
                    "completion_receipt_task_ids",
                    "manual_completion_authority_affected_goal_ids",
                    "manual_completion_authority_dependency_task_ids",
                    "manual_completion_authority_required_task_ids",
                    "manual_completion_authority_task_ids",
                    "manual_completion_renewal_quarantined_task_ids",
                    "manual_completion_revalidation_only_task_ids",
                    "manual_completion_revalidation_task_ids",
                    "quarantined_manual_completion_status_task_ids",
                    "released_retry_budget_strategy_block_task_ids",
                    "retry_budget_rearmed_task_ids",
                    "retry_budget_reset_deferred_task_ids",
                    "retry_budget_reset_task_ids",
                    "shared_active_merge_task_ids",
                )
            )
        ):
            return None

        evidence: dict[str, Any] = {
            "schema": (
                DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_SCHEMA
            ),
            **exact_attempt,
            "task_alias": task_alias,
            "attempt_root_key": paths.root.name,
            "attempt_authority_root_digest": _sha256_bytes(
                str(self.attempt_root).encode("utf-8")
            ),
            "attempt_root_digest": _sha256_bytes(
                str(paths.root).encode("utf-8")
            ),
            "attempt_directory_names_digest": _sha256_bytes(
                _canonical_json(sorted(directory_names))
            ),
            "binding_id": str(binding.get("binding_id") or ""),
            "binding_admission_id": str(
                durable_binding.get("record_id") or ""
            ),
            "binding_admission_digest": _sha256_bytes(
                _canonical_json(dict(durable_binding))
            ),
            "projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
            "task_revision": int(binding.get("task_revision") or 0),
            "board_namespace": str(identity.get("board_namespace") or ""),
            "nested_task_cid": nested_task_cid,
            "nested_attempt": 0,
            "terminal_reconciliation_evidence_id": link_evidence_id,
            "prepared_reconciliation_receipt_id": str(
                prepared.get("receipt_id") or ""
            ),
            "commit_barrier_receipt_id": str(
                commit_barrier.get("receipt_id") or ""
            ),
            "event_stream_id": str(manifest.get("stream_id") or ""),
            "event_snapshot_id": str(manifest.get("snapshot_id") or ""),
            "event_manifest_digest": str(
                manifest.get("manifest_digest") or ""
            ),
            "event_count": len(events),
            "event_head_sequence": int(manifest.get("latest_sequence") or 0),
            "event_head_id": str(manifest.get("last_event_id") or ""),
            "resource_deferred_daemon_pass_event_id": str(
                daemon_pass.get("event_id") or ""
            ),
            "diagnostic_event_count": len(diagnostic_events),
            "diagnostic_event_ids_digest": _sha256_bytes(
                _canonical_json(
                    [str(event.get("event_id") or "") for event in diagnostic_events]
                )
            ),
            "shutdown_reconciliation_event_count": len(shutdown_events),
            "shutdown_reconciliation_event_ids_digest": _sha256_bytes(
                _canonical_json(
                    [str(event.get("event_id") or "") for event in shutdown_events]
                )
            ),
            "state_digest": state_digest,
            "outer_block_receipt_digest": _sha256_bytes(
                _canonical_json(dict(receipt))
            ),
            "selection_idle_reason": selection_idle_reason,
            "nested_reconciliation_reason": "already_quiesced",
            "provider_runner_fence_reason": (
                "ordinary_provider_runner_receipt_absent"
            ),
            "provider_dispatched": False,
            "implementation_dispatched": False,
            "task_never_selected": True,
            "attempt_consumed": False,
            "validation_attempted": False,
            "commit_created": False,
            "merge_attempted": False,
            "acceptance_inferred": False,
            "route_deferred": True,
            "nested_state_quiescent": True,
        }
        if set(evidence) != (
            DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_FIELDS
            - {"evidence_id"}
        ):
            raise DatabasePortalBridgeError(
                "terminal quiescent deferred evidence field construction drifted"
            )
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

    def no_provider_dispatch_rearm_evidence(
        self,
        attempt: Any,
        *,
        outer_block_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Prove one exact nested recovery is eligible for bounded rearm.

        A populated terminal-reconciliation link first reaches the exact
        interrupted-implementation verifier.  Under the standard policy only,
        a distinct successor may then prove that the nested task was never
        selected because its route remained resource-deferred.  Neither route
        may fall through to legacy setup-failure evidence, and the stale
        dispatch migration policy remains exclusive to its migration proof.
        """

        receipt = dict(outer_block_receipt)
        expected_identity = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-retry-budget@1"
            ),
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "lease_id": str(attempt.lease_id),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "retry_exhausted": True,
            "forced_block": True,
            "authority_outcome": "unknown",
        }
        standard_policy = bool(
            receipt.get("operation") == "database_unknown_outcome_blocked"
            and receipt.get("reason") == "callback_authority_incomplete_blocked"
        )
        stale_dispatch_migration_policy = bool(
            receipt.get("operation") == "database_unknown_outcome_blocked"
            and receipt.get("reason") == "provider_dispatch_outcome_unknown"
            and isinstance(receipt.get("terminal_reconciliation"), Mapping)
            and bool(receipt.get("terminal_reconciliation"))
        )
        if (
            str(getattr(attempt, "status", "") or "") != "failed"
            or str(getattr(attempt, "committed_phase", "") or "") != "failed"
            or any(
                type(receipt.get(key)) is not type(value)
                or receipt.get(key) != value
                for key, value in expected_identity.items()
            )
            or not (standard_policy or stale_dispatch_migration_policy)
            or not str(receipt.get("process_instance_id") or "").strip()
        ):
            return None

        terminal_link_missing = object()
        terminal_link = receipt.get(
            "terminal_reconciliation",
            terminal_link_missing,
        )
        if terminal_link is not terminal_link_missing:
            if type(terminal_link) is not dict:
                return None
            if terminal_link:
                interrupted_evidence = (
                    self._interrupted_implementation_rearm_evidence(
                        attempt,
                        receipt,
                        expected_evidence_schema=(
                            DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA
                            if stale_dispatch_migration_policy
                            else DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
                        ),
                    )
                )
                if interrupted_evidence is not None:
                    return interrupted_evidence
                if not standard_policy:
                    return None
                return self._terminal_quiescent_deferred_rearm_evidence(
                    attempt,
                    receipt,
                )

        paths = self._paths(attempt)
        expected_root = self.attempt_root / hashlib.sha256(
            str(attempt.attempt_id).encode("utf-8")
        ).hexdigest()[:24]
        if paths.root != expected_root:
            return None
        try:
            sealed = self._pinned_no_provider_snapshot(paths)
            binding = dict(sealed["binding"])
            self._verify_binding_identity(binding)
            durable_binding = (
                self._binding_lookup(attempt)
                if self._binding_lookup is not None
                else None
            )
            projection = str(sealed["projection"])
            if (
                _projection_immutable_digest(projection)
                != str(binding.get("projection_immutable_digest") or "")
                or _HEADER.findall(projection)
                != [str(binding.get("task_alias") or "")]
            ):
                raise DatabasePortalBridgeError(
                    "database Portal task projection identity changed"
                )
            projection_task = self._projection_task(paths, binding, projection)
            identity = {
                "task_id": projection_task.task_id,
                "title": str(projection_task.title or ""),
                "canonical_task_key": str(
                    projection_task.canonical_task_key or ""
                ),
                "canonical_task_cid": str(
                    projection_task.canonical_task_cid or ""
                ),
                "board_namespace": str(
                    projection_task.board_namespace or ""
                ),
            }
            if any(not value for value in identity.values()):
                raise DatabasePortalBridgeError(
                    "Portal task projection lacks a complete canonical identity"
                )
            projection_track = str(projection_task.track or "")
            if not projection_track:
                raise DatabasePortalBridgeError(
                    "Portal task projection lacks its task track"
                )
            projection_status = _projection_status(projection)
            state = dict(sealed["state"])
            state_digest = str(sealed["state_digest"])
            events = list(sealed["events"])
            manifest = dict(sealed["manifest"])
            directory_names = list(sealed["directory_names"])
        except (DatabasePortalBridgeError, OSError, TypeError, ValueError):
            return None
        authority_root = self.attempt_root
        selected_root = paths.root
        if not isinstance(durable_binding, Mapping):
            return None
        durable_expected = {
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "binding_id": str(binding.get("binding_id") or ""),
            "projection_immutable_digest": str(
                binding.get("projection_immutable_digest") or ""
            ),
            "stage": "portal_entered",
        }
        binding_expected = {
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
        }
        if (
            any(
                type(durable_binding.get(key)) is not type(value)
                or durable_binding.get(key) != value
                for key, value in durable_expected.items()
            )
            or any(
                type(binding.get(key)) is not type(value)
                or binding.get(key) != value
                for key, value in binding_expected.items()
            )
        ):
            return None

        terminal_no_effect_evidence = (
            self._terminal_no_effect_route_rearm_evidence(
                attempt,
                receipt=receipt,
                paths=paths,
                binding=binding,
                durable_binding=durable_binding,
                identity=identity,
                projection_track=projection_track,
                projection_status=projection_status,
                directory_names=directory_names,
                state=state,
                state_digest=state_digest,
                events=events,
                manifest=manifest,
                log_raw=sealed.get("terminal_no_effect_log_raw"),
                log_relative_path=str(
                    sealed.get("terminal_no_effect_log_relative_path") or ""
                ),
                log_identity_digest=str(
                    sealed.get("terminal_no_effect_log_identity_digest") or ""
                ),
            )
        )
        if terminal_no_effect_evidence is not None:
            return terminal_no_effect_evidence

        deferred_provider_evidence = self._deferred_provider_rearm_evidence(
            attempt,
            receipt=receipt,
            paths=paths,
            binding=binding,
            durable_binding=durable_binding,
            identity=identity,
            projection_track=projection_track,
            projection_status=projection_status,
            directory_names=directory_names,
            state=state,
            state_digest=state_digest,
            events=events,
            manifest=manifest,
        )
        if deferred_provider_evidence is not None:
            return deferred_provider_evidence

        task_alias = str(binding["task_alias"])
        nested_task_cid = str(identity.get("canonical_task_cid") or "")
        selected = [
            event
            for event in events
            if event.get("type") == "task_selected"
            and event.get("task_id") == task_alias
            and event.get("canonical_task_cid") == nested_task_cid
        ]
        cleanup = [
            event
            for event in events
            if event.get("type") == "failed_setup_worktree_cleanup"
            and event.get("task_id") == task_alias
        ]
        exceptions = [
            event
            for event in events
            if event.get("type") == "implementation_exception"
            and event.get("task_id") == task_alias
        ]
        finished = [
            event
            for event in events
            if event.get("type") == "implementation_finished"
            and event.get("task_id") == task_alias
        ]
        if not (
            len(selected) == len(cleanup) == len(exceptions) == len(finished) == 1
        ):
            return None
        selected_event = selected[0]
        cleanup_event = cleanup[0]
        exception_event = exceptions[0]
        finished_event = finished[0]
        nested_attempt = finished_event.get("attempt")
        if (
            isinstance(nested_attempt, bool)
            or not isinstance(nested_attempt, int)
            or nested_attempt < 1
            or cleanup_event.get("attempt") != nested_attempt
            or exception_event.get("attempt") != nested_attempt
        ):
            return None
        exact_attempt_events = [
            event
            for event in events
            if event.get("task_id") == task_alias
            and event.get("attempt") == nested_attempt
        ]
        if {str(event.get("type") or "") for event in exact_attempt_events} != {
            "failed_setup_worktree_cleanup",
            "implementation_exception",
            "implementation_finished",
        }:
            return None
        try:
            selected_sequence = int(selected_event["sequence"])
            cleanup_sequence = int(cleanup_event["sequence"])
            exception_sequence = int(exception_event["sequence"])
            finished_sequence = int(finished_event["sequence"])
            adjacent_cleanup_event = events[cleanup_sequence - 2]
            daemon_pass = events[finished_sequence]
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        setup_events = events[selected_sequence: cleanup_sequence - 2]
        # These are the only diagnostics emitted by the exact PCTDD-034
        # worktree-setup route before its terminal cleanup receipt.  They do
        # not cross a callback boundary.  Keeping this vocabulary closed is
        # important: a merely unfamiliar event is not negative evidence for
        # provider dispatch.
        allowed_setup_event_types = {
            "nested_submodule_initialization_guarded",
            "submodule_worktree_base_ref_retried",
        }
        if not (
            selected_sequence == 1
            and cleanup_sequence > selected_sequence
            and all(
                str(event.get("type") or "") in allowed_setup_event_types
                and not str(event.get("task_id") or "")
                and event.get("provider_dispatched") is not True
                and not str(event.get("implementation_commit") or "")
                and not isinstance(event.get("validation_result"), Mapping)
                and not isinstance(event.get("commit_result"), Mapping)
                and not isinstance(event.get("merge_result"), Mapping)
                for event in setup_events
            )
            and exception_sequence == cleanup_sequence + 1
            and finished_sequence == exception_sequence + 1
            and adjacent_cleanup_event.get("type") == "cleanup_finished"
            and not str(adjacent_cleanup_event.get("task_id") or "")
            and cleanup_event.get("previous_event_id")
            == adjacent_cleanup_event.get("event_id")
            and exception_event.get("previous_event_id")
            == cleanup_event.get("event_id")
            and finished_event.get("previous_event_id")
            == exception_event.get("event_id")
            and daemon_pass.get("type") == "daemon_pass"
            and daemon_pass.get("sequence") == finished_sequence + 1
            and daemon_pass.get("previous_event_id")
            == finished_event.get("event_id")
            and daemon_pass.get("event_id") == manifest.get("last_event_id")
            and not str(daemon_pass.get("task_id") or "")
        ):
            return None

        cleanup_result = cleanup_event.get("cleanup_result")
        exception_cleanup = exception_event.get("cleanup_result")
        finished_cleanup = finished_event.get("cleanup_result")
        lifecycle = (
            cleanup_result.get("lifecycle_finalize")
            if isinstance(cleanup_result, Mapping)
            else None
        )
        finished_lifecycle = finished_event.get("lifecycle_finalize")
        exception_result = finished_event.get("exception_result")
        validation = finished_event.get("validation_result")
        commit = finished_event.get("commit_result")
        merge = finished_event.get("merge_result")
        board = finished_event.get("board_completion")
        workspace_setup = finished_event.get("workspace_setup")
        worktree_path = str(finished_event.get("worktree_path") or "")
        branch = str(finished_event.get("branch") or "")
        cleanup_fields = {
            "started_at",
            "finished_at",
            "worktree_path",
            "branch",
            "removed_worktree",
            "deleted_branch",
            "cleaned",
            "submodule_cleanup",
            "lifecycle_finalize",
        }
        lifecycle_fields = {"fence", "finalized", "reason", "state"}
        finished_lifecycle_fields = {
            "fence",
            "finalized",
            "prior_reason",
            "reason",
            "state",
        }
        exception_fields = {
            "phase",
            "exception_type",
            "message",
            "worktree_path",
            "branch",
        }
        finished_exception_fields = exception_fields | {"cleanup_result"}
        if not (
            isinstance(cleanup_result, Mapping)
            and set(cleanup_result) == cleanup_fields
            and cleanup_result.get("cleaned") is True
            and cleanup_result == exception_cleanup == finished_cleanup
            and cleanup_result.get("worktree_path") == worktree_path
            and cleanup_result.get("branch") == branch
            and self._successful_submodule_cleanup(
                cleanup_result.get("submodule_cleanup"),
                outer_branch=branch,
            )
            and all(
                adjacent_cleanup_event.get(field) == value
                for field, value in cleanup_result.items()
            )
            and isinstance(lifecycle, Mapping)
            and set(lifecycle) == lifecycle_fields
            and isinstance(lifecycle.get("fence"), int)
            and not isinstance(lifecycle.get("fence"), bool)
            and lifecycle.get("finalized") is True
            and lifecycle.get("state") == "terminal"
            and lifecycle.get("reason") == "worktree_cleaned"
            and isinstance(finished_lifecycle, Mapping)
            and set(finished_lifecycle) == finished_lifecycle_fields
            and isinstance(finished_lifecycle.get("fence"), int)
            and not isinstance(finished_lifecycle.get("fence"), bool)
            and finished_lifecycle.get("finalized") is True
            and finished_lifecycle.get("state") == "terminal"
            and finished_lifecycle.get("prior_reason") == "worktree_cleaned"
            and finished_lifecycle.get("reason")
            == "implementation_attempt_finished"
            and exception_event.get("phase") == "worktree_setup"
            and exception_event.get("worktree_path") == worktree_path
            and exception_event.get("branch") == branch
            and isinstance(cleanup_event.get("exception_result"), Mapping)
            and set(cleanup_event["exception_result"]) == exception_fields
            and dict(cleanup_event["exception_result"])
            == {
                name: exception_event.get(name)
                for name in exception_fields
            }
            and isinstance(exception_result, Mapping)
            and set(exception_result) == finished_exception_fields
            and exception_result.get("phase") == "worktree_setup"
            and exception_result.get("worktree_path") == worktree_path
            and exception_result.get("branch") == branch
            and exception_result.get("exception_type")
            == exception_event.get("exception_type")
            and exception_result.get("message") == exception_event.get("message")
            and exception_result.get("cleanup_result") == cleanup_result
            and all(
                event.get("canonical_task_key")
                == identity.get("canonical_task_key")
                and event.get("canonical_task_cid") == nested_task_cid
                and event.get("board_namespace")
                == identity.get("board_namespace")
                for event in (
                    selected_event,
                    cleanup_event,
                    exception_event,
                    finished_event,
                )
            )
            and selected_event.get("track") == projection_track
            and finished_event.get("task_cid") == nested_task_cid
            and finished_event.get("provider_dispatched") is False
            and finished_event.get("attempt_consumed") is True
            and isinstance(finished_event.get("returncode"), int)
            and not isinstance(finished_event.get("returncode"), bool)
            and int(finished_event["returncode"]) != 0
            and str(finished_event.get("implementation_commit") or "") == ""
            and isinstance(commit, Mapping)
            and dict(commit) == {"committed": False}
            and isinstance(validation, Mapping)
            and dict(validation)
            == {
                "attempted": False,
                "passed": True,
                "reason": "not_run",
                "results": [],
                "returncode": 0,
            }
            and isinstance(merge, Mapping)
            and dict(merge) == {"merged": False, "reason": "not_attempted"}
            and isinstance(board, Mapping)
            and dict(board)
            == {
                "complete": False,
                "pending_merge": False,
                "reason": "implementation_or_validation_failed",
            }
            and finished_event.get("failed_preservation_result") == {}
            and isinstance(workspace_setup, Mapping)
            and set(workspace_setup)
            == {
                "cache_hit",
                "pool_enabled",
                "reused",
                "saved_duration_seconds",
                "setup_duration_seconds",
            }
            and workspace_setup.get("cache_hit") is False
            and workspace_setup.get("reused") is False
            and isinstance(workspace_setup.get("pool_enabled"), bool)
            and isinstance(finished_event.get("cache_hit"), bool)
            and finished_event.get("cache_hit") is False
            and daemon_pass.get("active_task_id") == ""
            and daemon_pass.get("selection_idle_reason") == ""
            and daemon_pass.get("ordinary_provider_dispatch_allowed") is True
            and daemon_pass.get("max_task_attempts") == 1
            and daemon_pass.get("execution_slice_task_statuses")
            == {task_alias: "ready"}
            and daemon_pass.get("execution_slice_task_cids_by_id")
            == {task_alias: nested_task_cid}
        ):
            return None
        if any(
            event.get("provider_dispatched") is True
            or bool(str(event.get("implementation_commit") or ""))
            or (
                isinstance(event.get("validation_result"), Mapping)
                and event["validation_result"].get("attempted") is True
            )
            or (
                isinstance(event.get("commit_result"), Mapping)
                and event["commit_result"].get("committed") is True
            )
            or (
                isinstance(event.get("merge_result"), Mapping)
                and any(
                    event["merge_result"].get(field) is True
                    for field in ("attempted", "queued", "merged")
                )
            )
            for event in events
        ):
            return None

        state_identity = state.get("task_identities")
        state_attempts = state.get("implementation_attempts")
        state_attempts_by_cid = state.get(
            "implementation_attempts_by_cid"
        )
        state_statuses = state.get("task_statuses")
        state_identity_record = (
            state_identity.get(task_alias)
            if isinstance(state_identity, Mapping)
            else None
        )
        expected_state_identity_fields = {
            "board_namespace",
            "canonical_task_cid",
            "canonical_task_key",
            "display_task_id",
            "identity_version",
            "semantic_fingerprint",
            "source_path",
        }
        canonical_task_key = str(identity.get("canonical_task_key") or "")
        semantic_fingerprint = canonical_task_key.rsplit("/", 1)[-1]
        if not (
            state.get("implementation_in_progress") is False
            and state.get("active_task_id") == ""
            and state.get("active_task_key") == ""
            and state.get("active_task_cid") == ""
            and state.get("active_task_title") == ""
            and state.get("active_task_track") == ""
            and state.get("active_task_started_at") == ""
            and state.get("active_attempt") == 0
            and state.get("active_phase") == ""
            and state.get("active_phase_started_at") == ""
            and state.get("active_phase_detail") == ""
            and state.get("active_log_path") == ""
            and state.get("active_worktree_path") == ""
            and state.get("active_branch") == ""
            and state.get("active_provider_runner") == {}
            and state.get("last_implementation_task_id") == task_alias
            and state.get("last_implementation_task_key")
            == canonical_task_key
            and state.get("last_implementation_task_cid") == nested_task_cid
            and state.get("last_implementation_returncode")
            == finished_event.get("returncode")
            and state.get("last_implementation_worktree_path") == worktree_path
            and state.get("last_implementation_branch") == branch
            and state.get("last_implementation_commit") == ""
            and isinstance(state.get("last_implementation_started_at"), str)
            and bool(state.get("last_implementation_started_at"))
            and isinstance(state.get("last_implementation_finished_at"), str)
            and bool(state.get("last_implementation_finished_at"))
            and isinstance(state.get("last_implementation_log_path"), str)
            and bool(state.get("last_implementation_log_path"))
            and state.get("last_proof_workflow") == {}
            and state.get("last_merge_started_at") == ""
            and state.get("last_merge_finished_at") == ""
            and state.get("last_merge_branch") == ""
            and state.get("last_merge_commit") == ""
            and state.get("last_merge_returncode") is None
            and state.get("last_merge_error") == "not_attempted"
            and isinstance(state_identity, Mapping)
            and set(state_identity) == {task_alias}
            and isinstance(state_identity_record, Mapping)
            and set(state_identity_record) == expected_state_identity_fields
            and state_identity_record.get("display_task_id") == task_alias
            and state_identity_record.get("canonical_task_key")
            == canonical_task_key
            and state_identity_record.get("canonical_task_cid")
            == nested_task_cid
            and state_identity_record.get("board_namespace")
            == identity.get("board_namespace")
            and isinstance(
                state_identity_record.get("identity_version"), int
            )
            and not isinstance(
                state_identity_record.get("identity_version"), bool
            )
            and state_identity_record.get("identity_version") == 1
            and state_identity_record.get("semantic_fingerprint")
            == semantic_fingerprint
            and state_identity_record.get("source_path")
            == str(paths.task_projection)
            and isinstance(state_attempts, Mapping)
            and dict(state_attempts) == {task_alias: nested_attempt}
            and isinstance(state_attempts_by_cid, Mapping)
            and dict(state_attempts_by_cid)
            == {nested_task_cid: nested_attempt}
            and isinstance(state_statuses, Mapping)
            and dict(state_statuses) == {task_alias: "ready"}
            and state.get("ready_task_ids") == [task_alias]
            and state.get("selectable_ready_task_ids") == [task_alias]
            and state.get("eligible_ready_task_ids") == [task_alias]
            and state.get("completed_task_ids") == []
            and state.get("external_reserved_task_ids") == []
            and state.get("assumed_completed_task_ids") == []
            and state.get("strict_deprioritized_ready_task_ids") == []
            and state.get("waiting_task_ids") == []
            and state.get("blocked_task_ids") == []
            and state.get("task_count") == 1
            and state.get("ready_count") == 1
            and state.get("selectable_ready_count") == 1
            and state.get("eligible_ready_count") == 1
            and state.get("completed_count") == 0
            and state.get("external_reserved_count") == 0
            and state.get("assumed_completed_count") == 0
            and state.get("strict_deprioritized_ready_count") == 0
            and state.get("waiting_count") == 0
            and state.get("blocked_count") == 0
            and state.get("recommended_task_id") == ""
            and state.get("recommended_actions") == []
            and state.get("task_artifacts") == {task_alias: []}
            and isinstance(state.get("task_validation"), Mapping)
            and set(state["task_validation"]) == {task_alias}
            and isinstance(state["task_validation"][task_alias], list)
            and bool(state["task_validation"][task_alias])
            and all(
                isinstance(command, str) and bool(command)
                for command in state["task_validation"][task_alias]
            )
            and state.get("protected_implementation_attempts") == {}
            and state.get("retry_budget_repair_receipts") == {}
            and state.get("retry_budget_repair_rearm_receipts") == {}
            and state.get("stale_proposal_replay_rearm_receipts") == {}
            and state.get("validation_obsolescence_rearm_receipts") == {}
            and state.get("strategy_generation") == 0
            and state.get("selection_idle_reason") == ""
            and all(
                daemon_pass.get(field) == state.get(field)
                for field in (
                    "completed_count",
                    "ready_count",
                    "selectable_ready_count",
                    "eligible_ready_count",
                    "strict_deprioritized_ready_count",
                    "waiting_count",
                    "blocked_count",
                    "active_task_id",
                    "selection_idle_reason",
                )
            )
            and daemon_pass.get("shared_completed_task_ids")
            == state.get("completed_task_ids")
            and all(
                daemon_pass.get(field) == []
                for field in (
                    "attempt_limited_task_ids",
                    "completion_receipt_task_ids",
                    "manual_completion_authority_affected_goal_ids",
                    "manual_completion_authority_dependency_task_ids",
                    "manual_completion_authority_required_task_ids",
                    "manual_completion_authority_task_ids",
                    "manual_completion_renewal_quarantined_task_ids",
                    "manual_completion_revalidation_only_task_ids",
                    "manual_completion_revalidation_task_ids",
                    "quarantined_manual_completion_status_task_ids",
                    "released_retry_budget_strategy_block_task_ids",
                    "retry_budget_rearmed_task_ids",
                    "retry_budget_reset_deferred_task_ids",
                    "retry_budget_reset_task_ids",
                    "shared_active_merge_task_ids",
                )
            )
            and daemon_pass.get(
                "manual_completion_authority_revalidation_only"
            )
            is False
            and daemon_pass.get("virgin_task_transfer")
            == {
                "granted_away_task_ids": [],
                "granted_to_lane_task_ids": [],
                "mode": "",
                "request_task_id": "",
            }
            and daemon_pass.get("projection_delta_keys")
            == [
                "active_task_cid",
                "active_task_id",
                "active_task_key",
                "active_task_started_at",
                "active_task_title",
                "active_task_track",
                "eligible_ready_count",
                "eligible_ready_task_ids",
                "heartbeat_at",
                "last_progress_at",
                "ready_count",
                "ready_task_ids",
                "recommended_actions",
                "recommended_task_id",
                "selectable_ready_count",
                "selectable_ready_task_ids",
                "task_artifacts",
                "task_count",
                "task_identities",
                "task_statuses",
                "task_validation",
            ]
            and daemon_pass.get("protected_path_conflicts") == {}
        ):
            return None

        evidence: dict[str, Any] = {
            "schema": DATABASE_PORTAL_NO_PROVIDER_REARM_EVIDENCE_SCHEMA,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": task_alias,
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "attempt_root_key": paths.root.name,
            "attempt_authority_root_digest": _sha256_bytes(
                str(authority_root).encode("utf-8")
            ),
            "attempt_root_digest": _sha256_bytes(
                str(selected_root).encode("utf-8")
            ),
            "binding_id": str(binding["binding_id"]),
            "binding_admission_id": str(durable_binding.get("record_id") or ""),
            "binding_admission_digest": _sha256_bytes(
                _canonical_json(dict(durable_binding))
            ),
            "projection_immutable_digest": str(
                binding["projection_immutable_digest"]
            ),
            "nested_task_cid": nested_task_cid,
            "nested_attempt": nested_attempt,
            "event_stream_id": str(manifest["stream_id"]),
            "event_snapshot_id": str(manifest["snapshot_id"]),
            "event_manifest_digest": str(manifest.get("manifest_digest") or ""),
            "event_count": len(events),
            "event_head_sequence": int(manifest["latest_sequence"]),
            "event_head_id": str(manifest["last_event_id"]),
            "task_selected_event_id": str(selected_event["event_id"]),
            "setup_event_count": len(setup_events),
            "setup_event_ids_digest": _sha256_bytes(
                _canonical_json(
                    [str(event["event_id"]) for event in setup_events]
                )
            ),
            "cleanup_event_id": str(cleanup_event["event_id"]),
            "exception_event_id": str(exception_event["event_id"]),
            "finished_event_id": str(finished_event["event_id"]),
            "state_digest": state_digest,
            "outer_block_receipt_digest": _sha256_bytes(
                _canonical_json(receipt)
            ),
            "provider_dispatched": False,
            "validation_attempted": False,
            "commit_created": False,
            "merge_attempted": False,
            "cleanup_terminal": True,
        }
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

    @staticmethod
    def _blocked_terminal_landed_recovery_is_authorized(
        *,
        attempt: Any,
        record: Any,
        terminal_reconciliation: Any,
        durable_binding: Any,
        observed_binding: Mapping[str, Any],
    ) -> bool:
        """Validate the existing terminal link before historical recovery.

        A terminal database status necessarily changes the rendered task body
        and revision, so its original Portal projection is historical by the
        time an already-landed result can be recovered.  The historical
        projection is usable only when the canonical blocked receipt carries
        the exact immutable terminal-reconciliation link for this attempt and
        that link selects the already-admitted Portal binding.  This grants no
        provider authority; it only makes the provider-forbidden landed
        reconciliation adapter reachable.
        """

        if terminal_reconciliation is None:
            return False
        if not isinstance(terminal_reconciliation, Mapping):
            raise DatabasePortalBridgeError(
                "blocked terminal landed recovery link is malformed"
            )
        link = dict(terminal_reconciliation)
        if (
            set(link) != set(DATABASE_PORTAL_TERMINAL_RECONCILIATION_LINK_FIELDS)
            or link.get("schema")
            != DATABASE_PORTAL_TERMINAL_RECONCILIATION_LINK_SCHEMA
        ):
            raise DatabasePortalBridgeError(
                "blocked terminal landed recovery link is not closed"
            )
        unsigned = dict(link)
        evidence_id = str(unsigned.pop("evidence_id", "") or "")
        if not evidence_id or content_identity(unsigned) != evidence_id:
            raise DatabasePortalBridgeError(
                "blocked terminal landed recovery link identity failed"
            )
        body = getattr(record, "body", None)
        receipt = (
            dict(body.get("completion_receipt") or {})
            if isinstance(body, Mapping)
            else {}
        )
        record_status = str(getattr(record, "status", "") or "").strip().lower()
        expected_attempt = {
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "lease_id": str(attempt.lease_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
        }
        if (
            record_status != "blocked"
            or receipt.get("schema")
            != (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-retry-budget@1"
            )
            or receipt.get("operation") != "database_unknown_outcome_blocked"
            or receipt.get("reason") != "callback_authority_incomplete_blocked"
            or receipt.get("forced_block") is not True
            or receipt.get("authority_outcome") != "unknown"
            or receipt.get("terminal_reconciliation") != link
            or link.get("nested_reconciled") is not True
            or link.get("intended_database_disposition")
            != "blocked_unknown_outcome"
            or any(
                type(link.get(name)) is not type(expected)
                or link.get(name) != expected
                for name, expected in expected_attempt.items()
            )
            or any(
                type(receipt.get(name)) is not type(expected)
                or receipt.get(name) != expected
                for name, expected in expected_attempt.items()
            )
            or not isinstance(durable_binding, Mapping)
            or durable_binding.get("stage") != "portal_entered"
            or durable_binding.get("binding_id") != link.get("binding_id")
            or observed_binding.get("binding_id") != link.get("binding_id")
            or durable_binding.get("binding_id")
            != observed_binding.get("binding_id")
            or durable_binding.get("projection_immutable_digest")
            != observed_binding.get("projection_immutable_digest")
        ):
            raise DatabasePortalBridgeError(
                "blocked terminal landed recovery changed attempt authority"
            )
        return True

    def reconcile_quiesced_attempt(
        self,
        attempt: Any,
        *,
        terminal_reconciliation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reconcile the exact nested Portal state for one active DB attempt.

        The database attempt id selects one content-stable directory.  A
        sibling directory, task alias, or merely similar Portal state can
        never nominate work for cleanup.  Missing artifacts are an explicit
        pre-provider state; partial or mismatched artifacts fail closed.
        """

        paths = self._paths(attempt)
        try:
            confined_parent = paths.root.parent.resolve()
            authority_root = self.attempt_root.resolve()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt root is unavailable"
            ) from exc
        if (
            confined_parent != authority_root
            or self.attempt_root.is_symlink()
            or (
                self.attempt_root.exists()
                and not self.attempt_root.is_dir()
            )
            or paths.root.is_symlink()
            or any(
                path.is_symlink()
                for path in (
                    paths.binding,
                    paths.task_projection,
                    paths.state,
                )
            )
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt artifacts escape their exact root"
            )
        present = {
            "binding": paths.binding.exists(),
            "projection": paths.task_projection.exists(),
            "state": paths.state.exists(),
        }
        durable_binding = (
            self._binding_lookup(attempt)
            if self._binding_lookup is not None
            else None
        )
        record: Any | None = None
        seed = ""
        expected: Mapping[str, Any] | None = None
        try:
            record = self._record_for_attempt(self.task_source, attempt)
            seed = self._render_projection(attempt, record)
            expected = self._binding(attempt, record, seed)
        except DatabasePortalBridgeError:
            # A deleted/replaced canonical projection cannot authorize new
            # files, but a prior exact DB admission may still prove that the
            # provider boundary had not been crossed.
            record = None
        admitted_current = bool(
            isinstance(durable_binding, Mapping)
            and isinstance(expected, Mapping)
            and durable_binding.get("binding_id") == expected.get("binding_id")
            and durable_binding.get("projection_immutable_digest")
            == expected.get("projection_immutable_digest")
        )
        durable_stage = str((durable_binding or {}).get("stage") or "")

        def preportal_result(*, historical: bool) -> dict[str, Any]:
            return {
                "reconciled": True,
                "blocked": False,
                "reason": "admitted_preportal_artifacts_absent",
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(getattr(attempt, "task_alias", "") or ""),
                "attempt_number": int(attempt.attempt_number),
                "owner_session_id": str(attempt.owner_session_id),
                "attempt_root": str(paths.root),
                "binding_id": str(
                    (durable_binding or {}).get("binding_id") or ""
                ),
                "historical_binding": bool(historical),
                "nested_state": {
                    "present": False,
                    "state_path": str(paths.state),
                    "state_digest": "",
                    "active": False,
                },
                "provider_runner_fence": {
                    "applicable": False,
                    "safe_to_restart": True,
                    "fenced": False,
                    "reason": "admitted_preportal_boundary",
                },
                "provider_runner_reconciliation_authority": "not_applicable",
                "terminal_provider_evidence": False,
                "terminal_provider_receipt_id": "",
            }

        known_preportal_names = {
            paths.binding.name,
            paths.task_projection.name,
        }
        recoverable_temps: list[Path] = []
        unexpected_preportal_children: list[Path] = []
        invalid_reconciliation_store = False
        if paths.root.is_dir():
            temp_name = re.compile(
                r"^\.(?:database-attempt-binding\.json|task-projection\.md)\."
                r"[A-Za-z0-9_-]+\.tmp$"
            )
            for child in paths.root.iterdir():
                if child == paths.reconciliation:
                    if child.is_symlink() or not child.is_dir():
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                        continue
                    try:
                        receipt_paths = sorted(child.iterdir())
                    except OSError:
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                        continue
                    try:
                        recovery_targets: set[Path] = set()
                        temporary_recovery_count = 0
                        for receipt_path in receipt_paths:
                            if re.fullmatch(
                                r"[0-9a-f]{64}\.json",
                                receipt_path.name,
                            ):
                                recovery_targets.add(receipt_path)
                                continue
                            temporary_match = re.fullmatch(
                                r"\.([0-9a-f]{64}\.json)\."
                                r"[A-Za-z0-9_-]+\.(?:tmp|stage)",
                                receipt_path.name,
                            )
                            if temporary_match is not None:
                                temporary_recovery_count += 1
                                if temporary_recovery_count > 256:
                                    raise DatabasePortalBridgeError(
                                        "database Portal immutable evidence has "
                                        "too many temporary publications"
                                    )
                                # A SIGKILL may leave either a non-authoritative
                                # stage or a fully fsynced ready temp without a
                                # final pathname.  Derive only the closed
                                # content-addressed final name; locked recovery
                                # discards safe stages and strictly validates
                                # ready bytes before promotion.
                                recovery_targets.add(
                                    child / temporary_match.group(1)
                                )
                        for receipt_path in sorted(recovery_targets):
                            _recover_immutable_link_publication(receipt_path)
                        receipt_paths = sorted(child.iterdir())
                    except (OSError, DatabasePortalBridgeError):
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                        continue
                    valid_receipts = True
                    for receipt_path in receipt_paths:
                        match = re.fullmatch(r"([0-9a-f]{64})\.json", receipt_path.name)
                        try:
                            receipt_stat = receipt_path.lstat()
                        except OSError:
                            valid_receipts = False
                            break
                        if (
                            match is None
                            or receipt_path.is_symlink()
                            or not receipt_path.is_file()
                            or int(receipt_stat.st_nlink) != 1
                            or int(receipt_stat.st_size) > 1024 * 1024
                        ):
                            valid_receipts = False
                            break
                        try:
                            self.load_reconciliation_receipt(
                                attempt,
                                "sha256:" + match.group(1),
                            )
                        except DatabasePortalBridgeError:
                            valid_receipts = False
                            break
                    if not valid_receipts:
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                    continue
                if child.name in known_preportal_names:
                    continue
                try:
                    stat = child.lstat()
                except OSError:
                    unexpected_preportal_children.append(child)
                    continue
                if (
                    temp_name.fullmatch(child.name)
                    and child.is_file()
                    and not child.is_symlink()
                    and int(stat.st_nlink) == 1
                    and int(stat.st_size) <= 1024 * 1024
                    and len(recoverable_temps) < 8
                ):
                    recoverable_temps.append(child)
                else:
                    unexpected_preportal_children.append(child)
        if invalid_reconciliation_store:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation evidence store is not exact"
            )
        clean_preportal_root = bool(
            not paths.root.exists()
            or (
                paths.root.is_dir()
                and not unexpected_preportal_children
            )
        )
        if (
            isinstance(durable_binding, Mapping)
            and durable_stage == "prepared"
            and clean_preportal_root
        ):
            for temporary in recoverable_temps:
                try:
                    temporary.unlink()
                except OSError as exc:
                    raise DatabasePortalBridgeError(
                        "admitted database Portal temporary artifact could not "
                        "be quarantined"
                    ) from exc
            if recoverable_temps:
                directory_fd = os.open(paths.root, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        if not any(present.values()):
            if isinstance(durable_binding, Mapping):
                if durable_stage != "prepared" or not clean_preportal_root:
                    raise DatabasePortalBridgeError(
                        "database Portal artifacts disappeared after its "
                        "published or entered boundary"
                    )
                return preportal_result(historical=not admitted_current)
            else:
                return {
                    "reconciled": True,
                    "blocked": False,
                    "reason": "portal_attempt_artifacts_absent",
                    "attempt_id": str(attempt.attempt_id),
                    "claim_id": str(attempt.claim_id),
                    "task_cid": str(attempt.task_cid),
                    "task_alias": str(getattr(attempt, "task_alias", "") or ""),
                    "attempt_root": str(paths.root),
                    "binding_id": "",
                    "nested_state": {
                        "present": False,
                        "state_path": str(paths.state),
                        "state_digest": "",
                        "active": False,
                    },
                    "terminal_provider_evidence": False,
                }
        if present["binding"] and not present["projection"]:
            if (
                not isinstance(durable_binding, Mapping)
                or durable_stage != "prepared"
                or not clean_preportal_root
            ):
                raise DatabasePortalBridgeError(
                    "database Portal active attempt has partial binding artifacts"
                )
            observed_preportal = self._read_binding(paths.binding)
            self._verify_binding_identity(observed_preportal)
            if (
                durable_binding.get("binding_id")
                != observed_preportal.get("binding_id")
                or durable_binding.get("projection_immutable_digest")
                != observed_preportal.get("projection_immutable_digest")
            ):
                raise DatabasePortalBridgeError(
                    "database Portal partial binding lacks exact DB admission"
                )
            if admitted_current and observed_preportal == expected:
                # The durable prepared admission proves Portal construction
                # has not begun.  Recreate only the missing immutable
                # projection from the still-current canonical task; this is
                # preparation repair, not provider execution.
                _atomic_write(paths.task_projection, seed.encode("utf-8"))
                self._verify_projection(paths, expected)
                return preportal_result(historical=False)
            else:
                control_claim = dict(
                    getattr(attempt, "body", {}).get("control_claim") or {}
                )
                historical_expected = {
                    "attempt_id": str(attempt.attempt_id),
                    "claim_id": str(attempt.claim_id),
                    "task_cid": str(attempt.task_cid),
                    "task_alias": str(getattr(attempt, "task_alias", "") or ""),
                    "task_revision": int(control_claim.get("revision") or 0),
                    "fencing_token": int(attempt.fencing_token),
                    "fence_epoch": int(attempt.fence_epoch),
                    "lease_id": str(attempt.lease_id),
                }
                if any(
                    observed_preportal.get(field) != value
                    for field, value in historical_expected.items()
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal partial historical binding changed "
                        "attempt authority"
                    )
                return preportal_result(historical=True)
        if present["projection"] and not present["binding"]:
            raise DatabasePortalBridgeError(
                "database Portal active attempt has partial binding artifacts"
            )
        if not present["binding"] or not present["projection"]:
            raise DatabasePortalBridgeError(
                "database Portal active attempt has partial binding artifacts"
            )
        observed = self._read_binding(paths.binding)
        self._verify_binding_identity(observed)
        expected_root = self.attempt_root / hashlib.sha256(
            str(observed.get("attempt_id") or "").encode("utf-8")
        ).hexdigest()[:24]
        if paths.root != expected_root:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is stored under the wrong root"
            )
        if isinstance(durable_binding, Mapping) and durable_stage in {
            "prepared",
            "published",
        }:
            if not clean_preportal_root or present["state"]:
                raise DatabasePortalBridgeError(
                    "database Portal pre-entry binding has unexpected active "
                    "artifacts"
                )
            if (
                durable_binding.get("binding_id")
                != observed.get("binding_id")
                or durable_binding.get("projection_immutable_digest")
                != observed.get("projection_immutable_digest")
            ):
                raise DatabasePortalBridgeError(
                    "database Portal pre-entry binding changed DB admission"
                )
            return preportal_result(historical=observed != expected)
        if (
            not isinstance(durable_binding, Mapping)
            and self._reconciliation_binding_recorder is not None
        ):
            # One-way migration for the released predecessor: full historical
            # artifacts for the still-current task may already have crossed
            # the Portal boundary.  Admit them directly as entered; never
            # reinterpret them as prelaunch.  A replaced/deleted task cannot
            # validate a pre-admission filesystem pair and therefore remains
            # fail closed.
            if expected is None or observed != expected:
                raise DatabasePortalBridgeError(
                    "database Portal predecessor binding lacks durable admission"
                )
            self._reconciliation_binding_recorder(
                attempt,
                observed,
                "portal_entered",
            )
            durable_binding = self._binding_lookup(attempt)
            durable_stage = str((durable_binding or {}).get("stage") or "")
        if isinstance(durable_binding, Mapping) and durable_stage != "portal_entered":
            raise DatabasePortalBridgeError(
                "database Portal binding has an unknown execution stage"
            )
        terminal_landed_recovery = (
            self._blocked_terminal_landed_recovery_is_authorized(
                attempt=attempt,
                record=record,
                terminal_reconciliation=terminal_reconciliation,
                durable_binding=durable_binding,
                observed_binding=observed,
            )
            if terminal_reconciliation is not None
            else False
        )
        historical_binding = expected is None or observed != expected
        if historical_binding:
            # A legitimate canonical task replacement cannot reproduce the
            # old body/projection digests.  Validate the sealed historical
            # projection against the running attempt's immutable control
            # claim instead; the database daemon later decides whether that
            # exact old attempt is superseded.  This never makes the old
            # projection authoritative for the replacement task.
            control_claim = dict(
                getattr(attempt, "body", {}).get("control_claim") or {}
            )
            historical_expected = {
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(getattr(attempt, "task_alias", "") or ""),
                "task_revision": int(control_claim.get("revision") or 0),
                "fencing_token": int(attempt.fencing_token),
                "fence_epoch": int(attempt.fence_epoch),
                "lease_id": str(attempt.lease_id),
            }
            durable_binding = (
                self._binding_lookup(attempt)
                if self._binding_lookup is not None
                else None
            )
            if (
                not control_claim
                or any(
                    observed.get(field) != value
                    for field, value in historical_expected.items()
                )
                or not str(control_claim.get("execution_spec_cid") or "")
                or not str(control_claim.get("validation_spec_cid") or "")
                or not isinstance(durable_binding, Mapping)
                or durable_binding.get("binding_id")
                != observed.get("binding_id")
                or durable_binding.get("projection_immutable_digest")
                != observed.get("projection_immutable_digest")
            ):
                raise DatabasePortalBridgeError(
                    "database Portal historical binding does not match the "
                    "attempt control claim"
                )
            expected = observed
        projection = self._verify_projection(paths, expected)
        identity = self._projection_task_identity(paths, expected, projection)
        strict_state, strict_state_digest = self._strict_state_record(
            paths.state
        )
        nested_state = self._verify_nested_state_identity(
            paths,
            expected,
            identity,
            payload=strict_state,
            state_digest=strict_state_digest,
        )
        if terminal_landed_recovery and str(
            terminal_reconciliation.get("nested_state_digest") or ""
        ) != str(nested_state.get("state_digest") or ""):
            raise DatabasePortalBridgeError(
                "blocked terminal landed recovery nested state changed"
            )
        _state_before_fence, state_digest_before_fence = (
            self._strict_state_record(paths.state)
        )
        if state_digest_before_fence != strict_state_digest:
            return {
                "reconciled": False,
                "blocked": True,
                "reason": "nested_state_changed_before_provider_fence",
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(expected.get("task_alias") or ""),
                "attempt_root": str(paths.root),
                "binding_id": str(expected.get("binding_id") or ""),
                "nested_state": nested_state,
                "provider_runner_fence": {
                    "applicable": bool(nested_state.get("active")),
                    "safe_to_restart": False,
                    "fenced": False,
                    "reason": "nested_state_changed_before_provider_fence",
                },
                "terminal_provider_evidence": False,
            }
        from .supervisor import fence_ordinary_provider_runner

        provider_runner_fence = self._validated_provider_runner_fence(
            fence_ordinary_provider_runner(
                strict_state or {},
                grace_seconds=1.0,
            ),
            expected_runner_receipt=(strict_state or {}).get(
                "active_provider_runner"
            ),
        )
        provider_runner_reconciliation_authority = (
            # The sealed runner has a distinct descriptor/latch authority.
            # There is no public standalone validator for a dead sealed
            # receipt, so preserve that predecessor boundary and let the
            # existing Portal reconciliation path decide it.  This bridge
            # never signals a process from the sealed schema alone.
            "delegated_to_portal_sealed_authority"
            if provider_runner_fence.get("reason")
            == "sealed_provider_runner_receipt_not_applicable"
            else (
                "ordinary_provider_runner_fence"
                if provider_runner_fence.get("applicable") is True
                else "not_applicable"
            )
        )
        if (
            provider_runner_fence.get("safe_to_restart") is not True
            or (
                nested_state.get("active_phase") == "implementing"
                and provider_runner_fence.get("applicable") is not True
                and provider_runner_fence.get("reason")
                != "sealed_provider_runner_receipt_not_applicable"
            )
        ):
            fence_reason = (
                "nested_active_provider_runner_fence_missing"
                if nested_state.get("active_phase") == "implementing"
                and provider_runner_fence.get("applicable") is not True
                else "nested_provider_runner_fence_unproven"
            )
            return {
                "reconciled": False,
                "blocked": True,
                "reason": fence_reason,
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(expected.get("task_alias") or ""),
                "attempt_root": str(paths.root),
                "binding_id": str(expected.get("binding_id") or ""),
                "nested_state": nested_state,
                "provider_runner_fence": dict(provider_runner_fence),
                "provider_runner_reconciliation_authority": (
                    provider_runner_reconciliation_authority
                ),
                "terminal_provider_evidence": False,
            }
        _state_after_fence, state_digest_after_fence = (
            self._strict_state_record(paths.state)
        )
        if state_digest_after_fence != nested_state["state_digest"]:
            return {
                "reconciled": False,
                "blocked": True,
                "reason": "nested_state_changed_during_provider_fence",
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(expected.get("task_alias") or ""),
                "attempt_root": str(paths.root),
                "binding_id": str(expected.get("binding_id") or ""),
                "nested_state": nested_state,
                "provider_runner_fence": dict(provider_runner_fence),
                "provider_runner_reconciliation_authority": (
                    provider_runner_reconciliation_authority
                ),
                "terminal_provider_evidence": False,
            }
        daemon = self.portal_factory(
            paths,
            str(expected.get("task_alias") or attempt.task_cid),
        )
        reconcile = getattr(daemon, "reconcile_quiesced_active_attempt", None)
        if not callable(reconcile):
            raise DatabasePortalBridgeError(
                "portal_factory does not expose quiesced-attempt reconciliation"
            )
        terminal_provider = (
            None if historical_binding else self.recover_provider_result(attempt)
        )
        interrupted_validation_evidence = (
            None
            if terminal_landed_recovery
            else self._interrupted_validation_recovery_evidence(
                attempt,
                expected,
            )
        )
        reconcile_interrupted_validation = getattr(
            daemon,
            "reconcile_interrupted_database_validation_attempt",
            None,
        )
        interrupted_implementation_evidence = (
            None
            if (
                terminal_landed_recovery
                or terminal_provider
                or interrupted_validation_evidence is not None
            )
            else self._interrupted_implementation_retry_evidence(
                attempt,
                expected,
            )
        )
        reconcile_interrupted_implementation = getattr(
            daemon,
            "reconcile_interrupted_database_implementation_attempt",
            None,
        )
        try:
            if (
                interrupted_validation_evidence is not None
                and callable(reconcile_interrupted_validation)
            ):
                raw_reconciliation = reconcile_interrupted_validation(
                    interrupted_validation_evidence
                )
            elif (
                interrupted_implementation_evidence is not None
                and callable(reconcile_interrupted_implementation)
            ):
                raw_reconciliation = reconcile_interrupted_implementation(
                    interrupted_implementation_evidence
                )
            else:
                raw_reconciliation = reconcile()
            if not isinstance(raw_reconciliation, Mapping):
                raise DatabasePortalBridgeError(
                    "Portal nested reconciliation returned a non-object"
                )
            reconciliation = dict(raw_reconciliation)
            recover_terminal = getattr(
                daemon,
                "reconcile_provider_forbidden_terminal_result",
                None,
            )
            if (
                (not historical_binding or terminal_landed_recovery)
                and reconciliation.get("reconciled") is True
                and reconciliation.get("blocked") is not True
                and callable(recover_terminal)
            ):
                raw_terminal_recovery = recover_terminal(
                    expected_task_identity=identity,
                )
                if not isinstance(raw_terminal_recovery, Mapping):
                    raise DatabasePortalBridgeError(
                        "Portal provider-forbidden terminal recovery returned "
                        "a non-object"
                    )
                terminal_recovery = dict(raw_terminal_recovery)
                reconciliation[
                    "provider_forbidden_terminal_recovery"
                ] = terminal_recovery
                if terminal_recovery.get("blocked") is True:
                    reconciliation.update(
                        {
                            "reconciled": False,
                            "blocked": True,
                            "reason": (
                                "provider_forbidden_terminal_recovery_blocked"
                            ),
                        }
                    )
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(
                daemon, "close", None
            )
            if callable(close):
                close()
        if reconciliation.get("reconciled") is not True or reconciliation.get(
            "blocked"
        ) is True:
            return {
                "reconciled": False,
                "blocked": True,
                "reason": "nested_portal_attempt_reconciliation_blocked",
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(expected.get("task_alias") or ""),
                "attempt_root": str(paths.root),
                "binding_id": str(expected.get("binding_id") or ""),
                "nested_state": nested_state,
                "provider_runner_fence": dict(provider_runner_fence),
                "provider_runner_reconciliation_authority": (
                    provider_runner_reconciliation_authority
                ),
                "portal_reconciliation": reconciliation,
                "terminal_provider_evidence": False,
            }
        if terminal_landed_recovery:
            terminal_recovery = reconciliation.get(
                "provider_forbidden_terminal_recovery"
            )
            terminal_recovery_not_applicable = _closed_typed_record_matches(
                terminal_recovery,
                {
                    "applicable": False,
                    "blocked": False,
                    "implementation_dispatched": False,
                    "provider_dispatched": False,
                    "reason": (
                        "provider_forbidden_terminal_recovery_not_applicable"
                    ),
                    "reconciled": False,
                },
            )
            if terminal_recovery_not_applicable:
                terminal_provider_evidence = None
            elif (
                not isinstance(terminal_recovery, Mapping)
                or terminal_recovery.get("applicable") is not True
                or terminal_recovery.get("reconciled") is not True
                or terminal_recovery.get("blocked") is True
                or terminal_recovery.get("provider_dispatched") is not False
                or terminal_recovery.get("implementation_dispatched")
                is not False
            ):
                raise DatabasePortalBridgeError(
                    "blocked terminal landed recovery lacked exact provider-free "
                    "completion"
                )
            else:
                terminal_provider_evidence = self._acceptance_receipt(
                    attempt=attempt,
                    paths=paths,
                    binding=expected,
                    summaries=(),
                )
        else:
            terminal_provider_evidence = (
                None
                if historical_binding
                else terminal_provider or self.recover_provider_result(attempt)
            )
        result = {
            "reconciled": True,
            "blocked": False,
            "reason": "nested_portal_attempt_reconciled",
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(expected.get("task_alias") or ""),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "attempt_root": str(paths.root),
            "binding_id": str(expected.get("binding_id") or ""),
            "historical_binding": historical_binding,
            "nested_state": nested_state,
            "provider_runner_fence": dict(provider_runner_fence),
            "provider_runner_reconciliation_authority": (
                provider_runner_reconciliation_authority
            ),
            "portal_reconciliation": reconciliation,
            "terminal_provider_evidence": bool(terminal_provider_evidence),
            "terminal_provider_receipt_id": str(
                (terminal_provider_evidence or {}).get("receipt_id") or ""
            ),
        }
        if terminal_landed_recovery:
            result["terminal_provider_receipt"] = dict(
                terminal_provider_evidence or {}
            )
        return result

    def persist_reconciliation_receipt(
        self,
        attempt: Any,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist bounded task/attempt-linked terminal reconciliation evidence."""

        paths = self._paths(attempt)
        authoritative = {
            "schema": DATABASE_PORTAL_ATTEMPT_RECONCILIATION_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "attempt_root": str(paths.root),
        }
        supplied = dict(payload)
        supplied.pop("receipt_id", None)
        unknown = set(supplied) - _ATTEMPT_RECONCILIATION_FIELDS
        if unknown:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation payload has unknown fields: "
                + ", ".join(sorted(str(field) for field in unknown))
            )
        conflicts = [
            field
            for field, expected in authoritative.items()
            if field in supplied and supplied[field] != expected
        ]
        if conflicts:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation payload overrides authority: "
                + ", ".join(sorted(conflicts))
            )
        receipt = {**supplied, **authoritative}
        stage = str(receipt.get("stage") or "")
        if stage not in {"prepared", "commit_barrier", "terminal", "blocked"}:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation stage is not closed"
            )
        if not isinstance(receipt.get("trigger"), str) or not str(
            receipt.get("trigger") or ""
        ).strip():
            raise DatabasePortalBridgeError(
                "database Portal reconciliation trigger is missing"
            )
        if not isinstance(receipt.get("reconciled_at"), str) or not str(
            receipt.get("reconciled_at") or ""
        ).strip():
            raise DatabasePortalBridgeError(
                "database Portal reconciliation timestamp is missing"
            )
        try:
            json.dumps(receipt, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation payload is not strict JSON"
            ) from exc
        if any(receipt[field] != expected for field, expected in authoritative.items()):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation authority changed"
            )
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        try:
            confined_parent = paths.root.parent.resolve()
            authority_root = self.attempt_root.resolve()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation root is unavailable"
            ) from exc
        if (
            confined_parent != authority_root
            or self.attempt_root.is_symlink()
            or paths.root.is_symlink()
            or paths.reconciliation.is_symlink()
            or (
                paths.reconciliation.exists()
                and not paths.reconciliation.is_dir()
            )
        ):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation store escapes its exact root"
            )
        receipt_path = paths.reconciliation / (
            receipt["receipt_id"].removeprefix("sha256:") + ".json"
        )
        encoded = (
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        _atomic_write_if_absent(receipt_path, encoded)
        receipt["receipt_path"] = str(receipt_path)
        return receipt

    def load_reconciliation_receipt(
        self,
        attempt: Any,
        receipt_id: str,
        *,
        required_stage: str = "",
    ) -> dict[str, Any]:
        """Load one exact immutable reconciliation object fail closed.

        This is the replay authority for the narrow crash window after the
        canonical task CAS but before claim release/local attempt
        terminalization.  The receipt id selects one file below the exact
        attempt-id-derived root; no directory enumeration or task re-render is
        involved.
        """

        normalized_id = str(receipt_id or "").strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized_id):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt id is malformed"
            )
        paths = self._paths(attempt)
        try:
            authority_root = self.attempt_root.resolve()
            confined_parent = paths.root.parent.resolve()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation root is unavailable"
            ) from exc
        if (
            confined_parent != authority_root
            or self.attempt_root.is_symlink()
            or paths.root.is_symlink()
            or paths.reconciliation.is_symlink()
            or not paths.reconciliation.is_dir()
        ):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation store escapes its exact root"
            )
        receipt_path = paths.reconciliation / (
            normalized_id.removeprefix("sha256:") + ".json"
        )
        _recover_immutable_link_publication(receipt_path)
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt is not a regular file"
            )

        def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise DatabasePortalBridgeError(
                        "database Portal reconciliation receipt contains "
                        "duplicate keys"
                    )
                result[key] = value
            return result

        try:
            raw = receipt_path.read_bytes()
            if len(raw) > 262_144:
                raise DatabasePortalBridgeError(
                    "database Portal reconciliation receipt is oversized"
                )
            receipt = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=closed_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    DatabasePortalBridgeError(
                        "database Portal reconciliation receipt contains a "
                        "nonfinite value"
                    )
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt is unreadable"
            ) from exc
        if not isinstance(receipt, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt is malformed"
            )
        receipt = dict(receipt)
        unknown = set(receipt) - _ATTEMPT_RECONCILIATION_FIELDS
        if unknown:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt has unknown fields"
            )
        authoritative = {
            "schema": DATABASE_PORTAL_ATTEMPT_RECONCILIATION_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "attempt_root": str(paths.root),
            "receipt_id": normalized_id,
        }
        mismatched = [
            field
            for field, expected in authoritative.items()
            if type(receipt.get(field)) is not type(expected)
            or receipt.get(field) != expected
        ]
        if mismatched:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt changed authority: "
                + ", ".join(sorted(mismatched))
            )
        stage = str(receipt.get("stage") or "")
        if stage not in {
            "prepared",
            "commit_barrier",
            "terminal",
            "blocked",
        } or (
            required_stage and stage != str(required_stage)
        ):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt has the wrong stage"
            )
        unsigned = dict(receipt)
        unsigned.pop("receipt_id", None)
        if _sha256_bytes(_canonical_json(unsigned)) != normalized_id:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt identity does not verify"
            )
        return receipt

    def _interrupted_validation_recovery_evidence(
        self,
        attempt: Any,
        binding: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return one exact durable post-provider/pre-validation crash proof.

        The nested Portal state is mutable and may already have been cleared by
        an earlier crash-reconciliation pass.  Recovery may therefore use only
        an immutable database-attempt reconciliation receipt that captured the
        exact ``validating`` state while fencing the exact ordinary provider
        birth.  Repeated receipts for the same exact recovery identity are
        coalesced; distinct qualifying identities remain ambiguous.
        """

        paths = self._paths(attempt)
        if not paths.reconciliation.exists():
            return None
        if (
            paths.reconciliation.is_symlink()
            or not paths.reconciliation.is_dir()
        ):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation evidence store is not exact"
            )
        try:
            receipt_paths = sorted(paths.reconciliation.iterdir())
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation evidence store is unreadable"
            ) from exc
        task_alias = str(binding.get("task_alias") or "")
        binding_id = str(binding.get("binding_id") or "")
        matches: list[dict[str, Any]] = []
        for receipt_path in receipt_paths:
            match = re.fullmatch(r"([0-9a-f]{64})\.json", receipt_path.name)
            if match is None:
                raise DatabasePortalBridgeError(
                    "database Portal reconciliation evidence name is malformed"
                )
            receipt = self.load_reconciliation_receipt(
                attempt,
                "sha256:" + match.group(1),
            )
            if receipt.get("stage") != "blocked":
                continue
            nested = receipt.get("nested_state")
            fence = receipt.get("provider_runner_fence")
            portal = receipt.get("portal_reconciliation")
            if not (
                receipt.get("blocked") is True
                and receipt.get("reconciled") is False
                and receipt.get("reason")
                == "nested_portal_attempt_reconciliation_blocked"
                and receipt.get("binding_id") == binding_id
                and receipt.get("task_alias") == task_alias
                and receipt.get("terminal_provider_evidence") is False
                and receipt.get("provider_runner_reconciliation_authority")
                == "ordinary_provider_runner_fence"
                and isinstance(nested, Mapping)
                and nested.get("active") is True
                and nested.get("active_phase") == "validating"
                and nested.get("active_task_id") == task_alias
                and isinstance(nested.get("active_attempt"), int)
                and not isinstance(nested.get("active_attempt"), bool)
                and int(nested.get("active_attempt") or 0) > 0
                and isinstance(nested.get("active_worktree_path"), str)
                and bool(str(nested.get("active_worktree_path") or "").strip())
                and isinstance(nested.get("active_branch"), str)
                and bool(str(nested.get("active_branch") or "").strip())
                and nested.get("state_path") == str(paths.state)
                and isinstance(fence, Mapping)
                and fence.get("applicable") is True
                and fence.get("fenced") is True
                and fence.get("safe_to_restart") is True
                and fence.get("reason")
                == "ordinary_provider_runner_exact_birth_fenced"
                and isinstance(portal, Mapping)
                and portal.get("blocked") is True
                and portal.get("reconciled") is False
                and portal.get("reason")
                == "task_claim_reconciliation_blocked"
            ):
                continue

            protected = portal.get("protected_path_reconciliation")
            lifecycle = portal.get("worktree_lifecycle_reconciliation")
            claim = portal.get("task_claim_reconciliation")
            attempt_recovery = portal.get("attempt_recovery")
            active_attempt = int(nested["active_attempt"])
            workspace = str(nested["active_worktree_path"])
            if not (
                isinstance(protected, Mapping)
                and protected.get("blocked") is False
                and protected.get("reason") == "crash_reconciliation_unchanged"
                and protected.get("task_id") == task_alias
                and protected.get("workspace_path") == workspace
                and isinstance(lifecycle, Mapping)
                and lifecycle.get("blocked") is False
                and lifecycle.get("reconciled") is True
                and lifecycle.get("state") == "terminal"
                and lifecycle.get("task_id") == task_alias
                and lifecycle.get("workspace_path") == workspace
                and type(lifecycle.get("attempt")) is int
                and lifecycle.get("attempt") == active_attempt
                and isinstance(claim, Mapping)
                and claim.get("blocked") is True
                and claim.get("reconciled") is False
                and claim.get("reason") == "canonical_task_not_terminal"
                and claim.get("task_id") == task_alias
                and isinstance(attempt_recovery, Mapping)
                and attempt_recovery.get("consumed") is False
                and type(attempt_recovery.get("attempt")) is int
                and attempt_recovery.get("attempt") == active_attempt
                and attempt_recovery.get("task_id") == task_alias
            ):
                continue
            matches.append(receipt)

        if not matches:
            return None
        recovery_groups: dict[str, list[dict[str, Any]]] = {}
        for receipt in matches:
            nested = dict(receipt["nested_state"])
            portal = dict(receipt["portal_reconciliation"])
            lifecycle = dict(portal["worktree_lifecycle_reconciliation"])
            claim = dict(portal["task_claim_reconciliation"])
            recovery_identity = _sha256_bytes(
                _canonical_json(
                    {
                        "binding_id": binding_id,
                        "nested_state_digest": nested.get("state_digest"),
                        "active_attempt": nested.get("active_attempt"),
                        "active_task_id": nested.get("active_task_id"),
                        "active_worktree_path": nested.get(
                            "active_worktree_path"
                        ),
                        "active_branch": nested.get("active_branch"),
                        "lifecycle_record_id": lifecycle.get("record_id"),
                        "lifecycle_fence": lifecycle.get("fence"),
                        "canonical_task_cid": claim.get(
                            "canonical_task_cid"
                        ),
                    }
                )
            )
            recovery_groups.setdefault(recovery_identity, []).append(receipt)
        if len(recovery_groups) != 1:
            raise DatabasePortalBridgeError(
                "database Portal interrupted validation evidence is ambiguous"
            )
        recovery_identity, equivalent_receipts = next(
            iter(recovery_groups.items())
        )
        selected = min(
            equivalent_receipts,
            key=lambda item: str(item.get("receipt_id") or ""),
        )
        evidence = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-interrupted-validation-recovery@1"
            ),
            "binding": dict(binding),
            "recovery_identity": recovery_identity,
            "equivalent_receipt_ids": sorted(
                str(item.get("receipt_id") or "")
                for item in equivalent_receipts
            ),
            "reconciliation_receipt": selected,
        }
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

    def _stale_dispatch_migration_retry_evidence(
        self,
        attempt: Any,
        binding: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Nominate the one structural first-clear for the migration suffix."""

        paths = self._paths(attempt)
        if not paths.reconciliation.exists():
            return None
        if paths.reconciliation.is_symlink() or not paths.reconciliation.is_dir():
            raise DatabasePortalBridgeError(
                "database Portal reconciliation evidence store is not exact"
            )
        matches: list[dict[str, Any]] = []
        for path in sorted(paths.reconciliation.iterdir()):
            match = re.fullmatch(r"([0-9a-f]{64})\.json", path.name)
            if match is None:
                raise DatabasePortalBridgeError(
                    "database Portal reconciliation evidence name is malformed"
                )
            receipt = self.load_reconciliation_receipt(
                attempt,
                "sha256:" + match.group(1),
            )
            nested = receipt.get("nested_state")
            portal = receipt.get("portal_reconciliation")
            fence = receipt.get("provider_runner_fence")
            if not (
                receipt.get("stage") == "blocked"
                and receipt.get("blocked") is True
                and receipt.get("reconciled") is False
                and receipt.get("binding_id") == binding.get("binding_id")
                and receipt.get("task_alias") == binding.get("task_alias")
                and receipt.get("reason")
                == "nested_portal_attempt_reconciliation_blocked"
                and receipt.get("terminal_provider_evidence") is False
                and receipt.get("provider_runner_reconciliation_authority")
                == "ordinary_provider_runner_fence"
                and isinstance(nested, Mapping)
                and nested.get("active") is True
                and nested.get("active_phase") == "implementing"
                and nested.get("active_task_id") == binding.get("task_alias")
                and type(nested.get("active_attempt")) is int
                and int(nested.get("active_attempt") or 0) > 0
                and nested.get("state_path") == str(paths.state)
                and isinstance(fence, Mapping)
                and fence.get("applicable") is True
                and fence.get("fenced") is True
                and fence.get("safe_to_restart") is True
                and type(fence.get("pid")) is int
                and int(fence.get("pid") or 0) > 0
                and fence.get("reason")
                == "ordinary_provider_runner_exact_birth_fenced"
                and isinstance(portal, Mapping)
                and portal.get("blocked") is True
                and portal.get("reconciled") is False
                and portal.get("reason")
                == "task_claim_reconciliation_blocked"
            ):
                continue
            protected = portal.get("protected_path_reconciliation")
            lifecycle = portal.get("worktree_lifecycle_reconciliation")
            claim = portal.get("task_claim_reconciliation")
            recovery = portal.get("attempt_recovery")
            if not (
                _closed_typed_record_matches(
                    protected,
                    {
                        "blocked": False,
                        "critical_section_entered": False,
                        "reason": "no_active_snapshot",
                        "scan_outside_lease": True,
                    },
                )
                and isinstance(lifecycle, Mapping)
                and lifecycle.get("blocked") is False
                and lifecycle.get("reconciled") is True
                and lifecycle.get("state") == "terminal"
                and lifecycle.get("task_id") == nested.get("active_task_id")
                and lifecycle.get("canonical_task_cid")
                and type(lifecycle.get("attempt")) is int
                and lifecycle.get("attempt") == nested.get("active_attempt")
                and lifecycle.get("workspace_path")
                == nested.get("active_worktree_path")
                and lifecycle.get("record_id")
                and type(lifecycle.get("fence")) is int
                and int(lifecycle.get("fence") or 0) > 0
                and isinstance(claim, Mapping)
                and claim.get("blocked") is True
                and claim.get("reconciled") is False
                and claim.get("reason") == "canonical_task_not_terminal"
                and claim.get("observed_task_status") == "todo"
                and claim.get("task_id") == nested.get("active_task_id")
                and claim.get("canonical_task_cid")
                and isinstance(recovery, Mapping)
                and recovery.get("consumed") is False
                and type(recovery.get("attempt")) is int
                and recovery.get("attempt") == nested.get("active_attempt")
                and recovery.get("task_id") == nested.get("active_task_id")
                and recovery.get("canonical_task_cid")
                == claim.get("canonical_task_cid")
                and type(recovery.get("previous_display_count")) is int
                and recovery.get("previous_display_count")
                == nested.get("active_attempt")
                and type(recovery.get("previous_cid_count")) is int
                and recovery.get("previous_cid_count")
                == nested.get("active_attempt")
            ):
                continue
            matches.append(receipt)
        if len(matches) > 1:
            raise DatabasePortalBridgeError(
                "database Portal stale-dispatch migration evidence is ambiguous"
            )
        if not matches:
            return None
        evidence = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "stale-dispatch-release-migration-retry@1"
            ),
            "binding_id": str(binding.get("binding_id") or ""),
            "reconciliation_receipt": matches[0],
        }
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

    def _interrupted_implementation_retry_evidence(
        self,
        attempt: Any,
        binding: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Nominate one immutable implementing first-clear receipt."""

        paths = self._paths(attempt)
        if not paths.reconciliation.exists():
            return None
        if paths.reconciliation.is_symlink() or not paths.reconciliation.is_dir():
            raise DatabasePortalBridgeError(
                "database Portal reconciliation evidence store is not exact"
            )
        matches: list[dict[str, Any]] = []
        for path in sorted(paths.reconciliation.iterdir()):
            match = re.fullmatch(r"([0-9a-f]{64})\.json", path.name)
            if match is None:
                raise DatabasePortalBridgeError(
                    "database Portal reconciliation evidence name is malformed"
                )
            receipt = self.load_reconciliation_receipt(
                attempt,
                "sha256:" + match.group(1),
            )
            nested = receipt.get("nested_state")
            portal = receipt.get("portal_reconciliation")
            fence = receipt.get("provider_runner_fence")
            if not (
                receipt.get("stage") == "blocked"
                and receipt.get("blocked") is True
                and receipt.get("reconciled") is False
                and receipt.get("binding_id") == binding.get("binding_id")
                and receipt.get("task_alias")
                == binding.get("task_alias")
                and receipt.get("reason")
                == "nested_portal_attempt_reconciliation_blocked"
                and receipt.get("terminal_provider_evidence") is False
                and receipt.get("provider_runner_reconciliation_authority")
                == "ordinary_provider_runner_fence"
                and isinstance(nested, Mapping)
                and nested.get("active") is True
                and nested.get("active_phase") == "implementing"
                and nested.get("active_task_id") == binding.get("task_alias")
                and isinstance(nested.get("active_attempt"), int)
                and not isinstance(nested.get("active_attempt"), bool)
                and int(nested.get("active_attempt") or 0) > 0
                and nested.get("state_path") == str(paths.state)
                and isinstance(fence, Mapping)
                and fence.get("applicable") is True
                and fence.get("fenced") is True
                and fence.get("safe_to_restart") is True
                and isinstance(fence.get("pid"), int)
                and not isinstance(fence.get("pid"), bool)
                and int(fence.get("pid") or 0) > 0
                and fence.get("reason")
                == "ordinary_provider_runner_exact_birth_fenced"
                and isinstance(portal, Mapping)
                and portal.get("blocked") is True
                and portal.get("reconciled") is False
                and portal.get("reason")
                == "task_claim_reconciliation_blocked"
            ):
                continue
            protected = portal.get("protected_path_reconciliation")
            lifecycle = portal.get("worktree_lifecycle_reconciliation")
            claim = portal.get("task_claim_reconciliation")
            recovery = portal.get("attempt_recovery")
            if not (
                isinstance(protected, Mapping)
                and protected.get("blocked") is False
                and protected.get("reason") == "crash_reconciliation_unchanged"
                and protected.get("task_id") == nested.get("active_task_id")
                and protected.get("workspace_path")
                == nested.get("active_worktree_path")
                and isinstance(lifecycle, Mapping)
                and lifecycle.get("blocked") is False
                and lifecycle.get("reconciled") is True
                and lifecycle.get("state") == "terminal"
                and lifecycle.get("task_id") == nested.get("active_task_id")
                and type(lifecycle.get("attempt")) is int
                and lifecycle.get("attempt") == nested.get("active_attempt")
                and lifecycle.get("workspace_path")
                == nested.get("active_worktree_path")
                and lifecycle.get("record_id")
                and isinstance(lifecycle.get("fence"), int)
                and not isinstance(lifecycle.get("fence"), bool)
                and int(lifecycle.get("fence") or 0) > 0
                and isinstance(claim, Mapping)
                and claim.get("blocked") is True
                and claim.get("reconciled") is False
                and claim.get("reason") == "canonical_task_not_terminal"
                and claim.get("task_id") == nested.get("active_task_id")
                and claim.get("observed_task_status") == "todo"
                and claim.get("canonical_task_cid")
                and isinstance(recovery, Mapping)
                and recovery.get("consumed") is False
                and type(recovery.get("attempt")) is int
                and recovery.get("attempt") == nested.get("active_attempt")
                and recovery.get("task_id") == nested.get("active_task_id")
                and recovery.get("canonical_task_cid")
                == claim.get("canonical_task_cid")
                and type(recovery.get("previous_display_count")) is int
                and recovery.get("previous_display_count")
                == nested.get("active_attempt")
                and type(recovery.get("previous_cid_count")) is int
                and recovery.get("previous_cid_count")
                == nested.get("active_attempt")
            ):
                continue
            matches.append(receipt)
        if len(matches) > 1:
            raise DatabasePortalBridgeError(
                "database Portal interrupted implementation evidence is ambiguous"
            )
        if not matches:
            return None
        receipt = matches[0]
        evidence = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-interrupted-implementation-retry@1"
            ),
            "binding_id": str(binding.get("binding_id") or ""),
            "reconciliation_receipt": receipt,
        }
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

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
    "DATABASE_PORTAL_ATTEMPT_RECONCILIATION_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA",
    "DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_FIELDS",
    "DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_SCHEMA",
    "DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_FIELDS",
    "DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA",
    "DatabasePortalAttemptPaths",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalProviderRouteDeferred",
    "DatabasePortalBridgeError",
    "DatabasePortalPreEntryPublicationDeferred",
    "DatabasePortalExecutionBridge",
    "PortalDaemonFactory",
)
