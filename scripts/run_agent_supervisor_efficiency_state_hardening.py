#!/usr/bin/env python3
"""Bootstrap and operate the bounded ASEH board on the existing supervisor.

This is a program adapter, not a supervisor. It materializes sealed Markdown
once into DatabaseTaskSource, starts the existing Quack owner, enables the
PID-bound typed-grant handoff repaired by ASEH-BOOTSTRAP-001, and invokes the
existing configured-board implementation supervisor in the foreground.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (  # noqa: E402
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor import (  # noqa: E402
    KNOWN_NON_WORKTREE_PHASES,
)

DEFAULT_CONFIG: Final = Path(
    "config/agent_supervisor_efficiency_state_hardening_scheduler.json"
)
PROGRAM: Final = "agent-supervisor-efficiency-and-state-hardening-v1"
OPERATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-program-operator@1"
)
POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-population@1"
)
BOOTSTRAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-bootstrap@1"
)
REPAIR_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-bootstrap-repair-transition@1"
)
REPAIR_TRANSITION_TASK_ID: Final = "ASEH-BOOTSTRAP-002"
REPAIR_TRANSITION_BASE_HEAD: Final = (
    "6b1bbc34510fde1e11760c7eef520369dafdd3b5"
)
REPAIR_TRANSITION_CHANGED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/merge/merge_queue.py",
    "ipfs_accelerate_py/agent_supervisor/merge/merge_train.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_board_scheduler.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_database_portal_bridge.py",
    "test/api/test_agent_supervisor_merge_train.py",
)
REPAIR_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_merge_train.py", "-k",
        (
            "portal_projection or "
            "completed_queue_row_is_not_task_completion or "
            "integrated_pending_validation or false_completion"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_board_scheduler.py", "-k",
        (
            "non_dumpable_root_omitted_by_profile_scan or "
            "opaque_detached_child_before_root_term_exit"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py", "-k",
        (
            "blocked_reconciliation or canonical_merge_suffix or "
            "repair_transition or offline_continuity_replay"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_portal_bridge.py", "-k",
        (
            "post_merge_rearm_endpoints_fail_closed or "
            "false_completion_recheck_reuses_observation_timestamp"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "-k",
        (
            "reconcile_rearms_blocked_portal_provider_failed or "
            "reconcile_rearms_blocked_checkout_contention or "
            "portal_setup_error_requeues"
        ),
    ),
)
REPAIR_FOLLOWUP_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "aseh-bootstrap-repair-followup-transition@1"
)
REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD: Final = (
    "73a06a7d6f8303cbfbeed4662e847cde5d71a4d4"
)
REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT: Final = (
    "a398362913bba4bce0d3ecf1b62cee7f67e0c72a"
)
REPAIR_FOLLOWUP_TRANSITION_CANDIDATE: Final = (
    "7d4f19ef5a6f59beacffd34804c3465a76f079c5"
)
REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS: Final = "ASEH-001"
REPAIR_FOLLOWUP_TRANSITION_CHANGED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/worktrees.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
    "test/api/test_agent_supervisor_database_portal_bridge.py",
    "test/api/test_agent_supervisor_incremental_runtime.py",
    "test/api/test_agent_supervisor_quack_transport_defaults.py",
    "test/api/test_agent_supervisor_todo_daemon_port.py",
)
REPAIR_FOLLOWUP_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "-k", "canonical_merge_suffix or repair_transition",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_incremental_runtime.py", "-k",
        (
            "worktree_pool_discard_removes_locked_missing_registration_and_sidecar "
            "or worktree_pool_discard_preserves_state_until_registry_is_verified"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_todo_daemon_port.py", "-k",
        (
            "implementation_daemon_repairs_locked_missing_merged_worktree_registration "
            "or implementation_daemon_run_once_cleans_already_merged_worktree "
            "or implementation_daemon_fences_preparing_worktree_from_peer_merged_cleanup "
            "or implementation_supervisor_repairs_locked_missing_merged_registration_under_checkout_lock "
            "or implementation_supervisor_tolerates_worktree_removed_during_cleanup "
            "or implementation_supervisor_defers_worktree_cleanup_behind_checkout_lock "
            "or implementation_supervisor_keeps_peer_lane_active_worktree"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_portal_bridge.py", "-k",
        (
            "post_merge_wrapper_accepts_false_completion_reintegration_schema "
            "or post_merge_rearm_endpoints_fail_closed_on_invalid_payloads"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py", "-k",
        (
            "post_merge_recovery_settles_before_claiming_next_task "
            "or quack_transport_unavailable_defers_whole_pass_without_claim "
            "or quack_transport_unavailable_after_preflight_marks_effects_unknown "
            "or quack_transport_deferral_rejects_untyped_or_foreign_endpoint_errors "
            "or database_portal_reason_does_not_remint_application_failures_as_quack "
            "or quack_attach_contention_defers_instead_of_crashing "
            "or quack_attach_contention_requests_owner_board_unstall "
            "or quack_attach_contention_still_expires_running_attempts"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_quack_transport_defaults.py",
    ),
)
REPAIR_CLEAN_LAUNCH_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "aseh-bootstrap-repair-clean-launch-transition@1"
)
REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD: Final = (
    "3760a6fccb1634fef98b5e68d6697832079d6fe1"
)
REPAIR_CLEAN_LAUNCH_TRANSITION_CHANGED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_quack_transport_defaults.py",
)
REPAIR_CLEAN_LAUNCH_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "-k", "repair_clean_launch_transition",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_quack_transport_defaults.py",
        "-k", (
            "quack_mutation_timeout_is_unknown_outcome_without_internal_replay "
            "or quack_token_vault_path_anchors_relative_database_to_admitted_root "
            "or resolve_quack_attach_token_persists_missing_vault"
        ),
    ),
)
REPAIR_RUNTIME_HARDENING_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "aseh-bootstrap-repair-runtime-hardening-transition@1"
)
REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD: Final = (
    "205f3b514cbdf74af67cabdfc045b008e03900c2"
)
REPAIR_RUNTIME_HARDENING_TRANSITION_CHANGED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
    "ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_board_scheduler.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_database_coordination.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
    "test/api/test_agent_supervisor_database_portal_bridge.py",
    "test/api/test_agent_supervisor_todo_daemon_port.py",
    "test/api/test_agent_supervisor_worktree_lifecycle.py",
)
REPAIR_RUNTIME_HARDENING_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "-k", (
            "startup_honors_admitted_blocked_recovery_past_thirty_seconds "
            "or startup_fails_when_blocked_recovery_admission_is_lost "
            "or post_admission_grace_is_exclusive_to_typed_lane_loss "
            "or external_status_accepts_only_monotonic_replica_successor "
            "or repair_runtime_hardening_transition"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_board_scheduler.py",
        "-k", "multi_runner_stop_tracks",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "-k", (
            "deterministic_reconciliation or "
            "reclaims_only_dead_exact_lane_portal_lifecycle_claims"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_coordination.py",
        "-k", "released_same_key_retry_creates_new_claim",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "-k", (
            "false_completion_claim_settles_without_provider_or_effect "
            "or false_completion_target_consumer_contention_is_typed_and_replayable "
            "or false_completion_target_advance_before_fenced_cas_fails_closed "
            "or false_completion_deferral_preserves_zero_provider_retry_lineage "
            "or same_status_control_replay "
            "or reconcile_rearms_blocked_portal_provider_failed "
            "or reconcile_rearms_blocked_checkout_contention"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_todo_daemon_port.py",
        "-k", "database_deterministic_reconciliation_ignores_stale_merge_completion",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_worktree_lifecycle.py",
        "-k", "controlled_restart_reclaims_only_dead_same_lane_owner",
    ),
)
REPAIR_QUACK_RECOVERY_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "aseh-bootstrap-repair-quack-recovery-transition@1"
)
REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD: Final = (
    "e96e2b722f12a407848fb835659a4528ea3f8cd6"
)
REPAIR_QUACK_RECOVERY_TRANSITION_CHANGED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_database_coordination.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
    "test/api/test_agent_supervisor_database_portal_bridge.py",
    "test/api/test_agent_supervisor_quack_transport_defaults.py",
)
REPAIR_QUACK_RECOVERY_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "-k", "repair_quack_recovery_transition",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_coordination.py",
        "-k", (
            "open_rebuilds_stale_ready_index_before_updates "
            "or released_same_key_retry_creates_new_claim"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "-k", (
            "bridge_types_current_quack_refusal_before_portal_dispatch "
            "or bridge_does_not_type_foreign_quack_refusal_as_pre_dispatch "
            "or bridge_seals_historical_quack_preprojection_absence "
            "or bridge_quack_preprojection_recovery_rejects_any_projection_artifact "
            "or bridge_quack_preprojection_recovery_rejects_foreign_endpoint "
            "or configured_runner_binds_post_merge_recovery_when_queue_is_target_bound"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "-k", (
            "reconcile_recovers_exact_quack_preprojection_transport_failure "
            "or quack_preprojection_recovery_rejects_a_committed_provider_outcome "
            "or quack_transport_deferral_does_not_consume_model_or_spin_budget"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_quack_transport_defaults.py",
    ),
)
REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "aseh-bootstrap-repair-parallel-blocked-startup-transition@1"
)
REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD: Final = (
    "c742ae30ca72440ac3598eb07a677ff0e124e611"
)
REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_CHANGED_PATHS: Final = (
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
)
REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "-k", (
            "repair_parallel_blocked_startup_transition "
            "or health_admits_parallel_blocked_recovery_only_during_startup "
            "or startup_fails_when_blocked_recovery_admission_is_lost"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "-k", "bridge_seals_historical_quack_preprojection_absence",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "-k", "reconcile_recovers_exact_quack_preprojection_transport_failure",
    ),
)
REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "aseh-bootstrap-repair-quack-publication-contention-transition@1"
)
REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD: Final = (
    "386fd3626e210495f99116637c96ed43035901fa"
)
REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_CHANGED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
    "database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
    "implementation_daemon.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
    "test/api/test_agent_supervisor_database_portal_bridge.py",
)
REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "-k", (
            "repair_quack_publication_contention_transition "
            "or status_sample_retries_exact_owner_publication_races "
            "or status_sample_does_not_retry_foreign_replica_failures "
            "or health_admits_parallel_blocked_recovery_only_during_startup"
        ),
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "-k", "quack_refusal_during_portal_construction",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "-k", (
            "quack_preprojection_recovery_supersedes_only_expired_same_task_queue_lineage "
            "or quack_preprojection_recovery_rejects_live_or_current_queue_lineage "
            "or reconcile_recovers_exact_quack_preprojection_transport_failure"
        ),
    ),
)
REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_AUTHORITY: Final = (
    "the operator explicitly directed the bootstrap engineering agent to fix "
    "the existing supervisor so exact Quack replica-publication races defer "
    "and recover automatically without weakening single-writer, identity, "
    "CAS, fencing, validation, or unknown-outcome gates"
)
REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "aseh-bootstrap-repair-quack-recovery-replay-transition@1"
)
REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD: Final = (
    "43df7ebbc0a78d8acd58079a32abfed3c26f55bb"
)
REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_CHANGED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
    "implementation_daemon.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
)
REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "-k", "repair_quack_recovery_replay_transition",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "-k", (
            "quack_preprojection_recovery_supersedes_only_expired_same_task_queue_lineage "
            "or quack_preprojection_recovery_rejects_live_or_current_queue_lineage "
            "or reconcile_recovers_exact_quack_preprojection_transport_failure"
        ),
    ),
)
REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_AUTHORITY: Final = (
    "the operator explicitly directed the bootstrap engineering agent to fix "
    "the existing supervisor so canonical Quack recovery receipts with "
    "expired same-task predecessor lineage replay idempotently without "
    "weakening closed-receipt, lease, fence, queue, provider, effect, or lane "
    "health gates"
)
REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "aseh-bootstrap-repair-control-receipt-lifecycle-transition@1"
)
REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD: Final = (
    "fd70443429b1d345812995929a88b8096405cac7"
)
REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_CHANGED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
    "implementation_daemon.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
)
REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_VALIDATIONS: Final = (
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "-k", "repair_control_receipt_lifecycle_transition",
    ),
    (
        sys.executable, "-m", "pytest", "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "-k", (
            "recovery_control_receipts_bind_only_exact_preserved_reopen_count "
            "or quack_preprojection_recovery_supersedes_only_expired_same_task_queue_lineage "
            "or restart_accepts_exact_validation_retry_recovery_projection "
            "or reconcile_rearms_blocked_checkout_contention "
            "or reconcile_reopens_inflight_deferral_budget_block"
        ),
    ),
)
REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_AUTHORITY: Final = (
    "the operator explicitly directed the bootstrap engineering agent to fix "
    "the existing supervisor so the canonical state store's monotonic "
    "unknown-callback reopen counter survives and replays through every "
    "closed recovery receipt without permitting any other receipt extension "
    "or weakening task, lease, fence, provider, effect, or lane health gates"
)
BOOTSTRAP_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema", "source_head", "repository_tree_id", "plan_root_cid",
        "source_forest", "source_identities", "database_task_source_receipt",
        "snapshot", "integrity", "initial_ready_task_ids",
        "bootstrap_validation", "recovered_after_interrupted_materialization",
        "authority", "ducklake_projection", "bootstrap_receipt_id",
    }
)
REPAIR_TRANSITION_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema", "task_id", "stable_identity", "program_id",
        "bootstrap_receipt_id", "plan_root_cid", "repository_tree_id",
        "base_head", "base_tree", "repair_head", "repair_tree",
        "changed_paths", "patch_digest", "dependencies",
        "owning_repository", "risk_class", "authority_requirement",
        "validation_results", "terminal_success_criteria",
        "terminal_non_success_criteria", "semantic_corpus_changed",
        "database_mutated", "authorized_at", "receipt_cid",
    }
)
REPAIR_FOLLOWUP_TRANSITION_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema", "task_id", "stable_identity", "program_id",
        "transition_revision", "bootstrap_receipt_id",
        "previous_receipt_cid", "base_integration_witness",
        "authorization_task_observation",
        "plan_root_cid", "repository_tree_id", "base_head", "base_tree",
        "repair_head", "repair_tree", "changed_paths", "patch_digest",
        "dependencies", "owning_repository", "risk_class",
        "authority_requirement", "validation_results",
        "terminal_success_criteria", "terminal_non_success_criteria",
        "semantic_corpus_changed", "database_mutated", "authorized_at",
        "receipt_cid",
    }
)
REPAIR_CLEAN_LAUNCH_TRANSITION_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema", "task_id", "stable_identity", "program_id",
        "transition_revision", "bootstrap_receipt_id",
        "previous_receipt_cid", "plan_root_cid", "repository_tree_id",
        "base_head", "base_tree", "repair_head", "repair_tree",
        "changed_paths", "patch_digest", "dependencies",
        "owning_repository", "risk_class", "authority_requirement",
        "validation_results", "terminal_success_criteria",
        "terminal_non_success_criteria", "semantic_corpus_changed",
        "database_mutated", "authorized_at", "receipt_cid",
    }
)
REPAIR_RUNTIME_HARDENING_TRANSITION_RECEIPT_FIELDS: Final = (
    REPAIR_CLEAN_LAUNCH_TRANSITION_RECEIPT_FIELDS
)
REPAIR_QUACK_RECOVERY_TRANSITION_RECEIPT_FIELDS: Final = (
    REPAIR_CLEAN_LAUNCH_TRANSITION_RECEIPT_FIELDS
)
REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_RECEIPT_FIELDS: Final = (
    REPAIR_CLEAN_LAUNCH_TRANSITION_RECEIPT_FIELDS
)
REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_RECEIPT_FIELDS: Final = (
    REPAIR_CLEAN_LAUNCH_TRANSITION_RECEIPT_FIELDS
)
REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_RECEIPT_FIELDS: Final = (
    REPAIR_CLEAN_LAUNCH_TRANSITION_RECEIPT_FIELDS
)
REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_RECEIPT_FIELDS: Final = (
    REPAIR_CLEAN_LAUNCH_TRANSITION_RECEIPT_FIELDS
)
REPAIR_FOLLOWUP_BASE_WITNESS_FIELDS: Final = frozenset(
    {
        "schema", "base_head", "base_tree", "target_head", "target_tree",
        "request_id", "merge_request_cid", "task_id", "task_cid",
        "candidate_commit", "candidate_tree", "integration_commit",
        "integration_tree", "baseline_ref", "changed_paths",
        "validation_proof_cid", "completion_authoritative",
        "task_completion_admitted", "receipt_cid",
    }
)
REPAIR_FOLLOWUP_TASK_OBSERVATION_FIELDS: Final = frozenset(
    {
        "task_id", "task_cid", "status", "revision",
        "completion_authoritative", "observed_at",
    }
)
DUCKLAKE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-ducklake-projection@1"
)
GOAL_RE: Final = re.compile(r"^## (ASEH-G\d{3}) (.+)$", re.MULTILINE)
META_RE: Final = re.compile(r"^- ([^:\n]+):[ \t]*(.*)$", re.MULTILINE)
READY_STATUSES: Final = frozenset(
    {"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"}
)
ACTIVE_STATUSES: Final = frozenset({"claimed", "in_progress", "running"})
COMPLETED_STATUSES: Final = frozenset({"complete", "completed", "done", "skipped"})
TERMINAL_STATUSES: Final = frozenset(
    {*COMPLETED_STATUSES, "cancelled", "failed", "quarantined", "rejected"}
)
LIVE_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-live-status@2"
)
OWNER_LOCK_SUFFIX: Final = ".state-owner.lock"
OWNER_MARKER_SUFFIX: Final = ".state-owner.json"
STATUS_SAMPLE_INTERVAL_SECONDS: Final = 0.5
STATUS_REPLICA_STABILITY_ATTEMPTS: Final = 8
STATUS_REPLICA_RETRY_DELAY_SECONDS: Final = 0.05
STATUS_RECEIPT_MAX_BYTES: Final = 1_048_576
LIVE_REPLAY_MAX_BYTES: Final = 1_073_741_824
LIVE_REPLAY_IO_TIMEOUT_SECONDS: Final = 60.0
LIVE_REPLAY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/live-projection-shadow-replay@1"
)
LIVE_REPLAY_FIELDS: Final = frozenset(
    {
        "schema", "method", "authoritative", "mutation_authority",
        "projection_matches_events", "replica", "projection_cid",
        "event_cursor", "cache_key", "witness_cid",
    }
)
_LIVE_REPLAY_CACHE: dict[str, Any] = {}
_LIVE_REPLAY_CACHE_LOCK = threading.Lock()


class OperatorError(RuntimeError):
    """Fail-closed ASEH program error."""


class OperatorStopRequested(OperatorError):
    """The operator received an orderly process-stop signal."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(f"operator stop requested by signal {self.signum}")


@contextmanager
def _stop_signal_handlers(
    requested: threading.Event, received: dict[str, int]
) -> Any:
    """Install reversible SIGINT/SIGTERM handlers for orderly shutdown."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    prior: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        received.setdefault("signum", int(signum))
        requested.set()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            prior[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        yield
    finally:
        for signum, handler in prior.items():
            signal.signal(signum, handler)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_create(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable JSON object without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    temporary = path.with_name(
        f".{path.name}.create.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            raise OperatorError(
                "immutable runtime receipt already exists; reload and validate it"
            )
        descriptor = os.open(temporary, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            # Every cooperating authorizer holds the directory inode lock and
            # rechecks absence before this rename.  The final path therefore
            # becomes visible only after complete bytes are durable, without
            # the two-link crash window of a hard-link publication.
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            os.fsync(directory)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        try:
            fcntl.flock(directory, fcntl.LOCK_UN)
        finally:
            os.close(directory)


def _secure_runtime_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    """Read one same-UID, single-link runtime object without following links."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise OperatorError("runtime authority reads require O_NOFOLLOW")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size <= 0
            or opened.st_size > max_bytes
        ):
            raise OperatorError("runtime authority file identity is unsafe")
        chunks: list[bytes] = []
        remaining = int(opened.st_size)
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                raise OperatorError("runtime authority file was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = os.lstat(path)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev, value.st_ino, value.st_mode, value.st_uid,
            value.st_nlink, value.st_size, value.st_mtime_ns,
        )

    if (
        identity(opened) != identity(after)
        or identity(opened) != identity(path_after)
        or stat.S_ISLNK(path_after.st_mode)
    ):
        raise OperatorError("runtime authority file changed during read")
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError("runtime authority file is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise OperatorError("runtime authority JSON must be an object")
    return payload


def _bootstrap_receipt_id(payload: Mapping[str, Any]) -> str:
    """Validate the closed bootstrap receipt and return its content identity."""

    if (
        payload.get("schema") != BOOTSTRAP_SCHEMA
        or set(payload) != BOOTSTRAP_RECEIPT_FIELDS
    ):
        raise OperatorError("bootstrap receipt schema or fields are invalid")
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("bootstrap_receipt_id", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError("bootstrap receipt CID is invalid")
    return receipt_id


def _repair_transition_receipt_id(payload: Mapping[str, Any]) -> str:
    """Validate the one bounded bootstrap-repair transition receipt."""

    if (
        payload.get("schema") != REPAIR_TRANSITION_SCHEMA
        or set(payload) != REPAIR_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError("bootstrap repair transition schema is invalid")
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError("bootstrap repair transition CID is invalid")
    return receipt_id


def _repair_followup_transition_receipt_id(
    payload: Mapping[str, Any],
) -> str:
    """Validate the closed revision-2 receipt without changing revision 1."""

    if (
        payload.get("schema") != REPAIR_FOLLOWUP_TRANSITION_SCHEMA
        or set(payload) != REPAIR_FOLLOWUP_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("transition_revision") != 2
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError("bootstrap repair follow-up schema is invalid")
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError("bootstrap repair follow-up CID is invalid")
    return receipt_id


def _repair_clean_launch_transition_receipt_id(
    payload: Mapping[str, Any],
) -> str:
    """Validate the closed revision-3 launch-cleanliness receipt."""

    if (
        payload.get("schema") != REPAIR_CLEAN_LAUNCH_TRANSITION_SCHEMA
        or set(payload) != REPAIR_CLEAN_LAUNCH_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("transition_revision") != 3
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError("bootstrap repair clean-launch schema is invalid")
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError("bootstrap repair clean-launch CID is invalid")
    return receipt_id


def _repair_runtime_hardening_transition_receipt_id(
    payload: Mapping[str, Any],
) -> str:
    """Validate the closed revision-4 runtime-hardening receipt."""

    if (
        payload.get("schema") != REPAIR_RUNTIME_HARDENING_TRANSITION_SCHEMA
        or set(payload) != REPAIR_RUNTIME_HARDENING_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("transition_revision") != 4
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError(
            "bootstrap repair runtime-hardening schema is invalid"
        )
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError(
            "bootstrap repair runtime-hardening CID is invalid"
        )
    return receipt_id


def _repair_quack_recovery_transition_receipt_id(
    payload: Mapping[str, Any],
) -> str:
    """Validate the closed revision-5 Quack recovery receipt."""

    if (
        payload.get("schema") != REPAIR_QUACK_RECOVERY_TRANSITION_SCHEMA
        or set(payload) != REPAIR_QUACK_RECOVERY_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("transition_revision") != 5
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery schema is invalid"
        )
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError(
            "bootstrap repair Quack-recovery CID is invalid"
        )
    return receipt_id


def _repair_parallel_blocked_startup_transition_receipt_id(
    payload: Mapping[str, Any],
) -> str:
    """Validate the closed revision-6 startup recovery receipt."""

    if (
        payload.get("schema")
        != REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_SCHEMA
        or set(payload)
        != REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("transition_revision") != 6
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError(
            "bootstrap repair parallel-blocked-startup schema is invalid"
        )
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError(
            "bootstrap repair parallel-blocked-startup CID is invalid"
        )
    return receipt_id


def _repair_quack_publication_contention_transition_receipt_id(
    payload: Mapping[str, Any],
) -> str:
    """Validate the closed revision-7 Quack publication receipt."""

    if (
        payload.get("schema")
        != REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_SCHEMA
        or set(payload)
        != REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("transition_revision") != 7
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError(
            "bootstrap repair Quack-publication-contention schema is invalid"
        )
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError(
            "bootstrap repair Quack-publication-contention CID is invalid"
        )
    return receipt_id


def _repair_quack_recovery_replay_transition_receipt_id(
    payload: Mapping[str, Any],
) -> str:
    """Validate the closed revision-8 Quack recovery replay receipt."""

    if (
        payload.get("schema")
        != REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_SCHEMA
        or set(payload)
        != REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("transition_revision") != 8
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery-replay schema is invalid"
        )
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError(
            "bootstrap repair Quack-recovery-replay CID is invalid"
        )
    return receipt_id


def _repair_control_receipt_lifecycle_transition_receipt_id(
    payload: Mapping[str, Any],
) -> str:
    """Validate the closed revision-9 control lifecycle receipt."""

    if (
        payload.get("schema")
        != REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_SCHEMA
        or set(payload)
        != REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_RECEIPT_FIELDS
        or payload.get("task_id") != REPAIR_TRANSITION_TASK_ID
        or payload.get("program_id") != PROGRAM
        or payload.get("transition_revision") != 9
        or payload.get("semantic_corpus_changed") is not False
        or payload.get("database_mutated") is not False
    ):
        raise OperatorError(
            "bootstrap repair control-receipt-lifecycle schema is invalid"
        )
    unsigned = dict(payload)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError(
            "bootstrap repair control-receipt-lifecycle CID is invalid"
        )
    return receipt_id


def _run(
    argv: Sequence[str],
    *,
    timeout: float = 600.0,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(argv), cwd=ROOT, env=None if env is None else dict(env),
        text=True, capture_output=True, check=False, timeout=timeout,
    )


def _git(*args: str, cwd: Path | None = None) -> str:
    repository = ROOT if cwd is None else cwd
    completed = subprocess.run(
        ("git", *args), cwd=repository, text=True, capture_output=True,
        check=False, timeout=60,
    )
    if completed.returncode != 0:
        raise OperatorError(
            f"git {' '.join(args)} failed: {completed.stderr[-1000:]}"
        )
    return completed.stdout.strip()


def _git_bytes(*args: str, cwd: Path | None = None) -> bytes:
    repository = ROOT if cwd is None else cwd
    completed = subprocess.run(
        ("git", *args), cwd=repository, capture_output=True, check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise OperatorError(f"git {' '.join(args)} failed: {stderr[-1000:]}")
    return completed.stdout


def _git_changed_paths(base: str, target: str) -> tuple[str, ...]:
    payload = _git_bytes("diff", "--name-only", "-z", base, target, "--")
    values = payload.split(b"\0")
    if values and values[-1] == b"":
        values.pop()
    paths: list[str] = []
    for raw in values:
        try:
            path = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OperatorError("Git changed path is not UTF-8") from exc
        candidate = Path(path)
        if (
            not path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or path != candidate.as_posix()
        ):
            raise OperatorError("Git changed path is unsafe")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise OperatorError("Git changed path list contains duplicates")
    return tuple(paths)


def _git_patch_digest(base: str, target: str) -> str:
    return _identity(
        _git_bytes(
            "diff", "--binary", "--full-index", "--no-ext-diff",
            base, target, "--",
        )
    )


def _run_repair_transition_validations() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for command in REPAIR_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair transition validation failed: "
                + " ".join(command)
            )
    return results


def _run_repair_followup_transition_validations() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for command in REPAIR_FOLLOWUP_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair follow-up validation failed: "
                + " ".join(command)
            )
    return results


def _run_repair_clean_launch_transition_validations() -> list[dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair clean-launch validation requires a clean checkout"
        )
    results: list[dict[str, Any]] = []
    for command in REPAIR_CLEAN_LAUNCH_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair clean-launch validation failed: "
                + " ".join(command)
            )
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair clean-launch validation dirtied the checkout"
        )
    return results


def _run_repair_runtime_hardening_transition_validations(
) -> list[dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair runtime-hardening validation requires a clean "
            "checkout"
        )
    results: list[dict[str, Any]] = []
    for command in REPAIR_RUNTIME_HARDENING_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair runtime-hardening validation failed: "
                + " ".join(command)
            )
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair runtime-hardening validation dirtied the "
            "checkout"
        )
    return results


def _run_repair_quack_recovery_transition_validations(
) -> list[dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair Quack-recovery validation requires a clean "
            "checkout"
        )
    results: list[dict[str, Any]] = []
    for command in REPAIR_QUACK_RECOVERY_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair Quack-recovery validation failed: "
                + " ".join(command)
            )
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair Quack-recovery validation dirtied the checkout"
        )
    return results


def _run_repair_parallel_blocked_startup_transition_validations(
) -> list[dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair parallel-blocked-startup validation requires "
            "a clean checkout"
        )
    results: list[dict[str, Any]] = []
    for command in REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair parallel-blocked-startup validation failed: "
                + " ".join(command)
            )
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair parallel-blocked-startup validation dirtied "
            "the checkout"
        )
    return results


def _run_repair_quack_publication_contention_transition_validations(
) -> list[dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair Quack-publication-contention validation "
            "requires a clean checkout"
        )
    results: list[dict[str, Any]] = []
    for command in REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair Quack-publication-contention validation "
                "failed: " + " ".join(command)
            )
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair Quack-publication-contention validation "
            "dirtied the checkout"
        )
    return results


def _run_repair_quack_recovery_replay_transition_validations(
) -> list[dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair Quack-recovery-replay validation requires a "
            "clean checkout"
        )
    results: list[dict[str, Any]] = []
    for command in REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair Quack-recovery-replay validation failed: "
                + " ".join(command)
            )
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair Quack-recovery-replay validation dirtied the "
            "checkout"
        )
    return results


def _run_repair_control_receipt_lifecycle_transition_validations(
) -> list[dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair control-receipt-lifecycle validation requires "
            "a clean checkout"
        )
    results: list[dict[str, Any]] = []
    for command in REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_VALIDATIONS:
        completed = _run(command, timeout=900)
        result = {
            "argv": list(command),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout.encode("utf-8")),
            "stderr_digest": _identity(completed.stderr.encode("utf-8")),
        }
        results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "bootstrap repair control-receipt-lifecycle validation "
                "failed: " + " ".join(command)
            )
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise OperatorError(
            "bootstrap repair control-receipt-lifecycle validation dirtied "
            "the checkout"
        )
    return results


def _safe_path(value: str, *, field: str) -> Path:
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise OperatorError(f"{field} escapes repository") from exc
    return candidate


def _load(config_path: Path) -> tuple[Any, dict[str, Any]]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
    )

    path = config_path if config_path.is_absolute() else ROOT / config_path
    board = load_configured_board(path, repo_root=ROOT)
    payload = dict(board.payload)
    if payload.get("program_identifier") != PROGRAM:
        raise OperatorError("scheduler is not the ASEH program")
    return board, payload


def _paths(board: Any) -> dict[str, Path]:
    program = board.resolved_database_program()
    runtime = board.path(board.runtime_paths["root"])
    raw = board.payload.get("runtime_paths")
    raw = raw if isinstance(raw, Mapping) else {}
    ducklake = board.payload.get("ducklake_projection_program")
    ducklake = ducklake if isinstance(ducklake, Mapping) else {}
    result = {
        "runtime": runtime,
        "database": _safe_path(program.store_id, field="database_program.store_id"),
        "owner": _safe_path(
            str(raw.get("quack_owner") or f"{board.runtime_paths['root']}/quack-owner"),
            field="runtime_paths.quack_owner",
        ),
        "evidence": _safe_path(
            str(raw.get("evidence") or f"{board.runtime_paths['root']}/evidence"),
            field="runtime_paths.evidence",
        ),
        "merge_queue": _safe_path(
            str(
                raw.get("merge_queue")
                or board.runtime_paths.get("merge_queue")
                or f"{board.runtime_paths['root']}/merge-queue"
            ),
            field="runtime_paths.merge_queue",
        ),
        "ducklake_catalog": _safe_path(
            str(ducklake.get("catalog_path") or f"{board.runtime_paths['root']}/ducklake/catalog.duckdb"),
            field="ducklake_projection_program.catalog_path",
        ),
        "ducklake_data": _safe_path(
            str(ducklake.get("data_path") or f"{board.runtime_paths['root']}/ducklake/data"),
            field="ducklake_projection_program.data_path",
        ),
    }
    registry = _safe_path(
        program.runtime_registry_path,
        field="database_program.runtime_registry_path",
    )
    if registry != result["owner"]:
        raise OperatorError(
            "database runtime registry must equal the canonical Quack owner "
            "directory"
        )
    result["registry"] = registry
    result["bootstrap_receipt"] = (
        result["evidence"] / "bootstrap" / "bootstrap-materialization.json"
    )
    result["ducklake_receipt"] = (
        result["evidence"] / "bootstrap" / "ducklake-history-projection.json"
    )
    result["repair_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-transition.json"
    )
    result["repair_followup_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-followup-transition.json"
    )
    result["repair_clean_launch_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-clean-launch-transition.json"
    )
    result["repair_runtime_hardening_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-runtime-hardening-transition.json"
    )
    result["repair_quack_recovery_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-quack-recovery-transition.json"
    )
    result["repair_parallel_blocked_startup_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-parallel-blocked-startup-transition.json"
    )
    result["repair_quack_publication_contention_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-quack-publication-contention-transition.json"
    )
    result["repair_quack_recovery_replay_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-quack-recovery-replay-transition.json"
    )
    result["repair_control_receipt_lifecycle_transition_receipt"] = (
        result["evidence"]
        / "bootstrap"
        / "bootstrap-repair-control-receipt-lifecycle-transition.json"
    )
    result["status_receipt"] = (
        result["evidence"] / "control-plane" / "live-status.json"
    )
    result["inbox_failure_receipt"] = (
        result["evidence"] / "control-plane" / "owner-inbox-failure.json"
    )
    for name, path in result.items():
        if name == "runtime":
            continue
        try:
            path.relative_to(runtime)
        except ValueError as exc:
            raise OperatorError(f"{name} must remain below runtime root") from exc
    return result


def _tracked_bytes(path: Path, *, head: str) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ("git", "show", f"{head}:{relative}"),
        cwd=ROOT, capture_output=True, check=False, timeout=60,
    )
    if completed.returncode != 0:
        raise OperatorError(f"control input is not tracked at HEAD: {relative}")
    observed = path.read_bytes()
    if observed != completed.stdout:
        raise OperatorError(f"control input differs from HEAD: {relative}")
    return observed


def _metadata_value(value: Any) -> str:
    return str(value or "").strip()


def _goal_blocks(text: str) -> list[tuple[str, str, dict[str, str]]]:
    matches = list(GOAL_RE.finditer(text))
    rows: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields = {
            key.lower().replace(" ", "_").replace("-", "_"): value.strip()
            for key, value in META_RE.findall(text[match.end():end])
        }
        rows.append((match.group(1), match.group(2).strip(), fields))
    return rows


def _split(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _assert_clean_tree(board: Any) -> tuple[str, str]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise OperatorError("materialization/launch requires a clean checkout")
    branch = _git("branch", "--show-current")
    if branch != board.merge_target_branch:
        raise OperatorError("current branch differs from sealed merge target")
    return _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")


def _source_forest(
    board: Any,
    *,
    outer_head: str,
    outer_tree: str,
) -> dict[str, Any]:
    """Build the exact clean source identity for every task owner."""

    if _git("rev-parse", "--show-toplevel") != str(ROOT):
        raise OperatorError("accelerator repository root differs")
    entries: list[dict[str, Any]] = [
        {
            "owning_repository": "ipfs_accelerate_py",
            "path": ".",
            "commit": outer_head,
            "tree": outer_tree,
            "gitlink_commit": "",
        }
    ]
    for relative in board.worktree_submodule_paths:
        lexical_path = board.path(relative)
        observed = os.lstat(lexical_path)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise OperatorError(
                f"configured source owner is not a real directory: {relative}"
            )
        path = lexical_path.resolve(strict=True)
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise OperatorError(
                f"configured source owner escapes repository: {relative}"
            ) from exc
        if _git("rev-parse", "--show-toplevel", cwd=path) != str(path):
            raise OperatorError(
                f"configured source owner root differs: {relative}"
            )
        nested_status = _git(
            "status", "--porcelain=v1", "--untracked-files=all", cwd=path
        )
        if nested_status:
            raise OperatorError(
                f"configured source owner is not clean: {relative}"
            )
        nested_head = _git("rev-parse", "HEAD", cwd=path)
        nested_tree = _git("rev-parse", "HEAD^{tree}", cwd=path)
        gitlink_commit = _git("rev-parse", f"{outer_head}:{relative}")
        expected_row = f"160000 commit {nested_head}\t{relative}"
        if (
            gitlink_commit != nested_head
            or _git("ls-tree", outer_head, "--", relative) != expected_row
        ):
            raise OperatorError(
                f"configured source owner differs from gitlink: {relative}"
            )
        entries.append(
            {
                "owning_repository": Path(relative).name,
                "path": relative,
                "commit": nested_head,
                "tree": nested_tree,
                "gitlink_commit": gitlink_commit,
            }
        )
    by_owner = {
        str(entry["owning_repository"]): dict(entry) for entry in entries
    }
    if set(by_owner) != {
        "ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py"
    }:
        raise OperatorError(
            "source forest must contain exactly Accelerate, Datasets, and Kit"
        )
    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-source-forest@1",
        "entries": [
            {**entry, "source_identity": _identity(entry)}
            for entry in entries
        ],
    }
    return {
        **body,
        "forest_cid": _identity(body),
        "by_owner": {
            owner: {**entry, "source_identity": _identity(entry)}
            for owner, entry in sorted(by_owner.items())
        },
    }


def _population(board: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        parse_todo_blocks,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
        split_validation_commands,
    )

    head, tree = _assert_clean_tree(board)
    source_forest = _source_forest(
        board, outer_head=head, outer_tree=tree,
    )
    source_paths = {
        "config": board.config_path,
        "taskboard": board.path(board.taskboard_path),
        "objectives": board.path(board.objectives_path),
        "plan": board.path(board.plan_path),
        "program_registry": ROOT / "docs/architecture/agent_supervisor/PROGRAMS.md",
        "validator": board.path(board.validator_path),
        "operator": Path(__file__).resolve(),
        "baseline": ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/bootstrap_baseline.json",
        "bootstrap_test": ROOT / "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "requirements": board.path(str(config.get("requirements_path") or "")),
    }
    sources = {
        name: _tracked_bytes(path, head=head)
        for name, path in source_paths.items()
    }
    if _identity(sources["requirements"]) != config.get("requirements_digest"):
        raise OperatorError("protected requirements digest differs from config")
    plan_root = content_identity(
        {
            "schema": "aseh-plan-root@1",
            "source_head": head,
            "repository_tree_id": tree,
            "source_forest_cid": source_forest["forest_cid"],
            "sources": {
                name: _identity(payload) for name, payload in sorted(sources.items())
            },
        }
    )

    goal_rows = _goal_blocks(sources["objectives"].decode("utf-8"))
    if [item[0] for item in goal_rows] != [
        "ASEH-G000", "ASEH-G010", "ASEH-G020", "ASEH-G030", "ASEH-G040",
        "ASEH-G050", "ASEH-G060", "ASEH-G070", "ASEH-G080",
    ]:
        raise OperatorError("goal heap differs from sealed ASEH IDs/order")
    goal_cids = {
        goal_id: content_identity(
            {
                "goal_id": goal_id, "title": title, "metadata": fields,
                "plan_root_cid": plan_root,
            }
        )
        for goal_id, title, fields in goal_rows
    }
    goals: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    observed_goals: set[str] = set()
    for ordinal, (goal_id, title, fields) in enumerate(goal_rows, start=1):
        parent = fields.get("parent", "")
        if parent and parent not in observed_goals:
            raise OperatorError(f"{goal_id} parent must precede it")
        goal = {
            "goal_cid": goal_cids[goal_id],
            "goal_id": goal_id,
            "goal_alias": goal_id,
            "title": title,
            "ordinal": ordinal,
            "status": fields.get("status", "active").lower(),
            "objective_id": "objective:aseh-root" if goal_id == "ASEH-G000" else "",
            "objective_alias": "ASEH-G000",
            "priority": fields.get("priority", "P0"),
            "body": dict(fields),
        }
        if parent:
            goal["parent_goal_cid"] = goal_cids[parent]
            edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in _split(fields.get("depends_on")):
            if dependency not in goal_cids:
                raise OperatorError(f"{goal_id} has an unknown dependency")
            edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
        goals.append(goal)
        observed_goals.add(goal_id)

    parsed = parse_todo_blocks(
        sources["taskboard"].decode("utf-8"),
        task_header_prefix="## ASEH-",
    )
    expected_ids = [
        "ASEH-000", "ASEH-001",
        *[f"ASEH-{group}{item}" for group, count in (
            ("01", 6), ("02", 5), ("03", 6), ("04", 6), ("05", 6),
            ("06", 3), ("07", 6),
        ) for item in range(count)],
    ]
    # The compact expression above would produce 010..015, 020..024, etc.
    task_ids = [item[0] for item in parsed]
    if task_ids != expected_ids:
        raise OperatorError("task board differs from the sealed 40-task ID/order")
    normalized = [
        (
            task_id, title, source_line,
            {key: _metadata_value(value) for key, value in fields.items()},
        )
        for task_id, title, source_line, fields in parsed
    ]
    task_cids = {
        task_id: content_identity(
            {
                "task_id": task_id,
                "title": title,
                "source_line": source_line,
                "metadata": fields,
                "plan_root_cid": plan_root,
                "source_forest_cid": source_forest["forest_cid"],
                "owner_source": source_forest["by_owner"].get(
                    fields.get("owning_repository", "")
                ),
            }
        )
        for task_id, title, source_line, fields in normalized
    }
    tasks: list[dict[str, Any]] = []
    observed_tasks: set[str] = set()
    for ordinal, (task_id, title, source_line, fields) in enumerate(normalized, start=1):
        dependencies = _split(fields.get("depends_on"))
        future = [item for item in dependencies if item not in observed_tasks]
        if future:
            raise OperatorError(f"{task_id} dependencies must precede it: {future}")
        goal_id = fields.get("subgoal_id") or fields.get("goal_id") or ""
        if goal_id not in goal_cids:
            raise OperatorError(f"{task_id} refers to an unknown goal")
        owner = fields.get("owning_repository", "")
        owner_source = source_forest["by_owner"].get(owner)
        if not isinstance(owner_source, Mapping):
            raise OperatorError(f"{task_id} has an unknown source owner: {owner}")
        outputs = _split(
            fields.get("exact_declared_outputs") or fields.get("outputs")
        )
        task = dict(fields)
        task.update(
            {
                "task_cid": task_cids[task_id],
                "task_id": task_id,
                "task_alias": task_id,
                "title": title,
                "source_line": source_line,
                "goal_cid": goal_cids[goal_id],
                "goal_id": goal_id,
                "plan_cid": plan_root,
                "objective_id": "objective:aseh-root",
                "ordinal": ordinal,
                "status": "todo",
                "priority": fields.get("priority", "P1"),
                "dependencies": [task_cids[item] for item in dependencies],
                "depends_on": [task_cids[item] for item in dependencies],
                "outputs": [
                    {
                        "path": path,
                        "effect_id": content_identity(
                            {"task_cid": task_cids[task_id], "path": path}
                        ),
                    }
                    for path in outputs
                ],
                "acceptance": [fields.get("acceptance_conditions", "")],
                "validations": list(
                    split_validation_commands(fields.get("validation", ""))
                ),
                "accepted_plan_root_cid": plan_root,
                "source_forest_cid": source_forest["forest_cid"],
                "base_revision": owner_source["commit"],
                "base_repository_tree_id": owner_source["tree"],
                "owner_source_identity": owner_source["source_identity"],
                "owning_repository": owner,
            }
        )
        tasks.append(task)
        observed_tasks.add(task_id)

    projection = config.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    dependency_count = sum(
        len(_split(item[3].get("depends_on"))) for item in normalized
    )
    if (
        int(projection.get("task_count", -1)) != len(tasks)
        or int(projection.get("goal_count", -1)) != len(goals)
        or int(projection.get("task_dependency_count", -1)) != dependency_count
    ):
        raise OperatorError("materialized population differs from initial projection")
    return {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": tree,
        "source_head": head,
        "plan_root_cid": plan_root,
        "source_forest": source_forest,
        "source_identities": {
            name: _identity(payload) for name, payload in sorted(sources.items())
        },
        "objectives": goals,
        "goal_edges": edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "ASEH-PLAN-R1",
                "goal_cid": goal_cids["ASEH-G000"],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
    }


def _validate_bootstrap(board: Any) -> dict[str, Any]:
    commands = (
        (sys.executable, str(board.path(board.validator_path)), "--check-all", "--json"),
        (
            sys.executable, "-m", "pytest", "-q",
            "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        ),
    )
    results: list[dict[str, Any]] = []
    for command in commands:
        completed = _run(command, timeout=900)
        observation = {
            "argv": list(command),
            "returncode": completed.returncode,
            "stdout_digest": _identity(completed.stdout.encode()),
            "stderr_digest": _identity(completed.stderr.encode()),
        }
        results.append(observation)
        if completed.returncode != 0:
            raise OperatorError(
                f"bootstrap validation failed: {' '.join(command)}"
            )
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-bootstrap-validation@1",
        "hermetic": True,
        "live": False,
        "commands": results,
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def _ducklake_projection(
    *,
    paths: Mapping[str, Path],
    population: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "schema": DUCKLAKE_SCHEMA,
        "authoritative": False,
        "scheduler_gate": False,
        "acceptance_gate": False,
        "completion_gate": False,
        "status": "unavailable",
        "reason_code": "ducklake_projection_unavailable",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
    }
    try:
        import duckdb

        catalog = paths["ducklake_catalog"]
        data_path = paths["ducklake_data"]
        catalog.parent.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("LOAD ducklake")
            catalog_sql = str(catalog).replace("'", "''")
            data_sql = str(data_path).replace("'", "''")
            connection.execute(
                f"ATTACH 'ducklake:{catalog_sql}' AS aseh_history "
                f"(DATA_PATH '{data_sql}')"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS aseh_history.bootstrap_history (
                    event_id VARCHAR,
                    observed_at_epoch DOUBLE,
                    source_head VARCHAR,
                    repository_tree_id VARCHAR,
                    plan_root_cid VARCHAR,
                    projection_cid VARCHAR,
                    task_count BIGINT,
                    goal_count BIGINT,
                    body_json VARCHAR
                )
                """
            )
            event_id = _identity(
                {
                    "source_head": population["source_head"],
                    "plan_root_cid": population["plan_root_cid"],
                    "projection_cid": control_receipt.get("projection_cid"),
                }
            )
            if int(connection.execute(
                "SELECT COUNT(*) FROM aseh_history.bootstrap_history WHERE event_id = ?",
                [event_id],
            ).fetchone()[0]) == 0:
                connection.execute(
                    "INSERT INTO aseh_history.bootstrap_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        event_id, time.time(), population["source_head"],
                        population["repository_tree_id"], population["plan_root_cid"],
                        str(control_receipt.get("projection_cid") or ""),
                        int(control_receipt.get("task_count") or 0),
                        int(control_receipt.get("goal_count") or 0),
                        json.dumps(
                            {
                                "control_authority": "DuckDB/DatabaseTaskSource@1",
                                "transport": "QuackStateServer@1",
                                "projection": "DuckLake/non-authoritative",
                            },
                            sort_keys=True,
                        ),
                    ],
                )
            count = int(connection.execute(
                "SELECT COUNT(*) FROM aseh_history.bootstrap_history"
            ).fetchone()[0])
            connection.execute("DETACH aseh_history")
        finally:
            connection.close()
        projection.update(
            {
                "status": "available", "reason_code": "", "event_id": event_id,
                "row_count": count,
                "catalog_path": str(catalog.relative_to(ROOT)),
                "data_path": str(data_path.relative_to(ROOT)),
            }
        )
    except Exception as exc:
        projection["error_class"] = type(exc).__name__
    projection["projection_receipt_id"] = _identity(projection)
    _atomic_json(paths["ducklake_receipt"], projection)
    return projection


@contextmanager
def _offline_database_guard(paths: Mapping[str, Path]) -> Any:
    """Hold the Quack owner's exact lock for every direct-file operation."""

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        owner_liveness,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        OwnerMarker,
    )

    database = paths["database"]
    lock_path = database.with_name(f".{database.name}{OWNER_LOCK_SUFFIX}")
    marker_path = database.with_name(f".{database.name}{OWNER_MARKER_SUFFIX}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OperatorError(
                "offline database access refused while the Quack owner is live"
            ) from exc
        if marker_path.exists():
            try:
                marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
                marker = OwnerMarker.from_dict(marker_payload)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise OperatorError(
                    "offline database access refused: owner marker is invalid"
                ) from exc
            liveness = owner_liveness(marker.process_birth)
            if liveness is not OwnerLiveness.DEAD:
                raise OperatorError(
                    "offline database access refused: owner liveness is not dead"
                )
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def _offline_merge_queue_guard(paths: Mapping[str, Path]) -> Any:
    """Freeze the existing merge queue for one read-only continuity view."""

    queue_dir = paths["merge_queue"]
    database = queue_dir / "merge_queue.duckdb"
    consumer_lock = queue_dir / "train" / "consumer.lock"
    database_lock = database.with_name(f".{database.name}.lock")
    if not database.is_file():
        raise OperatorError("canonical merge queue database is absent")
    handles: list[Any] = []
    try:
        for path in (consumer_lock, database_lock):
            if not path.is_file():
                raise OperatorError(
                    f"canonical merge queue lock is absent: {path.name}"
                )
            handle = path.open("rb")
            handles.append(handle)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OperatorError(
                    "offline merge queue access refused while a lane is live"
                ) from exc
        yield database
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _immutable_goal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project the immutable semantic identity of one admitted goal."""

    return {
        "goal_cid": str(value.get("goal_cid") or ""),
        "goal_alias": str(value.get("goal_alias") or ""),
        "objective_id": str(value.get("objective_id") or ""),
        "parent_goal_cid": str(value.get("parent_goal_cid") or ""),
        "ordinal": int(value.get("ordinal") or 0),
        "title": str(value.get("title") or ""),
        "body": dict(value.get("body") or {}),
    }


def _goal_edge_sort_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("parent_goal_cid") or ""),
        str(value.get("child_goal_cid") or ""),
        str(value.get("edge_kind") or ""),
    )


def _immutable_plan_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project the immutable semantic identity of the admitted plan root."""

    return {
        "plan_cid": str(value.get("plan_cid") or ""),
        "goal_cid": str(value.get("goal_cid") or ""),
        "plan_alias": str(value.get("plan_alias") or ""),
        "body": dict(value.get("body") or {}),
    }


def _immutable_objective_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project immutable semantic authority for the sole ASEH objective."""

    return {
        "objective_id": str(value.get("objective_id") or ""),
        "objective_alias": str(value.get("objective_alias") or ""),
        "parent_objective_id": str(value.get("parent_objective_id") or ""),
        "title": str(value.get("title") or ""),
        "priority": str(value.get("priority") or ""),
        "body": dict(value.get("body") or {}),
        "extension_schema": str(value.get("extension_schema") or ""),
        "extension": dict(value.get("extension") or {}),
    }


def _objective_record_from_projection(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    objectives = projection.get("objectives")
    if not isinstance(objectives, list) or len(objectives) != 1:
        raise OperatorError("ASEH projection must contain exactly one objective")
    objective = objectives[0]
    if not isinstance(objective, Mapping):
        raise OperatorError("ASEH objective projection is malformed")
    record = _immutable_objective_record(objective)
    if not record["objective_id"] or not record["objective_alias"]:
        raise OperatorError("ASEH objective identity is incomplete")
    return record


def _expected_objective_record(population: Mapping[str, Any]) -> dict[str, Any]:
    roots = [
        item
        for item in population["objectives"]
        if isinstance(item, Mapping) and str(item.get("objective_id") or "")
    ]
    if len(roots) != 1:
        raise OperatorError("sealed ASEH population must define one objective")
    root = roots[0]
    return _immutable_objective_record(
        {
            "objective_id": root["objective_id"],
            "objective_alias": root.get("objective_alias")
            or root["objective_id"],
            "parent_objective_id": "",
            "title": root.get("title") or root["objective_id"],
            "priority": root.get("priority") or "P2",
            "body": {
                key: value
                for key, value in root.items()
                if key
                not in {
                    "objective_id", "objective_alias", "title", "status",
                    "priority",
                }
            },
            "extension_schema": "",
            "extension": {},
        }
    )


def _expected_task_authority_spec_cids(
    population: Mapping[str, Any],
) -> dict[str, str]:
    """Derive complete immutable task authority from the sealed source.

    This intentionally mirrors the public ``DatabaseTaskSource.materialize``
    contract rather than trusting the database projection that it validates.
    The resulting identity covers body, scope/outputs, acceptance, validation,
    dependencies, and owner/tree identity while excluding lifecycle receipts.
    """

    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        task_authority_spec_cid,
    )

    repository_tree_id = str(population["repository_tree_id"])
    result: dict[str, str] = {}
    excluded_body_fields = {
        "task_cid", "task_id", "task_alias", "cid", "goal_cid",
        "goal_id", "depends_on", "dependencies", "effects", "outputs",
        "acceptance_criteria", "acceptance", "validation_commands",
        "validations", "status", "priority", "ordinal", "plan_cid",
        "objective_id",
    }
    for index, item in enumerate(population["tasks"]):
        task_cid = str(item["task_cid"])
        task_alias = str(item["task_alias"])
        outputs = []
        for ordinal, raw in enumerate(item.get("outputs") or ()):
            output = dict(raw)
            outputs.append(
                {
                    "ordinal": ordinal,
                    "path": str(
                        output.get("path")
                        or output.get("effect_id")
                        or f"output:{ordinal}"
                    ),
                    "effect": output,
                }
            )
        acceptance = []
        for ordinal, raw in enumerate(item.get("acceptance") or ()):
            if isinstance(raw, str):
                criterion = raw.strip()
                policy: dict[str, Any] = {"criterion": criterion}
            else:
                policy = dict(raw)
                criterion = str(
                    policy.get("criterion")
                    or policy.get("statement")
                    or policy.get("criterion_key")
                    or f"criterion:{ordinal}"
                ).strip()
            acceptance.append(
                {
                    "ordinal": ordinal,
                    "criterion": criterion,
                    "evidence_policy": policy,
                }
            )
        validations = []
        for ordinal, raw in enumerate(item.get("validations") or ()):
            if isinstance(raw, str):
                argv = [raw]
                policy = {}
            elif isinstance(raw, Mapping):
                validation = dict(raw)
                raw_argv = validation.get("argv") or validation.get(
                    "validation_commands"
                )
                if isinstance(raw_argv, str):
                    argv = [raw_argv]
                elif isinstance(raw_argv, Sequence):
                    argv = [str(part) for part in raw_argv]
                else:
                    argv = [
                        str(
                            validation.get("command")
                            or f"validation:{ordinal}"
                        )
                    ]
                policy = {
                    key: value
                    for key, value in validation.items()
                    if key not in {"argv", "validation_commands", "command"}
                }
            else:
                argv = [str(part) for part in raw]
                policy = {}
            validations.append(
                {"ordinal": ordinal, "argv": argv, "policy": policy}
            )
        projected = {
            "task_cid": task_cid,
            "task_alias": task_alias,
            "goal_cid": str(item["goal_cid"]),
            "objective_id": str(item.get("objective_id") or ""),
            "ordinal": int(item.get("ordinal") or index + 1),
            "priority": str(item.get("priority") or "P2"),
            "identity": {
                "task_cid": task_cid,
                "task_alias": task_alias,
                "repository_tree_id": repository_tree_id,
            },
            "body": {
                key: value
                for key, value in item.items()
                if key not in excluded_body_fields
            },
            "extension_schema": "",
            "extension": {},
            "dependencies": [
                {"dependency_task_cid": str(value), "kind": "depends_on"}
                for value in sorted(
                    item.get("depends_on") or item.get("dependencies") or ()
                )
            ],
            "outputs": outputs,
            "acceptance": acceptance,
            "validations": validations,
        }
        result[task_alias] = task_authority_spec_cid(projected)
    return dict(sorted(result.items()))


def _task_authority_spec_cids(
    projection: Mapping[str, Any],
) -> dict[str, str]:
    """Compute complete authority identities from one read-only projection."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        task_authority_spec_cid,
    )

    tasks = projection.get("tasks")
    if not isinstance(tasks, list):
        raise OperatorError("materialized plan projection has no task population")
    result: dict[str, str] = {}
    for raw in tasks:
        if not isinstance(raw, Mapping):
            raise OperatorError("materialized plan projection task is malformed")
        alias = str(raw.get("task_alias") or "")
        if not alias or alias in result:
            raise OperatorError("materialized task authority aliases are invalid")
        result[alias] = task_authority_spec_cid(raw)
    return dict(sorted(result.items()))


def _verify_materialized_source(
    source: Any,
    *,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    require_initial_frontier: bool = True,
    projection_matches_events: bool | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Prove the bounded projection and its admitted-event replay agree."""

    expected_tasks = list(population["tasks"])
    expected_aliases = [str(item["task_alias"]) for item in expected_tasks]
    expected_cids = {
        str(item["task_alias"]): str(item["task_cid"])
        for item in expected_tasks
    }
    projection_matches = (
        source.projection_matches_events()
        if projection_matches_events is None
        else projection_matches_events
    )
    if projection_matches is not True:
        raise OperatorError("materialized projection differs from admitted events")
    snapshot = source.snapshot().to_dict()
    page = source.list_tasks(limit=100)
    if page.next_cursor:
        raise OperatorError("materialized task population exceeds its sealed bound")
    observed_aliases = [item.task_alias for item in page.tasks]
    if observed_aliases != expected_aliases:
        raise OperatorError("materialized task order/aliases differ from the board")
    for item in page.tasks:
        if item.task_cid != expected_cids.get(item.task_alias):
            raise OperatorError(
                f"materialized task identity differs: {item.task_alias}"
            )
        expected = expected_tasks[expected_aliases.index(item.task_alias)]
        if tuple(sorted(item.dependencies)) != tuple(
            sorted(expected.get("dependencies") or ())
        ):
            raise OperatorError(
                f"materialized dependency edges differ: {item.task_alias}"
            )
        for key in (
            "owning_repository",
            "base_revision",
            "base_repository_tree_id",
            "source_forest_cid",
            "owner_source_identity",
        ):
            if item.body.get(key) != expected.get(key):
                raise OperatorError(
                    f"materialized owner binding differs: {item.task_alias}:{key}"
                )
    plan_projection = source.plan_projection(
        task_cids=[str(item["task_cid"]) for item in expected_tasks]
    )
    task_authority_spec_cids = _task_authority_spec_cids(plan_projection)
    expected_authority_spec_cids = _expected_task_authority_spec_cids(population)
    if task_authority_spec_cids != expected_authority_spec_cids:
        raise OperatorError(
            "materialized task authority differs from the sealed board"
        )
    objective_record = _objective_record_from_projection(plan_projection)
    if objective_record != _expected_objective_record(population):
        raise OperatorError(
            "materialized objective authority differs from the sealed board"
        )
    ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    expected_ready = list(config["initial_projection"]["ready_task_ids"])
    if require_initial_frontier and ready != expected_ready:
        raise OperatorError(
            f"initial ready frontier differs: expected {expected_ready}, "
            f"observed {ready}"
        )
    if (
        len(ready) != len(set(ready))
        or not set(ready).issubset(expected_aliases)
    ):
        raise OperatorError("materialized ready frontier is not task-corpus bound")
    projection = config["initial_projection"]
    expected_counts = {
        "task_count": int(projection["task_count"]),
        "goal_count": int(projection["goal_count"]),
        "dependency_count": int(projection["task_dependency_count"]),
        "objective_count": len(
            {
                str(item.get("objective_id") or "")
                for item in population["objectives"]
                if str(item.get("objective_id") or "")
            }
        ),
        "plan_count": len(population["plans"]),
    }
    for key, expected in expected_counts.items():
        if int(snapshot.get(key, -1)) != expected:
            raise OperatorError(f"materialized {key} differs from the board")

    goal_records: dict[str, dict[str, Any]] = {}
    for expected in population["objectives"]:
        goal_alias = str(expected["goal_alias"])
        observed = source.get_goal(str(expected["goal_cid"]))
        if not isinstance(observed, Mapping):
            raise OperatorError(f"materialized goal is missing: {goal_alias}")
        observed_record = _immutable_goal_record(observed)
        expected_record = _immutable_goal_record(
            {
                "goal_cid": expected["goal_cid"],
                "goal_alias": goal_alias,
                "objective_id": expected.get("objective_id") or "",
                "parent_goal_cid": expected.get("parent_goal_cid") or "",
                "ordinal": expected["ordinal"],
                "title": expected["title"],
                "body": {
                    key: value
                    for key, value in expected.items()
                    if key
                    not in {
                        "goal_cid", "goal_id", "goal_alias", "title",
                        "status", "ordinal", "objective_id",
                    }
                },
            }
        )
        if observed_record != expected_record:
            raise OperatorError(f"materialized goal differs: {goal_alias}")
        goal_records[goal_alias] = observed_record

    observed_edges = sorted(
        (dict(item) for item in source.list_goal_edges(limit=100)),
        key=_goal_edge_sort_key,
    )
    expected_edges = sorted(
        (dict(item) for item in population["goal_edges"]),
        key=_goal_edge_sort_key,
    )
    if observed_edges != expected_edges:
        raise OperatorError("materialized goal edges differ from the board")

    expected_plans = list(population["plans"])
    if len(expected_plans) != 1:
        raise OperatorError("sealed ASEH population must have exactly one plan")
    expected_plan = expected_plans[0]
    observed_plan = source.get_plan(str(expected_plan["plan_cid"]))
    if not isinstance(observed_plan, Mapping):
        raise OperatorError("materialized plan root is missing")
    plan_record = _immutable_plan_record(observed_plan)
    if plan_record != _immutable_plan_record(
        {
            "plan_cid": expected_plan["plan_cid"],
            "goal_cid": expected_plan["goal_cid"],
            "plan_alias": expected_plan["plan_alias"],
            "body": dict(expected_plan),
        }
    ):
        raise OperatorError("materialized plan root differs from the board")
    integrity = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-integrity@1",
        "projection_matches_events": True,
        "projection_cid": snapshot["projection_cid"],
        "event_cursor": snapshot["event_cursor"],
        "task_statuses": {
            item.task_alias: str(item.status or "").lower()
            for item in page.tasks
        },
        "task_revisions": {
            item.task_alias: int(item.revision) for item in page.tasks
        },
        "task_cids": {
            item.task_alias: item.task_cid for item in page.tasks
        },
        "task_owner_bindings": {
            item.task_alias: {
                key: item.body.get(key)
                for key in (
                    "owning_repository", "base_revision",
                    "base_repository_tree_id", "source_forest_cid",
                    "owner_source_identity",
                )
            }
            for item in page.tasks
        },
        "task_dependencies": {
            item.task_alias: list(item.dependencies) for item in page.tasks
        },
        "task_authority_spec_cids": task_authority_spec_cids,
        "objective_record": objective_record,
        "goal_records": dict(sorted(goal_records.items())),
        "goal_edges": observed_edges,
        "plan_record": plan_record,
        **expected_counts,
    }
    integrity["integrity_receipt_id"] = _identity(integrity)
    return snapshot, ready, integrity


def _verify_materialized_source_from_bootstrap(
    source: Any,
    *,
    bootstrap: Mapping[str, Any],
    projection_matches_events: bool | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any], dict[str, tuple[str, ...]]]:
    """Verify live lifecycle state without reminting the sealed task corpus."""

    sealed = bootstrap.get("integrity")
    if not isinstance(sealed, Mapping):
        raise OperatorError("bootstrap integrity is absent")
    sealed_task_cids = sealed.get("task_cids")
    sealed_dependencies = sealed.get("task_dependencies")
    sealed_owners = sealed.get("task_owner_bindings")
    sealed_authority = sealed.get("task_authority_spec_cids")
    if not all(
        isinstance(value, Mapping)
        for value in (
            sealed_task_cids, sealed_dependencies, sealed_owners,
            sealed_authority,
        )
    ):
        raise OperatorError("bootstrap task corpus is incomplete")
    expected_aliases = list(sealed_task_cids)
    projection_matches = (
        source.projection_matches_events()
        if projection_matches_events is None
        else projection_matches_events
    )
    if projection_matches is not True:
        raise OperatorError("materialized projection differs from admitted events")
    snapshot = source.snapshot().to_dict()
    page = source.list_tasks(limit=100)
    if page.next_cursor:
        raise OperatorError("materialized task population exceeds its sealed bound")
    if [item.task_alias for item in page.tasks] != expected_aliases:
        raise OperatorError("materialized task order/aliases differ from bootstrap")
    for item in page.tasks:
        alias = item.task_alias
        if item.task_cid != sealed_task_cids.get(alias):
            raise OperatorError(f"materialized task identity differs: {alias}")
        if tuple(sorted(item.dependencies)) != tuple(
            sorted(sealed_dependencies.get(alias) or ())
        ):
            raise OperatorError(f"materialized dependency edges differ: {alias}")
        observed_owner = {
            key: item.body.get(key)
            for key in (
                "owning_repository", "base_revision",
                "base_repository_tree_id", "source_forest_cid",
                "owner_source_identity",
            )
        }
        if observed_owner != sealed_owners.get(alias):
            raise OperatorError(f"materialized owner binding differs: {alias}")

    plan_projection = source.plan_projection(
        task_cids=[str(sealed_task_cids[alias]) for alias in expected_aliases]
    )
    authority = _task_authority_spec_cids(plan_projection)
    if authority != sealed_authority:
        raise OperatorError("materialized task authority differs from bootstrap")
    objective_record = _objective_record_from_projection(plan_projection)
    if objective_record != sealed.get("objective_record"):
        raise OperatorError("materialized objective authority differs from bootstrap")

    transient_outputs: dict[str, tuple[str, ...]] = {}
    projected_tasks = plan_projection.get("tasks")
    if not isinstance(projected_tasks, list):
        raise OperatorError("materialized plan projection lacks tasks")
    for task in projected_tasks:
        if not isinstance(task, Mapping):
            raise OperatorError("materialized plan task is malformed")
        alias = str(task.get("task_alias") or "")
        outputs = task.get("outputs")
        if alias not in sealed_task_cids or not isinstance(outputs, list):
            raise OperatorError("materialized task outputs are malformed")
        paths: list[str] = []
        for output in outputs:
            if not isinstance(output, Mapping):
                raise OperatorError("materialized task output is malformed")
            path = str(output.get("path") or "").strip()
            if not path or path in paths:
                raise OperatorError("materialized task output path is invalid")
            paths.append(path)
        transient_outputs[alias] = tuple(paths)
    if set(transient_outputs) != set(expected_aliases):
        raise OperatorError("materialized task output corpus differs")

    sealed_goals = sealed.get("goal_records")
    if not isinstance(sealed_goals, Mapping):
        raise OperatorError("bootstrap goal corpus is absent")
    goal_records: dict[str, dict[str, Any]] = {}
    for alias, expected in sealed_goals.items():
        if not isinstance(expected, Mapping):
            raise OperatorError("bootstrap goal record is malformed")
        observed = source.get_goal(str(expected.get("goal_cid") or ""))
        if not isinstance(observed, Mapping):
            raise OperatorError(f"materialized goal is missing: {alias}")
        record = _immutable_goal_record(observed)
        if record != expected:
            raise OperatorError(f"materialized goal differs: {alias}")
        goal_records[str(alias)] = record
    observed_edges = sorted(
        (dict(item) for item in source.list_goal_edges(limit=100)),
        key=_goal_edge_sort_key,
    )
    if observed_edges != sealed.get("goal_edges"):
        raise OperatorError("materialized goal edges differ from bootstrap")
    sealed_plan = sealed.get("plan_record")
    if not isinstance(sealed_plan, Mapping):
        raise OperatorError("bootstrap plan record is absent")
    observed_plan = source.get_plan(str(sealed_plan.get("plan_cid") or ""))
    if (
        not isinstance(observed_plan, Mapping)
        or _immutable_plan_record(observed_plan) != sealed_plan
    ):
        raise OperatorError("materialized plan root differs from bootstrap")

    for field in (
        "task_count", "goal_count", "dependency_count", "objective_count",
        "plan_count",
    ):
        if int(snapshot.get(field, -1)) != int(sealed.get(field, -2)):
            raise OperatorError(f"materialized {field} differs from bootstrap")
    ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    if len(ready) != len(set(ready)) or not set(ready).issubset(expected_aliases):
        raise OperatorError("materialized ready frontier is not task-corpus bound")
    integrity = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-integrity@1",
        "projection_matches_events": True,
        "projection_cid": snapshot["projection_cid"],
        "event_cursor": snapshot["event_cursor"],
        "task_statuses": {
            item.task_alias: str(item.status or "").lower()
            for item in page.tasks
        },
        "task_revisions": {
            item.task_alias: int(item.revision) for item in page.tasks
        },
        "task_cids": {
            item.task_alias: item.task_cid for item in page.tasks
        },
        "task_owner_bindings": {
            item.task_alias: {
                key: item.body.get(key)
                for key in (
                    "owning_repository", "base_revision",
                    "base_repository_tree_id", "source_forest_cid",
                    "owner_source_identity",
                )
            }
            for item in page.tasks
        },
        "task_dependencies": {
            item.task_alias: list(item.dependencies) for item in page.tasks
        },
        "task_authority_spec_cids": authority,
        "objective_record": objective_record,
        "goal_records": dict(goal_records),
        "goal_edges": observed_edges,
        "plan_record": _immutable_plan_record(observed_plan),
        **{
            field: int(snapshot[field])
            for field in (
                "task_count", "goal_count", "dependency_count",
                "objective_count", "plan_count",
            )
        },
    }
    integrity["integrity_receipt_id"] = _identity(integrity)
    _admit_current_projection_against_bootstrap(bootstrap, snapshot, integrity)
    return snapshot, ready, integrity, transient_outputs


_IMMUTABLE_SNAPSHOT_FIELDS = (
    "source_schema",
    "schema_version",
    "plan_root_cid",
    "repository_tree_id",
    "formal_plan_id",
    "task_count",
    "goal_count",
    "dependency_count",
    "objective_count",
    "plan_count",
)
_IMMUTABLE_INTEGRITY_FIELDS = (
    "schema",
    "projection_matches_events",
    "task_cids",
    "task_owner_bindings",
    "task_dependencies",
    "task_authority_spec_cids",
    "objective_record",
    "goal_records",
    "goal_edges",
    "plan_record",
    "task_count",
    "goal_count",
    "dependency_count",
    "objective_count",
    "plan_count",
)


def _admit_current_projection_against_bootstrap(
    bootstrap: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> None:
    """Admit monotonic lifecycle state without rewriting initial truth."""

    sealed_snapshot = bootstrap.get("snapshot")
    sealed_snapshot = (
        sealed_snapshot if isinstance(sealed_snapshot, Mapping) else {}
    )
    sealed_integrity = bootstrap.get("integrity")
    sealed_integrity = (
        sealed_integrity if isinstance(sealed_integrity, Mapping) else {}
    )
    if any(
        sealed_snapshot.get(field) in (None, "")
        or snapshot.get(field) != sealed_snapshot.get(field)
        for field in _IMMUTABLE_SNAPSHOT_FIELDS
    ):
        raise OperatorError(
            "current projection immutable snapshot differs from bootstrap"
        )
    for candidate in (sealed_snapshot, snapshot):
        expected_source_identity = content_identity(
            {
                "plan_root_cid": candidate.get("plan_root_cid"),
                "repository_tree_id": candidate.get("repository_tree_id"),
                "projection_cid": candidate.get("projection_cid"),
            }
        )
        if (
            not candidate.get("projection_cid")
            or candidate.get("source_identity") != expected_source_identity
        ):
            raise OperatorError("projection source identity is not self-authenticating")
    if any(
        integrity.get(field) != sealed_integrity.get(field)
        for field in _IMMUTABLE_INTEGRITY_FIELDS
    ):
        raise OperatorError(
            "current projection immutable corpus differs from bootstrap"
        )
    sealed_revisions = sealed_integrity.get("task_revisions")
    current_revisions = integrity.get("task_revisions")
    sealed_statuses = sealed_integrity.get("task_statuses")
    current_statuses = integrity.get("task_statuses")
    if not all(
        isinstance(value, Mapping)
        for value in (
            sealed_revisions, current_revisions, sealed_statuses,
            current_statuses,
        )
    ) or not (
        set(sealed_revisions)
        == set(current_revisions)
        == set(sealed_statuses)
        == set(current_statuses)
    ):
        raise OperatorError("current lifecycle task corpus differs from bootstrap")
    for task_alias in sealed_revisions:
        sealed_revision = sealed_revisions[task_alias]
        current_revision = current_revisions[task_alias]
        if type(sealed_revision) is not int or type(current_revision) is not int:
            raise OperatorError("task lifecycle revisions are not exact integers")
        if current_revision < sealed_revision:
            raise OperatorError("task lifecycle revision regressed from bootstrap")
        if (
            current_revision == sealed_revision
            and current_statuses[task_alias] != sealed_statuses[task_alias]
        ):
            raise OperatorError("task status changed without a revision advance")
    sealed_cursor = sealed_snapshot.get("event_cursor")
    current_cursor = snapshot.get("event_cursor")
    if (
        type(sealed_cursor) is not int
        or type(current_cursor) is not int
        or current_cursor < sealed_cursor
        or integrity.get("event_cursor") != current_cursor
        or integrity.get("projection_cid") != snapshot.get("projection_cid")
    ):
        raise OperatorError("current event cursor regressed or diverged")


def _read_completed_merge_requests(database: Path) -> tuple[Any, ...]:
    """Read the authoritative queue under the caller's offline queue locks."""

    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeRequest
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        connect_duckdb_with_policy,
    )

    import duckdb

    connection = connect_duckdb_with_policy(duckdb, database, read_only=True)
    try:
        cursor = connection.execute(
            """SELECT request_id, branch_name, task_id, priority, lane_id,
                      enqueued_at, attempt, metadata_json, commit_sha,
                      canonical_task_id, canonical_task_key, status,
                      claimed_at, consumer_id, failure_count, failure_reason,
                      claim_token, claim_generation, retry_not_before
                 FROM merge_requests
                WHERE status='completed'
                ORDER BY request_id"""
        )
        rows = cursor.fetchall()
        columns = [str(item[0]) for item in cursor.description]
    finally:
        connection.close()
    if len(rows) > 256:
        raise OperatorError("completed merge population exceeds continuity bound")
    requests: list[Any] = []
    for row in rows:
        values = dict(zip(columns, row, strict=True))
        try:
            metadata = json.loads(str(values.pop("metadata_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise OperatorError("completed merge metadata is invalid") from exc
        if not isinstance(metadata, dict):
            raise OperatorError("completed merge metadata is not an object")
        requests.append(MergeRequest.from_dict({**values, "metadata": metadata}))
    return tuple(requests)


def _git_tree_entry(commit: str, path: str) -> bytes:
    return _git_bytes("ls-tree", "-z", commit, "--", f":(literal){path}")


def _admit_canonical_merge_suffix(
    board: Any,
    *,
    base_head: str,
    target_head: str,
    bootstrap: Mapping[str, Any],
    integrity: Mapping[str, Any],
    task_outputs: Mapping[str, Sequence[str]],
    completed_requests: Sequence[Any],
    admission_mode: str = "canonical_completion",
) -> dict[str, Any]:
    """Prove every first-parent advance is one exact admitted queue merge."""

    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_repository_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.merge_train import (
        integrated_candidate_handoff_proof,
    )

    base = str(base_head or "").strip().casefold()
    target = str(target_head or "").strip().casefold()
    if admission_mode not in {
        "canonical_completion", "followup_repair_base",
    }:
        raise OperatorError("continuity admission mode is invalid")
    followup_repair_base = admission_mode == "followup_repair_base"
    if followup_repair_base and (
        base != REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT
        or target != REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD
    ):
        raise OperatorError("follow-up repair base identity differs")
    if _git("rev-parse", "--verify", f"{base}^{{commit}}") != base:
        raise OperatorError("continuity base commit is unavailable")
    if _git("merge-base", "--is-ancestor", base, target) != "":
        # Successful merge-base --is-ancestor emits no output.
        raise OperatorError("continuity ancestry command returned output")
    protected = frozenset(str(path) for path in board.protected_paths)
    repository_id = checkout_repository_id(ROOT)
    statuses = integrity.get("task_statuses")
    revisions = integrity.get("task_revisions")
    task_cids = integrity.get("task_cids")
    sealed_revisions = bootstrap.get("integrity", {}).get("task_revisions")
    if not all(
        isinstance(value, Mapping)
        for value in (statuses, revisions, task_cids, sealed_revisions)
    ):
        raise OperatorError("continuity task lifecycle evidence is incomplete")

    raw_suffix = _git(
        "rev-list", "--first-parent", "--reverse", f"{base}..{target}"
    )
    commits = tuple(item for item in raw_suffix.splitlines() if item)
    if followup_repair_base and commits != (
        REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD,
    ):
        raise OperatorError("follow-up repair base is not one exact integration")
    previous = base
    admitted: list[dict[str, Any]] = []
    used_requests: set[str] = set()
    for integration_commit in commits:
        parent_fields = _git(
            "show", "-s", "--format=%P", integration_commit
        ).split()
        if len(parent_fields) != 2 or parent_fields[0] != previous:
            raise OperatorError(
                "continuity suffix contains a non-canonical integration commit"
            )
        candidate = parent_fields[1].casefold()
        matches = [
            request
            for request in completed_requests
            if str(request.commit_sha or "").strip().casefold() == candidate
            and str(request.request_id or "") not in used_requests
        ]
        if len(matches) != 1:
            raise OperatorError(
                "continuity integration lacks one completed queue request"
            )
        request = matches[0]
        metadata = request.metadata
        alias = str(request.task_id or "").strip()
        task_cid = str(request.canonical_task_id or "").strip()
        status = str(statuses.get(alias) or "").strip().casefold()
        lifecycle_admitted = status in COMPLETED_STATUSES
        if followup_repair_base:
            lifecycle_admitted = (
                integration_commit == REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD
                and candidate == REPAIR_FOLLOWUP_TRANSITION_CANDIDATE
                and alias == REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS
                and bool(status)
            )
        if (
            request.status != "completed"
            or metadata.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/merge-candidate@3"
            or metadata.get("target_binding_schema")
            != "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
            or str(metadata.get("target_repository_id") or "") != repository_id
            or str(metadata.get("target_branch") or "")
            != board.merge_target_branch
            or task_cids.get(alias) != task_cid
            or request.canonical_task_key != task_cid
            or not lifecycle_admitted
            or type(revisions.get(alias)) is not int
            or int(revisions[alias]) <= int(sealed_revisions.get(alias, -1))
            or metadata.get("completion_task_cids") != {alias: task_cid}
        ):
            raise OperatorError("continuity queue/task authority binding differs")
        completion = metadata.get("completion")
        if isinstance(completion, Mapping) and (
            completion.get("accepted") is False
            or completion.get("acceptance_pending") is True
            or str(completion.get("status") or "")
            == "integrated_pending_validation"
        ):
            raise OperatorError("continuity merge acceptance is not admitted")

        validation = metadata.get("validation_proof")
        candidate_tree = _git("rev-parse", f"{candidate}^{{tree}}")
        if (
            not isinstance(validation, Mapping)
            or validation.get("passed") is not True
            or str(validation.get("target_commit") or "").casefold()
            != candidate
            or str(validation.get("target_tree") or "").casefold()
            != candidate_tree
            or str(metadata.get("candidate_tree") or "").casefold()
            != candidate_tree
            or str(metadata.get("repository_tree_id") or "")
            != f"git-tree:{candidate_tree}"
        ):
            raise OperatorError("continuity candidate validation binding differs")
        task = metadata.get("task")
        expected_outputs = tuple(str(path) for path in task_outputs.get(alias, ()))
        declared_outputs = (
            tuple(str(path) for path in task.get("outputs") or ())
            if isinstance(task, Mapping)
            else ()
        )
        if not expected_outputs or declared_outputs != expected_outputs:
            raise OperatorError("continuity declared outputs differ from task authority")
        output_set = frozenset(expected_outputs)
        if output_set & protected:
            raise OperatorError("continuity task attempts protected-path mutation")
        baseline_ref = str(metadata.get("baseline_ref") or "").casefold()
        if (
            re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", baseline_ref)
            is None
            or _git("rev-parse", "--verify", f"{baseline_ref}^{{commit}}")
            != baseline_ref
        ):
            raise OperatorError(
                "continuity candidate baseline is not an exact commit"
            )
        # Parallel candidates may share an older baseline, but it must be an
        # immutable ancestor of both the validated candidate and the target
        # state onto which that candidate was integrated.
        _git("merge-base", "--is-ancestor", baseline_ref, candidate)
        _git("merge-base", "--is-ancestor", baseline_ref, previous)
        candidate_paths = _git_changed_paths(baseline_ref, candidate)
        landed_paths = _git_changed_paths(previous, integration_commit)
        if (
            not candidate_paths
            or not set(candidate_paths).issubset(output_set)
            or not set(landed_paths).issubset(output_set)
            or not set(landed_paths).issubset(set(candidate_paths))
            or set(landed_paths) & protected
        ):
            raise OperatorError("continuity merge changed out-of-scope paths")
        # A merge may have no target-relative change for an output already
        # present with identical bytes, so path-set equality is too strict.
        # Nevertheless every path changed by the validated candidate must
        # have its exact tree entry (mode, type, and object ID) in the
        # integration commit; a partial/ours merge cannot be admitted.
        for path in candidate_paths:
            if _git_tree_entry(candidate, path) != _git_tree_entry(
                integration_commit, path
            ):
                raise OperatorError(
                    "continuity merge output differs from validated candidate"
                )
        proof = integrated_candidate_handoff_proof(
            ROOT,
            candidate_commit=candidate,
            target_commit=integration_commit,
            changed_submodule_paths=metadata.get("changed_submodule_paths"),
        )
        if proof.get("passed") is not True:
            raise OperatorError("continuity candidate handoff is not integrated")
        integration_tree = _git("rev-parse", f"{integration_commit}^{{tree}}")
        integration = {
            "request_id": str(request.request_id),
            "task_alias": alias,
            "task_cid": task_cid,
            "candidate_commit": candidate,
            "candidate_tree": candidate_tree,
            "integration_commit": integration_commit,
            "integration_tree": integration_tree,
            "changed_paths": list(landed_paths),
        }
        if followup_repair_base:
            integration.update(
                {
                    "merge_request_cid": _identity(request.to_dict()),
                    "baseline_ref": baseline_ref,
                    "validation_proof_cid": _identity(validation),
                    "completion_authoritative": False,
                    "task_completion_admitted": False,
                    "observed_task_status": status,
                    "observed_task_revision": int(revisions[alias]),
                }
            )
        admitted.append(integration)
        used_requests.add(str(request.request_id))
        previous = integration_commit
    if previous != target:
        raise OperatorError("continuity suffix does not end at the current target")
    result = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            + (
                "aseh-nonterminal-integration-base@1"
                if followup_repair_base
                else "aseh-canonical-merge-suffix@1"
            )
        ),
        "base_head": base,
        "target_head": target,
        "target_tree": _git("rev-parse", f"{target}^{{tree}}"),
        "integrations": admitted,
    }
    if followup_repair_base:
        result["completion_authoritative"] = False
        result["task_completion_admitted"] = False
    return {**result, "receipt_cid": _identity(result)}


def _recovered_control_receipt(
    population: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-task-source@1",
        "plan_root_cid": population["plan_root_cid"],
        "repository_tree_id": population["repository_tree_id"],
        "projection_cid": snapshot["projection_cid"],
        "task_count": len(population["tasks"]),
        "goal_count": len(population["objectives"]),
        "goal_edge_count": len(population["goal_edges"]),
        "plan_count": len(population["plans"]),
        "event_watermark": snapshot["event_cursor"],
        "task_cids": [str(item["task_cid"]) for item in population["tasks"]],
        "recovered_from_exact_projection": True,
    }


def _write_bootstrap_receipt(
    *,
    paths: Mapping[str, Path],
    population: Mapping[str, Any],
    validation: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    integrity: Mapping[str, Any],
    ready: Sequence[str],
    recovered: bool,
) -> dict[str, Any]:
    ducklake = _ducklake_projection(
        paths=paths, population=population, control_receipt=control_receipt,
    )
    receipt = {
        "schema": BOOTSTRAP_SCHEMA,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "source_forest": population["source_forest"],
        "source_identities": population["source_identities"],
        "database_task_source_receipt": dict(control_receipt),
        "snapshot": dict(snapshot),
        "integrity": dict(integrity),
        "initial_ready_task_ids": list(ready),
        "bootstrap_validation": dict(validation),
        "recovered_after_interrupted_materialization": bool(recovered),
        "authority": {
            "operational_state": "DuckDB/DatabaseTaskSource@1",
            "live_transport": "QuackStateServer@1/TypedStateOwnerCommandGateway@1",
            "ducklake": "non_authoritative_history_projection",
        },
        "ducklake_projection": ducklake,
    }
    receipt["bootstrap_receipt_id"] = _identity(receipt)
    _atomic_json(paths["bootstrap_receipt"], receipt)
    return receipt


def materialize(config_path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    board, config = _load(config_path)
    paths = _paths(board)
    population = _population(board, config)
    validation = _validate_bootstrap(board)
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    database = paths["database"]
    bootstrap = paths["bootstrap_receipt"]
    stage = database.with_name(f".{database.name}.bootstrap-stage")
    with _offline_database_guard(paths):
        if _population(board, config) != population:
            raise OperatorError(
                "sealed source population changed after bootstrap validation"
            )
        if bootstrap.exists() and not database.is_file():
            raise OperatorError("bootstrap receipt exists without its database")
        if database.exists() and not database.is_file():
            raise OperatorError("database authority is not a regular file")
        if database.is_file():
            has_bootstrap_receipt = bootstrap.is_file()
            with DatabaseTaskSource(
                database, owner_id="aseh-bootstrap:verify",
                install_schema=False,
                repository_tree_id=population["repository_tree_id"],
                plan_root_cid=population["plan_root_cid"],
            ) as source:
                snapshot, ready, integrity = _verify_materialized_source(
                    source,
                    population=population,
                    config=config,
                    require_initial_frontier=not has_bootstrap_receipt,
                )
            if has_bootstrap_receipt:
                prior = _secure_runtime_json(
                    bootstrap, max_bytes=STATUS_RECEIPT_MAX_BYTES
                )
                _bootstrap_receipt_id(prior)
                if any(
                    prior.get(key) != population.get(key)
                    for key in (
                        "source_head", "repository_tree_id", "plan_root_cid",
                    )
                ) or (
                    prior.get("source_forest") != population["source_forest"]
                    or prior.get("source_identities")
                    != population["source_identities"]
                ):
                    raise OperatorError(
                        "existing authority differs from the sealed source forest"
                    )
                _admit_current_projection_against_bootstrap(
                    prior, snapshot, integrity
                )
                return {
                    "schema": OPERATOR_SCHEMA, "command": "materialize",
                    "ok": True, "idempotent_replay": True,
                    "bootstrap_receipt": prior, "snapshot": snapshot,
                    "ready_task_ids": ready,
                }
            control_receipt = _recovered_control_receipt(population, snapshot)
            receipt = _write_bootstrap_receipt(
                paths=paths, population=population, validation=validation,
                control_receipt=control_receipt, snapshot=snapshot,
                integrity=integrity, ready=ready, recovered=True,
            )
            return {
                "schema": OPERATOR_SCHEMA, "command": "materialize",
                "ok": True, "idempotent_replay": False,
                "recovered": True, "bootstrap_receipt": receipt,
                "snapshot": snapshot,
            }

        for candidate in (stage, Path(f"{stage}.wal")):
            candidate.unlink(missing_ok=True)
        try:
            with DatabaseTaskSource(
                stage, owner_id="aseh-bootstrap:single-writer",
                repository_tree_id=population["repository_tree_id"],
                plan_root_cid=population["plan_root_cid"],
            ) as source:
                control_receipt = dict(source.materialize(population))
                snapshot, ready, integrity = _verify_materialized_source(
                    source, population=population, config=config,
                )
            with stage.open("rb") as handle:
                os.fsync(handle.fileno())
            if _population(board, config) != population:
                raise OperatorError(
                    "sealed source population changed before database publication"
                )
            os.replace(stage, database)
            directory = os.open(
                database.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            stage.unlink(missing_ok=True)
            Path(f"{stage}.wal").unlink(missing_ok=True)
            raise
        receipt = _write_bootstrap_receipt(
            paths=paths, population=population, validation=validation,
            control_receipt=control_receipt, snapshot=snapshot,
            integrity=integrity, ready=ready, recovered=False,
        )
    return {
        "schema": OPERATOR_SCHEMA, "command": "materialize",
        "ok": True, "idempotent_replay": False,
        "bootstrap_receipt": receipt, "snapshot": snapshot,
    }


def _validate_repair_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only the exact single-commit ASEH-BOOTSTRAP-002 repair."""

    receipt_id = _repair_transition_receipt_id(receipt)
    base = str(receipt.get("base_head") or "").casefold()
    repair = str(receipt.get("repair_head") or "").casefold()
    expected_stable_identity = (
        f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R1"
    )
    expected_authority = (
        "the operator explicitly directed the bootstrap engineering agent "
        "to repair the existing canonical handoff and resume through it"
    )
    if (
        base != REPAIR_TRANSITION_BASE_HEAD
        or re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or receipt.get("stable_identity") != expected_stable_identity
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("changed_paths")
        != list(REPAIR_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-001", "ASEH-000"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement") != expected_authority
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError("bootstrap repair transition authority differs")
    parents = _git("show", "-s", "--format=%P", repair).split()
    if parents != [base]:
        raise OperatorError("bootstrap repair must be one exact child commit")
    base_tree = _git("rev-parse", f"{base}^{{tree}}")
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(base, repair) != REPAIR_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest") != _git_patch_digest(base, repair)
    ):
        raise OperatorError("bootstrap repair transition Git proof differs")
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError("bootstrap repair changed a sibling authority")
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results) != len(REPAIR_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError("bootstrap repair validation receipt differs")
    for stored, command in zip(
        stored_results, REPAIR_TRANSITION_VALIDATIONS, strict=True
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(stored.get("stdout_digest") or "")
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(stored.get("stderr_digest") or "")
            )
            is None
        ):
            raise OperatorError("bootstrap repair validation receipt differs")
    if rerun_validations:
        validation_results = _run_repair_transition_validations()
        if [item["argv"] for item in validation_results] != [
            item.get("argv")
            for item in stored_results
        ]:
            raise OperatorError("bootstrap repair validation commands differ")
    result = {
        "schema": REPAIR_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "base_head": base,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(REPAIR_TRANSITION_CHANGED_PATHS),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "receipt_cid": receipt_id,
    }
    return result


def _repair_followup_witness_from_proof(
    proof: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate immutable integration evidence from its live lifecycle view."""

    integrations = proof.get("integrations")
    if (
        proof.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-nonterminal-integration-base@1"
        )
        or proof.get("base_head")
        != REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT
        or proof.get("target_head") != REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD
        or proof.get("completion_authoritative") is not False
        or proof.get("task_completion_admitted") is not False
        or not isinstance(integrations, list)
        or len(integrations) != 1
        or not isinstance(integrations[0], Mapping)
    ):
        raise OperatorError("follow-up nonterminal integration proof differs")
    integration = integrations[0]
    if (
        integration.get("task_alias")
        != REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS
        or integration.get("candidate_commit")
        != REPAIR_FOLLOWUP_TRANSITION_CANDIDATE
        or integration.get("integration_commit")
        != REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD
        or integration.get("completion_authoritative") is not False
        or integration.get("task_completion_admitted") is not False
    ):
        raise OperatorError("follow-up integration identity differs")
    witness = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-nonterminal-integration-witness@1"
        ),
        "base_head": REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT,
        "base_tree": _git(
            "rev-parse",
            f"{REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT}^{{tree}}",
        ),
        "target_head": REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD,
        "target_tree": str(proof.get("target_tree") or ""),
        "request_id": str(integration.get("request_id") or ""),
        "merge_request_cid": str(
            integration.get("merge_request_cid") or ""
        ),
        "task_id": REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS,
        "task_cid": str(integration.get("task_cid") or ""),
        "candidate_commit": REPAIR_FOLLOWUP_TRANSITION_CANDIDATE,
        "candidate_tree": str(integration.get("candidate_tree") or ""),
        "integration_commit": REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD,
        "integration_tree": str(integration.get("integration_tree") or ""),
        "baseline_ref": str(integration.get("baseline_ref") or ""),
        "changed_paths": list(integration.get("changed_paths") or ()),
        "validation_proof_cid": str(
            integration.get("validation_proof_cid") or ""
        ),
        "completion_authoritative": False,
        "task_completion_admitted": False,
    }
    witness["receipt_cid"] = _identity(witness)
    observation = {
        "task_id": REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS,
        "task_cid": witness["task_cid"],
        "status": str(integration.get("observed_task_status") or ""),
        "revision": integration.get("observed_task_revision"),
        "completion_authoritative": False,
        "observed_at": time.time(),
    }
    return witness, observation


def _validate_repair_followup_base_witness(
    witness: Mapping[str, Any],
) -> str:
    if (
        set(witness) != REPAIR_FOLLOWUP_BASE_WITNESS_FIELDS
        or witness.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-nonterminal-integration-witness@1"
        )
        or witness.get("base_head")
        != REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT
        or witness.get("target_head")
        != REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD
        or witness.get("integration_commit")
        != REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD
        or witness.get("candidate_commit")
        != REPAIR_FOLLOWUP_TRANSITION_CANDIDATE
        or witness.get("task_id")
        != REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS
        or witness.get("completion_authoritative") is not False
        or witness.get("task_completion_admitted") is not False
        or not str(witness.get("request_id") or "")
        or not str(witness.get("task_cid") or "")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(witness.get("merge_request_cid") or ""),
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(witness.get("validation_proof_cid") or ""),
        )
        is None
        or not isinstance(witness.get("changed_paths"), list)
    ):
        raise OperatorError("bootstrap repair follow-up witness differs")
    if (
        witness.get("base_tree")
        != _git(
            "rev-parse",
            f"{REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT}^{{tree}}",
        )
        or witness.get("target_tree")
        != _git(
            "rev-parse", f"{REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD}^{{tree}}"
        )
        or witness.get("candidate_tree")
        != _git(
            "rev-parse", f"{REPAIR_FOLLOWUP_TRANSITION_CANDIDATE}^{{tree}}"
        )
        or witness.get("integration_tree") != witness.get("target_tree")
        or _git(
            "show", "-s", "--format=%P",
            REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD,
        ).split()
        != [
            REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT,
            REPAIR_FOLLOWUP_TRANSITION_CANDIDATE,
        ]
    ):
        raise OperatorError("bootstrap repair follow-up Git witness differs")
    unsigned = dict(witness)
    receipt_id = str(unsigned.pop("receipt_cid", "") or "")
    if receipt_id != _identity(unsigned):
        raise OperatorError("bootstrap repair follow-up witness CID is invalid")
    return receipt_id


def _validate_repair_followup_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    previous_receipt: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only revision 2 chained to the immutable revision-1 receipt."""

    receipt_id = _repair_followup_transition_receipt_id(receipt)
    previous_receipt_id = _repair_transition_receipt_id(previous_receipt)
    expected_authority = (
        "the operator explicitly directed the bootstrap engineering agent "
        "to fix the existing canonical supervisor so it automatically "
        "recovers ASEH runtime faults without state-writer contention"
    )
    if (
        receipt.get("stable_identity")
        != f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R2"
        or receipt.get("previous_receipt_cid") != previous_receipt_id
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("base_head")
        != REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD
        or receipt.get("changed_paths")
        != list(REPAIR_FOLLOWUP_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R1", "ASEH-001"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement") != expected_authority
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError("bootstrap repair follow-up authority differs")
    repair = str(receipt.get("repair_head") or "").strip().casefold()
    if (
        re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or _git("show", "-s", "--format=%P", repair).split()
        != [REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD]
    ):
        raise OperatorError("bootstrap repair follow-up must be one exact child")
    base_tree = _git(
        "rev-parse", f"{REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD}^{{tree}}"
    )
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD, repair)
        != REPAIR_FOLLOWUP_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest")
        != _git_patch_digest(REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD, repair)
    ):
        raise OperatorError("bootstrap repair follow-up Git proof differs")
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError("bootstrap repair follow-up changed a sibling")
    witness = receipt.get("base_integration_witness")
    if not isinstance(witness, Mapping):
        raise OperatorError("bootstrap repair follow-up witness is absent")
    witness_id = _validate_repair_followup_base_witness(witness)
    observation = receipt.get("authorization_task_observation")
    sealed_revisions = bootstrap.get("integrity", {}).get("task_revisions")
    sealed_revision = (
        sealed_revisions.get(REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS)
        if isinstance(sealed_revisions, Mapping)
        else None
    )
    if (
        not isinstance(observation, Mapping)
        or set(observation) != REPAIR_FOLLOWUP_TASK_OBSERVATION_FIELDS
        or observation.get("task_id")
        != REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS
        or observation.get("task_cid") != witness.get("task_cid")
        or observation.get("status") not in {"blocked", "retrying"}
        or type(observation.get("revision")) is not int
        or type(sealed_revision) is not int
        or int(observation["revision"]) <= sealed_revision
        or observation.get("completion_authoritative") is not False
        or type(observation.get("observed_at")) not in {int, float}
        or float(observation["observed_at"]) <= 0.0
    ):
        raise OperatorError("bootstrap repair follow-up observation differs")
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results) != len(REPAIR_FOLLOWUP_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError("bootstrap repair follow-up validation differs")
    for stored, command in zip(
        stored_results, REPAIR_FOLLOWUP_TRANSITION_VALIDATIONS, strict=True
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(stored.get("stdout_digest") or "")
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(stored.get("stderr_digest") or "")
            )
            is None
        ):
            raise OperatorError("bootstrap repair follow-up validation differs")
    if rerun_validations:
        rerun = _run_repair_followup_transition_validations()
        if [item["argv"] for item in rerun] != [
            item.get("argv") for item in stored_results
        ]:
            raise OperatorError("bootstrap repair follow-up commands differ")
    return {
        "schema": REPAIR_FOLLOWUP_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "transition_revision": 2,
        "base_head": REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(REPAIR_FOLLOWUP_TRANSITION_CHANGED_PATHS),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "previous_receipt_cid": previous_receipt_id,
        "base_integration_witness_cid": witness_id,
        "base_integration_witness": dict(witness),
        "authorization_task_observation": dict(observation),
        "receipt_cid": receipt_id,
    }


def _validate_repair_clean_launch_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    previous_receipt: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only revision 3 chained to the immutable revision-2 repair."""

    receipt_id = _repair_clean_launch_transition_receipt_id(receipt)
    previous_receipt_id = _repair_followup_transition_receipt_id(
        previous_receipt
    )
    expected_authority = (
        "the operator explicitly directed the bootstrap engineering agent "
        "to fix the existing supervisor so admission validation cannot "
        "materialize credentials in or dirty the launch checkout"
    )
    if (
        receipt.get("stable_identity")
        != f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R3"
        or receipt.get("previous_receipt_cid") != previous_receipt_id
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("base_head")
        != REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD
        or receipt.get("changed_paths")
        != list(REPAIR_CLEAN_LAUNCH_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R2"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement") != expected_authority
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError("bootstrap repair clean-launch authority differs")
    repair = str(receipt.get("repair_head") or "").strip().casefold()
    if (
        re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or _git("show", "-s", "--format=%P", repair).split()
        != [REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD]
    ):
        raise OperatorError(
            "bootstrap repair clean-launch must be one exact child"
        )
    base_tree = _git(
        "rev-parse", f"{REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD}^{{tree}}"
    )
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD, repair)
        != REPAIR_CLEAN_LAUNCH_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest")
        != _git_patch_digest(REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD, repair)
    ):
        raise OperatorError("bootstrap repair clean-launch Git proof differs")
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError(
                "bootstrap repair clean-launch changed a sibling"
            )
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results)
        != len(REPAIR_CLEAN_LAUNCH_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError("bootstrap repair clean-launch validation differs")
    for stored, command in zip(
        stored_results,
        REPAIR_CLEAN_LAUNCH_TRANSITION_VALIDATIONS,
        strict=True,
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(stored.get("stdout_digest") or "")
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(stored.get("stderr_digest") or "")
            )
            is None
        ):
            raise OperatorError(
                "bootstrap repair clean-launch validation differs"
            )
    if rerun_validations:
        rerun = _run_repair_clean_launch_transition_validations()
        if [item["argv"] for item in rerun] != [
            item.get("argv") for item in stored_results
        ]:
            raise OperatorError(
                "bootstrap repair clean-launch commands differ"
            )
    return {
        "schema": REPAIR_CLEAN_LAUNCH_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "transition_revision": 3,
        "base_head": REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(REPAIR_CLEAN_LAUNCH_TRANSITION_CHANGED_PATHS),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "previous_receipt_cid": previous_receipt_id,
        "receipt_cid": receipt_id,
    }


def _validate_repair_runtime_hardening_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    previous_receipt: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only revision 4 chained to the immutable revision-3 repair."""

    receipt_id = _repair_runtime_hardening_transition_receipt_id(receipt)
    previous_receipt_id = _repair_clean_launch_transition_receipt_id(
        previous_receipt
    )
    expected_authority = (
        "the operator explicitly directed the bootstrap engineering agent "
        "to fix the existing supervisor so it automatically recovers ASEH "
        "false-completion, startup, shutdown, and dead-attempt lifecycle "
        "faults without state-writer or checkout contention"
    )
    if (
        receipt.get("stable_identity")
        != f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R4"
        or receipt.get("previous_receipt_cid") != previous_receipt_id
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("base_head")
        != REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD
        or receipt.get("changed_paths")
        != list(REPAIR_RUNTIME_HARDENING_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R3"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement") != expected_authority
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError(
            "bootstrap repair runtime-hardening authority differs"
        )
    repair = str(receipt.get("repair_head") or "").strip().casefold()
    if (
        re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or _git("show", "-s", "--format=%P", repair).split()
        != [REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD]
    ):
        raise OperatorError(
            "bootstrap repair runtime-hardening must be one exact child"
        )
    base_tree = _git(
        "rev-parse",
        f"{REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD}^{{tree}}",
    )
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(
            REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD,
            repair,
        )
        != REPAIR_RUNTIME_HARDENING_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest")
        != _git_patch_digest(
            REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD,
            repair,
        )
    ):
        raise OperatorError(
            "bootstrap repair runtime-hardening Git proof differs"
        )
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError(
                "bootstrap repair runtime-hardening changed a sibling"
            )
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results)
        != len(REPAIR_RUNTIME_HARDENING_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError(
            "bootstrap repair runtime-hardening validation differs"
        )
    for stored, command in zip(
        stored_results,
        REPAIR_RUNTIME_HARDENING_TRANSITION_VALIDATIONS,
        strict=True,
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stdout_digest") or ""),
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stderr_digest") or ""),
            )
            is None
        ):
            raise OperatorError(
                "bootstrap repair runtime-hardening validation differs"
            )
    if rerun_validations:
        rerun = _run_repair_runtime_hardening_transition_validations()
        if [item["argv"] for item in rerun] != [
            item.get("argv") for item in stored_results
        ]:
            raise OperatorError(
                "bootstrap repair runtime-hardening commands differ"
            )
    return {
        "schema": REPAIR_RUNTIME_HARDENING_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "transition_revision": 4,
        "base_head": REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(
            REPAIR_RUNTIME_HARDENING_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "previous_receipt_cid": previous_receipt_id,
        "receipt_cid": receipt_id,
    }


def _validate_repair_quack_recovery_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    previous_receipt: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only revision 5 chained to immutable runtime hardening."""

    receipt_id = _repair_quack_recovery_transition_receipt_id(receipt)
    previous_receipt_id = _repair_runtime_hardening_transition_receipt_id(
        previous_receipt
    )
    expected_authority = (
        "the operator explicitly directed the bootstrap engineering agent "
        "to fix the existing supervisor so it automatically classifies and "
        "recovers exact preprojection Quack transport failures, rebuilds only "
        "the observed lane-private stale coordination index under its sole "
        "writer lock, and resumes ASEH without state-writer contention"
    )
    if (
        receipt.get("stable_identity")
        != f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R5"
        or receipt.get("previous_receipt_cid") != previous_receipt_id
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("base_head")
        != REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD
        or receipt.get("changed_paths")
        != list(REPAIR_QUACK_RECOVERY_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R4"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement") != expected_authority
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery authority differs"
        )
    repair = str(receipt.get("repair_head") or "").strip().casefold()
    if (
        re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or _git("show", "-s", "--format=%P", repair).split()
        != [REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD]
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery must be one exact child"
        )
    base_tree = _git(
        "rev-parse",
        f"{REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD}^{{tree}}",
    )
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(
            REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD,
            repair,
        )
        != REPAIR_QUACK_RECOVERY_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest")
        != _git_patch_digest(
            REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD,
            repair,
        )
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery Git proof differs"
        )
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError(
                "bootstrap repair Quack-recovery changed a sibling"
            )
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results)
        != len(REPAIR_QUACK_RECOVERY_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery validation differs"
        )
    for stored, command in zip(
        stored_results,
        REPAIR_QUACK_RECOVERY_TRANSITION_VALIDATIONS,
        strict=True,
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stdout_digest") or ""),
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stderr_digest") or ""),
            )
            is None
        ):
            raise OperatorError(
                "bootstrap repair Quack-recovery validation differs"
            )
    if rerun_validations:
        rerun = _run_repair_quack_recovery_transition_validations()
        if [item["argv"] for item in rerun] != [
            item.get("argv") for item in stored_results
        ]:
            raise OperatorError(
                "bootstrap repair Quack-recovery commands differ"
            )
    return {
        "schema": REPAIR_QUACK_RECOVERY_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "transition_revision": 5,
        "base_head": REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(
            REPAIR_QUACK_RECOVERY_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "previous_receipt_cid": previous_receipt_id,
        "receipt_cid": receipt_id,
    }


def _validate_repair_parallel_blocked_startup_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    previous_receipt: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only revision 6 chained to immutable Quack recovery."""

    receipt_id = _repair_parallel_blocked_startup_transition_receipt_id(
        receipt
    )
    previous_receipt_id = _repair_quack_recovery_transition_receipt_id(
        previous_receipt
    )
    expected_authority = (
        "the operator explicitly directed the bootstrap engineering agent "
        "to fix the existing supervisor so exact recoverable blocked tasks "
        "can rearm while a parallel ready frontier exists, without weakening "
        "single-writer, freshness, validation, or bounded startup gates"
    )
    if (
        receipt.get("stable_identity")
        != f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R6"
        or receipt.get("previous_receipt_cid") != previous_receipt_id
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("base_head")
        != REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD
        or receipt.get("changed_paths")
        != list(REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R5"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement") != expected_authority
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError(
            "bootstrap repair parallel-blocked-startup authority differs"
        )
    repair = str(receipt.get("repair_head") or "").strip().casefold()
    if (
        re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or _git("show", "-s", "--format=%P", repair).split()
        != [REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD]
    ):
        raise OperatorError(
            "bootstrap repair parallel-blocked-startup must be one exact child"
        )
    base_tree = _git(
        "rev-parse",
        f"{REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD}^{{tree}}",
    )
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(
            REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD,
            repair,
        )
        != REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest")
        != _git_patch_digest(
            REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD,
            repair,
        )
    ):
        raise OperatorError(
            "bootstrap repair parallel-blocked-startup Git proof differs"
        )
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError(
                "bootstrap repair parallel-blocked-startup changed a sibling"
            )
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results)
        != len(REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError(
            "bootstrap repair parallel-blocked-startup validation differs"
        )
    for stored, command in zip(
        stored_results,
        REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_VALIDATIONS,
        strict=True,
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stdout_digest") or ""),
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stderr_digest") or ""),
            )
            is None
        ):
            raise OperatorError(
                "bootstrap repair parallel-blocked-startup validation differs"
            )
    if rerun_validations:
        rerun = _run_repair_parallel_blocked_startup_transition_validations()
        if [item["argv"] for item in rerun] != [
            item.get("argv") for item in stored_results
        ]:
            raise OperatorError(
                "bootstrap repair parallel-blocked-startup commands differ"
            )
    return {
        "schema": REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "transition_revision": 6,
        "base_head": REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(
            REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "previous_receipt_cid": previous_receipt_id,
        "receipt_cid": receipt_id,
    }


def _validate_repair_quack_publication_contention_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    previous_receipt: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only revision 7 chained to immutable startup recovery."""

    receipt_id = _repair_quack_publication_contention_transition_receipt_id(
        receipt
    )
    previous_receipt_id = (
        _repair_parallel_blocked_startup_transition_receipt_id(
            previous_receipt
        )
    )
    if (
        receipt.get("stable_identity")
        != f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R7"
        or receipt.get("previous_receipt_cid") != previous_receipt_id
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("base_head")
        != REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD
        or receipt.get("changed_paths")
        != list(REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R6"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement")
        != REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_AUTHORITY
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError(
            "bootstrap repair Quack-publication-contention authority differs"
        )
    repair = str(receipt.get("repair_head") or "").strip().casefold()
    if (
        re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or _git("show", "-s", "--format=%P", repair).split()
        != [REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD]
    ):
        raise OperatorError(
            "bootstrap repair Quack-publication-contention must be one exact "
            "child"
        )
    base_tree = _git(
        "rev-parse",
        f"{REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD}^{{tree}}",
    )
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(
            REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD,
            repair,
        )
        != REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest")
        != _git_patch_digest(
            REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD,
            repair,
        )
    ):
        raise OperatorError(
            "bootstrap repair Quack-publication-contention Git proof differs"
        )
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError(
                "bootstrap repair Quack-publication-contention changed a "
                "sibling"
            )
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results)
        != len(REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError(
            "bootstrap repair Quack-publication-contention validation differs"
        )
    for stored, command in zip(
        stored_results,
        REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_VALIDATIONS,
        strict=True,
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stdout_digest") or ""),
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stderr_digest") or ""),
            )
            is None
        ):
            raise OperatorError(
                "bootstrap repair Quack-publication-contention validation "
                "differs"
            )
    if rerun_validations:
        rerun = (
            _run_repair_quack_publication_contention_transition_validations()
        )
        if [item["argv"] for item in rerun] != [
            item.get("argv") for item in stored_results
        ]:
            raise OperatorError(
                "bootstrap repair Quack-publication-contention commands differ"
            )
    return {
        "schema": REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "transition_revision": 7,
        "base_head": REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(
            REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "previous_receipt_cid": previous_receipt_id,
        "receipt_cid": receipt_id,
    }


def _validate_repair_quack_recovery_replay_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    previous_receipt: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only revision 8 chained to immutable publication recovery."""

    receipt_id = _repair_quack_recovery_replay_transition_receipt_id(receipt)
    previous_receipt_id = (
        _repair_quack_publication_contention_transition_receipt_id(
            previous_receipt
        )
    )
    if (
        receipt.get("stable_identity")
        != f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R8"
        or receipt.get("previous_receipt_cid") != previous_receipt_id
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("base_head")
        != REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD
        or receipt.get("changed_paths")
        != list(REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R7"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement")
        != REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_AUTHORITY
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery-replay authority differs"
        )
    repair = str(receipt.get("repair_head") or "").strip().casefold()
    if (
        re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or _git("show", "-s", "--format=%P", repair).split()
        != [REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD]
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery-replay must be one exact child"
        )
    base_tree = _git(
        "rev-parse",
        f"{REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD}^{{tree}}",
    )
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(
            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD,
            repair,
        )
        != REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest")
        != _git_patch_digest(
            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD,
            repair,
        )
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery-replay Git proof differs"
        )
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError(
                "bootstrap repair Quack-recovery-replay changed a sibling"
            )
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results)
        != len(REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError(
            "bootstrap repair Quack-recovery-replay validation differs"
        )
    for stored, command in zip(
        stored_results,
        REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_VALIDATIONS,
        strict=True,
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stdout_digest") or ""),
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stderr_digest") or ""),
            )
            is None
        ):
            raise OperatorError(
                "bootstrap repair Quack-recovery-replay validation differs"
            )
    if rerun_validations:
        rerun = _run_repair_quack_recovery_replay_transition_validations()
        if [item["argv"] for item in rerun] != [
            item.get("argv") for item in stored_results
        ]:
            raise OperatorError(
                "bootstrap repair Quack-recovery-replay commands differ"
            )
    return {
        "schema": REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "transition_revision": 8,
        "base_head": REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(
            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "previous_receipt_cid": previous_receipt_id,
        "receipt_cid": receipt_id,
    }


def _validate_repair_control_receipt_lifecycle_transition(
    receipt: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    previous_receipt: Mapping[str, Any],
    rerun_validations: bool,
) -> dict[str, Any]:
    """Admit only revision 9 chained to immutable recovery replay."""

    receipt_id = _repair_control_receipt_lifecycle_transition_receipt_id(
        receipt
    )
    previous_receipt_id = (
        _repair_quack_recovery_replay_transition_receipt_id(previous_receipt)
    )
    if (
        receipt.get("stable_identity")
        != f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R9"
        or receipt.get("previous_receipt_cid") != previous_receipt_id
        or receipt.get("bootstrap_receipt_id")
        != bootstrap.get("bootstrap_receipt_id")
        or receipt.get("plan_root_cid") != bootstrap.get("plan_root_cid")
        or receipt.get("repository_tree_id")
        != bootstrap.get("repository_tree_id")
        or receipt.get("base_head")
        != REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD
        or receipt.get("changed_paths")
        != list(REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_CHANGED_PATHS)
        or receipt.get("dependencies")
        != ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R8"]
        or receipt.get("owning_repository") != "ipfs_accelerate_py"
        or receipt.get("risk_class")
        != "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
        or receipt.get("authority_requirement")
        != REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_AUTHORITY
        or type(receipt.get("authorized_at")) not in {int, float}
        or float(receipt["authorized_at"]) <= 0.0
    ):
        raise OperatorError(
            "bootstrap repair control-receipt-lifecycle authority differs"
        )
    repair = str(receipt.get("repair_head") or "").strip().casefold()
    if (
        re.fullmatch(r"[0-9a-f]{40}", repair) is None
        or _git("show", "-s", "--format=%P", repair).split()
        != [REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD]
    ):
        raise OperatorError(
            "bootstrap repair control-receipt-lifecycle must be one exact "
            "child"
        )
    base_tree = _git(
        "rev-parse",
        f"{REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD}^{{tree}}",
    )
    repair_tree = _git("rev-parse", f"{repair}^{{tree}}")
    if (
        receipt.get("base_tree") != base_tree
        or receipt.get("repair_tree") != repair_tree
        or _git_changed_paths(
            REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD,
            repair,
        )
        != REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_CHANGED_PATHS
        or receipt.get("patch_digest")
        != _git_patch_digest(
            REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD,
            repair,
        )
    ):
        raise OperatorError(
            "bootstrap repair control-receipt-lifecycle Git proof differs"
        )
    forest = bootstrap.get("source_forest")
    by_owner = forest.get("by_owner") if isinstance(forest, Mapping) else None
    if not isinstance(by_owner, Mapping):
        raise OperatorError("bootstrap source forest owner binding is absent")
    for owner, path in (
        ("ipfs_datasets_py", "ipfs_datasets_py"),
        ("ipfs_kit_py", "ipfs_kit_py"),
    ):
        expected = by_owner.get(owner)
        if (
            not isinstance(expected, Mapping)
            or _git("rev-parse", f"{repair}:{path}")
            != expected.get("commit")
        ):
            raise OperatorError(
                "bootstrap repair control-receipt-lifecycle changed a sibling"
            )
    stored_results = receipt.get("validation_results")
    if (
        not isinstance(stored_results, list)
        or len(stored_results)
        != len(REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_VALIDATIONS)
    ):
        raise OperatorError(
            "bootstrap repair control-receipt-lifecycle validation differs"
        )
    for stored, command in zip(
        stored_results,
        REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_VALIDATIONS,
        strict=True,
    ):
        if (
            not isinstance(stored, Mapping)
            or set(stored)
            != {"argv", "returncode", "stdout_digest", "stderr_digest"}
            or stored.get("argv") != list(command)
            or stored.get("returncode") != 0
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stdout_digest") or ""),
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(stored.get("stderr_digest") or ""),
            )
            is None
        ):
            raise OperatorError(
                "bootstrap repair control-receipt-lifecycle validation differs"
            )
    if rerun_validations:
        rerun = _run_repair_control_receipt_lifecycle_transition_validations()
        if [item["argv"] for item in rerun] != [
            item.get("argv") for item in stored_results
        ]:
            raise OperatorError(
                "bootstrap repair control-receipt-lifecycle commands differ"
            )
    return {
        "schema": REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "transition_revision": 9,
        "base_head": REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD,
        "base_tree": base_tree,
        "repair_head": repair,
        "repair_tree": repair_tree,
        "changed_paths": list(
            REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": str(receipt.get("patch_digest") or ""),
        "previous_receipt_cid": previous_receipt_id,
        "receipt_cid": receipt_id,
    }


def _projection_matches_events_on_disposable_copy(database: Path) -> bool:
    """Replay projections on a private clone, never on authoritative bytes."""

    import shutil
    import tempfile

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    def identity(path: Path) -> tuple[int, int, int, int, int]:
        value = path.stat()
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    database_before = identity(database)
    wal = Path(f"{database}.wal")
    wal_before = identity(wal) if wal.is_file() else None
    with tempfile.TemporaryDirectory(prefix="aseh-event-replay-") as raw:
        clone = Path(raw) / database.name
        shutil.copy2(database, clone)
        if wal_before is not None:
            shutil.copy2(wal, Path(f"{clone}.wal"))
        if identity(database) != database_before:
            raise OperatorError("control database changed during private replay copy")
        if (identity(wal) if wal.is_file() else None) != wal_before:
            raise OperatorError("control database WAL changed during private replay copy")
        with DatabaseTaskSource(
            clone,
            owner_id="aseh-private-event-replay",
            install_schema=False,
        ) as replay:
            return replay.projection_matches_events() is True


@contextmanager
def _read_only_database_task_source(
    database: Path,
    *,
    owner_id: str,
    repository_tree_id: str,
    plan_root_cid: str,
) -> Any:
    """Bind DatabaseTaskSource reads to one policy-locked read-only handle."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        connect_duckdb_with_policy,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        open_intent_repository,
    )

    import duckdb

    connection = connect_duckdb_with_policy(duckdb, database, read_only=True)
    source: DatabaseTaskSource | None = None
    try:
        intent = open_intent_repository(
            database,
            bound_connection=connection,
            owner_id=owner_id,
            install_schema=False,
        )
        source = DatabaseTaskSource(
            intent=intent,
            owner_id=owner_id,
            repository_tree_id=repository_tree_id,
            plan_root_cid=plan_root_cid,
        )
        yield source
    finally:
        if source is not None:
            source.close()
        connection.close()


def _read_continuity_state(
    board: Any,
    paths: Mapping[str, Path],
    bootstrap: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[str],
    dict[str, Any],
    dict[str, tuple[str, ...]],
    tuple[Any, ...],
]:
    """Take one mutually excluded read-only DB and merge-queue snapshot."""

    with _offline_database_guard(paths):
        with _offline_merge_queue_guard(paths) as queue_database:
            projection_matches = _projection_matches_events_on_disposable_copy(
                paths["database"]
            )
            with _read_only_database_task_source(
                paths["database"],
                owner_id="aseh-launch-continuity:read-only",
                repository_tree_id=str(bootstrap["repository_tree_id"]),
                plan_root_cid=str(bootstrap["plan_root_cid"]),
            ) as source:
                snapshot, ready, integrity, outputs = (
                    _verify_materialized_source_from_bootstrap(
                        source,
                        bootstrap=bootstrap,
                        projection_matches_events=projection_matches,
                    )
                )
            requests = _read_completed_merge_requests(queue_database)
    return snapshot, ready, integrity, outputs, requests


def _admit_repair_followup_base(
    board: Any,
    *,
    bootstrap: Mapping[str, Any],
    integrity: Mapping[str, Any],
    task_outputs: Mapping[str, Sequence[str]],
    completed_requests: Sequence[Any],
    stored_followup: Mapping[str, Any] | None = None,
    require_nonterminal: bool = False,
) -> dict[str, Any]:
    """Admit exact integrated bytes without converting them to completion."""

    statuses = integrity.get("task_statuses")
    revisions = integrity.get("task_revisions")
    task_cids = integrity.get("task_cids")
    alias = REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS
    if not all(
        isinstance(value, Mapping) for value in (statuses, revisions, task_cids)
    ):
        raise OperatorError("follow-up lifecycle authority is incomplete")
    status = str(statuses.get(alias) or "").strip().casefold()
    revision = revisions.get(alias)
    task_cid = str(task_cids.get(alias) or "")
    if type(revision) is not int or not task_cid:
        raise OperatorError("follow-up lifecycle identity is incomplete")
    historical: Mapping[str, Any] = {}
    stored_witness: Mapping[str, Any] = {}
    if stored_followup is not None:
        raw_historical = stored_followup.get("authorization_task_observation")
        raw_witness = stored_followup.get("base_integration_witness")
        if not isinstance(raw_historical, Mapping) or not isinstance(
            raw_witness, Mapping
        ):
            raise OperatorError("follow-up historical evidence is absent")
        historical = raw_historical
        stored_witness = raw_witness
        if (
            historical.get("task_cid") != task_cid
            or type(historical.get("revision")) is not int
            or revision < int(historical["revision"])
            or (
                revision == int(historical["revision"])
                and status != str(historical.get("status") or "")
            )
            or stored_witness.get("task_cid") != task_cid
        ):
            raise OperatorError("follow-up task identity or revision regressed")
    if require_nonterminal and status not in {"blocked", "retrying"}:
        raise OperatorError(
            "follow-up authorization requires blocked or retrying ASEH-001"
        )
    witness_proof = _admit_canonical_merge_suffix(
        board,
        base_head=REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT,
        target_head=REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD,
        bootstrap=bootstrap,
        integrity=integrity,
        task_outputs=task_outputs,
        completed_requests=completed_requests,
        admission_mode="followup_repair_base",
    )
    witness, current_observation = _repair_followup_witness_from_proof(
        witness_proof
    )
    if stored_followup is not None and witness != dict(stored_witness):
        raise OperatorError("follow-up immutable integration witness drifted")
    result = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-followup-base-continuity@1"
        ),
        "admission": "immutable_integration_witness",
        "task_completion_admitted": False,
        "proof": witness_proof,
        "base_integration_witness": witness,
        "authorization_task_observation": (
            dict(historical)
            if stored_followup is not None
            else current_observation
        ),
    }
    if status in COMPLETED_STATUSES:
        result["canonical_completion_proof"] = _admit_canonical_merge_suffix(
            board,
            base_head=REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT,
            target_head=REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs=task_outputs,
            completed_requests=completed_requests,
        )
        result["admission"] = "canonical_completion"
        result["task_completion_admitted"] = True
    return result


def authorize_repair_transition(config_path: Path) -> dict[str, Any]:
    """Authorize the one user-directed bootstrap repair after it is committed."""

    board, _config = _load(config_path)
    paths = _paths(board)
    population = _population(board, _config)
    head = str(population["source_head"])
    if head == REPAIR_TRANSITION_BASE_HEAD:
        raise OperatorError("bootstrap repair transition has not been committed")
    bootstrap = _secure_runtime_json(
        paths["bootstrap_receipt"], max_bytes=STATUS_RECEIPT_MAX_BYTES
    )
    bootstrap_id = _bootstrap_receipt_id(bootstrap)
    if paths["repair_transition_receipt"].is_file():
        prior = _secure_runtime_json(
            paths["repair_transition_receipt"],
            max_bytes=STATUS_RECEIPT_MAX_BYTES,
        )
        prior_transition = _validate_repair_transition(
            prior, bootstrap=bootstrap, rerun_validations=False
        )
        repair_head = str(prior_transition["repair_head"])
        _git("merge-base", "--is-ancestor", repair_head, head)
        followup_path = paths.get("repair_followup_transition_receipt")
        if isinstance(followup_path, Path) and followup_path.is_file():
            followup = _secure_runtime_json(
                followup_path, max_bytes=STATUS_RECEIPT_MAX_BYTES
            )
            advanced = followup.get("repair_head") != head
            followup_transition = _validate_repair_followup_transition(
                followup,
                bootstrap=bootstrap,
                previous_receipt=prior,
                rerun_validations=not advanced,
            )
            _git(
                "merge-base", "--is-ancestor",
                str(followup_transition["repair_head"]), head,
            )
            clean_launch_path = paths.get(
                "repair_clean_launch_transition_receipt"
            )
            if (
                isinstance(clean_launch_path, Path)
                and clean_launch_path.is_file()
            ):
                clean_launch = _secure_runtime_json(
                    clean_launch_path, max_bytes=STATUS_RECEIPT_MAX_BYTES
                )
                clean_launch_transition = (
                    _validate_repair_clean_launch_transition(
                        clean_launch,
                        bootstrap=bootstrap,
                        previous_receipt=followup,
                        rerun_validations=(
                            clean_launch.get("repair_head") == head
                        ),
                    )
                )
                _git(
                    "merge-base", "--is-ancestor",
                    str(clean_launch_transition["repair_head"]), head,
                )
                runtime_path = paths.get(
                    "repair_runtime_hardening_transition_receipt"
                )
                if isinstance(runtime_path, Path) and runtime_path.is_file():
                    runtime_hardening = _secure_runtime_json(
                        runtime_path,
                        max_bytes=STATUS_RECEIPT_MAX_BYTES,
                    )
                    runtime_transition = (
                        _validate_repair_runtime_hardening_transition(
                            runtime_hardening,
                            bootstrap=bootstrap,
                            previous_receipt=clean_launch,
                            rerun_validations=(
                                runtime_hardening.get("repair_head") == head
                            ),
                        )
                    )
                    _git(
                        "merge-base",
                        "--is-ancestor",
                        str(runtime_transition["repair_head"]),
                        head,
                    )
                    quack_recovery_path = paths.get(
                        "repair_quack_recovery_transition_receipt"
                    )
                    if (
                        isinstance(quack_recovery_path, Path)
                        and quack_recovery_path.is_file()
                    ):
                        quack_recovery = _secure_runtime_json(
                            quack_recovery_path,
                            max_bytes=STATUS_RECEIPT_MAX_BYTES,
                        )
                        quack_recovery_transition = (
                            _validate_repair_quack_recovery_transition(
                                quack_recovery,
                                bootstrap=bootstrap,
                                previous_receipt=runtime_hardening,
                                rerun_validations=(
                                    quack_recovery.get("repair_head") == head
                                ),
                            )
                        )
                        _git(
                            "merge-base",
                            "--is-ancestor",
                            str(quack_recovery_transition["repair_head"]),
                            head,
                        )
                        parallel_startup_path = paths.get(
                            "repair_parallel_blocked_startup_transition_receipt"
                        )
                        if (
                            isinstance(parallel_startup_path, Path)
                            and parallel_startup_path.is_file()
                        ):
                            parallel_startup = _secure_runtime_json(
                                parallel_startup_path,
                                max_bytes=STATUS_RECEIPT_MAX_BYTES,
                            )
                            parallel_startup_transition = (
                                _validate_repair_parallel_blocked_startup_transition(
                                    parallel_startup,
                                    bootstrap=bootstrap,
                                    previous_receipt=quack_recovery,
                                    rerun_validations=(
                                        parallel_startup.get("repair_head")
                                        == head
                                    ),
                                )
                            )
                            _git(
                                "merge-base",
                                "--is-ancestor",
                                str(
                                    parallel_startup_transition["repair_head"]
                                ),
                                head,
                            )
                            publication_path = paths.get(
                                "repair_quack_publication_contention_transition_receipt"
                            )
                            if (
                                isinstance(publication_path, Path)
                                and publication_path.is_file()
                            ):
                                publication = _secure_runtime_json(
                                    publication_path,
                                    max_bytes=STATUS_RECEIPT_MAX_BYTES,
                                )
                                publication_transition = (
                                    _validate_repair_quack_publication_contention_transition(
                                        publication,
                                        bootstrap=bootstrap,
                                        previous_receipt=parallel_startup,
                                        rerun_validations=(
                                            publication.get("repair_head")
                                            == head
                                        ),
                                    )
                                )
                                _git(
                                    "merge-base",
                                    "--is-ancestor",
                                    str(publication_transition["repair_head"]),
                                    head,
                                )
                                replay_path = paths.get(
                                    "repair_quack_recovery_replay_transition_receipt"
                                )
                                if (
                                    isinstance(replay_path, Path)
                                    and replay_path.is_file()
                                ):
                                    replay = _secure_runtime_json(
                                        replay_path,
                                        max_bytes=STATUS_RECEIPT_MAX_BYTES,
                                    )
                                    replay_transition = (
                                        _validate_repair_quack_recovery_replay_transition(
                                            replay,
                                            bootstrap=bootstrap,
                                            previous_receipt=publication,
                                            rerun_validations=(
                                                replay.get("repair_head")
                                                == head
                                            ),
                                        )
                                    )
                                    _git(
                                        "merge-base",
                                        "--is-ancestor",
                                        str(replay_transition["repair_head"]),
                                        head,
                                    )
                                    lifecycle_path = paths.get(
                                        "repair_control_receipt_lifecycle_transition_receipt"
                                    )
                                    if (
                                        isinstance(lifecycle_path, Path)
                                        and lifecycle_path.is_file()
                                    ):
                                        lifecycle = _secure_runtime_json(
                                            lifecycle_path,
                                            max_bytes=STATUS_RECEIPT_MAX_BYTES,
                                        )
                                        lifecycle_transition = (
                                            _validate_repair_control_receipt_lifecycle_transition(
                                                lifecycle,
                                                bootstrap=bootstrap,
                                                previous_receipt=replay,
                                                rerun_validations=(
                                                    lifecycle.get(
                                                        "repair_head"
                                                    )
                                                    == head
                                                ),
                                            )
                                        )
                                        _git(
                                            "merge-base",
                                            "--is-ancestor",
                                            str(
                                                lifecycle_transition[
                                                    "repair_head"
                                                ]
                                            ),
                                            head,
                                        )
                                        current_admission = (
                                            _admit_materialized_launch(
                                                board, _config, paths
                                            )
                                        )
                                        admitted_repair = (
                                            current_admission.get(
                                                "repair_transition"
                                            )
                                        )
                                        admitted_continuity = (
                                            current_admission.get(
                                                "canonical_continuity"
                                            )
                                        )
                                        if (
                                            not isinstance(
                                                admitted_repair, Mapping
                                            )
                                            or admitted_repair.get(
                                                "repair_head"
                                            )
                                            != lifecycle_transition[
                                                "repair_head"
                                            ]
                                            or not isinstance(
                                                admitted_continuity, Mapping
                                            )
                                            or "repair_to_current"
                                            not in admitted_continuity
                                        ):
                                            raise OperatorError(
                                                "current admission does not "
                                                "retain the control-receipt-"
                                                "lifecycle repair transition"
                                            )
                                        return {
                                            "schema": OPERATOR_SCHEMA,
                                            "command": (
                                                "authorize-repair-transition"
                                            ),
                                            "ok": True,
                                            "idempotent_replay": True,
                                            "repair_transition_receipt": (
                                                lifecycle
                                            ),
                                            "repair_transition_chain": [
                                                prior,
                                                followup,
                                                clean_launch,
                                                runtime_hardening,
                                                quack_recovery,
                                                parallel_startup,
                                                publication,
                                                replay,
                                                lifecycle,
                                            ],
                                            "current_admission_cid": (
                                                current_admission[
                                                    "admission_cid"
                                                ]
                                            ),
                                            "runtime_source_head": (
                                                current_admission[
                                                    "runtime_source_head"
                                                ]
                                            ),
                                        }
                                    if replay.get("repair_head") != head:
                                        parents = _git(
                                            "show", "-s", "--format=%P", head
                                        ).split()
                                        if parents != [
                                            REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD
                                        ]:
                                            raise OperatorError(
                                                "bootstrap repair control-"
                                                "receipt-lifecycle must be one "
                                                "child of the exact revision-8 "
                                                "repair"
                                            )
                                        if _git_changed_paths(
                                            REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD,
                                            head,
                                        ) != (
                                            REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_CHANGED_PATHS
                                        ):
                                            raise OperatorError(
                                                "bootstrap repair control-"
                                                "receipt-lifecycle changed-path "
                                                "set differs"
                                            )
                                        validation_results = (
                                            _run_repair_control_receipt_lifecycle_transition_validations()
                                        )
                                        receipt = {
                                            "schema": (
                                                REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_SCHEMA
                                            ),
                                            "task_id": REPAIR_TRANSITION_TASK_ID,
                                            "stable_identity": (
                                                f"{PROGRAM}/"
                                                f"{REPAIR_TRANSITION_TASK_ID}"
                                                "@ASEH-PLAN-R9"
                                            ),
                                            "program_id": PROGRAM,
                                            "transition_revision": 9,
                                            "bootstrap_receipt_id": (
                                                bootstrap_id
                                            ),
                                            "previous_receipt_cid": (
                                                replay_transition[
                                                    "receipt_cid"
                                                ]
                                            ),
                                            "plan_root_cid": bootstrap[
                                                "plan_root_cid"
                                            ],
                                            "repository_tree_id": bootstrap[
                                                "repository_tree_id"
                                            ],
                                            "base_head": (
                                                REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD
                                            ),
                                            "base_tree": _git(
                                                "rev-parse",
                                                REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD
                                                + "^{tree}",
                                            ),
                                            "repair_head": head,
                                            "repair_tree": _git(
                                                "rev-parse", f"{head}^{{tree}}"
                                            ),
                                            "changed_paths": list(
                                                REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_CHANGED_PATHS
                                            ),
                                            "patch_digest": _git_patch_digest(
                                                REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_BASE_HEAD,
                                                head,
                                            ),
                                            "dependencies": [
                                                "ASEH-BOOTSTRAP-002@ASEH-PLAN-R8"
                                            ],
                                            "owning_repository": (
                                                "ipfs_accelerate_py"
                                            ),
                                            "risk_class": (
                                                "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
                                            ),
                                            "authority_requirement": (
                                                REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_AUTHORITY
                                            ),
                                            "validation_results": (
                                                validation_results
                                            ),
                                            "terminal_success_criteria": (
                                                "The canonical nonnegative "
                                                "unknown-callback reopen count "
                                                "is identical in task body and "
                                                "closed recovery receipt, every "
                                                "recovery verifier replays it, "
                                                "and affected lanes remain live."
                                            ),
                                            "terminal_non_success_criteria": (
                                                "Any missing counterpart, "
                                                "mismatch, boolean, string, "
                                                "negative count, foreign field, "
                                                "identity or sibling drift, "
                                                "database mutation during "
                                                "validation, or validation "
                                                "failure is rejected."
                                            ),
                                            "semantic_corpus_changed": False,
                                            "database_mutated": False,
                                            "authorized_at": time.time(),
                                        }
                                        receipt["receipt_cid"] = _identity(
                                            receipt
                                        )
                                        _validate_repair_control_receipt_lifecycle_transition(
                                            receipt,
                                            bootstrap=bootstrap,
                                            previous_receipt=replay,
                                            rerun_validations=False,
                                        )
                                        if not isinstance(
                                            lifecycle_path, Path
                                        ):
                                            raise OperatorError(
                                                "bootstrap repair control-"
                                                "receipt-lifecycle path is "
                                                "absent"
                                            )
                                        _atomic_json_create(
                                            lifecycle_path, receipt
                                        )
                                        return {
                                            "schema": OPERATOR_SCHEMA,
                                            "command": (
                                                "authorize-repair-transition"
                                            ),
                                            "ok": True,
                                            "idempotent_replay": False,
                                            "repair_transition_receipt": (
                                                receipt
                                            ),
                                            "repair_transition_chain": [
                                                prior,
                                                followup,
                                                clean_launch,
                                                runtime_hardening,
                                                quack_recovery,
                                                parallel_startup,
                                                publication,
                                                replay,
                                                receipt,
                                            ],
                                        }
                                    current_admission = (
                                        _admit_materialized_launch(
                                            board, _config, paths
                                        )
                                    )
                                    admitted_repair = current_admission.get(
                                        "repair_transition"
                                    )
                                    admitted_continuity = (
                                        current_admission.get(
                                            "canonical_continuity"
                                        )
                                    )
                                    if (
                                        not isinstance(
                                            admitted_repair, Mapping
                                        )
                                        or admitted_repair.get("repair_head")
                                        != replay_transition["repair_head"]
                                        or not isinstance(
                                            admitted_continuity, Mapping
                                        )
                                        or "repair_to_current"
                                        not in admitted_continuity
                                    ):
                                        raise OperatorError(
                                            "current admission does not retain "
                                            "the Quack-recovery-replay repair "
                                            "transition"
                                        )
                                    return {
                                        "schema": OPERATOR_SCHEMA,
                                        "command": (
                                            "authorize-repair-transition"
                                        ),
                                        "ok": True,
                                        "idempotent_replay": True,
                                        "repair_transition_receipt": replay,
                                        "repair_transition_chain": [
                                            prior,
                                            followup,
                                            clean_launch,
                                            runtime_hardening,
                                            quack_recovery,
                                            parallel_startup,
                                            publication,
                                            replay,
                                        ],
                                        "current_admission_cid": (
                                            current_admission["admission_cid"]
                                        ),
                                        "runtime_source_head": (
                                            current_admission[
                                                "runtime_source_head"
                                            ]
                                        ),
                                    }
                                if publication.get("repair_head") != head:
                                    parents = _git(
                                        "show", "-s", "--format=%P", head
                                    ).split()
                                    if parents != [
                                        REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD
                                    ]:
                                        raise OperatorError(
                                            "bootstrap repair Quack-recovery-"
                                            "replay must be one child of the "
                                            "exact revision-7 repair"
                                        )
                                    if _git_changed_paths(
                                        REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD,
                                        head,
                                    ) != (
                                        REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_CHANGED_PATHS
                                    ):
                                        raise OperatorError(
                                            "bootstrap repair Quack-recovery-"
                                            "replay changed-path set differs"
                                        )
                                    validation_results = (
                                        _run_repair_quack_recovery_replay_transition_validations()
                                    )
                                    receipt = {
                                        "schema": (
                                            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_SCHEMA
                                        ),
                                        "task_id": REPAIR_TRANSITION_TASK_ID,
                                        "stable_identity": (
                                            f"{PROGRAM}/"
                                            f"{REPAIR_TRANSITION_TASK_ID}"
                                            "@ASEH-PLAN-R8"
                                        ),
                                        "program_id": PROGRAM,
                                        "transition_revision": 8,
                                        "bootstrap_receipt_id": bootstrap_id,
                                        "previous_receipt_cid": (
                                            publication_transition[
                                                "receipt_cid"
                                            ]
                                        ),
                                        "plan_root_cid": bootstrap[
                                            "plan_root_cid"
                                        ],
                                        "repository_tree_id": bootstrap[
                                            "repository_tree_id"
                                        ],
                                        "base_head": (
                                            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD
                                        ),
                                        "base_tree": _git(
                                            "rev-parse",
                                            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD
                                            + "^{tree}",
                                        ),
                                        "repair_head": head,
                                        "repair_tree": _git(
                                            "rev-parse", f"{head}^{{tree}}"
                                        ),
                                        "changed_paths": list(
                                            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_CHANGED_PATHS
                                        ),
                                        "patch_digest": _git_patch_digest(
                                            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_BASE_HEAD,
                                            head,
                                        ),
                                        "dependencies": [
                                            "ASEH-BOOTSTRAP-002@ASEH-PLAN-R7"
                                        ],
                                        "owning_repository": (
                                            "ipfs_accelerate_py"
                                        ),
                                        "risk_class": (
                                            "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
                                        ),
                                        "authority_requirement": (
                                            REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_AUTHORITY
                                        ),
                                        "validation_results": (
                                            validation_results
                                        ),
                                        "terminal_success_criteria": (
                                            "An exact optional predecessor-"
                                            "lineage field is closed, bound to "
                                            "the failed older same-task attempt "
                                            "and replacement queue receipt, and "
                                            "replays as an idempotent no-op "
                                            "without lane restart."
                                        ),
                                        "terminal_non_success_criteria": (
                                            "Any missing, unknown, malformed, "
                                            "foreign, live, current, unexpired, "
                                            "reused-queue, provider/effect, "
                                            "identity, sibling, or validation "
                                            "drift is rejected without relaxing "
                                            "lane health."
                                        ),
                                        "semantic_corpus_changed": False,
                                        "database_mutated": False,
                                        "authorized_at": time.time(),
                                    }
                                    receipt["receipt_cid"] = _identity(receipt)
                                    _validate_repair_quack_recovery_replay_transition(
                                        receipt,
                                        bootstrap=bootstrap,
                                        previous_receipt=publication,
                                        rerun_validations=False,
                                    )
                                    if not isinstance(replay_path, Path):
                                        raise OperatorError(
                                            "bootstrap repair Quack-recovery-"
                                            "replay path is absent"
                                        )
                                    _atomic_json_create(replay_path, receipt)
                                    return {
                                        "schema": OPERATOR_SCHEMA,
                                        "command": (
                                            "authorize-repair-transition"
                                        ),
                                        "ok": True,
                                        "idempotent_replay": False,
                                        "repair_transition_receipt": receipt,
                                        "repair_transition_chain": [
                                            prior,
                                            followup,
                                            clean_launch,
                                            runtime_hardening,
                                            quack_recovery,
                                            parallel_startup,
                                            publication,
                                            receipt,
                                        ],
                                    }
                                current_admission = _admit_materialized_launch(
                                    board, _config, paths
                                )
                                admitted_repair = current_admission.get(
                                    "repair_transition"
                                )
                                admitted_continuity = current_admission.get(
                                    "canonical_continuity"
                                )
                                if (
                                    not isinstance(admitted_repair, Mapping)
                                    or admitted_repair.get("repair_head")
                                    != publication_transition["repair_head"]
                                    or not isinstance(
                                        admitted_continuity, Mapping
                                    )
                                    or "repair_to_current"
                                    not in admitted_continuity
                                ):
                                    raise OperatorError(
                                        "current admission does not retain the "
                                        "Quack-publication-contention repair "
                                        "transition"
                                    )
                                return {
                                    "schema": OPERATOR_SCHEMA,
                                    "command": "authorize-repair-transition",
                                    "ok": True,
                                    "idempotent_replay": True,
                                    "repair_transition_receipt": publication,
                                    "repair_transition_chain": [
                                        prior,
                                        followup,
                                        clean_launch,
                                        runtime_hardening,
                                        quack_recovery,
                                        parallel_startup,
                                        publication,
                                    ],
                                    "current_admission_cid": current_admission[
                                        "admission_cid"
                                    ],
                                    "runtime_source_head": current_admission[
                                        "runtime_source_head"
                                    ],
                                }
                            if parallel_startup.get("repair_head") != head:
                                parents = _git(
                                    "show", "-s", "--format=%P", head
                                ).split()
                                if parents != [
                                    REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD
                                ]:
                                    raise OperatorError(
                                        "bootstrap repair Quack-publication-"
                                        "contention must be one child of the "
                                        "exact revision-6 repair"
                                    )
                                if _git_changed_paths(
                                    REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD,
                                    head,
                                ) != (
                                    REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_CHANGED_PATHS
                                ):
                                    raise OperatorError(
                                        "bootstrap repair Quack-publication-"
                                        "contention changed-path set differs"
                                    )
                                validation_results = (
                                    _run_repair_quack_publication_contention_transition_validations()
                                )
                                receipt = {
                                    "schema": (
                                        REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_SCHEMA
                                    ),
                                    "task_id": REPAIR_TRANSITION_TASK_ID,
                                    "stable_identity": (
                                        f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}"
                                        "@ASEH-PLAN-R7"
                                    ),
                                    "program_id": PROGRAM,
                                    "transition_revision": 7,
                                    "bootstrap_receipt_id": bootstrap_id,
                                    "previous_receipt_cid": (
                                        parallel_startup_transition[
                                            "receipt_cid"
                                        ]
                                    ),
                                    "plan_root_cid": bootstrap[
                                        "plan_root_cid"
                                    ],
                                    "repository_tree_id": bootstrap[
                                        "repository_tree_id"
                                    ],
                                    "base_head": (
                                        REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD
                                    ),
                                    "base_tree": _git(
                                        "rev-parse",
                                        REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD
                                        + "^{tree}",
                                    ),
                                    "repair_head": head,
                                    "repair_tree": _git(
                                        "rev-parse", f"{head}^{{tree}}"
                                    ),
                                    "changed_paths": list(
                                        REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_CHANGED_PATHS
                                    ),
                                    "patch_digest": _git_patch_digest(
                                        REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_BASE_HEAD,
                                        head,
                                    ),
                                    "dependencies": [
                                        "ASEH-BOOTSTRAP-002@ASEH-PLAN-R6"
                                    ],
                                    "owning_repository": (
                                        "ipfs_accelerate_py"
                                    ),
                                    "risk_class": (
                                        "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
                                    ),
                                    "authority_requirement": (
                                        REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_AUTHORITY
                                    ),
                                    "validation_results": validation_results,
                                    "terminal_success_criteria": (
                                        "Exact owner replica-publication races "
                                        "are retried without health loss, exact "
                                        "pre-dispatch Quack refusal is deferred, "
                                        "and only expired same-task prior queue "
                                        "lineage is superseded through canonical "
                                        "owner CAS without provider dispatch or "
                                        "a second writer."
                                    ),
                                    "terminal_non_success_criteria": (
                                        "Any foreign authentication, schema, "
                                        "policy, or endpoint failure; live or "
                                        "current queue lineage; unknown provider "
                                        "or external-effect outcome; identity or "
                                        "sibling drift; second writer; database "
                                        "mutation during validation; or validation "
                                        "failure is rejected."
                                    ),
                                    "semantic_corpus_changed": False,
                                    "database_mutated": False,
                                    "authorized_at": time.time(),
                                }
                                receipt["receipt_cid"] = _identity(receipt)
                                _validate_repair_quack_publication_contention_transition(
                                    receipt,
                                    bootstrap=bootstrap,
                                    previous_receipt=parallel_startup,
                                    rerun_validations=False,
                                )
                                if not isinstance(publication_path, Path):
                                    raise OperatorError(
                                        "bootstrap repair Quack-publication-"
                                        "contention path is absent"
                                    )
                                _atomic_json_create(publication_path, receipt)
                                return {
                                    "schema": OPERATOR_SCHEMA,
                                    "command": "authorize-repair-transition",
                                    "ok": True,
                                    "idempotent_replay": False,
                                    "repair_transition_receipt": receipt,
                                    "repair_transition_chain": [
                                        prior,
                                        followup,
                                        clean_launch,
                                        runtime_hardening,
                                        quack_recovery,
                                        parallel_startup,
                                        receipt,
                                    ],
                                }
                            current_admission = _admit_materialized_launch(
                                board, _config, paths
                            )
                            admitted_repair = current_admission.get(
                                "repair_transition"
                            )
                            admitted_continuity = current_admission.get(
                                "canonical_continuity"
                            )
                            if (
                                not isinstance(admitted_repair, Mapping)
                                or admitted_repair.get("repair_head")
                                != parallel_startup_transition["repair_head"]
                                or not isinstance(admitted_continuity, Mapping)
                                or "repair_to_current" not in admitted_continuity
                            ):
                                raise OperatorError(
                                    "current admission does not retain the "
                                    "parallel-blocked-startup repair transition"
                                )
                            return {
                                "schema": OPERATOR_SCHEMA,
                                "command": "authorize-repair-transition",
                                "ok": True,
                                "idempotent_replay": True,
                                "repair_transition_receipt": parallel_startup,
                                "repair_transition_chain": [
                                    prior,
                                    followup,
                                    clean_launch,
                                    runtime_hardening,
                                    quack_recovery,
                                    parallel_startup,
                                ],
                                "current_admission_cid": current_admission[
                                    "admission_cid"
                                ],
                                "runtime_source_head": current_admission[
                                    "runtime_source_head"
                                ],
                            }
                        if quack_recovery.get("repair_head") != head:
                            parents = _git(
                                "show", "-s", "--format=%P", head
                            ).split()
                            if parents != [
                                REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD
                            ]:
                                raise OperatorError(
                                    "bootstrap repair parallel-blocked-startup "
                                    "must be one child of the exact revision-5 "
                                    "repair"
                                )
                            if _git_changed_paths(
                                REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD,
                                head,
                            ) != (
                                REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_CHANGED_PATHS
                            ):
                                raise OperatorError(
                                    "bootstrap repair parallel-blocked-startup "
                                    "changed-path set differs"
                                )
                            validation_results = (
                                _run_repair_parallel_blocked_startup_transition_validations()
                            )
                            receipt = {
                                "schema": (
                                    REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_SCHEMA
                                ),
                                "task_id": REPAIR_TRANSITION_TASK_ID,
                                "stable_identity": (
                                    f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}"
                                    "@ASEH-PLAN-R6"
                                ),
                                "program_id": PROGRAM,
                                "transition_revision": 6,
                                "bootstrap_receipt_id": bootstrap_id,
                                "previous_receipt_cid": (
                                    quack_recovery_transition["receipt_cid"]
                                ),
                                "plan_root_cid": bootstrap["plan_root_cid"],
                                "repository_tree_id": bootstrap[
                                    "repository_tree_id"
                                ],
                                "base_head": (
                                    REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD
                                ),
                                "base_tree": _git(
                                    "rev-parse",
                                    REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD
                                    + "^{tree}",
                                ),
                                "repair_head": head,
                                "repair_tree": _git(
                                    "rev-parse", f"{head}^{{tree}}"
                                ),
                                "changed_paths": list(
                                    REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_CHANGED_PATHS
                                ),
                                "patch_digest": _git_patch_digest(
                                    REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_BASE_HEAD,
                                    head,
                                ),
                                "dependencies": [
                                    "ASEH-BOOTSTRAP-002@ASEH-PLAN-R5"
                                ],
                                "owning_repository": "ipfs_accelerate_py",
                                "risk_class": (
                                    "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
                                ),
                                "authority_requirement": (
                                    "the operator explicitly directed the "
                                    "bootstrap engineering agent to fix the "
                                    "existing supervisor so exact recoverable "
                                    "blocked tasks can rearm while a parallel "
                                    "ready frontier exists, without weakening "
                                    "single-writer, freshness, validation, or "
                                    "bounded startup gates"
                                ),
                                "validation_results": validation_results,
                                "terminal_success_criteria": (
                                    "A blocked board with an admitted parallel "
                                    "ready, active, or delayed frontier keeps "
                                    "the canonical lanes alive only during the "
                                    "configured startup grace; it remains "
                                    "unhealthy until blocked_count reaches zero; "
                                    "and the exact R5 Quack recoveries can rearm "
                                    "through canonical CAS."
                                ),
                                "terminal_non_success_criteria": (
                                    "Any healthy claim while blocked, recovery "
                                    "past startup grace, missing ready frontier, "
                                    "owner/broker/identity/corpus loss, stale "
                                    "lane outside grace, second writer, sibling "
                                    "change, database mutation during validation, "
                                    "or validation failure is rejected."
                                ),
                                "semantic_corpus_changed": False,
                                "database_mutated": False,
                                "authorized_at": time.time(),
                            }
                            receipt["receipt_cid"] = _identity(receipt)
                            _validate_repair_parallel_blocked_startup_transition(
                                receipt,
                                bootstrap=bootstrap,
                                previous_receipt=quack_recovery,
                                rerun_validations=False,
                            )
                            if not isinstance(parallel_startup_path, Path):
                                raise OperatorError(
                                    "bootstrap repair parallel-blocked-startup "
                                    "path is absent"
                                )
                            _atomic_json_create(parallel_startup_path, receipt)
                            return {
                                "schema": OPERATOR_SCHEMA,
                                "command": "authorize-repair-transition",
                                "ok": True,
                                "idempotent_replay": False,
                                "repair_transition_receipt": receipt,
                                "repair_transition_chain": [
                                    prior,
                                    followup,
                                    clean_launch,
                                    runtime_hardening,
                                    quack_recovery,
                                    receipt,
                                ],
                            }
                        current_admission = _admit_materialized_launch(
                            board, _config, paths
                        )
                        admitted_repair = current_admission.get(
                            "repair_transition"
                        )
                        admitted_continuity = current_admission.get(
                            "canonical_continuity"
                        )
                        if (
                            not isinstance(admitted_repair, Mapping)
                            or admitted_repair.get("repair_head")
                            != quack_recovery_transition["repair_head"]
                            or not isinstance(admitted_continuity, Mapping)
                            or "repair_to_current" not in admitted_continuity
                        ):
                            raise OperatorError(
                                "current admission does not retain the "
                                "Quack-recovery repair transition"
                            )
                        return {
                            "schema": OPERATOR_SCHEMA,
                            "command": "authorize-repair-transition",
                            "ok": True,
                            "idempotent_replay": True,
                            "repair_transition_receipt": quack_recovery,
                            "repair_transition_chain": [
                                prior,
                                followup,
                                clean_launch,
                                runtime_hardening,
                                quack_recovery,
                            ],
                            "current_admission_cid": current_admission[
                                "admission_cid"
                            ],
                            "runtime_source_head": current_admission[
                                "runtime_source_head"
                            ],
                        }
                    if runtime_hardening.get("repair_head") != head:
                        parents = _git(
                            "show", "-s", "--format=%P", head
                        ).split()
                        if parents != [
                            REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD
                        ]:
                            raise OperatorError(
                                "bootstrap repair Quack-recovery must be one "
                                "child of the exact revision-4 repair"
                            )
                        if _git_changed_paths(
                            REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD,
                            head,
                        ) != REPAIR_QUACK_RECOVERY_TRANSITION_CHANGED_PATHS:
                            raise OperatorError(
                                "bootstrap repair Quack-recovery changed-path "
                                "set differs"
                            )
                        validation_results = (
                            _run_repair_quack_recovery_transition_validations()
                        )
                        receipt = {
                            "schema": REPAIR_QUACK_RECOVERY_TRANSITION_SCHEMA,
                            "task_id": REPAIR_TRANSITION_TASK_ID,
                            "stable_identity": (
                                f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}"
                                "@ASEH-PLAN-R5"
                            ),
                            "program_id": PROGRAM,
                            "transition_revision": 5,
                            "bootstrap_receipt_id": bootstrap_id,
                            "previous_receipt_cid": runtime_transition[
                                "receipt_cid"
                            ],
                            "plan_root_cid": bootstrap["plan_root_cid"],
                            "repository_tree_id": bootstrap[
                                "repository_tree_id"
                            ],
                            "base_head": (
                                REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD
                            ),
                            "base_tree": _git(
                                "rev-parse",
                                REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD
                                + "^{tree}",
                            ),
                            "repair_head": head,
                            "repair_tree": _git(
                                "rev-parse", f"{head}^{{tree}}"
                            ),
                            "changed_paths": list(
                                REPAIR_QUACK_RECOVERY_TRANSITION_CHANGED_PATHS
                            ),
                            "patch_digest": _git_patch_digest(
                                REPAIR_QUACK_RECOVERY_TRANSITION_BASE_HEAD,
                                head,
                            ),
                            "dependencies": [
                                "ASEH-BOOTSTRAP-002@ASEH-PLAN-R4"
                            ],
                            "owning_repository": "ipfs_accelerate_py",
                            "risk_class": (
                                "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
                            ),
                            "authority_requirement": (
                                "the operator explicitly directed the "
                                "bootstrap engineering agent to fix the "
                                "existing supervisor so it automatically "
                                "classifies and recovers exact preprojection "
                                "Quack transport failures, rebuilds only the "
                                "observed lane-private stale coordination "
                                "index under its sole writer lock, and "
                                "resumes ASEH without state-writer contention"
                            ),
                            "validation_results": validation_results,
                            "terminal_success_criteria": (
                                "Exact current DuckDB Quack refusal text is "
                                "typed before Portal dispatch; the two sealed "
                                "historical preprojection failures rearm only "
                                "after independent zero-provider/effect proof; "
                                "committed, foreign, stale-fence, projected, "
                                "or redirected evidence fails closed; and the "
                                "lane-private ready index rebuilds under its "
                                "exclusive writer lock without changing task "
                                "or lease authority."
                            ),
                            "terminal_non_success_criteria": (
                                "Any second writable authority, direct control "
                                "table edit, broad text-only live retry, blind "
                                "provider replay, committed outcome replay, "
                                "foreign endpoint, stale fence, projection or "
                                "path escape, non-private index mutation, "
                                "sibling change, database mutation during "
                                "validation, or validation failure is rejected."
                            ),
                            "semantic_corpus_changed": False,
                            "database_mutated": False,
                            "authorized_at": time.time(),
                        }
                        receipt["receipt_cid"] = _identity(receipt)
                        _validate_repair_quack_recovery_transition(
                            receipt,
                            bootstrap=bootstrap,
                            previous_receipt=runtime_hardening,
                            rerun_validations=False,
                        )
                        if not isinstance(quack_recovery_path, Path):
                            raise OperatorError(
                                "bootstrap repair Quack-recovery path is absent"
                            )
                        _atomic_json_create(quack_recovery_path, receipt)
                        return {
                            "schema": OPERATOR_SCHEMA,
                            "command": "authorize-repair-transition",
                            "ok": True,
                            "idempotent_replay": False,
                            "repair_transition_receipt": receipt,
                            "repair_transition_chain": [
                                prior,
                                followup,
                                clean_launch,
                                runtime_hardening,
                                receipt,
                            ],
                        }
                    current_admission = _admit_materialized_launch(
                        board, _config, paths
                    )
                    admitted_repair = current_admission.get(
                        "repair_transition"
                    )
                    admitted_continuity = current_admission.get(
                        "canonical_continuity"
                    )
                    if (
                        not isinstance(admitted_repair, Mapping)
                        or admitted_repair.get("repair_head")
                        != runtime_transition["repair_head"]
                        or not isinstance(admitted_continuity, Mapping)
                        or "repair_to_current" not in admitted_continuity
                    ):
                        raise OperatorError(
                            "current admission does not retain the "
                            "runtime-hardening repair transition"
                        )
                    return {
                        "schema": OPERATOR_SCHEMA,
                        "command": "authorize-repair-transition",
                        "ok": True,
                        "idempotent_replay": True,
                        "repair_transition_receipt": runtime_hardening,
                        "repair_transition_chain": [
                            prior,
                            followup,
                            clean_launch,
                            runtime_hardening,
                        ],
                        "current_admission_cid": current_admission[
                            "admission_cid"
                        ],
                        "runtime_source_head": current_admission[
                            "runtime_source_head"
                        ],
                    }
                if clean_launch.get("repair_head") != head:
                    parents = _git(
                        "show", "-s", "--format=%P", head
                    ).split()
                    if parents != [
                        REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD
                    ]:
                        raise OperatorError(
                            "bootstrap repair runtime-hardening must be one "
                            "child of the exact revision-3 repair"
                        )
                    if _git_changed_paths(
                        REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD,
                        head,
                    ) != REPAIR_RUNTIME_HARDENING_TRANSITION_CHANGED_PATHS:
                        raise OperatorError(
                            "bootstrap repair runtime-hardening changed-path "
                            "set differs"
                        )
                    validation_results = (
                        _run_repair_runtime_hardening_transition_validations()
                    )
                    receipt = {
                        "schema": (
                            REPAIR_RUNTIME_HARDENING_TRANSITION_SCHEMA
                        ),
                        "task_id": REPAIR_TRANSITION_TASK_ID,
                        "stable_identity": (
                            f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}"
                            "@ASEH-PLAN-R4"
                        ),
                        "program_id": PROGRAM,
                        "transition_revision": 4,
                        "bootstrap_receipt_id": bootstrap_id,
                        "previous_receipt_cid": clean_launch_transition[
                            "receipt_cid"
                        ],
                        "plan_root_cid": bootstrap["plan_root_cid"],
                        "repository_tree_id": bootstrap[
                            "repository_tree_id"
                        ],
                        "base_head": (
                            REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD
                        ),
                        "base_tree": _git(
                            "rev-parse",
                            REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD
                            + "^{tree}",
                        ),
                        "repair_head": head,
                        "repair_tree": _git(
                            "rev-parse", f"{head}^{{tree}}"
                        ),
                        "changed_paths": list(
                            REPAIR_RUNTIME_HARDENING_TRANSITION_CHANGED_PATHS
                        ),
                        "patch_digest": _git_patch_digest(
                            REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD,
                            head,
                        ),
                        "dependencies": [
                            "ASEH-BOOTSTRAP-002@ASEH-PLAN-R3"
                        ],
                        "owning_repository": "ipfs_accelerate_py",
                        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
                        "authority_requirement": (
                            "the operator explicitly directed the bootstrap "
                            "engineering agent to fix the existing supervisor "
                            "so it automatically recovers ASEH "
                            "false-completion, startup, shutdown, and "
                            "dead-attempt lifecycle faults without state-writer "
                            "or checkout contention"
                        ),
                        "validation_results": validation_results,
                        "terminal_success_criteria": (
                            "The existing supervisor preserves the admitted "
                            "recovery window, stops all lanes concurrently, "
                            "reconciles an integrated false completion through "
                            "a fresh zero-provider/effect claim, fences the "
                            "exact target and board checkout through completion, "
                            "fences exact dead Portal lifecycle claims through "
                            "the canonical store, and rejects live, foreign, "
                            "stale-fence, and foreign same-status claims."
                        ),
                        "terminal_non_success_criteria": (
                            "Any second writable authority, blind provider "
                            "retry, provider/effect replay, foreign claim "
                            "receipt, target drift, stale-fence completion, "
                            "serialized lane orphan, checkout contention loss, "
                            "database mutation during validation, sibling "
                            "change, or validation failure is rejected."
                        ),
                        "semantic_corpus_changed": False,
                        "database_mutated": False,
                        "authorized_at": time.time(),
                    }
                    receipt["receipt_cid"] = _identity(receipt)
                    _validate_repair_runtime_hardening_transition(
                        receipt,
                        bootstrap=bootstrap,
                        previous_receipt=clean_launch,
                        rerun_validations=False,
                    )
                    if not isinstance(runtime_path, Path):
                        raise OperatorError(
                            "bootstrap repair runtime-hardening path is absent"
                        )
                    _atomic_json_create(runtime_path, receipt)
                    return {
                        "schema": OPERATOR_SCHEMA,
                        "command": "authorize-repair-transition",
                        "ok": True,
                        "idempotent_replay": False,
                        "repair_transition_receipt": receipt,
                        "repair_transition_chain": [
                            prior, followup, clean_launch, receipt
                        ],
                    }
                current_admission = _admit_materialized_launch(
                    board, _config, paths
                )
                admitted_repair = current_admission.get(
                    "repair_transition"
                )
                admitted_continuity = current_admission.get(
                    "canonical_continuity"
                )
                if (
                    not isinstance(admitted_repair, Mapping)
                    or admitted_repair.get("repair_head")
                    != clean_launch_transition["repair_head"]
                    or not isinstance(admitted_continuity, Mapping)
                    or "repair_to_current" not in admitted_continuity
                ):
                    raise OperatorError(
                        "current admission does not retain the clean-launch "
                        "repair transition"
                    )
                return {
                    "schema": OPERATOR_SCHEMA,
                    "command": "authorize-repair-transition",
                    "ok": True,
                    "idempotent_replay": True,
                    "repair_transition_receipt": clean_launch,
                    "repair_transition_chain": [
                        prior, followup, clean_launch
                    ],
                    "current_admission_cid": current_admission[
                        "admission_cid"
                    ],
                    "runtime_source_head": current_admission[
                        "runtime_source_head"
                    ],
                }
            if advanced:
                parents = _git("show", "-s", "--format=%P", head).split()
                if parents != [REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD]:
                    raise OperatorError(
                        "bootstrap repair clean-launch must be one child of "
                        "the exact revision-2 repair"
                    )
                if _git_changed_paths(
                    REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD, head
                ) != REPAIR_CLEAN_LAUNCH_TRANSITION_CHANGED_PATHS:
                    raise OperatorError(
                        "bootstrap repair clean-launch changed-path set differs"
                    )
                validation_results = (
                    _run_repair_clean_launch_transition_validations()
                )
                receipt = {
                    "schema": REPAIR_CLEAN_LAUNCH_TRANSITION_SCHEMA,
                    "task_id": REPAIR_TRANSITION_TASK_ID,
                    "stable_identity": (
                        f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}"
                        "@ASEH-PLAN-R3"
                    ),
                    "program_id": PROGRAM,
                    "transition_revision": 3,
                    "bootstrap_receipt_id": bootstrap_id,
                    "previous_receipt_cid": followup_transition[
                        "receipt_cid"
                    ],
                    "plan_root_cid": bootstrap["plan_root_cid"],
                    "repository_tree_id": bootstrap[
                        "repository_tree_id"
                    ],
                    "base_head": REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD,
                    "base_tree": _git(
                        "rev-parse",
                        f"{REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD}^{{tree}}",
                    ),
                    "repair_head": head,
                    "repair_tree": _git("rev-parse", f"{head}^{{tree}}"),
                    "changed_paths": list(
                        REPAIR_CLEAN_LAUNCH_TRANSITION_CHANGED_PATHS
                    ),
                    "patch_digest": _git_patch_digest(
                        REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD, head
                    ),
                    "dependencies": [
                        "ASEH-BOOTSTRAP-002@ASEH-PLAN-R2"
                    ],
                    "owning_repository": "ipfs_accelerate_py",
                    "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
                    "authority_requirement": (
                        "the operator explicitly directed the bootstrap "
                        "engineering agent to fix the existing supervisor so "
                        "admission validation cannot materialize credentials "
                        "in or dirty the launch checkout"
                    ),
                    "validation_results": validation_results,
                    "terminal_success_criteria": (
                        "Opaque state-store identities cannot resolve a token "
                        "vault from cwd, admission tests leave the exact "
                        "checkout clean, and launch remains fail closed."
                    ),
                    "terminal_non_success_criteria": (
                        "Any path escape, checkout credential artifact, dirty "
                        "post-validation tree, identity drift, sibling change, "
                        "database mutation, or validation failure is rejected."
                    ),
                    "semantic_corpus_changed": False,
                    "database_mutated": False,
                    "authorized_at": time.time(),
                }
                receipt["receipt_cid"] = _identity(receipt)
                if not isinstance(clean_launch_path, Path):
                    raise OperatorError(
                        "bootstrap repair clean-launch path is absent"
                    )
                _atomic_json_create(clean_launch_path, receipt)
                return {
                    "schema": OPERATOR_SCHEMA,
                    "command": "authorize-repair-transition",
                    "ok": True,
                    "idempotent_replay": False,
                    "repair_transition_receipt": receipt,
                    "repair_transition_chain": [prior, followup, receipt],
                }
            current_admission = _admit_materialized_launch(
                board, _config, paths
            )
            admitted_repair = current_admission.get("repair_transition")
            admitted_continuity = current_admission.get(
                "canonical_continuity"
            )
            if (
                not isinstance(admitted_repair, Mapping)
                or admitted_repair.get("repair_head")
                != followup_transition["repair_head"]
                or not isinstance(admitted_continuity, Mapping)
                or "repair_to_current" not in admitted_continuity
            ):
                raise OperatorError(
                    "current admission does not retain the repair transition"
                )
            return {
                "schema": OPERATOR_SCHEMA,
                "command": "authorize-repair-transition",
                "ok": True,
                "idempotent_replay": True,
                "repair_transition_receipt": followup,
                "repair_transition_chain": [prior, followup],
                "current_admission_cid": current_admission["admission_cid"],
                "runtime_source_head": current_admission[
                    "runtime_source_head"
                ],
            }

        if head == REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD:
            raise OperatorError(
                "bootstrap repair follow-up has not been committed"
            )
        if repair_head != REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT:
            raise OperatorError("prior bootstrap repair identity differs")
        snapshot, _ready, integrity, outputs, requests = (
            _read_continuity_state(board, paths, bootstrap)
        )
        base_proof = _admit_canonical_merge_suffix(
            board,
            base_head=str(bootstrap["source_head"]),
            target_head=REPAIR_TRANSITION_BASE_HEAD,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs=outputs,
            completed_requests=requests,
        )
        if not base_proof["integrations"]:
            raise OperatorError(
                "bootstrap repair base lacks a canonical merge suffix"
            )
        followup_base = _admit_repair_followup_base(
            board,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs=outputs,
            completed_requests=requests,
            require_nonterminal=True,
        )
        parents = _git("show", "-s", "--format=%P", head).split()
        if parents != [REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD]:
            raise OperatorError(
                "bootstrap repair follow-up must be one child of the exact base"
            )
        if _git_changed_paths(
            REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD, head
        ) != REPAIR_FOLLOWUP_TRANSITION_CHANGED_PATHS:
            raise OperatorError("bootstrap repair follow-up changed-path set differs")
        validation_results = _run_repair_followup_transition_validations()
        receipt = {
            "schema": REPAIR_FOLLOWUP_TRANSITION_SCHEMA,
            "task_id": REPAIR_TRANSITION_TASK_ID,
            "stable_identity": (
                f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R2"
            ),
            "program_id": PROGRAM,
            "transition_revision": 2,
            "bootstrap_receipt_id": bootstrap_id,
            "previous_receipt_cid": prior_transition["receipt_cid"],
            "base_integration_witness": followup_base[
                "base_integration_witness"
            ],
            "authorization_task_observation": followup_base[
                "authorization_task_observation"
            ],
            "plan_root_cid": bootstrap["plan_root_cid"],
            "repository_tree_id": bootstrap["repository_tree_id"],
            "base_head": REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD,
            "base_tree": _git(
                "rev-parse", f"{REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD}^{{tree}}"
            ),
            "repair_head": head,
            "repair_tree": _git("rev-parse", f"{head}^{{tree}}"),
            "changed_paths": list(REPAIR_FOLLOWUP_TRANSITION_CHANGED_PATHS),
            "patch_digest": _git_patch_digest(
                REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD, head
            ),
            "dependencies": [
                "ASEH-BOOTSTRAP-002@ASEH-PLAN-R1",
                "ASEH-001",
            ],
            "owning_repository": "ipfs_accelerate_py",
            "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
            "authority_requirement": (
                "the operator explicitly directed the bootstrap engineering "
                "agent to fix the existing canonical supervisor so it "
                "automatically recovers ASEH runtime faults without "
                "state-writer contention"
            ),
            "validation_results": validation_results,
            "terminal_success_criteria": (
                "The existing supervisor recovers exact missing Git worktree "
                "registrations, settles recovery before new claims, and "
                "defers exact Quack transport unavailability without a "
                "second writer or fabricated completion."
            ),
            "terminal_non_success_criteria": (
                "Any identity drift, broad error reminting, unverified cleanup, "
                "claim during recovery settlement, duplicate writer, task "
                "completion claim, validation failure, or sibling change is "
                "rejected."
            ),
            "semantic_corpus_changed": False,
            "database_mutated": False,
            "authorized_at": time.time(),
        }
        receipt["receipt_cid"] = _identity(receipt)
        if not isinstance(followup_path, Path):
            raise OperatorError("bootstrap repair follow-up path is absent")
        _atomic_json(followup_path, receipt)
        return {
            "schema": OPERATOR_SCHEMA,
            "command": "authorize-repair-transition",
            "ok": True,
            "idempotent_replay": False,
            "authoritative_event_cursor": snapshot["event_cursor"],
            "canonical_base_suffix": base_proof,
            "initial_repair_to_followup_base": followup_base,
            "repair_transition_receipt": receipt,
            "repair_transition_chain": [prior, receipt],
        }
    snapshot, _ready, integrity, outputs, requests = _read_continuity_state(
        board, paths, bootstrap
    )
    base_proof = _admit_canonical_merge_suffix(
        board,
        base_head=str(bootstrap["source_head"]),
        target_head=REPAIR_TRANSITION_BASE_HEAD,
        bootstrap=bootstrap,
        integrity=integrity,
        task_outputs=outputs,
        completed_requests=requests,
    )
    if not base_proof["integrations"]:
        raise OperatorError("bootstrap repair base lacks a canonical merge suffix")
    parents = _git("show", "-s", "--format=%P", head).split()
    if parents != [REPAIR_TRANSITION_BASE_HEAD]:
        raise OperatorError("bootstrap repair must be one child of the exact base")
    if _git_changed_paths(REPAIR_TRANSITION_BASE_HEAD, head) != (
        REPAIR_TRANSITION_CHANGED_PATHS
    ):
        raise OperatorError("bootstrap repair changed-path set differs")
    validation_results = _run_repair_transition_validations()
    receipt = {
        "schema": REPAIR_TRANSITION_SCHEMA,
        "task_id": REPAIR_TRANSITION_TASK_ID,
        "stable_identity": f"{PROGRAM}/{REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R1",
        "program_id": PROGRAM,
        "bootstrap_receipt_id": bootstrap_id,
        "plan_root_cid": bootstrap["plan_root_cid"],
        "repository_tree_id": bootstrap["repository_tree_id"],
        "base_head": REPAIR_TRANSITION_BASE_HEAD,
        "base_tree": _git(
            "rev-parse", f"{REPAIR_TRANSITION_BASE_HEAD}^{{tree}}"
        ),
        "repair_head": head,
        "repair_tree": _git("rev-parse", f"{head}^{{tree}}"),
        "changed_paths": list(REPAIR_TRANSITION_CHANGED_PATHS),
        "patch_digest": _git_patch_digest(REPAIR_TRANSITION_BASE_HEAD, head),
        "dependencies": ["ASEH-BOOTSTRAP-001", "ASEH-000"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            "the operator explicitly directed the bootstrap engineering agent "
            "to repair the existing canonical handoff and resume through it"
        ),
        "validation_results": validation_results,
        "terminal_success_criteria": (
            "False merge completion is rejected, exact blocked recovery is "
            "bounded, opaque managed roots are fenced, and immutable ASEH "
            "DuckDB task/plan identities remain unchanged."
        ),
        "terminal_non_success_criteria": (
            "Any other base, child, path, patch, sibling identity, validation "
            "result, task corpus, database mutation, or queue proof is rejected."
        ),
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": time.time(),
    }
    receipt["receipt_cid"] = _identity(receipt)
    _atomic_json(paths["repair_transition_receipt"], receipt)
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "authorize-repair-transition",
        "ok": True,
        "idempotent_replay": False,
        "authoritative_event_cursor": snapshot["event_cursor"],
        "canonical_base_suffix": base_proof,
        "repair_transition_receipt": receipt,
    }


def _admit_materialized_launch(
    board: Any,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Rebind the offline task store to the exact current sealed source."""

    population = _population(board, config)
    bootstrap = _secure_runtime_json(
        paths["bootstrap_receipt"], max_bytes=STATUS_RECEIPT_MAX_BYTES
    )
    receipt_id = _bootstrap_receipt_id(bootstrap)
    expected_fields = {
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "source_forest": population["source_forest"],
        "source_identities": population["source_identities"],
    }
    exact_bootstrap = not any(
        bootstrap.get(name) != value for name, value in expected_fields.items()
    )
    continuity: dict[str, Any] = {}
    repair_transition: dict[str, Any] = {}
    repair_transition_chain: list[dict[str, Any]] = []
    if exact_bootstrap:
        with _offline_database_guard(paths):
            projection_matches = _projection_matches_events_on_disposable_copy(
                paths["database"]
            )
            with _read_only_database_task_source(
                paths["database"],
                owner_id="aseh-launch-admission:read-only",
                repository_tree_id=population["repository_tree_id"],
                plan_root_cid=population["plan_root_cid"],
            ) as source:
                snapshot, ready, integrity = _verify_materialized_source(
                    source,
                    population=population,
                    config=config,
                    require_initial_frontier=False,
                    projection_matches_events=projection_matches,
                )
        _admit_current_projection_against_bootstrap(
            bootstrap, snapshot, integrity
        )
    else:
        repair_receipt_path = paths.get("repair_transition_receipt")
        if (
            not isinstance(repair_receipt_path, Path)
            or not repair_receipt_path.is_file()
        ):
            raise OperatorError(
                "bootstrap receipt differs from the exact current source "
                "forest and no repair transition is admitted"
            )
        repair_receipt = _secure_runtime_json(
            repair_receipt_path,
            max_bytes=STATUS_RECEIPT_MAX_BYTES,
        )
        repair_transition = _validate_repair_transition(
            repair_receipt, bootstrap=bootstrap, rerun_validations=True
        )
        repair_transition_chain = [repair_transition]
        snapshot, ready, integrity, outputs, requests = _read_continuity_state(
            board, paths, bootstrap
        )
        base_proof = _admit_canonical_merge_suffix(
            board,
            base_head=str(bootstrap["source_head"]),
            target_head=REPAIR_TRANSITION_BASE_HEAD,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs=outputs,
            completed_requests=requests,
        )
        followup_path = paths.get("repair_followup_transition_receipt")
        if isinstance(followup_path, Path) and followup_path.is_file():
            followup_receipt = _secure_runtime_json(
                followup_path, max_bytes=STATUS_RECEIPT_MAX_BYTES
            )
            followup_transition = _validate_repair_followup_transition(
                followup_receipt,
                bootstrap=bootstrap,
                previous_receipt=repair_receipt,
                rerun_validations=True,
            )
            if (
                repair_transition["repair_head"]
                != REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT
            ):
                raise OperatorError("prior bootstrap repair identity differs")
            followup_base = _admit_repair_followup_base(
                board,
                bootstrap=bootstrap,
                integrity=integrity,
                task_outputs=outputs,
                completed_requests=requests,
                stored_followup=followup_receipt,
            )
            active_transition = followup_transition
            clean_launch_transition: dict[str, Any] | None = None
            runtime_hardening_transition: dict[str, Any] | None = None
            quack_recovery_transition: dict[str, Any] | None = None
            parallel_blocked_startup_transition: dict[str, Any] | None = None
            quack_publication_contention_transition: (
                dict[str, Any] | None
            ) = None
            quack_recovery_replay_transition: dict[str, Any] | None = None
            control_receipt_lifecycle_transition: (
                dict[str, Any] | None
            ) = None
            clean_launch_receipt: dict[str, Any] | None = None
            clean_launch_path = paths.get(
                "repair_clean_launch_transition_receipt"
            )
            if (
                isinstance(clean_launch_path, Path)
                and clean_launch_path.is_file()
            ):
                clean_launch_receipt = _secure_runtime_json(
                    clean_launch_path, max_bytes=STATUS_RECEIPT_MAX_BYTES
                )
                clean_launch_transition = (
                    _validate_repair_clean_launch_transition(
                        clean_launch_receipt,
                        bootstrap=bootstrap,
                        previous_receipt=followup_receipt,
                        rerun_validations=True,
                    )
                )
                if (
                    clean_launch_transition["base_head"]
                    != followup_transition["repair_head"]
                ):
                    raise OperatorError(
                        "clean-launch repair does not extend revision 2"
                )
                active_transition = clean_launch_transition
                runtime_path = paths.get(
                    "repair_runtime_hardening_transition_receipt"
                )
                if (
                    isinstance(runtime_path, Path)
                    and runtime_path.is_file()
                ):
                    runtime_receipt = _secure_runtime_json(
                        runtime_path,
                        max_bytes=STATUS_RECEIPT_MAX_BYTES,
                    )
                    runtime_hardening_transition = (
                        _validate_repair_runtime_hardening_transition(
                            runtime_receipt,
                            bootstrap=bootstrap,
                            previous_receipt=clean_launch_receipt,
                            rerun_validations=True,
                        )
                    )
                    if (
                        runtime_hardening_transition["base_head"]
                        != clean_launch_transition["repair_head"]
                    ):
                        raise OperatorError(
                            "runtime-hardening repair does not extend "
                            "revision 3"
                        )
                    active_transition = runtime_hardening_transition
                    quack_recovery_path = paths.get(
                        "repair_quack_recovery_transition_receipt"
                    )
                    if (
                        isinstance(quack_recovery_path, Path)
                        and quack_recovery_path.is_file()
                    ):
                        quack_recovery_receipt = _secure_runtime_json(
                            quack_recovery_path,
                            max_bytes=STATUS_RECEIPT_MAX_BYTES,
                        )
                        quack_recovery_transition = (
                            _validate_repair_quack_recovery_transition(
                                quack_recovery_receipt,
                                bootstrap=bootstrap,
                                previous_receipt=runtime_receipt,
                                rerun_validations=True,
                            )
                        )
                        if (
                            quack_recovery_transition["base_head"]
                            != runtime_hardening_transition["repair_head"]
                        ):
                            raise OperatorError(
                                "Quack-recovery repair does not extend "
                                "revision 4"
                            )
                        active_transition = quack_recovery_transition
                        parallel_startup_path = paths.get(
                            "repair_parallel_blocked_startup_transition_receipt"
                        )
                        if (
                            isinstance(parallel_startup_path, Path)
                            and parallel_startup_path.is_file()
                        ):
                            parallel_startup_receipt = _secure_runtime_json(
                                parallel_startup_path,
                                max_bytes=STATUS_RECEIPT_MAX_BYTES,
                            )
                            parallel_blocked_startup_transition = (
                                _validate_repair_parallel_blocked_startup_transition(
                                    parallel_startup_receipt,
                                    bootstrap=bootstrap,
                                    previous_receipt=quack_recovery_receipt,
                                    rerun_validations=True,
                                )
                            )
                            if (
                                parallel_blocked_startup_transition["base_head"]
                                != quack_recovery_transition["repair_head"]
                            ):
                                raise OperatorError(
                                    "parallel-blocked-startup repair does not "
                                    "extend revision 5"
                                )
                            active_transition = (
                                parallel_blocked_startup_transition
                            )
                            publication_path = paths.get(
                                "repair_quack_publication_contention_transition_receipt"
                            )
                            if (
                                isinstance(publication_path, Path)
                                and publication_path.is_file()
                            ):
                                publication_receipt = _secure_runtime_json(
                                    publication_path,
                                    max_bytes=STATUS_RECEIPT_MAX_BYTES,
                                )
                                quack_publication_contention_transition = (
                                    _validate_repair_quack_publication_contention_transition(
                                        publication_receipt,
                                        bootstrap=bootstrap,
                                        previous_receipt=(
                                            parallel_startup_receipt
                                        ),
                                        rerun_validations=True,
                                    )
                                )
                                if (
                                    quack_publication_contention_transition[
                                        "base_head"
                                    ]
                                    != parallel_blocked_startup_transition[
                                        "repair_head"
                                    ]
                                ):
                                    raise OperatorError(
                                        "Quack-publication-contention repair "
                                        "does not extend revision 6"
                                    )
                                active_transition = (
                                    quack_publication_contention_transition
                                )
                                replay_path = paths.get(
                                    "repair_quack_recovery_replay_transition_receipt"
                                )
                                if (
                                    isinstance(replay_path, Path)
                                    and replay_path.is_file()
                                ):
                                    replay_receipt = _secure_runtime_json(
                                        replay_path,
                                        max_bytes=STATUS_RECEIPT_MAX_BYTES,
                                    )
                                    quack_recovery_replay_transition = (
                                        _validate_repair_quack_recovery_replay_transition(
                                            replay_receipt,
                                            bootstrap=bootstrap,
                                            previous_receipt=(
                                                publication_receipt
                                            ),
                                            rerun_validations=True,
                                        )
                                    )
                                    if (
                                        quack_recovery_replay_transition[
                                            "base_head"
                                        ]
                                        != quack_publication_contention_transition[
                                            "repair_head"
                                        ]
                                    ):
                                        raise OperatorError(
                                            "Quack-recovery-replay repair does "
                                            "not extend revision 7"
                                        )
                                    active_transition = (
                                        quack_recovery_replay_transition
                                    )
                                    lifecycle_path = paths.get(
                                        "repair_control_receipt_lifecycle_transition_receipt"
                                    )
                                    if (
                                        isinstance(lifecycle_path, Path)
                                        and lifecycle_path.is_file()
                                    ):
                                        lifecycle_receipt = (
                                            _secure_runtime_json(
                                                lifecycle_path,
                                                max_bytes=(
                                                    STATUS_RECEIPT_MAX_BYTES
                                                ),
                                            )
                                        )
                                        control_receipt_lifecycle_transition = (
                                            _validate_repair_control_receipt_lifecycle_transition(
                                                lifecycle_receipt,
                                                bootstrap=bootstrap,
                                                previous_receipt=(
                                                    replay_receipt
                                                ),
                                                rerun_validations=True,
                                            )
                                        )
                                        if (
                                            control_receipt_lifecycle_transition[
                                                "base_head"
                                            ]
                                            != quack_recovery_replay_transition[
                                                "repair_head"
                                            ]
                                        ):
                                            raise OperatorError(
                                                "control-receipt-lifecycle "
                                                "repair does not extend "
                                                "revision 8"
                                            )
                                        active_transition = (
                                            control_receipt_lifecycle_transition
                                        )
            current_proof = _admit_canonical_merge_suffix(
                board,
                base_head=str(active_transition["repair_head"]),
                target_head=str(population["source_head"]),
                bootstrap=bootstrap,
                integrity=integrity,
                task_outputs=outputs,
                completed_requests=requests,
            )
            repair_transition = active_transition
            repair_transition_chain.append(followup_transition)
            if clean_launch_transition is not None:
                repair_transition_chain.append(clean_launch_transition)
            if runtime_hardening_transition is not None:
                repair_transition_chain.append(
                    runtime_hardening_transition
                )
            if quack_recovery_transition is not None:
                repair_transition_chain.append(quack_recovery_transition)
            if parallel_blocked_startup_transition is not None:
                repair_transition_chain.append(
                    parallel_blocked_startup_transition
                )
            if quack_publication_contention_transition is not None:
                repair_transition_chain.append(
                    quack_publication_contention_transition
                )
            if quack_recovery_replay_transition is not None:
                repair_transition_chain.append(
                    quack_recovery_replay_transition
                )
            if control_receipt_lifecycle_transition is not None:
                repair_transition_chain.append(
                    control_receipt_lifecycle_transition
                )
            continuity = {
                "bootstrap_to_repair_base": base_proof,
                "initial_repair_to_followup_base": followup_base,
                "repair_to_current": current_proof,
            }
            if clean_launch_transition is not None:
                continuity["followup_to_clean_launch"] = (
                    clean_launch_transition
                )
            if runtime_hardening_transition is not None:
                continuity["clean_launch_to_runtime_hardening"] = (
                    runtime_hardening_transition
                )
            if quack_recovery_transition is not None:
                continuity["runtime_hardening_to_quack_recovery"] = (
                    quack_recovery_transition
                )
            if parallel_blocked_startup_transition is not None:
                continuity["quack_recovery_to_parallel_blocked_startup"] = (
                    parallel_blocked_startup_transition
                )
            if quack_publication_contention_transition is not None:
                continuity[
                    "parallel_blocked_startup_to_quack_publication_contention"
                ] = quack_publication_contention_transition
            if quack_recovery_replay_transition is not None:
                continuity[
                    "quack_publication_contention_to_quack_recovery_replay"
                ] = quack_recovery_replay_transition
            if control_receipt_lifecycle_transition is not None:
                continuity[
                    "quack_recovery_replay_to_control_receipt_lifecycle"
                ] = control_receipt_lifecycle_transition
        else:
            current_proof = _admit_canonical_merge_suffix(
                board,
                base_head=str(repair_transition["repair_head"]),
                target_head=str(population["source_head"]),
                bootstrap=bootstrap,
                integrity=integrity,
                task_outputs=outputs,
                completed_requests=requests,
            )
            continuity = {
                "bootstrap_to_repair_base": base_proof,
                "repair_to_current": current_proof,
            }
    admission = {
        "source_head": bootstrap["source_head"],
        "repository_tree_id": bootstrap["repository_tree_id"],
        "source_forest_cid": bootstrap["source_forest"]["forest_cid"],
        "runtime_source_head": population["source_head"],
        "runtime_repository_tree_id": population["repository_tree_id"],
        "runtime_source_forest_cid": population["source_forest"]["forest_cid"],
        "plan_root_cid": bootstrap["plan_root_cid"],
        "bootstrap_receipt_id": receipt_id,
        "repair_transition": repair_transition,
        "repair_transition_chain": repair_transition_chain,
        "canonical_continuity": continuity,
        "projection_cid": snapshot["projection_cid"],
        "event_cursor": snapshot["event_cursor"],
        "ready_task_ids": list(ready),
        "task_statuses": dict(integrity["task_statuses"]),
        "task_revisions": dict(integrity["task_revisions"]),
    }
    admission["admission_cid"] = _identity(admission)
    return admission


def _build_server(board: Any, paths: Mapping[str, Path]) -> Any:
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
        TYPED_STATE_OWNER_SOCKET_FILENAME,
    )

    program = board.resolved_database_program()
    endpoint = str(program.quack_endpoint)
    port = int(endpoint.rsplit(":", 1)[1])
    if Path.cwd().resolve() != ROOT:
        raise OperatorError(
            "the configured owner must start from the sealed repository root"
        )
    owner_relative = paths["owner"].relative_to(ROOT)
    socket_parent = Path("/proc/self/cwd") / owner_relative
    typed_socket = socket_parent / TYPED_STATE_OWNER_SOCKET_FILENAME
    broker_socket = socket_parent / TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME
    if any(
        len(os.fsencode(str(path))) >= 108
        for path in (typed_socket, broker_socket)
    ):
        raise OperatorError("configured owner socket aliases exceed AF_UNIX bounds")
    return build_server(
        database_path=paths["database"],
        state_dir=paths["owner"],
        repository_root=ROOT,
        host="127.0.0.1",
        port=port,
        repository_id=PROGRAM,
        store_id=program.store_id,
        secret_handle=program.endpoint_secret_handle,
        typed_command_socket_path=typed_socket,
    )


def _record_control_failure(
    paths: Mapping[str, Path],
    failure: dict[str, Any],
    failure_event: threading.Event,
    *,
    reason_code: str,
    error_type: str,
) -> None:
    if failure_event.is_set():
        return
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-control-failure@1",
        "reason_code": reason_code,
        "error_type": error_type,
        "observed_at": time.time(),
    }
    payload["receipt_cid"] = _identity(payload)
    failure.update(payload)
    _atomic_json(paths["inbox_failure_receipt"], payload)
    failure_event.set()


def _terminate_scheduler(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10.0)


def run_supervisor(config_path: Path, *, implement: bool, duration: float) -> int:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        preflight_configured_board,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        STATE_LIVE_SCHEMA_REVISION_ENV,
        STATE_SCHEMA_REVISION_ENV,
        STATE_STORE_GENERATION_ENV,
        STATE_STORE_LIVE_GENERATION_ENV,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        harden_state_authority_process,
        state_authority_pass_fds,
    )

    board, _config = _load(config_path)
    paths = _paths(board)
    if not paths["bootstrap_receipt"].is_file() or not paths["database"].is_file():
        raise OperatorError("materialize the sealed board before starting the owner")
    preflight = preflight_configured_board(board)
    if preflight.get("valid") is not True:
        raise OperatorError(
            "configured-board preflight failed: "
            + json.dumps(preflight.get("errors") or [])
        )
    launch_admission = _admit_materialized_launch(board, _config, paths)
    server = _build_server(board, paths)
    stop = threading.Event()
    failure_event = threading.Event()
    failure: dict[str, Any] = {}
    monitor_thread: threading.Thread | None = None
    scheduler: subprocess.Popen[Any] | None = None
    prior_environment: dict[str, str | None] = {}
    shutdown_requested = threading.Event()
    received_signal: dict[str, int] = {}
    with _stop_signal_handlers(shutdown_requested, received_signal):
        try:
            identity = server.start()
            launched_at = time.time()
            configured_program = board.resolved_database_program()
            if (
                str(identity.store_id) != configured_program.store_id
                or int(identity.generation) < 1
                or int(identity.schema_revision) < 0
                or int(identity.process_birth.pid) != os.getpid()
            ):
                raise OperatorError(
                    "live owner identity differs from the configured store"
                )
            program_environment = dict(
                configured_program.environment(repository_root=ROOT)
            )
            live_generation = str(identity.generation)
            live_schema_revision = str(identity.schema_revision)
            program_environment[STATE_STORE_GENERATION_ENV] = live_generation
            program_environment[STATE_SCHEMA_REVISION_ENV] = live_schema_revision
            program_environment[STATE_STORE_LIVE_GENERATION_ENV] = live_generation
            program_environment[STATE_LIVE_SCHEMA_REVISION_ENV] = live_schema_revision
            program_payload = json.loads(
                program_environment["IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON"]
            )
            program_payload["store_generation"] = live_generation
            program_payload["schema_revision"] = live_schema_revision
            program_environment[
                "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON"
            ] = json.dumps(program_payload, separators=(",", ":"), sort_keys=True)
            expected_mutations = str((paths["owner"] / "mutations").resolve())
            if program_environment.get(
                "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"
            ) != expected_mutations:
                raise OperatorError("scheduler and owner mutation inboxes differ")
            program_environment["IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET"] = str(
                server.typed_command_socket_path()
            )
            broker_environment = dict(server.start_supervisor_grant_broker())
            launch_environment = {**program_environment, **broker_environment}
            raw_token_name = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
            prior_environment[raw_token_name] = os.environ.get(raw_token_name)
            os.environ.pop(raw_token_name, None)
            for name, value in launch_environment.items():
                prior_environment[name] = os.environ.get(name)
                os.environ[name] = value
            harden_state_authority_process()
            argv = [
                sys.executable,
                str(ROOT / "scripts/ops/agent_supervisor/configured_board_scheduler.py"),
                "--repo-root", str(ROOT), "--config", str(board.config_path),
                "launch", "--foreground", "--duration-seconds", str(duration),
            ]
            if implement:
                argv.append("--implement")
            scheduler = subprocess.Popen(
                argv, cwd=ROOT, env=dict(os.environ), start_new_session=True,
                pass_fds=state_authority_pass_fds(os.environ),
            )
            initial_health, last_progress_at = _await_initial_health(
                board, paths, server, scheduler, launched_at=launched_at,
                failure=failure, failure_event=failure_event,
                shutdown_requested=shutdown_requested,
                received_signal=received_signal,
            )
            monitor_thread = threading.Thread(
                target=_status_monitor_loop,
                kwargs={
                    "board": board, "paths": paths, "server": server,
                    "scheduler": scheduler, "launched_at": launched_at,
                    "previous": initial_health["samples"][-1],
                    "last_progress_at": last_progress_at, "stop": stop,
                    "failure": failure, "failure_event": failure_event,
                },
                name="aseh-live-health-monitor", daemon=True,
            )
            monitor_thread.start()
            launch_record = {
                "schema": "ipfs_accelerate_py/agent-supervisor/aseh-owner-launch@1",
                "identity": identity.to_dict(),
                "typed_grant_broker": {
                    "available": True,
                    "socket_path": broker_environment[
                        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET"
                    ],
                    "secret_published": False,
                },
                "materialized_launch_admission": launch_admission,
                "initial_health_receipt_cid": initial_health["receipt_cid"],
                "live_owner_program": {
                    "store_id": identity.store_id,
                    "store_generation": live_generation,
                    "schema_revision": live_schema_revision,
                    "process_birth_id": identity.process_birth_id,
                },
                "implement": implement,
            }
            launch_record["receipt_cid"] = _identity(launch_record)
            _atomic_json(
                paths["evidence"] / "control-plane" / "owner-launch.json",
                launch_record,
            )
            while scheduler.poll() is None:
                if shutdown_requested.is_set():
                    raise OperatorStopRequested(
                        int(received_signal.get("signum") or signal.SIGTERM)
                    )
                if failure_event.wait(0.25):
                    _terminate_scheduler(scheduler)
                    raise OperatorError(
                        f"foreground control plane failed: {failure.get('reason_code')}"
                    )
            return int(scheduler.returncode or 0)
        except OperatorStopRequested as exc:
            return 128 + int(exc.signum)
        finally:
            stop.set()
            if scheduler is not None:
                _terminate_scheduler(scheduler)
            if monitor_thread is not None:
                monitor_thread.join(timeout=2.0)
            try:
                try:
                    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
                        reset_quack_transport_cache,
                    )

                    reset_quack_transport_cache()
                finally:
                    server.stop()
            finally:
                for name, prior in prior_environment.items():
                    if prior is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = prior


def _published_replica_binding(
    owner_status: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Admit one exact owner-published, non-authoritative replica identity."""

    identity = owner_status.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    replica = owner_status.get("read_replica")
    replica = replica if isinstance(replica, Mapping) else {}
    expected_path = paths["database"].with_name(
        f"{paths['database'].stem}.read-replica{paths['database'].suffix}"
    )
    digest = str(replica.get("sha256") or "")
    size_bytes = replica.get("size_bytes")
    refresh_sequence = replica.get("refresh_sequence")
    exact_identity = all(
        replica.get(field) == identity.get(field)
        for field in (
            "server_id", "database_uuid", "generation", "schema_revision",
            "schema_fingerprint",
        )
    )
    if (
        owner_status.get("lifecycle") != "ready"
        or identity.get("status") != "ready"
        or replica.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/read-replica-observation@1"
        or replica.get("authority") != "non_authoritative_read_replica"
        or replica.get("live") is not True
        or not exact_identity
        or Path(str(replica.get("path") or "")).resolve(strict=False)
        != expected_path.resolve(strict=False)
        or Path(str(replica.get("source_database_path") or "")).resolve(
            strict=False
        )
        != paths["database"].resolve(strict=False)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        or type(size_bytes) is not int
        or not 0 < size_bytes <= LIVE_REPLAY_MAX_BYTES
        or type(refresh_sequence) is not int
        or refresh_sequence < 1
        or not str(replica.get("storage_schema_fingerprint") or "")
        or replica.get("storage_schema_fingerprint")
        != owner_status.get("storage_schema_fingerprint")
    ):
        raise OperatorError("live owner replica identity is incomplete or stale")
    return {
        "path": str(expected_path),
        "source_database_path": str(paths["database"]),
        "server_id": str(identity["server_id"]),
        "database_uuid": str(identity["database_uuid"]),
        "generation": int(identity["generation"]),
        "schema_revision": int(identity["schema_revision"]),
        "schema_fingerprint": str(identity["schema_fingerprint"]),
        "storage_schema_fingerprint": str(
            replica["storage_schema_fingerprint"]
        ),
        "sha256": digest,
        "size_bytes": size_bytes,
        "refresh_sequence": refresh_sequence,
    }


def _copy_published_replica(
    binding: Mapping[str, Any], destination: Path
) -> None:
    """Copy exact stable replica bytes without following or racing a path."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise OperatorError("live projection replay requires O_NOFOLLOW")
    source = Path(str(binding["path"]))
    source_descriptor = os.open(
        source, os.O_RDONLY | os.O_CLOEXEC | nofollow
    )
    destination_descriptor = -1
    started = time.monotonic()
    try:
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != int(binding["size_bytes"])
        ):
            raise OperatorError("published replica file identity is unsafe")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | nofollow,
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while copied < before.st_size:
            if time.monotonic() - started > LIVE_REPLAY_IO_TIMEOUT_SECONDS:
                raise OperatorError("published replica shadow copy timed out")
            chunk = os.read(
                source_descriptor, min(1_048_576, before.st_size - copied)
            )
            if not chunk:
                raise OperatorError("published replica was truncated during copy")
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise OperatorError("published replica copy made no progress")
                remaining = remaining[written:]
            copied += len(chunk)
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        path_after = os.lstat(source)

        def file_identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev, value.st_ino, value.st_mode, value.st_uid,
                value.st_nlink, value.st_size, value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if (
            file_identity(before) != file_identity(after)
            or file_identity(before) != file_identity(path_after)
            or stat.S_ISLNK(path_after.st_mode)
            or copied != before.st_size
            or f"sha256:{digest.hexdigest()}" != binding["sha256"]
        ):
            raise OperatorError("published replica changed during shadow copy")
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def _published_replica_bytes_still_match(binding: Mapping[str, Any]) -> None:
    """Re-hash the live pathname before reusing a content-bound replay."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise OperatorError("live projection replay requires O_NOFOLLOW")
    source = Path(str(binding["path"]))
    descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    started = time.monotonic()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != int(binding["size_bytes"])
        ):
            raise OperatorError("published replica file identity is unsafe")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            if time.monotonic() - started > LIVE_REPLAY_IO_TIMEOUT_SECONDS:
                raise OperatorError("published replica verification timed out")
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                raise OperatorError("published replica was truncated during hash")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(source)

        def file_identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev, value.st_ino, value.st_mode, value.st_uid,
                value.st_nlink, value.st_size, value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if (
            file_identity(before) != file_identity(after)
            or file_identity(before) != file_identity(path_after)
            or stat.S_ISLNK(path_after.st_mode)
            or f"sha256:{digest.hexdigest()}" != binding["sha256"]
        ):
            raise OperatorError("published replica bytes differ from owner status")
    finally:
        os.close(descriptor)


def _retire_live_replay_directory(directory: Path) -> None:
    """Remove only exact same-UID files created by one private replay."""

    allowed = {
        "control.duckdb",
        "control.duckdb.wal",
        ".control.duckdb.lock",
        ".control.duckdb.intent.lock",
        ".control.duckdb.migration.lock",
    }
    unexpected: list[str] = []
    for child in directory.iterdir():
        if child.name not in allowed:
            unexpected.append(child.name)
            continue
        observed = os.lstat(child)
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_nlink != 1
        ):
            unexpected.append(child.name)
            continue
        child.unlink()
    if unexpected:
        raise OperatorError(
            "private live replay produced unexpected artifacts: "
            + ", ".join(sorted(unexpected))
        )
    directory.rmdir()


def _admit_live_projection_shadow_replay(
    *,
    paths: Mapping[str, Path],
    owner_status: Mapping[str, Any],
    expected_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay exact published bytes privately; never mutate live authority."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    binding = _published_replica_binding(owner_status, paths)
    cache_key = _identity(
        {
            "replica": binding,
            "projection_cid": expected_snapshot.get("projection_cid"),
            "event_cursor": expected_snapshot.get("event_cursor"),
            "plan_root_cid": expected_snapshot.get("plan_root_cid"),
            "repository_tree_id": expected_snapshot.get("repository_tree_id"),
        }
    )
    with _LIVE_REPLAY_CACHE_LOCK:
        cached = dict(_LIVE_REPLAY_CACHE) if _LIVE_REPLAY_CACHE.get(
            "cache_key"
        ) == cache_key else {}
    if cached:
        _published_replica_bytes_still_match(binding)
        return cached

    replay_directory = paths["runtime"] / (
        f".live-projection-replay.{os.getpid()}.{time.time_ns()}"
    )
    replay_directory.mkdir(mode=0o700)
    temporary = replay_directory / "control.duckdb"
    witness: dict[str, Any]
    try:
        _copy_published_replica(binding, temporary)
        with DatabaseTaskSource(
            temporary,
            install_schema=False,
            repository_tree_id=str(expected_snapshot.get("repository_tree_id") or ""),
            plan_root_cid=str(expected_snapshot.get("plan_root_cid") or ""),
        ) as replay:
            observed = replay.snapshot().to_dict()
            fields = (
                "source_schema", "schema_version", "plan_root_cid",
                "repository_tree_id", "projection_cid", "formal_plan_id",
                "source_identity", "revision", "event_cursor", "goal_count",
                "task_count", "dependency_count", "terminal",
                "objective_count", "plan_count",
            )
            if any(
                observed.get(field) != expected_snapshot.get(field)
                for field in fields
            ):
                raise OperatorError(
                    "published replica differs from authenticated Quack snapshot"
                )
            if replay.projection_matches_events() is not True:
                raise OperatorError(
                    "published replica projection differs from admitted events"
                )
        witness = {
            "schema": LIVE_REPLAY_SCHEMA,
            "method": "disposable_exact_owner_published_replica_replay",
            "authoritative": False,
            "mutation_authority": False,
            "projection_matches_events": True,
            "replica": binding,
            "projection_cid": str(expected_snapshot["projection_cid"]),
            "event_cursor": int(expected_snapshot["event_cursor"]),
            "cache_key": cache_key,
        }
        witness["witness_cid"] = _identity(witness)
    finally:
        try:
            _retire_live_replay_directory(replay_directory)
        except BaseException:
            # A replay is not admissible until every private artifact has
            # been retired.  Clear a concurrently published matching entry
            # as well, so cleanup failure cannot become a later cache hit.
            with _LIVE_REPLAY_CACHE_LOCK:
                if _LIVE_REPLAY_CACHE.get("cache_key") == cache_key:
                    _LIVE_REPLAY_CACHE.clear()
            raise
    with _LIVE_REPLAY_CACHE_LOCK:
        _LIVE_REPLAY_CACHE.clear()
        _LIVE_REPLAY_CACHE.update(witness)
    return witness


def _live_projection_reconciliation_admitted(
    sample: Mapping[str, Any], paths: Mapping[str, Path]
) -> bool:
    """Validate that one sample carries an exact current shadow replay."""

    authority = sample.get("authority")
    authority = authority if isinstance(authority, Mapping) else {}
    snapshot = authority.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    witness = authority.get("projection_reconciliation")
    witness = witness if isinstance(witness, Mapping) else {}
    owner_status = sample.get("owner_status")
    owner_status = owner_status if isinstance(owner_status, Mapping) else {}
    if (
        set(witness) != LIVE_REPLAY_FIELDS
        or witness.get("schema") != LIVE_REPLAY_SCHEMA
        or witness.get("method")
        != "disposable_exact_owner_published_replica_replay"
        or witness.get("authoritative") is not False
        or witness.get("mutation_authority") is not False
        or witness.get("projection_matches_events") is not True
        or witness.get("projection_cid") != snapshot.get("projection_cid")
        or witness.get("event_cursor") != snapshot.get("event_cursor")
    ):
        return False
    try:
        if witness.get("replica") != _published_replica_binding(
            owner_status, paths
        ):
            return False
    except (OSError, OperatorError, TypeError, ValueError):
        return False
    unsigned = dict(witness)
    witness_cid = unsigned.pop("witness_cid", "")
    if witness_cid != _identity(unsigned):
        return False
    expected_cache_key = _identity(
        {
            "replica": witness.get("replica"),
            "projection_cid": snapshot.get("projection_cid"),
            "event_cursor": snapshot.get("event_cursor"),
            "plan_root_cid": snapshot.get("plan_root_cid"),
            "repository_tree_id": snapshot.get("repository_tree_id"),
        }
    )
    return witness.get("cache_key") == expected_cache_key


def _broker_status_query(
    board: Any,
    paths: Mapping[str, Path],
    *,
    owner_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the live portfolio through the sealed-broker Quack path."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    program = board.resolved_database_program()
    broker_socket = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET", "")
        or ""
    )
    broker_fd = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD", "")
        or ""
    )
    if (
        Path(broker_socket).resolve(strict=False)
        != (paths["owner"] / "typed-state-owner-grants.sock").resolve(strict=False)
        or not broker_fd.isdecimal()
        or str(os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "") or "")
    ):
        raise OperatorError("live status lacks an exclusive sealed-broker binding")
    os.fstat(int(broker_fd))
    bootstrap = _secure_runtime_json(
        paths["bootstrap_receipt"], max_bytes=STATUS_RECEIPT_MAX_BYTES
    )
    _bootstrap_receipt_id(bootstrap)
    bootstrap_integrity = bootstrap.get("integrity")
    bootstrap_integrity = (
        bootstrap_integrity if isinstance(bootstrap_integrity, Mapping) else {}
    )
    query_started_at_ms = int(time.time() * 1_000)
    with DatabaseTaskSource(
        program.quack_endpoint,
        owner_id=f"aseh-status:{os.getpid()}:{time.time_ns()}",
        install_schema=False,
        repository_tree_id=str(bootstrap.get("repository_tree_id") or ""),
        plan_root_cid=str(bootstrap.get("plan_root_cid") or ""),
    ) as source:
        if source.intent.uses_quack_transport is not True:
            raise OperatorError("live status did not use the Quack transport")
        snapshot = source.snapshot().to_dict()
        page = source.list_tasks(limit=100)
        if page.next_cursor:
            raise OperatorError("live task portfolio exceeds the sealed bound")
        ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
        page_statuses = {
            item.task_alias: str(item.status or "").lower()
            for item in page.tasks
        }
        queue_entries: dict[str, dict[str, Any]] = {}
        # Queue cooldown state matters only at a zero-ready, zero-active
        # frontier. Avoid forty point queries on every ordinary health sample.
        if not ready and not any(
            status in ACTIVE_STATUSES for status in page_statuses.values()
        ):
            for item in page.tasks:
                if page_statuses[item.task_alias] not in READY_STATUSES:
                    continue
                entry = source.get_queue_entry(item.task_cid)
                if (
                    entry is not None
                    and int(entry.retry_not_before_ms) > query_started_at_ms
                ):
                    queue_entries[item.task_alias] = entry.to_dict()
        sealed_goal_records = bootstrap_integrity.get("goal_records")
        sealed_goal_records = (
            sealed_goal_records if isinstance(sealed_goal_records, Mapping) else {}
        )
        goal_records: dict[str, dict[str, Any]] = {}
        for goal_alias, sealed_record in sealed_goal_records.items():
            sealed_record = (
                sealed_record if isinstance(sealed_record, Mapping) else {}
            )
            goal = source.get_goal(str(sealed_record.get("goal_cid") or ""))
            if not isinstance(goal, Mapping):
                raise OperatorError(f"live goal is missing: {goal_alias}")
            goal_records[str(goal_alias)] = _immutable_goal_record(goal)
        goal_edges = sorted(
            (dict(item) for item in source.list_goal_edges(limit=100)),
            key=_goal_edge_sort_key,
        )
        plan = source.get_plan(str(bootstrap.get("plan_root_cid") or ""))
        if not isinstance(plan, Mapping):
            raise OperatorError("live plan root is missing")
        plan_record = _immutable_plan_record(plan)
        plan_projection = source.plan_projection(
            task_cids=[item.task_cid for item in page.tasks]
        )
        task_authority_spec_cids = _task_authority_spec_cids(plan_projection)
        objective_record = _objective_record_from_projection(plan_projection)
        binding_fields = (
            "server_id", "store_id", "database_uuid", "schema_revision",
            "schema_fingerprint", "generation", "process_birth_id",
            "listen_uri", "extension_fingerprint",
        )
        owner_identity = owner_status.get("identity")
        owner_identity = (
            owner_identity if isinstance(owner_identity, Mapping) else {}
        )
        storage_schema_fingerprint = owner_status.get(
            "storage_schema_fingerprint"
        )
        integer_binding_fields = {"schema_revision", "generation"}
        if (
            any(owner_identity.get(field) in (None, "") for field in binding_fields)
            or any(
                type(owner_identity.get(field)) is not int
                for field in integer_binding_fields
            )
            or any(
                not isinstance(owner_identity.get(field), str)
                for field in set(binding_fields) - integer_binding_fields
            )
            or storage_schema_fingerprint in (None, "")
            or not isinstance(storage_schema_fingerprint, str)
        ):
            raise OperatorError("published owner identity binding is incomplete")
        with source.intent._connection(write=False) as connection:  # noqa: SLF001
            raw_binding = getattr(connection, "_quack_mutation_binding", None)
            if not isinstance(raw_binding, Mapping):
                raise OperatorError("Quack status query lacks a live owner binding")
            identity_fields = tuple(
                field for field in binding_fields if field != "schema_fingerprint"
            )
            if any(
                type(raw_binding.get(field)) is not type(owner_identity.get(field))
                or raw_binding.get(field) != owner_identity.get(field)
                for field in identity_fields
            ):
                raise OperatorError(
                    "Quack status query owner binding differs from published owner"
                )
            # The transport binding names the physical DuckDB schema.  The
            # public owner identity names the StateServerIdentity contract.
            # Admit the former only against its dedicated owner-status field,
            # then publish the latter so every receipt has one canonical
            # incarnation identity.
            if (
                not isinstance(raw_binding.get("schema_fingerprint"), str)
                or raw_binding.get("schema_fingerprint")
                != storage_schema_fingerprint
            ):
                raise OperatorError(
                    "Quack status query storage schema differs from published owner"
                )
            owner_binding = {
                field: owner_identity.get(field) for field in binding_fields
            }
        if any(value in (None, "") for value in owner_binding.values()):
            raise OperatorError("Quack status query owner binding is incomplete")
    statuses: Counter[str] = Counter()
    aliases: dict[str, str] = {}
    revisions: dict[str, int] = {}
    task_cids: dict[str, str] = {}
    owner_bindings: dict[str, dict[str, Any]] = {}
    task_dependencies: dict[str, list[str]] = {}
    for task in page.tasks:
        status_name = str(task.status or "").lower()
        aliases[task.task_alias] = status_name
        revisions[task.task_alias] = int(task.revision)
        task_cids[task.task_alias] = task.task_cid
        owner_bindings[task.task_alias] = {
            key: task.body.get(key)
            for key in (
                "owning_repository", "base_revision",
                "base_repository_tree_id", "source_forest_cid",
                "owner_source_identity",
            )
        }
        task_dependencies[task.task_alias] = list(task.dependencies)
        statuses[status_name] += 1
    delayed_ready_task_ids = sorted(
        task_alias
        for task_alias, entry in queue_entries.items()
        if (
            aliases.get(task_alias) in READY_STATUSES
            and isinstance(entry, Mapping)
            and type(entry.get("retry_not_before_ms")) is int
            and int(entry["retry_not_before_ms"]) > query_started_at_ms
        )
    )
    replay_witness = _admit_live_projection_shadow_replay(
        paths=paths,
        owner_status=owner_status,
        expected_snapshot=snapshot,
    )
    return {
        "available": True,
        "transport": "quack",
        "credential_path": "sealed_memfd_broker",
        "projection_matches_events": replay_witness.get(
            "projection_matches_events"
        ) is True,
        "projection_reconciliation": replay_witness,
        "owner_binding": owner_binding,
        "snapshot": snapshot,
        "task_statuses": dict(sorted(aliases.items())),
        "task_revisions": dict(sorted(revisions.items())),
        "task_cids": dict(sorted(task_cids.items())),
        "task_owner_bindings": dict(sorted(owner_bindings.items())),
        "task_dependencies": dict(sorted(task_dependencies.items())),
        "task_authority_spec_cids": task_authority_spec_cids,
        "objective_record": objective_record,
        "queue_entries": dict(sorted(queue_entries.items())),
        "query_started_at_ms": query_started_at_ms,
        "delayed_ready_task_ids": delayed_ready_task_ids,
        "goal_records": dict(sorted(goal_records.items())),
        "goal_edges": goal_edges,
        "plan_record": plan_record,
        "status_counts": dict(sorted(statuses.items())),
        "ready_task_ids": ready,
        "ready_count": len(ready),
        "active_count": sum(statuses.get(item, 0) for item in ACTIVE_STATUSES),
        "blocked_count": int(statuses.get("blocked", 0)),
        "completed_count": sum(statuses.get(item, 0) for item in COMPLETED_STATUSES),
        "terminal_count": sum(statuses.get(item, 0) for item in TERMINAL_STATUSES),
        "objective_count": int(snapshot.get("objective_count", -1)),
        "plan_count": int(snapshot.get("plan_count", -1)),
        "event_cursor": int(snapshot.get("event_cursor") or 0),
    }


def _lane_status_observations(board: Any, *, now: float) -> list[dict[str, Any]]:
    state_root = board.path(board.runtime_paths["state"])
    prefix = re.sub(
        r"[^a-z0-9._-]+", "-", board.task_prefix.strip().lower()
    ).strip("-") or "configured-board"
    rows: list[dict[str, Any]] = []
    for index in range(board.max_lanes):
        path = (
            state_root / f"lane-{index}"
            / f"{prefix}_lane_{index}_supervisor_status.json"
        )
        row: dict[str, Any] = {
            "lane": index,
            "path": str(path.relative_to(ROOT)),
            "present": False,
            "fresh": False,
        }
        try:
            observed = path.stat()
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            rows.append(row)
            continue
        age = max(0.0, now - observed.st_mtime)
        worker_metrics_available = payload.get("worker_metrics_available")
        worker_metrics_available = (
            worker_metrics_available
            if type(worker_metrics_available) is bool
            else None
        )
        active_worker_count = payload.get("active_worker_count")
        active_worker_count = (
            active_worker_count
            if type(active_worker_count) is int and active_worker_count >= 0
            else None
        )
        active_worker_pids = payload.get("active_worker_pids")
        active_worker_pids = (
            list(active_worker_pids)
            if (
                isinstance(active_worker_pids, list)
                and all(
                    type(item) is int and item > 1
                    for item in active_worker_pids
                )
                and len(active_worker_pids) == len(set(active_worker_pids))
            )
            else None
        )
        worker_descendant_count = payload.get("worker_descendant_count")
        worker_descendant_count = (
            worker_descendant_count
            if (
                type(worker_descendant_count) is int
                and worker_descendant_count >= 0
            )
            else None
        )
        worker_descendant_pids = payload.get("worker_descendant_pids")
        worker_descendant_pids = (
            list(worker_descendant_pids)
            if (
                isinstance(worker_descendant_pids, list)
                and all(
                    type(item) is int and item > 1
                    for item in worker_descendant_pids
                )
                and len(worker_descendant_pids)
                == len(set(worker_descendant_pids))
            )
            else None
        )
        worker_root_pid = payload.get("worker_root_pid")
        worker_root_pid = (
            worker_root_pid
            if type(worker_root_pid) is int and worker_root_pid > 1
            else None
        )
        worker_root_start_time_ticks = payload.get(
            "worker_root_start_time_ticks"
        )
        worker_root_start_time_ticks = (
            worker_root_start_time_ticks
            if (
                type(worker_root_start_time_ticks) is int
                and worker_root_start_time_ticks > 0
            )
            else None
        )
        worker_root_boot_id = str(payload.get("worker_root_boot_id") or "")
        worker_root_identity_source = str(
            payload.get("worker_root_identity_source") or ""
        )
        run_id = str(payload.get("run_id") or "")
        worker_observed_at_ns = payload.get("worker_observed_at_ns")
        worker_observed_at_ns = (
            worker_observed_at_ns
            if type(worker_observed_at_ns) is int
            and worker_observed_at_ns > 0
            else None
        )
        observation_now = max(now, time.time())
        worker_observation_age_seconds = (
            observation_now - (worker_observed_at_ns / 1_000_000_000)
            if worker_observed_at_ns is not None
            else None
        )
        worker_observation_generation = str(
            payload.get("worker_observation_generation") or ""
        )
        daemon_pid = payload.get("daemon_pid")
        daemon_pid = (
            daemon_pid
            if type(daemon_pid) is int and daemon_pid > 1
            else None
        )
        worker_phase_guarded = payload.get("worker_phase_guarded")
        worker_phase_guarded = (
            worker_phase_guarded
            if type(worker_phase_guarded) is bool
            else None
        )
        worker_phase_available = payload.get("worker_phase_available")
        worker_phase_available = (
            worker_phase_available
            if type(worker_phase_available) is bool
            else None
        )
        worker_phase = payload.get("worker_phase")
        worker_phase = worker_phase if isinstance(worker_phase, str) else None
        worker_phase_known = payload.get("worker_phase_known")
        worker_phase_known = (
            worker_phase_known
            if type(worker_phase_known) is bool
            else None
        )
        worker_phase_known_non_worktree = payload.get(
            "worker_phase_known_non_worktree"
        )
        worker_phase_known_non_worktree = (
            worker_phase_known_non_worktree
            if type(worker_phase_known_non_worktree) is bool
            else None
        )
        worker_stall_evidence_available = payload.get(
            "worker_stall_evidence_available"
        )
        worker_stall_evidence_available = (
            worker_stall_evidence_available
            if type(worker_stall_evidence_available) is bool
            else None
        )
        worker_stall_unavailable_reason = str(
            payload.get("worker_stall_evidence_unavailable_reason") or ""
        )
        stalled_without_active_worker = payload.get(
            "stalled_without_active_worker"
        )
        stalled_without_active_worker = (
            stalled_without_active_worker
            if type(stalled_without_active_worker) is bool
            else None
        )
        worker_phase_age = payload.get("worker_phase_age_seconds")
        worker_phase_age = (
            float(worker_phase_age)
            if (
                not isinstance(worker_phase_age, bool)
                and isinstance(worker_phase_age, (int, float))
                and float(worker_phase_age) >= 0.0
            )
            else None
        )
        census_admissible = bool(
            worker_metrics_available is True
            and payload.get("worker_census_method")
            == "linux-procfs-descendant-census@1"
            and daemon_pid is not None
            and worker_root_pid == daemon_pid
            and worker_root_start_time_ticks is not None
            and bool(worker_root_boot_id)
            and worker_root_identity_source == "supervised_child_identity"
            and bool(run_id)
            and worker_observed_at_ns is not None
            and worker_observed_at_ns <= observed.st_mtime_ns
            and worker_observation_age_seconds is not None
            and 0.0 <= worker_observation_age_seconds <= 60.0
            and worker_observation_generation
            == (
                f"{run_id}:{worker_root_pid}:"
                f"{worker_root_start_time_ticks}:{worker_root_boot_id}"
            )
            and active_worker_count is not None
            and active_worker_pids is not None
            and active_worker_count == len(active_worker_pids)
            and worker_descendant_count is not None
            and worker_descendant_pids is not None
            and worker_descendant_count == len(worker_descendant_pids)
            and set(active_worker_pids).issubset(worker_descendant_pids)
            and worker_phase_known is True
        )
        stall_admissible = bool(
            (
                worker_stall_evidence_available is True
                and stalled_without_active_worker is not None
                and worker_phase_guarded is True
                and worker_phase_known_non_worktree is False
            )
            or (
                worker_stall_evidence_available is False
                and worker_phase_guarded is False
                and worker_phase in {"", *KNOWN_NON_WORKTREE_PHASES}
                and worker_phase_available is bool(worker_phase)
                and worker_phase_known_non_worktree
                is (worker_phase in KNOWN_NON_WORKTREE_PHASES)
                and stalled_without_active_worker is None
                and worker_stall_unavailable_reason == "phase_not_guarded"
            )
        )
        watchdog_admissible = bool(
            census_admissible and stall_admissible
        )
        row.update(
            {
                "present": True,
                "mtime_ns": observed.st_mtime_ns,
                "age_seconds": age,
                "fresh": age <= 60.0,
                "phase": worker_phase,
                "worker_metrics_available": worker_metrics_available,
                "worker_metrics_unavailable_reason": str(
                    payload.get("worker_metrics_unavailable_reason") or ""
                ),
                "worker_census_method": str(
                    payload.get("worker_census_method") or ""
                ),
                "worker_root_pid": worker_root_pid,
                "worker_root_start_time_ticks": (
                    worker_root_start_time_ticks
                ),
                "worker_root_boot_id": worker_root_boot_id,
                "worker_root_identity_source": worker_root_identity_source,
                "worker_observed_at_ns": worker_observed_at_ns,
                "worker_observation_age_seconds": (
                    worker_observation_age_seconds
                ),
                "worker_observation_generation": (
                    worker_observation_generation
                ),
                "active_worker_count": active_worker_count,
                "active_worker_pids": active_worker_pids,
                "worker_descendant_count": worker_descendant_count,
                "worker_descendant_pids": worker_descendant_pids,
                "worker_phase_guarded": worker_phase_guarded,
                "worker_phase_available": worker_phase_available,
                "worker_phase_known": worker_phase_known,
                "worker_phase_known_non_worktree": (
                    worker_phase_known_non_worktree
                ),
                "worker_phase_age_seconds": worker_phase_age,
                "worker_stall_evidence_available": (
                    worker_stall_evidence_available
                ),
                "worker_stall_evidence_unavailable_reason": (
                    worker_stall_unavailable_reason
                ),
                "stalled_without_active_worker": stalled_without_active_worker,
                "worker_census_admissible": census_admissible,
                "worker_stall_admissible": stall_admissible,
                "watchdog_admissible": watchdog_admissible,
                "admissible": bool(
                    payload.get("schema")
                    == (
                        "ipfs_accelerate_py.agent_supervisor."
                        "todo_implementation_supervisor.supervisor"
                    )
                    and payload.get("repo_root") == str(board.repo_root)
                    and payload.get("task_prefix") == board.task_header_prefix
                    and payload.get("state_prefix")
                    == f"{prefix}_lane_{index}"
                    and str(payload.get("status") or "")
                    in {
                        "starting",
                        "running",
                        "restarting",
                        "agentic_maintenance_started",
                        "agentic_maintenance_completed",
                    }
                ),
                "receipt_cid": (
                    payload.get("receipt_cid") or payload.get("status_cid") or ""
                ),
            }
        )
        rows.append(row)
    return rows


def _status_sample(
    board: Any,
    paths: Mapping[str, Path],
    server: Any,
    scheduler: subprocess.Popen[Any],
) -> dict[str, Any]:
    observed_at = time.time()
    owner_status_after: Mapping[str, Any] = {}
    authority: dict[str, Any] = {}
    for attempt in range(STATUS_REPLICA_STABILITY_ATTEMPTS):
        owner_status_before = server.status()
        retryable_publication_race = False
        try:
            authority = _broker_status_query(
                board, paths, owner_status=owner_status_before
            )
        except Exception as exc:
            # A canonical typed owner publishes each committed task mutation by
            # briefly withdrawing the read-only Quack endpoint, replacing the
            # exact replica, and rebinding the same owner identity.  Concurrent
            # health reads may therefore observe either DuckDB's exact
            # endpoint-bound refusal or a local byte-identity race.  Neither is
            # status evidence.  Retry the *whole* query from a fresh owner
            # status, while leaving every foreign/policy/schema error fail
            # closed and never manufacturing an available sample.
            from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
                quack_transport_error_is_unavailable,
                reset_quack_transport_cache,
            )

            program = board.resolved_database_program()
            quack_uri = str(getattr(program, "quack_endpoint", "") or "")
            transport_refresh = quack_transport_error_is_unavailable(
                exc, uri=quack_uri
            )
            local_replica_refresh = type(exc) is OperatorError and str(exc) in {
                "published replica changed during shadow copy",
                "published replica bytes differ from owner status",
            }
            retryable_publication_race = bool(
                transport_refresh or local_replica_refresh
            )
            if transport_refresh:
                reset_quack_transport_cache(quack_uri)
            authority = {
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ready_count": 0,
                "active_count": 0,
                "blocked_count": 0,
                "terminal_count": 0,
                "event_cursor": 0,
                "task_statuses": {},
                "task_revisions": {},
            }
        owner_status_after = server.status()
        if authority.get("available") is not True:
            if (
                retryable_publication_race
                and attempt + 1 < STATUS_REPLICA_STABILITY_ATTEMPTS
            ):
                time.sleep(STATUS_REPLICA_RETRY_DELAY_SECONDS)
                continue
            break
        try:
            stable_replica = bool(
                _published_replica_binding(owner_status_before, paths)
                == _published_replica_binding(owner_status_after, paths)
            )
        except Exception as exc:
            authority = {
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ready_count": 0,
                "active_count": 0,
                "blocked_count": 0,
                "terminal_count": 0,
                "event_cursor": 0,
                "task_statuses": {},
                "task_revisions": {},
            }
            break
        if stable_replica:
            break
        if attempt + 1 == STATUS_REPLICA_STABILITY_ATTEMPTS:
            authority = {
                "available": False,
                "error_type": "OperatorError",
                "error": (
                    "owner replica publication changed during status query"
                ),
                "ready_count": 0,
                "active_count": 0,
                "blocked_count": 0,
                "terminal_count": 0,
                "event_cursor": 0,
                "task_statuses": {},
                "task_revisions": {},
            }
    scheduler_returncode = scheduler.poll()
    try:
        scheduler_process_group = os.getpgid(scheduler.pid)
    except ProcessLookupError:
        scheduler_process_group = -1
    sample = {
        "observed_at": observed_at,
        "monotonic_ns": time.monotonic_ns(),
        "owner_status": owner_status_after,
        "scheduler": {
            "pid": scheduler.pid,
            "process_group": scheduler_process_group,
            "alive": scheduler_returncode is None,
            "returncode": scheduler_returncode,
        },
        "authority": authority,
        "lanes": _lane_status_observations(board, now=time.time()),
    }
    sample["sample_cid"] = _identity(sample)
    return sample


def _authoritative_progress_between(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[str]:
    """Return task-authority movement, never controller heartbeat movement."""

    pair = _task_authority_pair_invariants(before, after)
    if pair["admitted"] is not True or pair["revision_advanced"] is not True:
        return []
    evidence: list[str] = []
    # The event stream can contain evidence or accounting records that do not
    # move the task authority.  Admit its cursor only as corroboration for an
    # observed task status/revision delta, never as progress by itself.
    if pair["event_cursor_advanced"] is True:
        evidence.append("authoritative_event_advanced")
    if pair["status_changed"] is True:
        evidence.append("task_status_changed")
    if pair["revision_advanced"] is True:
        evidence.append("task_revision_changed")
    return evidence


def _task_authority_pair_invariants(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, bool]:
    """Validate two authenticated observations of one exact task population."""

    def authority(sample: Mapping[str, Any]) -> Mapping[str, Any] | None:
        value = sample.get("authority")
        if not isinstance(value, Mapping):
            return None
        if (
            value.get("available") is not True
            or value.get("transport") != "quack"
            or value.get("credential_path") != "sealed_memfd_broker"
        ):
            return None
        return value

    prior = authority(before)
    current = authority(after)
    empty = {
        "admitted": False,
        "event_cursor_monotonic": False,
        "event_cursor_advanced": False,
        "revision_monotonic": False,
        "revision_advanced": False,
        "status_revision_consistent": False,
        "status_changed": False,
    }
    if prior is None or current is None:
        return empty
    prior_statuses = prior.get("task_statuses")
    current_statuses = current.get("task_statuses")
    prior_revisions = prior.get("task_revisions")
    current_revisions = current.get("task_revisions")
    if not all(
        isinstance(value, Mapping)
        for value in (
            prior_statuses, current_statuses,
            prior_revisions, current_revisions,
        )
    ):
        return empty
    task_ids = set(prior_statuses)
    if (
        not task_ids
        or set(current_statuses) != task_ids
        or set(prior_revisions) != task_ids
        or set(current_revisions) != task_ids
        or any(
            type(prior_revisions[task_id]) is not int
            or type(current_revisions[task_id]) is not int
            for task_id in task_ids
        )
    ):
        return empty
    event_before = prior.get("event_cursor")
    event_after = current.get("event_cursor")
    if type(event_before) is not int or type(event_after) is not int:
        return empty
    revision_monotonic = all(
        int(current_revisions[task_id]) >= int(prior_revisions[task_id])
        for task_id in task_ids
    )
    advanced_tasks = {
        task_id
        for task_id in task_ids
        if int(current_revisions[task_id]) > int(prior_revisions[task_id])
    }
    changed_status_tasks = {
        task_id
        for task_id in task_ids
        if current_statuses[task_id] != prior_statuses[task_id]
    }
    event_cursor_monotonic = event_after >= event_before
    event_cursor_advanced = event_after > event_before
    status_revision_consistent = changed_status_tasks.issubset(advanced_tasks)
    admitted = bool(
        revision_monotonic
        and status_revision_consistent
        and event_cursor_monotonic
        and (not advanced_tasks or event_cursor_advanced)
    )
    return {
        "admitted": admitted,
        "event_cursor_monotonic": event_cursor_monotonic,
        "event_cursor_advanced": event_cursor_advanced,
        "revision_monotonic": revision_monotonic,
        "revision_advanced": bool(advanced_tasks),
        "status_revision_consistent": status_revision_consistent,
        "status_changed": bool(changed_status_tasks),
    }


def _lane_liveness_between(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[str]:
    """Return non-authoritative controller liveness observations."""

    evidence: list[str] = []
    prior_lanes = {
        int(item.get("lane", -1)): int(item.get("mtime_ns") or 0)
        for item in before.get("lanes", [])
        if isinstance(item, Mapping)
    }
    if any(
        int(item.get("mtime_ns") or 0)
        > prior_lanes.get(int(item.get("lane", -1)), 0)
        for item in after.get("lanes", [])
        if isinstance(item, Mapping)
    ):
        evidence.append("lane_heartbeat_advanced")
    return evidence


def _owner_identity_admitted(sample: Mapping[str, Any]) -> bool:
    """Require Quack and local owner status to name one exact incarnation."""

    authority = sample.get("authority")
    authority = authority if isinstance(authority, Mapping) else {}
    binding = authority.get("owner_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    owner_status = sample.get("owner_status")
    owner_status = owner_status if isinstance(owner_status, Mapping) else {}
    identity = owner_status.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    process_birth = identity.get("process_birth")
    process_birth = process_birth if isinstance(process_birth, Mapping) else {}
    process_pid = process_birth.get("pid")
    fields = (
        "server_id", "store_id", "database_uuid", "schema_revision",
        "schema_fingerprint", "generation", "process_birth_id", "listen_uri",
        "extension_fingerprint",
    )
    return bool(
        owner_status.get("lifecycle") == "ready"
        and authority.get("available") is True
        and all(identity.get(field) not in (None, "") for field in fields)
        and all(binding.get(field) == identity.get(field) for field in fields)
        and type(process_pid) is int
        and process_pid == os.getpid()
    )


def _bootstrap_authoritative_witness(
    paths: Mapping[str, Path], authority: Mapping[str, Any]
) -> list[str]:
    """Recognize a fast claim/status change observed before sample one."""

    bootstrap = _secure_runtime_json(
        paths["bootstrap_receipt"], max_bytes=STATUS_RECEIPT_MAX_BYTES
    )
    _bootstrap_receipt_id(bootstrap)
    bootstrap_snapshot = bootstrap.get("snapshot")
    bootstrap_snapshot = (
        bootstrap_snapshot if isinstance(bootstrap_snapshot, Mapping) else {}
    )
    if int(authority.get("event_cursor") or 0) <= int(
        bootstrap_snapshot.get("event_cursor") or 0
    ):
        return []
    evidence = ["authoritative_event_since_bootstrap"]
    integrity = bootstrap.get("integrity")
    integrity = integrity if isinstance(integrity, Mapping) else {}
    baseline_statuses = integrity.get("task_statuses")
    baseline_statuses = (
        baseline_statuses if isinstance(baseline_statuses, Mapping) else {}
    )
    baseline_revisions = integrity.get("task_revisions")
    baseline_revisions = (
        baseline_revisions if isinstance(baseline_revisions, Mapping) else {}
    )
    statuses = authority.get("task_statuses")
    statuses = statuses if isinstance(statuses, Mapping) else {}
    if baseline_statuses and statuses != baseline_statuses:
        evidence.append("authoritative_status_since_bootstrap")
    revisions = authority.get("task_revisions")
    revisions = revisions if isinstance(revisions, Mapping) else {}
    if baseline_revisions and any(
        type(value) is int
        and type(baseline_revisions.get(task_id)) is int
        and value > int(baseline_revisions[task_id])
        for task_id, value in revisions.items()
    ):
        evidence.append("authoritative_revision_since_bootstrap")
    return evidence if len(evidence) > 1 else []


def _health_receipt(
    board: Any,
    paths: Mapping[str, Path],
    *,
    samples: Sequence[Mapping[str, Any]],
    launched_at: float,
    last_progress_at: float,
    failure: dict[str, Any],
    require_authoritative_progress: bool = False,
) -> dict[str, Any]:
    if len(samples) != 2:
        raise OperatorError("health receipt requires exactly two samples")
    before, current = samples
    scheduler_samples = [
        item.get("scheduler")
        if isinstance(item.get("scheduler"), Mapping)
        else {}
        for item in (before, current)
    ]
    scheduler_alive = bool(
        all(item.get("alive") is True for item in scheduler_samples)
        and len({int(item.get("pid") or 0) for item in scheduler_samples}) == 1
        and all(
            int(item.get("process_group") or -1) == int(item.get("pid") or 0)
            for item in scheduler_samples
        )
    )
    prior_authority = before.get("authority")
    prior_authority = (
        prior_authority if isinstance(prior_authority, Mapping) else {}
    )
    authority = current.get("authority")
    authority = authority if isinstance(authority, Mapping) else {}
    owner_status = current.get("owner_status")
    owner_status = owner_status if isinstance(owner_status, Mapping) else {}
    progress = _authoritative_progress_between(before, current)
    liveness = _lane_liveness_between(before, current)
    now = float(current["observed_at"])
    stale_seconds = float(board.payload.get("stale_seconds") or 1800.0)
    startup_grace = float(
        board.payload.get("watchdog_startup_grace_seconds") or 300.0
    )
    lanes = [item for item in current.get("lanes", []) if isinstance(item, Mapping)]
    lane_fresh = bool(
        len(lanes) == board.max_lanes
        and all(
            item.get("fresh") is True
            and item.get("admissible") is True
            and item.get("watchdog_admissible") is True
            and int(item.get("mtime_ns") or 0) >= int(launched_at * 1_000_000_000)
            for item in lanes
        )
    )
    lane_stalled = any(
        item.get("stalled_without_active_worker") is True for item in lanes
    )
    owner_ready = owner_status.get("lifecycle") == "ready"
    broker = owner_status.get("configured_supervisor_credential_broker")
    broker = broker if isinstance(broker, Mapping) else {}
    broker_ready = bool(
        broker.get("available") is True
        and not str(broker.get("last_error_type") or "")
    )
    broker_samples_authenticated = all(
        sample.get("available") is True
        and sample.get("transport") == "quack"
        and sample.get("credential_path") == "sealed_memfd_broker"
        for sample in (prior_authority, authority)
    )
    owner_identity_admitted = all(
        _owner_identity_admitted(sample) for sample in (before, current)
    )
    prior_owner_binding = prior_authority.get("owner_binding")
    current_owner_binding = authority.get("owner_binding")
    owner_identity_admitted = bool(
        owner_identity_admitted
        and isinstance(prior_owner_binding, Mapping)
        and current_owner_binding == prior_owner_binding
    )
    task_authority_pair = _task_authority_pair_invariants(before, current)
    expected_tasks = int(board.payload["initial_projection"]["task_count"])
    snapshot = authority.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    task_count = int(snapshot.get("task_count", -1))
    expected_goals = int(board.payload["initial_projection"]["goal_count"])
    expected_dependencies = int(
        board.payload["initial_projection"]["task_dependency_count"]
    )
    goal_count = int(snapshot.get("goal_count", -1))
    dependency_count = int(snapshot.get("dependency_count", -1))
    bootstrap = _secure_runtime_json(
        paths["bootstrap_receipt"], max_bytes=STATUS_RECEIPT_MAX_BYTES
    )
    _bootstrap_receipt_id(bootstrap)
    bootstrap_snapshot = bootstrap.get("snapshot")
    bootstrap_snapshot = (
        bootstrap_snapshot if isinstance(bootstrap_snapshot, Mapping) else {}
    )
    immutable_snapshot_fields = (
        "source_schema", "schema_version", "plan_root_cid",
        "repository_tree_id", "formal_plan_id",
    )
    source_identity_admitted = all(
        isinstance(candidate.get("snapshot"), Mapping)
        and candidate.get("projection_matches_events") is True
        and _live_projection_reconciliation_admitted(sample, paths)
        and all(
            bootstrap_snapshot.get(field) not in (None, "")
            and candidate["snapshot"].get(field)
            == bootstrap_snapshot.get(field)
            for field in immutable_snapshot_fields
        )
        and bool(candidate["snapshot"].get("projection_cid"))
        and candidate["snapshot"].get("source_identity")
        == content_identity(
            {
                "plan_root_cid": candidate["snapshot"].get("plan_root_cid"),
                "repository_tree_id": candidate["snapshot"].get(
                    "repository_tree_id"
                ),
                "projection_cid": candidate["snapshot"].get("projection_cid"),
            }
        )
        for sample, candidate in (
            (before, prior_authority),
            (current, authority),
        )
    )
    bootstrap_integrity = bootstrap.get("integrity")
    bootstrap_integrity = (
        bootstrap_integrity if isinstance(bootstrap_integrity, Mapping) else {}
    )
    expected_objectives = int(bootstrap_integrity.get("objective_count", -1))
    expected_plans = int(bootstrap_integrity.get("plan_count", -1))
    objective_count = int(snapshot.get("objective_count", -1))
    plan_count = int(snapshot.get("plan_count", -1))
    sealed_task_cids = bootstrap_integrity.get("task_cids")
    sealed_task_cids = (
        sealed_task_cids if isinstance(sealed_task_cids, Mapping) else {}
    )
    sealed_owner_bindings = bootstrap_integrity.get("task_owner_bindings")
    sealed_owner_bindings = (
        sealed_owner_bindings
        if isinstance(sealed_owner_bindings, Mapping)
        else {}
    )
    sealed_task_dependencies = bootstrap_integrity.get("task_dependencies")
    sealed_task_dependencies = (
        sealed_task_dependencies
        if isinstance(sealed_task_dependencies, Mapping)
        else {}
    )
    sealed_task_authority_spec_cids = bootstrap_integrity.get(
        "task_authority_spec_cids"
    )
    sealed_task_authority_spec_cids = (
        sealed_task_authority_spec_cids
        if isinstance(sealed_task_authority_spec_cids, Mapping)
        else {}
    )
    sealed_goal_records = bootstrap_integrity.get("goal_records")
    sealed_goal_records = (
        sealed_goal_records if isinstance(sealed_goal_records, Mapping) else {}
    )
    sealed_goal_edges = bootstrap_integrity.get("goal_edges")
    sealed_goal_edges = (
        sealed_goal_edges if isinstance(sealed_goal_edges, list) else []
    )
    sealed_plan_record = bootstrap_integrity.get("plan_record")
    sealed_plan_record = (
        sealed_plan_record if isinstance(sealed_plan_record, Mapping) else {}
    )
    sealed_objective_record = bootstrap_integrity.get("objective_record")
    sealed_objective_record = (
        sealed_objective_record
        if isinstance(sealed_objective_record, Mapping)
        else {}
    )
    task_cids = authority.get("task_cids")
    task_cids = task_cids if isinstance(task_cids, Mapping) else {}
    task_owner_bindings = authority.get("task_owner_bindings")
    task_owner_bindings = (
        task_owner_bindings if isinstance(task_owner_bindings, Mapping) else {}
    )
    task_dependencies = authority.get("task_dependencies")
    task_dependencies = (
        task_dependencies if isinstance(task_dependencies, Mapping) else {}
    )
    task_authority_spec_cids = authority.get("task_authority_spec_cids")
    task_authority_spec_cids = (
        task_authority_spec_cids
        if isinstance(task_authority_spec_cids, Mapping)
        else {}
    )
    goal_records = authority.get("goal_records")
    goal_records = goal_records if isinstance(goal_records, Mapping) else {}
    goal_edges = authority.get("goal_edges")
    goal_edges = goal_edges if isinstance(goal_edges, list) else []
    plan_record = authority.get("plan_record")
    plan_record = plan_record if isinstance(plan_record, Mapping) else {}
    objective_record = authority.get("objective_record")
    objective_record = (
        objective_record if isinstance(objective_record, Mapping) else {}
    )
    task_statuses = authority.get("task_statuses")
    task_statuses = task_statuses if isinstance(task_statuses, Mapping) else {}
    task_revisions = authority.get("task_revisions")
    task_revisions = task_revisions if isinstance(task_revisions, Mapping) else {}
    def admitted_task_corpus(candidate: Mapping[str, Any]) -> bool:
        return bool(
            sealed_task_cids
            and candidate.get("task_cids") == sealed_task_cids
            and candidate.get("task_owner_bindings") == sealed_owner_bindings
            and candidate.get("task_dependencies") == sealed_task_dependencies
            and candidate.get("task_authority_spec_cids")
            == sealed_task_authority_spec_cids
            and isinstance(candidate.get("task_statuses"), Mapping)
            and isinstance(candidate.get("task_revisions"), Mapping)
            and set(candidate["task_statuses"]) == set(sealed_task_cids)
            and set(candidate["task_revisions"]) == set(sealed_task_cids)
        )

    task_corpus_admitted = all(
        admitted_task_corpus(candidate)
        for candidate in (prior_authority, authority)
    )

    def admitted_semantic_corpus(candidate: Mapping[str, Any]) -> bool:
        return bool(
            sealed_goal_records
            and candidate.get("goal_records") == sealed_goal_records
            and candidate.get("goal_edges") == sealed_goal_edges
            and candidate.get("plan_record") == sealed_plan_record
            and candidate.get("objective_record") == sealed_objective_record
        )

    semantic_corpus_admitted = all(
        admitted_semantic_corpus(candidate)
        for candidate in (prior_authority, authority)
    )
    blocked_count = int(authority.get("blocked_count") or 0)
    terminal_count = int(authority.get("terminal_count") or 0)
    ready_count = int(authority.get("ready_count") or 0)
    active_count = int(authority.get("active_count") or 0)
    delayed_ready_task_ids = authority.get("delayed_ready_task_ids")
    delayed_ready_task_ids = (
        delayed_ready_task_ids
        if isinstance(delayed_ready_task_ids, list)
        else []
    )
    delayed_frontier_admitted = bool(
        delayed_ready_task_ids
        and len(delayed_ready_task_ids) == len(set(delayed_ready_task_ids))
        and set(delayed_ready_task_ids).issubset(sealed_task_cids)
        and isinstance(authority.get("queue_entries"), Mapping)
        and set(authority["queue_entries"]) == set(delayed_ready_task_ids)
        and type(authority.get("query_started_at_ms")) is int
        and all(
            task_statuses.get(task_id) in READY_STATUSES
            and isinstance(authority["queue_entries"].get(task_id), Mapping)
            and authority["queue_entries"][task_id].get("task_cid")
            == sealed_task_cids.get(task_id)
            and type(
                authority["queue_entries"][task_id].get(
                    "retry_not_before_ms"
                )
            ) is int
            and int(
                authority["queue_entries"][task_id]["retry_not_before_ms"]
            )
            > int(authority["query_started_at_ms"])
            for task_id in delayed_ready_task_ids
        )
    )
    terminal = terminal_count == expected_tasks
    startup_active = now - launched_at <= startup_grace
    dependency_deadlock = bool(
        task_count == expected_tasks
        and not terminal
        and ready_count == 0
        and active_count == 0
        and not delayed_frontier_admitted
    )
    stuck = bool(
        lane_stalled
        or dependency_deadlock
        or (
            owner_ready
            and authority.get("available") is True
            and not terminal
            and not delayed_frontier_admitted
            and not startup_active
            and now - last_progress_at >= stale_seconds
        )
    )
    blocked = bool(blocked_count or dependency_deadlock or failure)
    bootstrap_progress = _bootstrap_authoritative_witness(paths, authority)
    admission_progress = bool(progress or bootstrap_progress or terminal)
    ready_task_ids = authority.get("ready_task_ids")
    ready_task_ids = ready_task_ids if isinstance(ready_task_ids, list) else []
    initial_ready = bootstrap.get("initial_ready_task_ids")
    initial_ready = initial_ready if isinstance(initial_ready, list) else []
    frontier_admitted = bool(
        ready_count == len(ready_task_ids)
        and len(ready_task_ids) == len(set(ready_task_ids))
        and set(ready_task_ids).issubset(sealed_task_cids)
        and (
            admission_progress
            or ready_task_ids == initial_ready
            or delayed_frontier_admitted
        )
    )
    healthy = bool(
        owner_ready
        and scheduler_alive
        and broker_ready
        and broker_samples_authenticated
        and owner_identity_admitted
        and task_authority_pair["admitted"] is True
        and source_identity_admitted
        and task_corpus_admitted
        and semantic_corpus_admitted
        and frontier_admitted
        and task_count == expected_tasks
        and goal_count == expected_goals
        and dependency_count == expected_dependencies
        and objective_count == expected_objectives == 1
        and plan_count == expected_plans == 1
        and lane_fresh
        and not blocked
        and not stuck
        and (ready_count or active_count or delayed_frontier_admitted or terminal)
        and (admission_progress or not require_authoritative_progress)
    )
    health_without_lane_admitted = bool(
        owner_ready
        and scheduler_alive
        and broker_ready
        and broker_samples_authenticated
        and owner_identity_admitted
        and task_authority_pair["admitted"] is True
        and source_identity_admitted
        and task_corpus_admitted
        and semantic_corpus_admitted
        and frontier_admitted
        and task_count == expected_tasks
        and goal_count == expected_goals
        and dependency_count == expected_dependencies
        and objective_count == expected_objectives == 1
        and plan_count == expected_plans == 1
        and not blocked
        and not stuck
        and (ready_count or active_count or delayed_frontier_admitted or terminal)
        and (admission_progress or not require_authoritative_progress)
    )
    blocked_recovery_window_seconds = min(
        stale_seconds,
        max(30.0, startup_grace),
    )
    blocked_recovery_scope = (
        "dependency_deadlock"
        if (
            blocked_count > 0
            and dependency_deadlock
            and now - last_progress_at <= blocked_recovery_window_seconds
        )
        else "parallel_startup"
        if (
            blocked_count > 0
            and startup_active
            and (
                ready_count > 0
                or active_count > 0
                or delayed_frontier_admitted
            )
        )
        else ""
    )
    blocked_recovery_admitted = bool(
        blocked_count > 0
        and blocked_recovery_scope
        and not terminal
        and not lane_stalled
        and not failure
        and owner_ready
        and scheduler_alive
        and broker_ready
        and broker_samples_authenticated
        and owner_identity_admitted
        and task_authority_pair["admitted"] is True
        and source_identity_admitted
        and task_corpus_admitted
        and semantic_corpus_admitted
        and frontier_admitted
        and task_count == expected_tasks
        and goal_count == expected_goals
        and dependency_count == expected_dependencies
        and objective_count == expected_objectives == 1
        and plan_count == expected_plans == 1
        # Fresh wrappers may not have replaced stale lane receipts yet on the
        # first post-launch samples.  The separate last-progress bound and
        # initial-health grace still cap this exception; after startup, a
        # fresh authenticated lane census is mandatory again.
        and (lane_fresh or startup_active)
        and (admission_progress or not require_authoritative_progress)
    )
    lane_active_worker_count = (
        sum(int(item["active_worker_count"]) for item in lanes)
        if lanes
        and all(type(item.get("active_worker_count")) is int for item in lanes)
        else None
    )
    receipt = {
        "schema": LIVE_STATUS_SCHEMA,
        "program_id": PROGRAM,
        "source_head": bootstrap.get("source_head"),
        "repository_tree_id": bootstrap.get("repository_tree_id"),
        "plan_root_cid": bootstrap.get("plan_root_cid"),
        "bootstrap_receipt_id": bootstrap.get("bootstrap_receipt_id"),
        "broker_authenticated": broker_samples_authenticated,
        "samples": [dict(before), dict(current)],
        "progress_evidence": [*bootstrap_progress, *progress],
        "liveness_evidence": liveness,
        "authoritative_progress_required": require_authoritative_progress,
        "authoritative_progress_admitted": admission_progress,
        "last_progress_at": last_progress_at,
        "startup_grace_active": startup_active,
        "lane_heartbeat_fresh": lane_fresh,
        "health_without_lane_admitted": health_without_lane_admitted,
        "blocked_recovery_admitted": blocked_recovery_admitted,
        "blocked_recovery_scope": blocked_recovery_scope,
        "blocked_recovery_window_seconds": (
            blocked_recovery_window_seconds
        ),
        "lane_stalled_without_active_worker": lane_stalled,
        "lane_active_worker_count": lane_active_worker_count,
        "owner_ready": owner_ready,
        "broker_ready": broker_ready,
        "owner_identity_admitted": owner_identity_admitted,
        "task_authority_pair": task_authority_pair,
        "source_identity_admitted": source_identity_admitted,
        "task_corpus_admitted": task_corpus_admitted,
        "semantic_corpus_admitted": semantic_corpus_admitted,
        "frontier_admitted": frontier_admitted,
        "delayed_frontier_admitted": delayed_frontier_admitted,
        "delayed_ready_task_ids": list(delayed_ready_task_ids),
        "task_cids": dict(sorted(task_cids.items())),
        "task_owner_bindings": dict(sorted(task_owner_bindings.items())),
        "task_dependencies": dict(sorted(task_dependencies.items())),
        "task_authority_spec_cids": dict(
            sorted(task_authority_spec_cids.items())
        ),
        "goal_records": dict(sorted(goal_records.items())),
        "goal_edges": list(goal_edges),
        "plan_record": dict(plan_record),
        "objective_record": dict(objective_record),
        "scheduler_alive": scheduler_alive,
        "healthy": healthy,
        "blocked": blocked,
        "stuck": stuck,
        "dependency_deadlock": dependency_deadlock,
        "terminal": terminal,
        "failure": dict(failure),
        "observed_at": now,
    }
    receipt["receipt_cid"] = _identity(receipt)
    return receipt


def _await_initial_health(
    board: Any,
    paths: Mapping[str, Path],
    server: Any,
    scheduler: subprocess.Popen[Any],
    *,
    launched_at: float,
    failure: Mapping[str, Any],
    failure_event: threading.Event,
    shutdown_requested: threading.Event,
    received_signal: Mapping[str, int],
) -> tuple[dict[str, Any], float]:
    first = _status_sample(board, paths, server, scheduler)
    last_progress_at = launched_at
    timeout = min(
        600.0,
        max(
            5.0,
            float(board.payload.get("watchdog_startup_grace_seconds") or 300.0),
        ),
    )
    deadline = time.monotonic() + timeout
    blocked_recovery_observed = False
    while time.monotonic() < deadline:
        if shutdown_requested.is_set():
            raise OperatorStopRequested(
                int(received_signal.get("signum") or signal.SIGTERM)
            )
        if scheduler.poll() is not None:
            _record_control_failure(
                paths, failure, failure_event,
                reason_code="scheduler_exited_before_health_admission",
                error_type="ASEHForegroundSchedulerExit",
            )
            raise OperatorError("scheduler exited before health admission")
        if failure_event.wait(STATUS_SAMPLE_INTERVAL_SECONDS):
            raise OperatorError("control failure occurred before health admission")
        if shutdown_requested.is_set():
            raise OperatorStopRequested(
                int(received_signal.get("signum") or signal.SIGTERM)
            )
        second = _status_sample(board, paths, server, scheduler)
        if _authoritative_progress_between(first, second):
            last_progress_at = float(second["observed_at"])
        receipt = _health_receipt(
            board, paths, samples=(first, second), launched_at=launched_at,
            last_progress_at=last_progress_at, failure=failure,
            require_authoritative_progress=True,
        )
        _atomic_json(paths["status_receipt"], receipt)
        if receipt.get("blocked") is True or receipt.get("stuck") is True:
            if receipt.get("blocked_recovery_admitted") is True:
                # The receipt is the bounded recovery authority.  It already
                # binds freshness, the configured recovery window, owner and
                # broker authority, sealed identities/corpora, lane safety,
                # and authoritative progress.  Do not replace that admission
                # with a shorter wall-clock or sample-count heuristic.
                blocked_recovery_observed = True
                first = second
                continue
            _record_control_failure(
                paths, failure, failure_event,
                reason_code=(
                    "authoritative_blocked_recovery_grace_exhausted"
                    if blocked_recovery_observed
                    else "authoritative_board_blocked"
                    if receipt.get("blocked") is True
                    else "authoritative_board_stuck"
                ),
                error_type="ASEHHealthGateFailure",
            )
            raise OperatorError("foreground health admission failed closed")
        prior_authority = first.get("authority")
        current_authority = second.get("authority")
        if not (
            isinstance(prior_authority, Mapping)
            and prior_authority.get("available") is True
        ) and not (
            isinstance(current_authority, Mapping)
            and current_authority.get("available") is True
        ):
            _record_control_failure(
                paths,
                failure,
                failure_event,
                reason_code="authoritative_status_unavailable_two_samples",
                error_type="ASEHHealthQueryFailure",
            )
            raise OperatorError(
                "two consecutive authoritative status samples unavailable"
            )
        if receipt.get("healthy") is True:
            return receipt, last_progress_at
        first = second
    _record_control_failure(
        paths, failure, failure_event,
        reason_code="foreground_health_admission_timeout",
        error_type="ASEHHealthAdmissionTimeout",
    )
    raise OperatorError("two-sample foreground health admission timed out")


def _post_admission_health_action(
    receipt: Mapping[str, Any],
    *,
    prior_available: bool,
    current_available: bool,
    unhealthy_edges: int,
) -> tuple[str, str, int]:
    """Return a bounded fail/stop/continue decision for an admitted launch."""

    if receipt.get("blocked") is True:
        if receipt.get("blocked_recovery_admitted") is True:
            # The authoritative receipt supplies the recovery bound.  Keep
            # the transient outage counter independent so an admitted repair
            # period cannot consume a later one/two-sample outage allowance.
            return "continue", "", 0
        return "fail", "authoritative_board_blocked", unhealthy_edges
    if receipt.get("stuck") is True:
        return "fail", "authoritative_board_stuck", unhealthy_edges
    if receipt.get("scheduler_alive") is not True:
        return "fail", "authoritative_scheduler_not_live", unhealthy_edges
    if receipt.get("owner_ready") is not True:
        return "fail", "authoritative_owner_not_ready", unhealthy_edges
    if receipt.get("broker_ready") is not True:
        return "fail", "authoritative_broker_not_ready", unhealthy_edges
    if not prior_available and not current_available:
        return (
            "fail",
            "authoritative_status_unavailable_two_samples",
            unhealthy_edges,
        )
    if receipt.get("terminal") is True:
        if receipt.get("healthy") is True:
            return "stop", "", 0
        return "fail", "authoritative_terminal_not_admitted", unhealthy_edges
    if receipt.get("healthy") is True:
        return "continue", "", 0
    if prior_available and current_available:
        # Lane census can flicker while claims continue.  Bound that to the
        # same recovery edges used for a single missing authority sample,
        # but keep semantic or identity loss fail-closed.
        lane_only_loss = bool(
            receipt.get("health_without_lane_admitted") is True
            and receipt.get("lane_heartbeat_fresh") is False
        )
        if lane_only_loss:
            next_edges = unhealthy_edges + 1
            if next_edges > 2:
                return (
                    "fail",
                    "authoritative_health_admission_lost",
                    next_edges,
                )
            return "continue", "", next_edges
        return "fail", "authoritative_health_admission_lost", unhealthy_edges
    next_edges = unhealthy_edges + 1
    if next_edges > 2:
        return (
            "fail",
            "authoritative_status_recovery_grace_exhausted",
            next_edges,
        )
    return "continue", "", next_edges


def _status_monitor_loop(
    board: Any,
    paths: Mapping[str, Path],
    server: Any,
    scheduler: subprocess.Popen[Any],
    *,
    launched_at: float,
    previous: Mapping[str, Any],
    last_progress_at: float,
    stop: threading.Event,
    failure: dict[str, Any],
    failure_event: threading.Event,
) -> None:
    interval = min(
        30.0,
        max(1.0, float(board.payload.get("check_interval_seconds") or 10.0)),
    )
    prior = dict(previous)
    unhealthy_edges = 0
    while not stop.wait(interval):
        try:
            current = _status_sample(board, paths, server, scheduler)
            if _authoritative_progress_between(prior, current):
                last_progress_at = float(current["observed_at"])
            receipt = _health_receipt(
                board, paths, samples=(prior, current), launched_at=launched_at,
                last_progress_at=last_progress_at, failure=failure,
            )
            _atomic_json(paths["status_receipt"], receipt)
            prior_authority = prior.get("authority")
            current_authority = current.get("authority")
            prior_available = (
                isinstance(prior_authority, Mapping)
                and prior_authority.get("available") is True
            )
            current_available = (
                isinstance(current_authority, Mapping)
                and current_authority.get("available") is True
            )
            action, reason_code, unhealthy_edges = (
                _post_admission_health_action(
                    receipt,
                    prior_available=prior_available,
                    current_available=current_available,
                    unhealthy_edges=unhealthy_edges,
                )
            )
            if action == "stop":
                return
            if action == "fail":
                _record_control_failure(
                    paths, failure, failure_event,
                    reason_code=reason_code,
                    error_type=(
                        "ASEHHealthQueryFailure"
                        if reason_code.startswith("authoritative_status_")
                        else "ASEHHealthGateFailure"
                    ),
                )
                return
            prior = current
        except Exception as exc:
            _record_control_failure(
                paths, failure, failure_event,
                reason_code="health_monitor_failed",
                error_type=type(exc).__name__,
            )
            return


def _read_live_status_receipt(
    board: Any, paths: Mapping[str, Path]
) -> tuple[dict[str, Any], float]:
    path = paths["status_receipt"]
    payload = _secure_runtime_json(path, max_bytes=STATUS_RECEIPT_MAX_BYTES)
    if payload.get("schema") != LIVE_STATUS_SCHEMA or payload.get("program_id") != PROGRAM:
        raise OperatorError("live status receipt identity differs")
    unsigned = dict(payload)
    receipt_cid = unsigned.pop("receipt_cid", "")
    if receipt_cid != _identity(unsigned):
        raise OperatorError("live status receipt CID is invalid")
    if not isinstance(payload.get("samples"), list) or len(payload["samples"]) != 2:
        raise OperatorError("live status receipt is not a two-sample observation")
    bootstrap = _secure_runtime_json(
        paths["bootstrap_receipt"], max_bytes=STATUS_RECEIPT_MAX_BYTES
    )
    _bootstrap_receipt_id(bootstrap)
    for field in (
        "source_head", "repository_tree_id", "plan_root_cid", "bootstrap_receipt_id",
    ):
        if payload.get(field) != bootstrap.get(field):
            raise OperatorError(f"live status receipt has stale {field}")
    age = max(0.0, time.time() - float(payload.get("observed_at") or 0.0))
    max_age = min(
        60.0,
        max(15.0, 3.0 * float(board.payload.get("check_interval_seconds") or 10.0)),
    )
    if age > max_age:
        raise OperatorError("live status receipt is stale")
    return payload, age


def _state_owner_process_birth_id(
    process_birth: Mapping[str, Any],
) -> str:
    """Derive StateServerIdentity.process_birth_id from canonical fields."""

    material = (
        f"{process_birth['pid']}:{process_birth['start_time_ticks']}:"
        f"{process_birth['boot_id']}:{process_birth['parent_pid']}"
    )
    return "birth:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _owner_incarnation_binding(
    owner_status: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Return the exact live owner incarnation and published replica."""

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    identity = owner_status.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    process_birth = identity.get("process_birth")
    process_birth = process_birth if isinstance(process_birth, Mapping) else {}
    fields = (
        "server_id", "store_id", "database_uuid", "schema_revision",
        "schema_fingerprint", "generation", "process_birth_id", "listen_uri",
        "extension_fingerprint",
    )
    selected = {field: identity.get(field) for field in fields}
    try:
        birth = ProcessBirthIdentity.from_dict(process_birth)
    except (TypeError, ValueError) as exc:
        raise OperatorError(
            "live owner process-birth identity is invalid"
        ) from exc
    birth_payload = birth.to_dict()
    derived_birth_id = _state_owner_process_birth_id(birth_payload)
    if (
        any(value in (None, "") for value in selected.values())
        or dict(process_birth) != birth_payload
        or birth.pid <= 0
        or birth.start_time_ticks <= 0
        or selected["process_birth_id"] != derived_birth_id
        or owner_liveness(birth) is not OwnerLiveness.ALIVE
    ):
        raise OperatorError(
            "live owner incarnation identity is incomplete or not alive"
        )
    return {
        "identity": selected,
        "process_birth": birth_payload,
        "replica": _published_replica_binding(owner_status, paths),
    }


def _published_replica_is_current_or_monotonic_successor(
    sampled: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Admit exact sampled bytes or a later replica from the same owner.

    A status query refreshes the non-authoritative read replica.  The refresh
    can complete after the health sampler publishes its final receipt but
    before the operator reads the owner's status document.  Treating that
    strictly newer publication as identity drift makes an otherwise healthy
    owner impossible to observe without winning a filesystem race.

    The sequence is only an ordering witness inside one fully matched replica
    authority.  Equal sequences still require byte-for-byte binding equality;
    rollbacks, identity changes, path changes, and schema/storage drift all
    fail closed.
    """

    authority_fields = (
        "path",
        "source_database_path",
        "server_id",
        "database_uuid",
        "generation",
        "schema_revision",
        "schema_fingerprint",
        "storage_schema_fingerprint",
    )
    if any(
        sampled.get(field) != current.get(field)
        for field in authority_fields
    ):
        return False
    sampled_sequence = sampled.get("refresh_sequence")
    current_sequence = current.get("refresh_sequence")
    if type(sampled_sequence) is not int or type(current_sequence) is not int:
        return False
    if current_sequence < sampled_sequence:
        return False
    if current_sequence == sampled_sequence:
        return dict(sampled) == dict(current)
    return True


def _admit_receipt_for_current_owner(
    receipt: Mapping[str, Any],
    owner_status: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Bind a two-sample health receipt to the current owner incarnation."""

    samples = receipt.get("samples")
    if not isinstance(samples, list) or len(samples) != 2:
        raise OperatorError("live status receipt lacks two owner samples")
    current = _owner_incarnation_binding(owner_status, paths)
    admitted_samples: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise OperatorError("live status receipt owner sample is invalid")
        sampled_status = sample.get("owner_status")
        sampled_status = (
            sampled_status if isinstance(sampled_status, Mapping) else {}
        )
        sampled = _owner_incarnation_binding(sampled_status, paths)
        authority = sample.get("authority")
        authority = authority if isinstance(authority, Mapping) else {}
        authority_binding = authority.get("owner_binding")
        authority_binding = (
            authority_binding
            if isinstance(authority_binding, Mapping)
            else {}
        )
        if (
            sampled["identity"] != current["identity"]
            or sampled["process_birth"] != current["process_birth"]
            or authority_binding != current["identity"]
        ):
            raise OperatorError(
                "live status receipt belongs to a different owner incarnation"
            )
        admitted_samples.append(sampled)
    # A task mutation may legitimately replace the replica between samples,
    # and the read-side status query itself may publish a strict monotonic
    # successor after the final sample.  Equal sequence values remain exact;
    # only a later publication by this same owner incarnation is admissible.
    if not _published_replica_is_current_or_monotonic_successor(
        admitted_samples[-1]["replica"], current["replica"]
    ):
        raise OperatorError(
            "live status receipt belongs to a different published replica"
        )
    return current


def status(config_path: Path, *, require_ready: bool) -> tuple[int, dict[str, Any]]:
    board, _config = _load(config_path)
    paths = _paths(board)
    owner_status: dict[str, Any] = {}
    status_path = paths["owner"] / "quack-state-server.status.json"
    if status_path.is_file():
        try:
            owner_status = _secure_runtime_json(
                status_path, max_bytes=STATUS_RECEIPT_MAX_BYTES
            )
        except Exception:
            owner_status = {}
    try:
        receipt, age = _read_live_status_receipt(board, paths)
        _admit_receipt_for_current_owner(receipt, owner_status, paths)
        receipt_available = True
        receipt_error: dict[str, Any] = {}
    except Exception as exc:
        receipt = {}
        age = float("inf")
        receipt_available = False
        receipt_error = {
            "error_type": type(exc).__name__,
            "reason": "live_status_receipt_unavailable_or_invalid",
        }
    owner_ready = owner_status.get("lifecycle") == "ready"
    healthy = bool(
        receipt_available and owner_ready and receipt.get("healthy") is True
    )
    report = {
        "schema": LIVE_STATUS_SCHEMA,
        "program_id": PROGRAM,
        "owner_ready": owner_ready,
        "owner_status": owner_status,
        "broker_authenticated_receipt": bool(
            receipt_available and receipt.get("broker_authenticated") is True
        ),
        "receipt_age_seconds": age if receipt_available else None,
        "receipt": receipt,
        "receipt_error": receipt_error,
        "healthy": healthy,
        "blocked": bool(receipt.get("blocked", False)),
        "stuck": bool(receipt.get("stuck", False)),
        "terminal": bool(receipt.get("terminal", False)),
        "observed_at": time.time(),
    }
    return (0 if healthy or not require_ready else 1), report


def preflight(config_path: Path) -> tuple[int, dict[str, Any]]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        preflight_configured_board,
    )

    board, _config = _load(config_path)
    report = preflight_configured_board(board)
    return (0 if report.get("valid") is True else 1), dict(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("materialize")
    commands.add_parser("authorize-repair-transition")
    commands.add_parser("preflight")
    run = commands.add_parser("run")
    run.add_argument("--implement", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--duration-seconds", type=float, default=float("inf"))
    show = commands.add_parser("status")
    show.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "materialize":
            payload = materialize(args.config)
            code = 0
        elif args.command == "authorize-repair-transition":
            payload = authorize_repair_transition(args.config)
            code = 0
        elif args.command == "preflight":
            code, payload = preflight(args.config)
        elif args.command == "status":
            code, payload = status(args.config, require_ready=args.require_ready)
        else:
            return run_supervisor(
                args.config,
                implement=bool(args.implement),
                duration=float(args.duration_seconds),
            )
    except (OperatorError, OSError, RuntimeError, ValueError) as exc:
        payload = {
            "schema": OPERATOR_SCHEMA, "command": args.command, "ok": False,
            "error_type": type(exc).__name__, "error": str(exc),
        }
        code = 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
