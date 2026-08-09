"""Fail-closed validation for the prompt-only v3 convergence bootstrap.

ASE3-000 deliberately treats historical ASE/ASE2 state as evidence, never as
completion authority.  This module validates the bounded evidence packet that
binds v3 work to a current-main seed, accounts for every rescue-branch commit
and changed path, records historical state contradictions, and proves that the
dirty source checkout was not used as the integration worktree.

The configured-board preflight invokes this module with ``--check-all``.  The
command always prints one JSON object containing at least ``valid`` and
``errors`` and exits non-zero when any check fails.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

CURRENT_MAIN_BASELINE_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-current-main-baseline@1"
)
HISTORICAL_CONTRADICTION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-historical-contradictions@1"
)
RESCUE_DISPOSITION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-rescue-dispositions@1"
)
CLEAN_WORKTREE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-clean-worktree-receipt@1"
)
CONVERGENCE_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-convergence-manifest@1"
)
CONVERGENCE_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-convergence-report@1"
)
CONVERGENCE_MANIFEST_CREATED_AT: Final = "2026-08-08T17:56:14Z"
POST_WAVE3_RESIDUAL_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.post-wave3-residual-report@1"
)
FALSE_COMPLETION_RECOVERY_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.false-completion-recovery@1"
)
PROVIDER_FALLBACK_POLICY_AUTHORIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.provider-fallback-policy-authorization@1"
)
ASE3_019_ATTEMPT2_SELF_HOST_INCIDENT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.ase3-019-attempt2-self-host-incident@1"
)
ASE3_019_ATTEMPT2_EVENT_BUNDLE_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.ase3-019-attempt2-event-bundle@1"
)

BOARD_NAMESPACE: Final = "agent-supervisor-prompt-only-self-improvement-v3"
POST_WAVE3_RESIDUAL_FILENAME: Final = "post_wave3_residuals_20260808.json"
FALSE_COMPLETION_RECOVERY_FILENAME: Final = (
    "false_completion_recovery_20260808.json"
)
FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME: Final = (
    "false_completion_merge_receipt_ase3_006_20260808.json"
)
FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME: Final = (
    "false_completion_merge_receipt_ase3_018_20260808.json"
)
FAILED_VALIDATION_EVENT_019_FILENAME: Final = (
    "failed_validation_event_ase3_019_20260808.json"
)
PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME: Final = (
    "provider_fallback_policy_authorization_20260808.json"
)
FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME: Final = (
    "failed_pre_dispatch_event_ase3_019_attempt_2_20260808.json"
)
FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME: Final = (
    "failed_pre_dispatch_log_ase3_019_attempt_2_20260808.txt"
)
SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME: Final = (
    "self_host_seed_failure_ase3_019_attempt_2_20260808.json"
)
OPERATOR_SALVAGE_RECEIPT_019_FILENAME: Final = (
    "operator_salvage_receipt_ase3_019_20260808.json"
)
JSON_ARTIFACT_FILENAMES: Final = (
    "current_main_baseline.json",
    "historical_state_contradictions.json",
    "rescue_artifact_dispositions.json",
    "clean_integration_worktree_receipt.json",
    POST_WAVE3_RESIDUAL_FILENAME,
    FALSE_COMPLETION_RECOVERY_FILENAME,
    FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
    FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
    FAILED_VALIDATION_EVENT_019_FILENAME,
    PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME,
    FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME,
    SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME,
)
TEXT_ARTIFACT_FILENAMES: Final = (FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME,)
ARTIFACT_FILENAMES: Final = (*JSON_ARTIFACT_FILENAMES, *TEXT_ARTIFACT_FILENAMES)
MANIFEST_FILENAME: Final = "convergence_manifest.json"
DEFAULT_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
PROMPT_V3_TASKBOARD_RELATIVE_PATH: Final = Path(
    "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md"
)
PROMPT_V3_OBJECTIVES_RELATIVE_PATH: Final = Path(
    "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.objectives.md"
)
PROMPT_V3_PLAN_RELATIVE_PATH: Final = Path(
    "docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md"
)
PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH: Final = Path(
    "config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json"
)
PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME: Final = (
    "provider_attempt_daemon_reload_receipt.json"
)
PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
)
PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME: Final = (
    "protected_runtime_activation_receipt.json"
)
PROTECTED_RUNTIME_ACTIVATION_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME
)
HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME: Final = (
    "hermetic_control_plane_identity_acceptance_receipt.json"
)
HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME
)
HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.ase3-030-hermetic-identity-acceptance@1"
)
DEFAULT_ARTIFACT_ROOT: Final = (
    DEFAULT_REPOSITORY_ROOT
    / "data"
    / "agent_supervisor"
    / "prompt_only_self_improvement_v3"
    / "convergence"
)
MAX_EVIDENCE_SNAPSHOT_BYTES: Final[int] = 1_048_576
_EVIDENCE_SNAPSHOT_BYTE_BOUNDS: Final = {
    FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME: 64 * 1024,
    FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME: 8 * 1024,
    SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME: 32 * 1024,
}
_EVIDENCE_READ_CHUNK_BYTES: Final[int] = 64 * 1024
_TASK_TITLE_KEY: Final = "__task_title__"
_TASK_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-identity@1"
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_TASK_IDS: Final = frozenset(f"ASE3-{index:03d}" for index in range(15))
_PROGRAM_CANONICAL_TASK_IDS: Final = (
    "ASE3-000",
    "ASE3-001",
    "ASE3-002",
    "ASE3-003",
    "ASE3-004",
    "ASE3-005",
    "ASE3-006",
    "ASE3-007",
    "ASE3-008",
    "ASE3-009",
    "ASE3-010",
    "ASE3-011",
    "ASE3-012",
    "ASE3-013",
    "ASE3-014",
    "ASE3-018",
    "ASE3-019",
    "ASE3-020",
    "ASE3-021",
    "ASE3-023",
    "ASE3-024",
    "ASE3-025",
    "ASE3-026",
    "ASE3-027",
    "ASE3-028",
    "ASE3-029",
    "ASE3-030",
)
_PROGRAM_NONCANONICAL_TASK_IDS: Final = (
    "ASE3-015",
    "ASE3-016",
    "ASE3-017",
    "ASE3-022",
)
_DISPOSITIONS: Final = frozenset({"port", "rewrite", "superseded", "discard"})
_REQUIRED_CONTRADICTIONS: Final = frozenset(
    {
        "source-board-vs-eligible-index",
        "bundle-index-vs-eligible-index",
        "stale-process-projections",
        "drained-without-refill",
        "branch-local-completion",
    }
)
_POST_WAVE3_CREATED_AT: Final = "2026-08-08T09:53:00Z"
_POST_WAVE3_REPOSITORY: Final = {
    "head": "4370931d7dc556d56962a88ed1db511487be39d2",
    "tree": "1d472b508368a0574e1dbfa87467158377797e23",
    "branch": "agent/prompt-self-improvement-v3",
}
_POST_WAVE3_COMPLETED_TASKS: Final = {
    "ASE3-005": {
        "implementation_commit": "8b82c968d829a1191fcacff3e20804be0c232b0a",
        "merge_commit": "8945d1b08e564fb1baf26a38d7ea6909012a104b",
        "status_commit": "4370931d7dc556d56962a88ed1db511487be39d2",
        "declared_current_tree_tests_passed": 13,
        "declared_current_tree_tests_failed": 0,
    },
    "ASE3-007": {
        "implementation_commit": "5c4098a8adf7c29e24602a18b699f9042b3ca9f6",
        "merge_commit": "023bb9972ca8d9eb6009f565c3293c2ce8a16aea",
        "status_commit": "05773ac5abcf361a870404428f4e82dcd15168ce",
        "declared_current_tree_tests_passed": 87,
        "declared_current_tree_tests_failed": 0,
    },
}
_POST_WAVE3_RESIDUALS: Final = {
    "trusted-context-canonical-composition": (
        "ASE3-018",
        frozenset({"ASE3-001", "ASE3-002", "ASE3-005"}),
    ),
    "signed-authority-and-durable-provider-attempt": (
        "ASE3-019",
        frozenset({"ASE3-002", "ASE3-006"}),
    ),
    "production-durable-refill-wiring": (
        "ASE3-021",
        frozenset({"ASE3-007"}),
    ),
    "transactional-run-truth-and-effect-recovery": (
        "ASE3-020",
        frozenset({"ASE3-003", "ASE3-005", "ASE3-007"}),
    ),
}
_POST_WAVE3_PROVIDER_INCIDENT: Final = {
    "task_id": "ASE3-006",
    "event_id": "sha256:e2dee32eb866a9a4216c809318f4066bc49bf33e1e0ef3290365cf4ccaf58f97",
    "log_sha256": "sha256:2724af1a5b52fadae7130b4a80081cf9849dabc0f0104f839033474fff332596",
    "failure": "grok_authentication_unavailable",
    "attempt": 1,
    "attempt_consumed": False,
    "fallback_dispatched": False,
    "workspace_changed": False,
    "operator_fenced_before_retry": True,
}
_POST_WAVE3_DISPOSITION: Final = {
    "historical_task_status_authoritative": False,
    "declared_test_success_authorizes_goal_completion": False,
    "operator_reviewed_refill_required": True,
    "target_tasks": ["ASE3-018", "ASE3-019", "ASE3-021", "ASE3-020"],
    "gate_task": "ASE3-008",
    "completion_authority": False,
    "provider_policy_broadening_authorized": False,
    "attempt_counter_mutation_authorized": False,
}
_FALSE_COMPLETION_RECOVERY_CREATED_AT: Final = "2026-08-08T16:35:00Z"
_FALSE_COMPLETION_RECOVERY_SOURCE: Final = {
    "branch": "agent/prompt-self-improvement-v3",
    "launch_stamp": "20260808T162057Z",
    "launch_base_head": "0c40afb32f9b95ca54d73b18e06a4a2c193469f7",
    "launch_base_tree": "917a307ed4f4854a8f9cfb74290e8a475e831d08",
    "recovery_parent_head": "733e63333f992477d091449319869e23912d4f9c",
    "recovery_parent_tree": "641c15fd65d64953b3fbf5971454fcc56143c4c6",
    "protected_parent_blobs": {
        "config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json": (
            "867dbb193029f28ac5d9face7c99f7c9cbeb63b0"
        ),
        "docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md": (
            "7dd1310ab239b6da56be704d4e475d4784809a0e"
        ),
        "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.objectives.md": (
            "7b9b1e86a9d5aeb4887dd53094f542b5624684d4"
        ),
        "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md": (
            "3765d05d04cde1866eedf1f6c7f5d22732a68ca9"
        ),
    },
}
_FALSE_COMPLETION_RECORDS: Final = {
    "ASE3-006": {
        "canonical_task_cid": (
            "baguqeeraz5pve2rmjuvo6qivduhrvf4o6nrclgu2jtdjo3kll7aqpnyipzkq"
        ),
        "attempt": 1,
        "implementation_commit": "159b298910a11ba4adbafa3c0192a9585639c53a",
        "implementation_tree": "f91558d857ca640066fd33637ebf126dbf442250",
        "merge_commit": "9d8f1062583f4e7b717ac535878716e04f2d7577",
        "status_commit": "78eeaad86de70b61d5cc03940aa043c84fd441d8",
        "source_merge_receipt_path": (
            "data/agent_supervisor/prompt_only_self_improvement_v3/live/"
            "merge-queue/train/receipts/"
            "90732ac34f5c00e9dfeeeee018d137ae9d057084caa64e55a32c64e9adf2c797.json"
        ),
        "merge_receipt_snapshot": FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
        "merge_receipt_sha256": (
            "sha256:838237cd8c8c5fbc47e4ca989d5e2e9f0a27aa7a8b1f9a3ba2552b67048acf0d"
        ),
        "acceptance_authoritative": False,
        "findings": [
            "adaptive scheduler and compiler have no production scheduler or runtime consumer",
            "execution_plan defines a duplicate InvocationBudget instead of consuming entrypoints.contracts.InvocationBudget",
            "standalone SQLite ledger duplicates production task-source and claim authority",
            "whole-plan and exact-slice restart adoption are not integrated or proven",
        ],
        "repair_task": "ASE3-023",
        "repair_goal": "ASE3-G040",
        "repair_strict_shard": 2,
    },
    "ASE3-018": {
        "canonical_task_cid": (
            "baguqeeraifrriecjdwkl2yz266asjxkgkeleqgq2uonuc5mxugyuuojeeehq"
        ),
        "attempt": 1,
        "implementation_commit": "23bf69ff8517de100f5cd4918e479af3667ec70a",
        "implementation_tree": "5a8be38132e792684dc6ab7de203998238bc9bd5",
        "merge_commit": "518830810013fde1b13599591146356001aa774a",
        "status_commit": "733e63333f992477d091449319869e23912d4f9c",
        "source_merge_receipt_path": (
            "data/agent_supervisor/prompt_only_self_improvement_v3/live/"
            "merge-queue/train/receipts/"
            "27843f980b7b5687fba68692b159f05b2cbdbc981a972e1ae0b957b6ad69d3ec.json"
        ),
        "merge_receipt_snapshot": FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
        "merge_receipt_sha256": (
            "sha256:aee69ba91156b9a6ec9ed75a191ec579d6389a4b4298a2a93af777840612f85b"
        ),
        "acceptance_authoritative": False,
        "findings": [
            "production defaults still resolve only repository instead of all nine required launch fields",
            "prefilled context values bypass real leaf-resolver composition",
            "profile authorization trusts signature shape and a caller profile_signed boolean",
            "caller-constructible UCAN evidence is treated as verified without cryptographic attenuation",
            "mixed mapping keys still raise TypeError in production receipt identity",
        ],
        "repair_task": "ASE3-027",
        "repair_goal": "ASE3-G020",
        "repair_strict_shard": 0,
    },
}
_FALSE_COMPLETION_FAILED_ATTEMPT: Final = {
    "task_id": "ASE3-019",
    "canonical_task_cid": (
        "baguqeeraw5jsn2ffbxmdxjvktktdfdbkxu7didced4edylidsa4hj44qyz2a"
    ),
    "attempt": 1,
    "implementation_commit": "eb68ff2a20e0719388f60ffef1f5bfcb90b79263",
    "implementation_tree": "695e2d6f07bc1c48bdc34ebb490342444de2cbef",
    "failed_event_id": (
        "sha256:6b6482b68fef226ab3bc631cb722de49ccfb863cc5350fdf91a79ac1e34cfce4"
    ),
    "failed_event_snapshot": FAILED_VALIDATION_EVENT_019_FILENAME,
    "failed_event_snapshot_sha256": (
        "sha256:df2cb757d0330996c3a586acfd649fdfcf7d76758bddd3c35689ea4998a1115e"
    ),
    "rescue_branch": (
        "rescue/ase3-019-b75326e8a50d-attempt-1-1786206062-failed-validation"
    ),
    "validation_returncode": 1,
    "merge_dispatched": False,
    "attempt_counter_mutation_authorized": False,
    "continuation": "same_identity_attempt_2_with_prior_attempt_seed",
    "retry_strict_shard": 1,
}
_FALSE_COMPLETION_FENCE: Final = {
    "master_pid": 1009686,
    "supervisor_pids": [1009840, 1009841, 1009842],
    "daemon_pids": [1012621, 1010240, 1010193],
    "shutdown_signal": "SIGTERM",
    "lane_statuses": ["stopped", "stopped", "stopped"],
    "lane_restart_counts": [0, 0, 0],
    "zero_owned_processes": True,
    "zero_scoped_provider_containers": True,
    "active_attempts_cleared": True,
}
_FALSE_COMPLETION_DISPOSITION: Final = {
    "completion_authority": False,
    "old_completion_satisfies_repair": False,
    "runtime_state_mutation_authorized": False,
    "attempt_counter_mutation_authorized": False,
    "queue_history_mutation_authorized": False,
    "legacy_refill_enablement_authorized": False,
    "repair_tasks": ["ASE3-023", "ASE3-027"],
    "retry_task": "ASE3-019",
    "reload_gate": "ASE3-022",
    "reload_gate_must_remain_blocked": True,
}
_ASE3_019_ATTEMPT2_CREATED_AT: Final = "2026-08-08T17:56:14Z"
_ASE3_019_ATTEMPT2_TASK_CID: Final = (
    "baguqeeraw5jsn2ffbxmdxjvktktdfdbkxu7didced4edylidsa4hj44qyz2a"
)
_ASE3_019_ATTEMPT2_TASK_KEY: Final = (
    "task/v1/b75326e8a50dd83ba6aa9aa6328c2abd3e340c441f083c2d03903874f390c674"
)
_ASE3_019_ATTEMPT2_EVENT_ID: Final = (
    "sha256:e0c223f22824466570825cae33c68ef5baceeaae267e7a968bbea13d5b2e9682"
)
_ASE3_019_ATTEMPT2_EVENT_SHA256: Final = (
    "sha256:8dfe081d00135789e7c6b6969125f643a4913e3227b2ad3d6a6fa76747ad1d62"
)
_ASE3_019_ATTEMPT2_LOG_SHA256: Final = (
    "sha256:24adec0b6d3cd97badb5586d26b7f17bdbaaa9851cfb344ebcf030773c50744e"
)
_ASE3_019_ATTEMPT2_INCIDENT_SHA256: Final = (
    "sha256:ae9e3e5349db4b0c21ac429259ee0c51ed95a4232825b933929d862080ce61b1"
)
_ASE3_019_ATTEMPT2_SEED_EVENT_ID: Final = (
    "sha256:6f69feb3b06607060c5d69e790066addf9434c7bbda66f51f83c0cb5b4357f09"
)
_ASE3_019_ATTEMPT2_STARTED_EVENT_ID: Final = (
    "sha256:190828f5596ac57a1790f87131dff3cd4db3bf07e4f6f32c2b59751d34f7e0aa"
)
_ASE3_019_ATTEMPT2_SHUTDOWN_EVENT_ID: Final = (
    "sha256:878bc907f4ee2f844c245cc7a0677496b87dd9b59e21aef4651929b8845aaefb"
)
_ASE3_019_ATTEMPT2_LAUNCH: Final = {
    "branch": "agent/prompt-self-improvement-v3",
    "launch_stamp": "20260808T173004Z",
    "launch_head": "e6f8e4a7771907372fc93b0f35cfde30170c2b2a",
    "launch_tree": "62a5b820bafa6c43387a2047b87e8c48941c83dc",
    "master_pid": 2329325,
    "supervisor_pids": [2330228, 2330230, 2330232],
    "daemon_pids": [2332798, 2330758, 2330811],
}
_ASE3_019_ATTEMPT2_BRANCH: Final = (
    "implementation/ase3-019-b75326e8a50d-attempt-2-1786210208"
)
_ASE3_019_ATTEMPT2_REPLAYED_PATHS: Final = (
    "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/llm_router.py",
    "test/api/test_agent_supervisor_prompt_v3_authority_hardening.py",
    "test/api/test_llm_router_agent_supervisor_fallback_route.py",
)
_ASE3_019_ATTEMPT2_PRIOR_SEED: Final = {
    "source_attempt": 1,
    "source_commit": "eb68ff2a20e0719388f60ffef1f5bfcb90b79263",
    "source_tree": "695e2d6f07bc1c48bdc34ebb490342444de2cbef",
    "source_rescue_ref": (
        "rescue/ase3-019-b75326e8a50d-attempt-1-1786206062-failed-validation"
    ),
    "merge_base": "0c40afb32f9b95ca54d73b18e06a4a2c193469f7",
    "attempt_2_branch": _ASE3_019_ATTEMPT2_BRANCH,
    "seed_event_id": _ASE3_019_ATTEMPT2_SEED_EVENT_ID,
    "started_event_id": _ASE3_019_ATTEMPT2_STARTED_EVENT_ID,
    "binary_full_index_delta_sha256": (
        "sha256:0dca974830907318ccc8b056e2fd190773b608082b91458bdce9b9393c904403"
    ),
    "replayed_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
}
_ASE3_019_ATTEMPT2_ACCEPTED_BLOBS: Final = {
    "ipfs_accelerate_py/agent_supervisor/__init__.py": (
        "346c809c0457f0d612d378672abbdb0324de1f47"
    ),
    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py": (
        "5774ec9fe78f7b80decb51b82d57cab775d7a615"
    ),
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py": (
        "70ae5eb357888b50cec3267aaac7281661425b93"
    ),
    "ipfs_accelerate_py/llm_router.py": (
        "0d974f6f085c05979470b34d156f7d4170f2df92"
    ),
}
_ASE3_019_ATTEMPT2_CANDIDATE_BLOBS: Final = {
    "ipfs_accelerate_py/agent_supervisor/__init__.py": (
        "346c809c0457f0d612d378672abbdb0324de1f47"
    ),
    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py": (
        "06c6ef1eeca773efc6173130597d44d2664370b4"
    ),
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py": (
        "bb5e7c469a47d2fb2f2cab287a17d182dff95f63"
    ),
    "ipfs_accelerate_py/llm_router.py": (
        "f4c27ebe41c0968529ebb933a8d7a7bc8315b1f5"
    ),
}
_ASE3_019_OPERATOR_SALVAGE_REQUIRED_FIELDS: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "task",
    "incident",
    "authority",
    "source_candidate",
    "salvage_base",
    "implementation",
    "merge",
    "validation",
    "review",
    "accepted_control_plane",
    "denials",
)
_ASE3_019_ATTEMPT2_NORMALIZED_ERROR: Final = (
    "agent implementation route binding fields are invalid"
)
_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID: Final = "ASE3-022"
_PROVIDER_ATTEMPT_RELOAD_GATE_DEPENDENCIES: Final = (
    "ASE3-006",
    "ASE3-018",
    "ASE3-019",
    "ASE3-023",
    "ASE3-027",
)
_PROVIDER_ATTEMPT_RELOAD_GATE_BLOCKED_REASON: Final = (
    "provider-attempt daemon reload boundary not yet accepted"
)
_PROVIDER_ATTEMPT_RELOAD_GATE_C1_CONTRACT_SHA256: Final = (
    "sha256:e38d159c1a30ebf74e171a1d7a00f7dba0773058dd17f96eca34e01f09810e4b"
)
_FALSE_COMPLETION_REPAIR_TASKS: Final = {
    "ASE3-023": {
        "title": "Repair production plan-bound adaptive parallel dispatch",
        "contract_sha256": (
            "sha256:c13240a72521f3f7f71b39e5d404daa5825581b1606e707a5dad8e693af73f25"
        ),
        "goal id": "ASE3-G040",
        "depends on": ("ASE3-003", "ASE3-004", "ASE3-005", "ASE3-006"),
        "outputs": (
            "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
            "test/api/test_agent_supervisor_prompt_v3_parallelism.py",
            "test/api/test_agent_supervisor_configured_board_scheduler.py",
            "test/api/test_agent_supervisor_implementation_supervisor_runner.py",
        ),
        "validation": (
            "python -m pytest test/api/test_agent_supervisor_prompt_v3_parallelism.py "
            "test/api/test_agent_supervisor_configured_board_scheduler.py "
            "test/api/test_agent_supervisor_implementation_supervisor_runner.py -q"
        ),
        "repairs task": "ASE3-006",
        "strict_shard": 2,
        "evidence_anchor": (
            "false_completion_recovery_20260808.json#false_completions/ASE3-006"
        ),
    },
    "ASE3-027": {
        "title": (
            "Repair production canonical resolver composition and verified trust evidence"
        ),
        "contract_sha256": (
            "sha256:69853f7f6174a9bd118b4fca13d5ba8e897e962def801d7fb012d9e4969f7d8c"
        ),
        "goal id": "ASE3-G020",
        "depends on": ("ASE3-001", "ASE3-018"),
        "outputs": (
            "ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py",
            "test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py",
        ),
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py "
            "test/api/test_agent_supervisor_prompt_v3_resolution.py "
            "test/api/test_agent_supervisor_inference_runtime.py "
            "test/api/test_agent_supervisor_target_resolver.py "
            "test/api/test_agent_supervisor_state_resolver.py "
            "test/api/test_agent_supervisor_profile_resolver.py "
            "test/api/test_agent_supervisor_objective_resolver.py "
            "test/api/test_agent_supervisor_capability_resolver.py "
            "test/api/test_agent_supervisor_authority_resolver.py -q"
        ),
        "repairs task": "ASE3-018",
        "strict_shard": 0,
        "evidence_anchor": (
            "false_completion_recovery_20260808.json#false_completions/ASE3-018"
        ),
    },
}
_PROGRAM_EXPANSION_TASKS: Final = {
    "ASE3-024": {
        "title": "Make prompt intake and goal planning crash-safe and router-owned",
        "contract_sha256": (
            "sha256:8bae1e3234974cc5aa160d03a61250edccb93a6482a6b955fefa71187c4e46e5"
        ),
        "canonical_task_cid": (
            "baguqeerawhpesa66k7agz2kqllpcxxvtzdl262fffigckkzieel37qos2zvq"
        ),
        "goal id": "ASE3-G030",
        "depends on": ("ASE3-003", "ASE3-004", "ASE3-028"),
        "outputs": (
            "ipfs_accelerate_py/llm_router.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/planning_policy.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/planning_effect.py",
            "ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py",
            "test/api/test_agent_supervisor_prompt_v3_prompt_transaction.py",
            "test/api/test_agent_supervisor_prompt_broker.py",
            "test/api/test_agent_supervisor_prompt_planning_policy.py",
            "test/api/test_agent_supervisor_prompt_goal_planner.py",
        ),
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_prompt_v3_prompt_transaction.py "
            "test/api/test_agent_supervisor_prompt_broker.py "
            "test/api/test_agent_supervisor_prompt_planning_policy.py "
            "test/api/test_agent_supervisor_prompt_goal_planner.py "
            "test/api/test_agent_supervisor_prompt_v3_provider_route.py -q"
        ),
    },
    "ASE3-025": {
        "title": (
            "Prove canonical generated boards execute through the real adaptive "
            "runtime"
        ),
        "contract_sha256": (
            "sha256:038598ffbd0a17d09486d43a3d4edd83c5d7e1419a9f63c786ff35febc1a0b93"
        ),
        "canonical_task_cid": (
            "baguqeerarcqpxaz2jt75eaipecwrybvvdryyhqyyoz7g4mlso3but63dxuoq"
        ),
        "goal id": "ASE3-G040",
        "depends on": ("ASE3-004", "ASE3-023", "ASE3-024"),
        "outputs": (
            "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py",
            "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_compiler.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/generated_program_task_source.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
            "test/api/test_agent_supervisor_prompt_v3_generated_board_e2e.py",
            "test/api/test_agent_supervisor_prompt_v3_plan_materializer.py",
            "test/api/test_agent_supervisor_prompt_workflow_contracts.py",
        ),
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_prompt_v3_generated_board_e2e.py "
            "test/api/test_agent_supervisor_prompt_v3_plan_materializer.py "
            "test/api/test_agent_supervisor_prompt_workflow_contracts.py "
            "test/api/test_agent_supervisor_markdown_task_source.py "
            "test/api/test_agent_supervisor_duckdb_task_source.py "
            "test/api/test_agent_supervisor_configured_board_scheduler.py -q"
        ),
    },
    "ASE3-028": {
        "title": "Restore router ownership and the package dependency direction",
        "contract_sha256": (
            "sha256:4acc7e0f9a94cf461ba54677505c69aa2e3bdc16573cfaa57517053e6d562434"
        ),
        "canonical_task_cid": (
            "baguqeeraetybichqkpnv2pnsc3eqmuy4rn76nm3gymtqavsobicogri57kfa"
        ),
        "goal id": "ASE3-G020",
        "depends on": ("ASE3-029",),
        "outputs": (
            "ipfs_accelerate_py/llm_router.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_provider_auto.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py",
            "test/api/test_implementation_provider_auto.py",
            "test/api/test_agent_supervisor_prompt_v3_provider_route.py",
            "test/api/test_agent_supervisor_router_owned_provider_decision.py",
        ),
        "validation": (
            "python -m pytest test/api/test_implementation_provider_auto.py "
            "test/api/test_agent_supervisor_prompt_v3_provider_route.py "
            "test/api/test_agent_supervisor_router_owned_provider_decision.py -q"
        ),
    },
    "ASE3-029": {
        "title": "Lower shared supervisor contracts into a neutral package",
        "contract_sha256": (
            "sha256:d34914edefc3d3625d36662b3a08882569937ee924bc1e65e9fb26250ee40b4e"
        ),
        "canonical_task_cid": (
            "baguqeeraaft6esbryems3slsxxeav7sioahafs2nuoighml2muu5r2mfh5qa"
        ),
        "goal id": "ASE3-G020",
        "depends on": ("ASE3-022",),
        "outputs": (
            "ipfs_accelerate_py/agent_supervisor/contracts/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/contracts/authority.py",
            "ipfs_accelerate_py/agent_supervisor/contracts/execution.py",
            "ipfs_accelerate_py/agent_supervisor/contracts/provider_capacity.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_provider_auto.py",
            "test/api/test_agent_supervisor_contract_layering.py",
        ),
        "validation": (
            "python -m pytest test/api/test_agent_supervisor_contract_layering.py "
            "test/api/test_agent_supervisor_configured_board_scheduler.py "
            "test/api/test_agent_supervisor_implementation_supervisor_runner.py "
            "test/api/test_agent_supervisor_implementation_provider_receipts.py "
            "test/api/test_implementation_provider_auto.py -q"
        ),
    },
    "ASE3-030": {
        "title": "Seal hermetic control-plane identity dependency closure",
        "contract_sha256": (
            "sha256:fe06816d222c538150df4f2c67773e722233c2d0cf4ad0199ae9968e11e52263"
        ),
        "canonical_task_cid": (
            "baguqeeraixg3vmaaqjjzelv2eh5hhib3y57ezmtrp2uq5aufnvuakmjnov6q"
        ),
        "goal id": "ASE3-G040",
        "depends on": ("ASE3-019",),
        "outputs": (
            "ipfs_accelerate_py/llm_router.py",
            "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
            "ipfs_accelerate_py/utils/cid_utils.py",
            "test/api/test_agent_supervisor_control_plane_capsule_identity.py",
            "test/api/test_agent_supervisor_control_plane.py",
            "test/api/test_agent_supervisor_multiformats_identity.py",
            "test/api/test_llm_router_agent_implementation_route.py",
            HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        ),
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_control_plane_capsule_identity.py "
            "test/api/test_agent_supervisor_control_plane.py "
            "test/api/test_agent_supervisor_multiformats_identity.py "
            "test/api/test_llm_router_agent_implementation_route.py -q"
        ),
    },
}
_PROGRAM_AMENDED_TASK_CIDS: Final = {
    "ASE3-008": "baguqeeraps4yiytww7kf5e7ybn3ctmtjjeacvwao2vvdzlpcq664l3ihpo6q",
    "ASE3-009": "baguqeera7ly4s4ddus5vo5iyaobxuz5mmlmoi4g3ajcvcuycrpfihtzlbykq",
    "ASE3-012": "baguqeerakpgeugpi6adjmmkv3vsqgaznlotedao7srb2fnynsco3rbgzjcpa",
    "ASE3-013": "baguqeeraxkgeu5kylsecwjoxwhi3pnjw3fnwddvtcmvj7heecl7qepwvwseq",
    "ASE3-020": "baguqeeraeofnvkxowsyssyahrjh362aembsxtbcmq6mv2drims225tnkggya",
    "ASE3-021": "baguqeeraycuz4hddho6gr2bqbl3pknpz5e2pqjuvgqtluo63j343gkq52jsq",
}
_PROGRAM_AMENDED_TASK_CONTRACT_SHA256S: Final = {
    "ASE3-008": (
        "sha256:2243ee5c6e3e749f3ef23ebba14676ece081b1bd015fee7354812670bf819e8f"
    ),
    "ASE3-009": (
        "sha256:82e0a373cc1423b6b2aa9dd1d750cb5f44e8955c20f4b2e25b12bb44b7ab1e5f"
    ),
    "ASE3-012": (
        "sha256:0b35be7e0aacbb9eb3e0540610e0b564969f5888a13d409be33a406ed3430b30"
    ),
    "ASE3-013": (
        "sha256:7869721e77a17ecc7b15be092478e733bb0d18833fbe042195707c850d685f23"
    ),
    "ASE3-020": (
        "sha256:1532bbae10cc65268df2f0ca87512f37375e557aafeaf6b63030e2071033f1d1"
    ),
    "ASE3-021": (
        "sha256:39c2c005fb56b4190b00ebf63b95098e678146054469bcebe69047950272baac"
    ),
}
_PROGRAM_UNCHANGED_FUTURE_TASK_CONTRACT_SHA256S: Final = {
    "ASE3-010": (
        "sha256:5e06e1a521e917d7936c4d71f8fd9b48edf513a546710a608d282b7d894d7efa"
    ),
    "ASE3-011": (
        "sha256:e00226a0ea674c095bfce8dbd88b8101927b45203c6084ff1cb1f42bc0b7e4f8"
    ),
    "ASE3-014": (
        "sha256:feb03791e0564d25f7c2e57844423bde13053679d95e1dcff21af7f6ac769025"
    ),
}
_PROGRAM_AMENDED_TASK_DEPENDENCIES: Final = {
    "ASE3-008": ("ASE3-006", "ASE3-020"),
    "ASE3-009": ("ASE3-005", "ASE3-008", "ASE3-026"),
    "ASE3-012": ("ASE3-010", "ASE3-011"),
    "ASE3-013": ("ASE3-008", "ASE3-012"),
    "ASE3-020": (
        "ASE3-003",
        "ASE3-005",
        "ASE3-018",
        "ASE3-019",
        "ASE3-021",
        "ASE3-024",
        "ASE3-025",
        "ASE3-028",
    ),
    "ASE3-021": (
        "ASE3-004",
        "ASE3-006",
        "ASE3-007",
        "ASE3-019",
        "ASE3-022",
        "ASE3-024",
        "ASE3-025",
    ),
}
_PROGRAM_AMENDED_TASK_REQUIREMENTS: Final = {
    "ASE3-008": ("DurableMonitorRunner", "client disconnect"),
    "ASE3-009": ("ProductionServiceCompositionManifest", "ASE3-026"),
    "ASE3-012": ("black-box", "production composition CID"),
    "ASE3-013": (
        "no preseeded objective or taskboard",
        "non-sentinel",
        "monitor_policy.canary_observation_seconds: 900",
    ),
    "ASE3-020": ("RequiredArgumentCoverageReceipt", "actual supervisor and daemon parsers"),
    "ASE3-021": (
        (
            "event -> current-tree residual -> append/adopt CAS -> active-plan "
            "invalidation -> recompile -> real descendant dispatch"
        ),
        "ASE3-025 canonical supervisor schema",
    ),
}
_PROTECTED_RUNTIME_ACTIVATION_TASK_ID: Final = "ASE3-026"
_PROTECTED_RUNTIME_ACTIVATION_BLOCKED_REASON: Final = (
    "protected runtime activation receipt not yet accepted"
)
_PROTECTED_RUNTIME_ACTIVATION_CONTRACT_SHA256: Final = (
    "sha256:b2f4f5afeecfdce68ace7509b072646c8f762ffd53692adc9e98431a4a9fe6ce"
)
_PROTECTED_RUNTIME_ACTIVATION_TASK_CID: Final = (
    "baguqeerah5rwdashtgibn3xqzdlo6w4ft4gy567vmov6zfq5vpxsekqxkqra"
)
_ASE3_019_TITLE: Final = (
    "Seal signed provider authority, authentication lifecycle, and once-only fallback"
)
_ASE3_019_CONTRACT_SHA256: Final = (
    "sha256:0b03746b83ab9a7316d6e8f5145fe092043819bf8ac2012ee8596277c44602a8"
)
_ASE3_023_FORBIDDEN_OUTPUTS: Final = frozenset(
    {
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/llm_router.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py",
    }
)
_PROVIDER_FALLBACK_AUTHORIZATION_CREATED_AT: Final = "2026-08-08T13:59:09Z"
_PROVIDER_FALLBACK_AUTHORIZATION_SOURCE: Final = {
    "kind": "explicit_operator_override",
    "source_head": "b9c1368a35cee206dff6ff34553782be851fc571",
    "source_tree": "7aeb7e4d78f5b45d2213173a10deebcf6114092f",
    "prospective_only": True,
    "requires_descendant_tree": True,
}
_PROVIDER_FALLBACK_AUTHORIZATION_ROUTE: Final = {
    "route_id": (
        "agent-supervisor-prompt-v3-grok45-terra56-high-auth-or-hard-quota-v1"
    ),
    "primary_provider_id": "grok_cli",
    "primary_model_id": "grok-4.5",
    "fallback_provider_id": "codex",
    "fallback_model_id": "gpt-5.6-terra",
    "fallback_reasoning_effort": "high",
    "allowed_trigger_classes": [
        "grok_authentication_unavailable",
        "grok_hard_quota_exhausted",
    ],
}
_PROVIDER_FALLBACK_OWNERSHIP_CONTRACT: Final = {
    "canonical_route_plan_owner": "ipfs_accelerate_py.llm_router",
    "typed_fallback_decision_owner": "ipfs_accelerate_py.llm_router",
    "route_plan_and_decision_exports_required_before_bootstrap_dispatch": True,
    "route_authority_binding_fields": [
        "board_namespace",
        "authorization_artifact_sha256",
        "authorization_source.kind",
        "authorization_source.source_head",
        "authorization_source.source_tree",
    ],
    "verified_authority_binding_must_reach_terminal_outcome_and_daemon_accounting": True,
    "ambient_six_field_route_profile_alone_authorizes_fallback": False,
    "runner_role": "isolation_process_effect_and_terminal_outcome_emitter",
    "daemon_role": "task_retry_accounting_only",
    "scheduler_role": "route_profile_input_only",
    "duplicate_route_policy_or_failure_classification_outside_router_allowed": False,
}
_PROVIDER_FALLBACK_BOOTSTRAP_GUARANTEES: Final = {
    "nonce_bound_auth_or_quota_finding_required": True,
    "primary_probe_is_fixed_no_tools": True,
    "direct_auth_signal_allowlist": ["not signed in", "not authenticated"],
    "ambiguous_direct_auth_signals_denied": [
        "401",
        "403",
        "forbidden",
        "unauthorized",
    ],
    "ambiguous_signal_may_continue_only_as_independently_confirmed_hard_quota": True,
    "hard_quota_independent_confirmation_required": True,
    "pre_effect_workspace_fingerprint_required": True,
    "explicit_codex_review_conflict_denied": True,
    "fallback_dispatch_scope": "once_per_runner_same_daemon_attempt",
    "fallback_remains_same_daemon_attempt": True,
    "durable_cross_process_restart_reservation_present": False,
    "full_signed_field_equality_present": False,
}
_PROVIDER_FALLBACK_ASE3_019_REQUIREMENTS: Final = {
    "typed_failure_evidence_required": True,
    "quota_evidence_must_be_independently_verified": True,
    "evidence_must_precede_repository_effect": True,
    "signed_equality_fields": [
        "invocation",
        "task",
        "prompt",
        "scope",
        "budget",
        "authority",
        "provider",
    ],
    "durable_cross_process_restart_once_only_cas_required": True,
    "restart_must_adopt_existing_reservation": True,
    "auth_signal_policy_expansion_requires_signed_typed_policy": True,
    "canonical_route_plan_and_typed_decision_must_remain_router_owned": True,
    "provider_capacity_attempt_restoration_must_remain_denied": True,
    "signed_reviewer_identity_and_provider_required": True,
    "fallback_implementer_and_reviewer_must_differ": True,
}
_PROVIDER_FALLBACK_DOCKER_BOUNDARY: Final = {
    "required": True,
    "runtime": "runc",
    "image_id": (
        "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
    ),
    "image_label": "2026-08-03-v2",
    "pull_allowed": False,
    "read_only_root": True,
    "cap_drop": "ALL",
    "no_new_privileges": True,
    "workspace_is_only_writable_bind_mount": True,
    "codex_auth_mount_read_only": True,
    "docker_socket_mounted": False,
    "host_home_mounted": False,
    "environment_sealed": True,
}
_PROVIDER_FALLBACK_DENIALS: Final = {
    "arbitrary_error_fallback_allowed": False,
    "rate_limit_fallback_allowed": False,
    "transport_error_fallback_allowed": False,
    "invalid_request_fallback_allowed": False,
    "unknown_error_fallback_allowed": False,
    "post_effect_fallback_allowed": False,
    "workspace_changed_before_fallback_allowed": False,
    "attempt_counter_mutation_authorized": False,
    "provider_capacity_attempt_restoration_allowed": False,
    "legacy_objective_refill_authorized": False,
    "legacy_codebase_refill_authorized": False,
}
_PROVIDER_FALLBACK_HISTORICAL_EVIDENCE: Final = {
    "post_wave3_residual_report_is_immutable": True,
    "historical_incident_reclassified": False,
    "incident_event_id": _POST_WAVE3_PROVIDER_INCIDENT["event_id"],
    "incident_log_sha256": _POST_WAVE3_PROVIDER_INCIDENT["log_sha256"],
}
_ASE3_019_REQUIRED_OUTPUTS: Final = (
    "ipfs_accelerate_py/llm_router.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "test/api/test_llm_router_agent_supervisor_fallback_route.py",
    "test/api/test_agent_supervisor_prompt_v3_authority_hardening.py",
)
_ASE3_019_REQUIRED_VALIDATION: Final = (
    "python -m pytest test/api/test_llm_router_agent_supervisor_fallback_route.py "
    "test/api/test_agent_supervisor_prompt_v3_authority_hardening.py "
    "test/api/test_agent_supervisor_prompt_v3_authority.py "
    "test/api/test_agent_supervisor_prompt_v3_provider_route.py "
    "test/api/test_agent_supervisor_grok_quota_terra_gate.py "
    "test/api/test_agent_supervisor_implementation_provider_receipts.py -q"
)
_ASE3_019_REQUIRED_INTERFACES: Final = (
    "AgentImplementationRoutePlan",
    "AgentImplementationFallbackDecision",
    "SignedSupervisorProfile",
    "SecureLocalIdentityStore",
    "SignedProfileLifecycleReceipt",
    "DurableProviderAttemptCAS",
    "AuthLifecycleFinding",
    "QuotaExhaustionEvidence",
    "ProviderFallbackReceipt",
)
_ASE3_019_REQUIRED_EFFECTS: Final = (
    "Export an immutable canonical implementation route plan and typed fallback "
    "decision from `ipfs_accelerate_py.llm_router` as the sole provider-policy "
    "source; bind the exact board namespace, authorization-artifact SHA-256, "
    "authorization kind, source HEAD, source tree, nonempty reviewer identity, and "
    "reviewer provider into every route plan and terminal outcome, deny when the "
    "reviewer identity or provider matches the chosen fallback implementer, and "
    "treat the ambient six-field provider/model/trigger/effort tuple as profile input "
    "that cannot authorize fallback by itself; make the scheduler pass only the "
    "route profile, the runner apply only isolation/process effects and emit the "
    "terminal outcome, and the daemon apply only task retry accounting; remove "
    "duplicate provider/model/trigger/effort, authentication/quota classification, "
    "and fallback allow/deny logic from those layers; sign exact repository, "
    "baseline tree, effects, budgets, resources, provider route, reviewer, and "
    "fallback bounds with a verifiable Ed25519 did:key identity stored as an owned "
    "regular nonsymlink 0600 file; persist signed rotation and revocation so copied "
    "old authority cannot revive; require either runner-produced typed pre-effect "
    "Grok authentication-unavailable evidence or independently verified native "
    "signed typed hard-quota evidence, with mandatory wall-clock freshness and exact "
    "nonempty invocation/task/prompt/scope/budget/authority/provider equality; "
    "reserve exactly one Codex `gpt-5.6-terra` fallback at `high` reasoning through "
    "durable compare-and-swap before any fallback effect, execute it only inside the "
    "pinned external Docker boundary, and adopt the winning receipt after crash as "
    "the same logical attempt without counter mutation or provider-capacity "
    "restoration; deny fallback for arbitrary errors, rate limits, transport "
    "failures, invalid requests, unknown failures, a changed workspace, or post-"
    "effect evidence."
)
_ASE3_019_REQUIRED_ACCEPTANCE: Final = (
    "Public immutable `AgentImplementationRoutePlan` and "
    "`AgentImplementationFallbackDecision` "
    "exports from `ipfs_accelerate_py.llm_router` are the only canonical provider-"
    "policy source; a missing or mismatched board namespace, authorization-artifact "
    "SHA-256, authorization kind, source HEAD, source tree, reviewer identity, or "
    "reviewer provider denies, a chosen fallback implementer cannot be its own "
    "reviewer, and the ambient six-field route profile alone never creates "
    "authority; the scheduler, runner, and daemon contain no independent route "
    "tuple, failure classifier, or fallback allow/deny branch, the runner executes "
    "only the router decision and emits its terminal outcome, and the daemon never "
    "reclassifies provider evidence and changes only task retry accounting; only a "
    "currently valid signed profile and the source-bound prospective authorization "
    "can authorize bounded effects; symlink, ownership, permission, substitution, "
    "copied-revoked-key, incomplete-bound, or non-descendant-tree cases fail closed; "
    "exactly one concurrent or restarted worker automatically admits a matching pre-"
    "effect Codex `gpt-5.6-terra` fallback at `high` reasoning for only typed Grok "
    "authentication-unavailable or independently verified hard-quota evidence and "
    "adopts it as the same logical attempt; arbitrary caller DTOs, optional/stale "
    "timestamps, empty equality fields, arbitrary/generic/rate-limit/transport/"
    "invalid/unknown errors, changed-workspace or post-effect evidence, self-review, "
    "and route mismatches deny; no fallback path mutates or restores attempt counters, "
    "including provider-capacity restoration, or enables legacy objective/codebase "
    "refill, and the historical `Not signed in` record remains uncharged, immutable "
    "evidence rather than being rewritten or reclassified."
)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _file_snapshot(status: os.stat_result) -> tuple[int, ...]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(stat.S_IFMT(status.st_mode)),
        int(status.st_nlink),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int = MAX_EVIDENCE_SNAPSHOT_BYTES,
) -> bytes:
    """Read one bounded, single-link, stable evidence-file snapshot."""

    if maximum_bytes < 0:
        raise ValueError(f"{path.name}: invalid evidence snapshot byte bound")
    initial = path.lstat()
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"{path.name}: expected a regular nonsymlink file")
    if initial.st_nlink != 1:
        raise ValueError(f"{path.name}: expected a single-link evidence file")
    if initial.st_size > maximum_bytes:
        raise ValueError(
            f"{path.name}: exceeds {maximum_bytes}-byte evidence snapshot bound"
        )

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _file_snapshot(opened) != _file_snapshot(initial)
        ):
            raise ValueError(
                f"{path.name}: evidence file changed before bounded read"
            )
        if opened.st_size > maximum_bytes:
            raise ValueError(
                f"{path.name}: exceeds {maximum_bytes}-byte evidence snapshot bound"
            )

        chunks: list[bytes] = []
        observed_bytes = 0
        while True:
            remaining = maximum_bytes + 1 - observed_bytes
            if remaining <= 0:
                raise ValueError(
                    f"{path.name}: exceeds {maximum_bytes}-byte evidence snapshot bound"
                )
            chunk = os.read(
                descriptor,
                min(_EVIDENCE_READ_CHUNK_BYTES, remaining),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed_bytes += len(chunk)
            if observed_bytes > maximum_bytes:
                raise ValueError(
                    f"{path.name}: exceeds {maximum_bytes}-byte evidence snapshot bound"
                )

        final_descriptor = os.fstat(descriptor)
        final_path = path.lstat()
        payload = b"".join(chunks)
        if (
            len(payload) != opened.st_size
            or _file_snapshot(final_descriptor) != _file_snapshot(opened)
            or _file_snapshot(final_path) != _file_snapshot(opened)
        ):
            raise ValueError(
                f"{path.name}: evidence file changed during bounded read"
            )
        return payload
    finally:
        os.close(descriptor)


def _load_json_bytes(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: expected UTF-8 JSON") from exc
    payload = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{name}: non-finite JSON constant {value!r}")
        ),
    )
    if not isinstance(payload, Mapping):
        # The document is JSON but its value violates this object's schema.
        raise ValueError(f"{name}: root must be a JSON object")  # noqa: TRY004
    return payload


def _load_json(path: Path) -> Mapping[str, Any]:
    return _load_json_bytes(_read_regular_bytes(path), name=path.name)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(_read_regular_bytes(path)).hexdigest()


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _require_hex40(errors: list[str], label: str, value: Any) -> None:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        errors.append(f"{label}: expected a lowercase 40-hex Git identity")


def _require_sha256(errors: list[str], label: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        errors.append(f"{label}: expected sha256:<64 lowercase hex>")


def _validate_exact_structure(
    errors: list[str],
    *,
    prefix: str,
    actual: Any,
    expected: Any,
) -> None:
    """Recursively require exact keys, sequence population, types, and values."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            errors.append(f"{prefix}: expected object")
            return
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            errors.append(f"{prefix}: exact key population required")
        for key in sorted(expected_keys):
            if key in actual:
                _validate_exact_structure(
                    errors,
                    prefix=f"{prefix}.{key}",
                    actual=actual[key],
                    expected=expected[key],
                )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, list):
            errors.append(f"{prefix}: expected array")
            return
        if len(actual) != len(expected):
            errors.append(f"{prefix}: exact population required")
        for index, expected_item in enumerate(expected):
            if index < len(actual):
                _validate_exact_structure(
                    errors,
                    prefix=f"{prefix}[{index}]",
                    actual=actual[index],
                    expected=expected_item,
                )
        return
    if isinstance(expected, bool):
        if actual is not expected:
            errors.append(f"{prefix}: expected {expected!r}")
        return
    if type(actual) is not type(expected) or actual != expected:
        errors.append(f"{prefix}: expected {expected!r}")


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


@dataclass(frozen=True)
class CurrentMainBaseline:
    """Immutable identities for current main, the seed, and rescue history."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CurrentMainBaseline:
        return cls(dict(payload))

    @property
    def integration_seed_commit(self) -> str:
        return str(self.payload.get("integration_seed", {}).get("commit", ""))

    @property
    def rescue_head(self) -> str:
        return str(self.payload.get("rescue", {}).get("head", ""))

    @property
    def merge_base(self) -> str:
        return str(self.payload.get("rescue", {}).get("merge_base", ""))

    @property
    def upstream_main_commit(self) -> str:
        return str(self.payload.get("upstream_main", {}).get("commit", ""))

    @property
    def integration_seed_tree(self) -> str:
        return str(self.payload.get("integration_seed", {}).get("tree", ""))

    @property
    def integration_branch(self) -> str:
        return str(self.payload.get("integration_seed", {}).get("branch", ""))

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.payload.get("schema") != CURRENT_MAIN_BASELINE_SCHEMA:
            errors.append("current_main_baseline.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("current_main_baseline.board_namespace: mismatch")
        for section, field in (
            ("upstream_main", "commit"),
            ("upstream_main", "tree"),
            ("integration_seed", "commit"),
            ("integration_seed", "tree"),
            ("rescue", "head"),
            ("rescue", "tree"),
            ("rescue", "merge_base"),
            ("rescue", "merge_base_tree"),
        ):
            block = self.payload.get(section, {})
            value = block.get(field) if isinstance(block, Mapping) else None
            _require_hex40(errors, f"current_main_baseline.{section}.{field}", value)
        rescue = self.payload.get("rescue", {})
        if not isinstance(rescue, Mapping):
            errors.append("current_main_baseline.rescue: expected object")
        else:
            for field in ("current_main_ahead", "rescue_ahead"):
                value = rescue.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    errors.append(
                        f"current_main_baseline.rescue.{field}: expected nonnegative integer"
                    )
        integration = self.payload.get("integration_seed", {})
        if not isinstance(integration, Mapping):
            errors.append("current_main_baseline.integration_seed: expected object")
        else:
            _require_hex40(
                errors,
                "current_main_baseline.integration_seed.parent",
                integration.get("parent"),
            )
            if not str(integration.get("branch", "")).strip():
                errors.append("current_main_baseline.integration_seed.branch: required")
        upstream = self.payload.get("upstream_main", {})
        if not isinstance(upstream, Mapping):
            errors.append("current_main_baseline.upstream_main: expected object")
        elif not str(upstream.get("branch", "")).strip():
            errors.append("current_main_baseline.upstream_main.branch: required")
        checkout = self.payload.get("original_checkout", {})
        if not isinstance(checkout, Mapping):
            errors.append("current_main_baseline.original_checkout: expected object")
        else:
            if checkout.get("clean") is not False:
                errors.append("current_main_baseline.original_checkout.clean: must be false")
            if not isinstance(checkout.get("dirty_entry_count"), int) or int(
                checkout.get("dirty_entry_count", 0)
            ) <= 0:
                errors.append(
                    "current_main_baseline.original_checkout.dirty_entry_count: must be positive"
                )
            _require_sha256(
                errors,
                "current_main_baseline.original_checkout.status_sha256",
                checkout.get("status_sha256"),
            )
            if checkout.get("preservation_policy") != "read-only-protected":
                errors.append(
                    "current_main_baseline.original_checkout.preservation_policy: must be read-only-protected"
                )
            if not isinstance(checkout.get("path"), str) or not Path(
                str(checkout.get("path", ""))
            ).is_absolute():
                errors.append(
                    "current_main_baseline.original_checkout.path: expected absolute path"
                )
            _require_hex40(
                errors,
                "current_main_baseline.original_checkout.head",
                checkout.get("head"),
            )
            if checkout.get("head") != self.upstream_main_commit:
                errors.append(
                    "current_main_baseline.original_checkout.head: must equal upstream main commit"
                )
            if isinstance(upstream, Mapping) and checkout.get("branch") != upstream.get(
                "branch"
            ):
                errors.append(
                    "current_main_baseline.original_checkout.branch: must equal upstream main branch"
                )
            if checkout.get("status_snapshot_is_historical") is not True:
                errors.append(
                    "current_main_baseline.original_checkout.status_snapshot_is_historical: must be true"
                )
        submodules = self.payload.get("submodules")
        if not isinstance(submodules, Sequence) or isinstance(submodules, (str, bytes)):
            errors.append("current_main_baseline.submodules: expected list")
        else:
            paths: set[str] = set()
            for index, item in enumerate(submodules):
                if not isinstance(item, Mapping):
                    errors.append(f"current_main_baseline.submodules[{index}]: expected object")
                    continue
                path = item.get("path")
                if not isinstance(path, str) or not _is_safe_relative_path(path):
                    errors.append(
                        f"current_main_baseline.submodules[{index}].path: unsafe path"
                    )
                elif path in paths:
                    errors.append(
                        f"current_main_baseline.submodules[{index}].path: duplicate {path}"
                    )
                else:
                    paths.add(path)
                _require_hex40(
                    errors,
                    f"current_main_baseline.submodules[{index}].gitlink_commit",
                    item.get("gitlink_commit"),
                )
        return tuple(errors)


@dataclass(frozen=True)
class HistoricalStateContradictionReport:
    """Contradictory historical projections that are explicitly non-authoritative."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> HistoricalStateContradictionReport:
        return cls(dict(payload))

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.payload.get("schema") != HISTORICAL_CONTRADICTION_SCHEMA:
            errors.append("historical_state_contradictions.schema: unsupported schema")
        if self.payload.get("authority") != "evidence-only":
            errors.append("historical_state_contradictions.authority: must be evidence-only")
        if self.payload.get("v3_completion_credit") is not False:
            errors.append(
                "historical_state_contradictions.v3_completion_credit: must be false"
            )
        sources = self.payload.get("sources")
        if not isinstance(sources, Mapping) or not sources:
            errors.append("historical_state_contradictions.sources: expected non-empty object")
        else:
            for source_id, source in sources.items():
                if not isinstance(source, Mapping):
                    errors.append(
                        f"historical_state_contradictions.sources.{source_id}: expected object"
                    )
                    continue
                _require_sha256(
                    errors,
                    f"historical_state_contradictions.sources.{source_id}.sha256",
                    source.get("sha256"),
                )
        records = self.payload.get("contradictions")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            errors.append(
                "historical_state_contradictions.contradictions: expected list"
            )
            return tuple(errors)
        codes: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}]: expected object"
                )
                continue
            code = record.get("code")
            if not isinstance(code, str) or not code:
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}].code: required"
                )
            elif code in codes:
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}].code: duplicate {code}"
                )
            else:
                codes.add(code)
            if record.get("authoritative") is not False:
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}].authoritative: must be false"
                )
            source_ids = record.get("source_ids")
            if not isinstance(source_ids, list) or not source_ids:
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}].source_ids: required"
                )
            elif isinstance(sources, Mapping):
                unknown = sorted(set(source_ids) - set(sources))
                if unknown:
                    errors.append(
                        f"historical_state_contradictions.contradictions[{index}].source_ids: unknown {unknown}"
                    )
        missing = sorted(_REQUIRED_CONTRADICTIONS - codes)
        if missing:
            errors.append(
                "historical_state_contradictions.contradictions: missing required "
                + ",".join(missing)
            )
        return tuple(errors)


@dataclass(frozen=True)
class RescueArtifactDisposition:
    """Disposition for one rescue commit or one changed path."""

    kind: str
    identity: str
    disposition: str
    target_tasks: tuple[str, ...]
    rationale: str
    current_state: str = ""
    target_tasks_is_list: bool = True

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, kind: str
    ) -> RescueArtifactDisposition:
        identity_key = "commit" if kind == "commit" else "path"
        tasks = payload.get("target_tasks", [])
        return cls(
            kind=kind,
            identity=str(payload.get(identity_key, "")),
            disposition=str(payload.get("disposition", "")),
            target_tasks=tuple(str(item) for item in tasks) if isinstance(tasks, list) else (),
            rationale=str(payload.get("rationale", "")),
            current_state=str(payload.get("current_state", "")),
            target_tasks_is_list=isinstance(tasks, list),
        )

    def validate(self, *, index: int) -> tuple[str, ...]:
        errors: list[str] = []
        prefix = f"rescue_artifact_dispositions.{self.kind}s[{index}]"
        if self.kind == "commit":
            _require_hex40(errors, f"{prefix}.commit", self.identity)
        elif not _is_safe_relative_path(self.identity):
            errors.append(f"{prefix}.path: unsafe path")
        if self.disposition not in _DISPOSITIONS:
            errors.append(f"{prefix}.disposition: unsupported {self.disposition!r}")
        if not self.target_tasks_is_list:
            errors.append(f"{prefix}.target_tasks: expected list")
        if self.disposition in {"port", "rewrite"} and not self.target_tasks:
            errors.append(f"{prefix}.target_tasks: required for {self.disposition}")
        for task in self.target_tasks:
            if task not in _TASK_IDS:
                errors.append(f"{prefix}.target_tasks: unknown task {task!r}")
        if len(self.target_tasks) != len(set(self.target_tasks)):
            errors.append(f"{prefix}.target_tasks: duplicate task")
        if not self.rationale.strip():
            errors.append(f"{prefix}.rationale: required")
        if self.kind == "file" and self.current_state not in {"missing", "diverged"}:
            errors.append(f"{prefix}.current_state: expected missing or diverged")
        return tuple(errors)


@dataclass(frozen=True)
class RescueDispositionReport:
    """Complete rescue population and its explicit convergence decisions."""

    payload: Mapping[str, Any]
    commits: tuple[RescueArtifactDisposition, ...]
    files: tuple[RescueArtifactDisposition, ...]
    shape_errors: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RescueDispositionReport:
        commits_payload = payload.get("commits", [])
        files_payload = payload.get("files", [])
        shape_errors: list[str] = []

        def parse_population(
            value: Any,
            *,
            field: str,
            kind: str,
        ) -> tuple[RescueArtifactDisposition, ...]:
            if not isinstance(value, list):
                shape_errors.append(
                    f"rescue_artifact_dispositions.{field}: expected list"
                )
                return ()
            parsed: list[RescueArtifactDisposition] = []
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    shape_errors.append(
                        f"rescue_artifact_dispositions.{field}[{index}]: expected object"
                    )
                    item = {}
                parsed.append(RescueArtifactDisposition.from_dict(item, kind=kind))
            return tuple(parsed)

        commits = parse_population(
            commits_payload,
            field="commits",
            kind="commit",
        )
        files = parse_population(files_payload, field="files", kind="file")
        return cls(dict(payload), commits, files, tuple(shape_errors))

    def validate(self, baseline: CurrentMainBaseline) -> tuple[str, ...]:
        errors: list[str] = list(self.shape_errors)
        if self.payload.get("schema") != RESCUE_DISPOSITION_SCHEMA:
            errors.append("rescue_artifact_dispositions.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("rescue_artifact_dispositions.board_namespace: mismatch")
        observed_at = self.payload.get("observed_at")
        if not isinstance(observed_at, str) or _UTC_TIMESTAMP.fullmatch(observed_at) is None:
            errors.append(
                "rescue_artifact_dispositions.observed_at: expected UTC timestamp"
            )
        if self.payload.get("historical_authority") != "evidence-only":
            errors.append(
                "rescue_artifact_dispositions.historical_authority: must be evidence-only"
            )
        if self.payload.get("bulk_merge_allowed") is not False:
            errors.append(
                "rescue_artifact_dispositions.bulk_merge_allowed: must be false"
            )
        for field, expected in (
            ("merge_base", baseline.merge_base),
            ("rescue_head", baseline.rescue_head),
            ("current_seed", baseline.integration_seed_commit),
        ):
            value = self.payload.get(field)
            _require_hex40(
                errors,
                f"rescue_artifact_dispositions.{field}",
                value,
            )
            if value != expected:
                errors.append(
                    f"rescue_artifact_dispositions.{field}: baseline mismatch"
                )
        if not str(self.payload.get("decision_rule", "")).strip():
            errors.append("rescue_artifact_dispositions.decision_rule: required")
        if len(self.commits) != 36:
            errors.append(
                f"rescue_artifact_dispositions.commits: expected 36, got {len(self.commits)}"
            )
        if len(self.files) != 35:
            errors.append(
                f"rescue_artifact_dispositions.files: expected 35, got {len(self.files)}"
            )
        for index, item in enumerate(self.commits):
            errors.extend(item.validate(index=index))
        for index, item in enumerate(self.files):
            errors.extend(item.validate(index=index))
        commit_ids = [item.identity for item in self.commits]
        file_paths = [item.identity for item in self.files]
        if len(commit_ids) != len(set(commit_ids)):
            errors.append("rescue_artifact_dispositions.commits: duplicate identity")
        if len(file_paths) != len(set(file_paths)):
            errors.append("rescue_artifact_dispositions.files: duplicate path")
        return tuple(errors)


@dataclass(frozen=True)
class CleanIntegrationWorktreeReceipt:
    """Receipt that separates the v3 integration lane from user changes."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> CleanIntegrationWorktreeReceipt:
        return cls(dict(payload))

    def validate(self, baseline: CurrentMainBaseline) -> tuple[str, ...]:
        errors: list[str] = []
        if self.payload.get("schema") != CLEAN_WORKTREE_RECEIPT_SCHEMA:
            errors.append("clean_worktree_receipt.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("clean_worktree_receipt.board_namespace: mismatch")
        worktree = self.payload.get("worktree", {})
        if not isinstance(worktree, Mapping):
            errors.append("clean_worktree_receipt.worktree: expected object")
        else:
            if worktree.get("clean_at_creation") is not True:
                errors.append("clean_worktree_receipt.worktree.clean_at_creation: must be true")
            if worktree.get("creation_head") != baseline.integration_seed_commit:
                errors.append(
                    "clean_worktree_receipt.worktree.creation_head: must equal integration seed"
                )
            _require_hex40(
                errors,
                "clean_worktree_receipt.worktree.creation_tree",
                worktree.get("creation_tree"),
            )
            if worktree.get("creation_tree") != baseline.integration_seed_tree:
                errors.append(
                    "clean_worktree_receipt.worktree.creation_tree: must equal integration seed tree"
                )
            if worktree.get("branch") != baseline.integration_branch:
                errors.append(
                    "clean_worktree_receipt.worktree.branch: must equal integration branch"
                )
            if worktree.get("isolated_from_source_checkout") is not True:
                errors.append(
                    "clean_worktree_receipt.worktree.isolated_from_source_checkout: must be true"
                )
            if worktree.get("working_tree_is_expected_to_change_after_receipt") is not True:
                errors.append(
                    "clean_worktree_receipt.worktree.working_tree_is_expected_to_change_after_receipt: must be true"
                )
            path = worktree.get("path")
            if not isinstance(path, str) or not Path(path).is_absolute():
                errors.append(
                    "clean_worktree_receipt.worktree.path: expected absolute path"
                )
        source = self.payload.get("protected_source_checkout", {})
        baseline_source = baseline.payload.get("original_checkout", {})
        if not isinstance(source, Mapping) or not isinstance(baseline_source, Mapping):
            errors.append("clean_worktree_receipt.protected_source_checkout: expected object")
        else:
            for field in (
                "path",
                "head",
                "status_sha256",
                "dirty_entry_count",
                "preservation_policy",
            ):
                if source.get(field) != baseline_source.get(field):
                    errors.append(
                        "clean_worktree_receipt.protected_source_checkout."
                        f"{field}: baseline mismatch"
                    )
            if source.get("modified_by_bootstrap") is not False:
                errors.append(
                    "clean_worktree_receipt.protected_source_checkout.modified_by_bootstrap: must be false"
                )
        state = self.payload.get("state_namespace", {})
        if not isinstance(state, Mapping):
            errors.append("clean_worktree_receipt.state_namespace: expected object")
        else:
            value = str(state.get("path", ""))
            normalized_value = value.replace("_", "-")
            if "prompt-only-self-improvement-v3" not in normalized_value:
                errors.append(
                    "clean_worktree_receipt.state_namespace.path: must be a fresh v3 namespace"
                )
            if "prompt-only-entrypoints-v2" in normalized_value:
                errors.append(
                    "clean_worktree_receipt.state_namespace.path: historical namespace forbidden"
                )
            if state.get("fresh_for_board") is not True:
                errors.append(
                    "clean_worktree_receipt.state_namespace.fresh_for_board: must be true"
                )
            if state.get("historical_import_allowed") is not False:
                errors.append(
                    "clean_worktree_receipt.state_namespace.historical_import_allowed: must be false"
                )
            if state.get("generated_runtime_artifacts_are_completion_authority") is not False:
                errors.append(
                    "clean_worktree_receipt.state_namespace.generated_runtime_artifacts_are_completion_authority: must be false"
                )
        downstream = self.payload.get("downstream_binding", {})
        if not isinstance(downstream, Mapping):
            errors.append("clean_worktree_receipt.downstream_binding: expected object")
        else:
            if downstream.get("required_ancestor") != baseline.integration_seed_commit:
                errors.append(
                    "clean_worktree_receipt.downstream_binding.required_ancestor: baseline mismatch"
                )
            if downstream.get("required_branch") != baseline.integration_branch:
                errors.append(
                    "clean_worktree_receipt.downstream_binding.required_branch: baseline mismatch"
                )
            if downstream.get("changed_revision_requires_fresh_validation") is not True:
                errors.append(
                    "clean_worktree_receipt.downstream_binding.changed_revision_requires_fresh_validation: must be true"
                )
            if downstream.get("historical_ase_or_ase2_receipt_satisfies_v3") is not False:
                errors.append(
                    "clean_worktree_receipt.downstream_binding.historical_ase_or_ase2_receipt_satisfies_v3: must be false"
                )
        return tuple(errors)


@dataclass(frozen=True)
class PostWave3ResidualReport:
    """Fail-closed residual audit that authorizes the post-wave-3 refill only."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PostWave3ResidualReport:
        return cls(dict(payload))

    @property
    def repository_head(self) -> str:
        repository = self.payload.get("repository", {})
        return str(repository.get("head", "")) if isinstance(repository, Mapping) else ""

    @property
    def repository_tree(self) -> str:
        repository = self.payload.get("repository", {})
        return str(repository.get("tree", "")) if isinstance(repository, Mapping) else ""

    @property
    def completed_task_evidence(self) -> Mapping[str, Any]:
        evidence = self.payload.get("completed_task_evidence", {})
        return evidence if isinstance(evidence, Mapping) else {}

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        prefix = "post_wave3_residuals"
        expected_fields = {
            "schema",
            "created_at",
            "board_namespace",
            "repository",
            "completed_task_evidence",
            "residuals",
            "provider_incident",
            "disposition",
        }
        if set(self.payload) != expected_fields:
            errors.append(f"{prefix}: field population mismatch")
        if self.payload.get("schema") != POST_WAVE3_RESIDUAL_SCHEMA:
            errors.append(f"{prefix}.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append(f"{prefix}.board_namespace: mismatch")
        created_at = self.payload.get("created_at")
        if (
            not isinstance(created_at, str)
            or _UTC_TIMESTAMP.fullmatch(created_at) is None
            or created_at != _POST_WAVE3_CREATED_AT
        ):
            errors.append(
                f"{prefix}.created_at: expected immutable UTC timestamp "
                f"{_POST_WAVE3_CREATED_AT}"
            )

        repository = self.payload.get("repository")
        if not isinstance(repository, Mapping):
            errors.append(f"{prefix}.repository: expected object")
        else:
            if set(repository) != set(_POST_WAVE3_REPOSITORY):
                errors.append(f"{prefix}.repository: field population mismatch")
            for field in ("head", "tree"):
                value = repository.get(field)
                _require_hex40(errors, f"{prefix}.repository.{field}", value)
                if value != _POST_WAVE3_REPOSITORY[field]:
                    errors.append(
                        f"{prefix}.repository.{field}: immutable identity mismatch"
                    )
            if repository.get("branch") != _POST_WAVE3_REPOSITORY["branch"]:
                errors.append(f"{prefix}.repository.branch: mismatch")

        completed = self.payload.get("completed_task_evidence")
        if not isinstance(completed, Mapping):
            errors.append(f"{prefix}.completed_task_evidence: expected object")
        else:
            if set(completed) != set(_POST_WAVE3_COMPLETED_TASKS):
                errors.append(
                    f"{prefix}.completed_task_evidence: expected exactly "
                    "ASE3-005 and ASE3-007"
                )
            for task_id, expected in _POST_WAVE3_COMPLETED_TASKS.items():
                item = completed.get(task_id)
                item_prefix = f"{prefix}.completed_task_evidence.{task_id}"
                if not isinstance(item, Mapping):
                    errors.append(f"{item_prefix}: expected object")
                    continue
                if set(item) != set(expected):
                    errors.append(f"{item_prefix}: field population mismatch")
                for field in (
                    "implementation_commit",
                    "merge_commit",
                    "status_commit",
                ):
                    value = item.get(field)
                    _require_hex40(errors, f"{item_prefix}.{field}", value)
                    if value != expected[field]:
                        errors.append(f"{item_prefix}.{field}: immutable identity mismatch")
                for field in (
                    "declared_current_tree_tests_passed",
                    "declared_current_tree_tests_failed",
                ):
                    value = item.get(field)
                    if type(value) is not int or value != expected[field]:
                        errors.append(
                            f"{item_prefix}.{field}: expected {expected[field]}"
                        )

        residuals = self.payload.get("residuals")
        observed_residuals: dict[str, Mapping[str, Any]] = {}
        if not isinstance(residuals, list):
            errors.append(f"{prefix}.residuals: expected list")
        else:
            if len(residuals) != len(_POST_WAVE3_RESIDUALS):
                errors.append(
                    f"{prefix}.residuals: expected exactly "
                    f"{len(_POST_WAVE3_RESIDUALS)} records"
                )
            residual_fields = {
                "gap_id",
                "severity",
                "source_tasks",
                "target_task",
                "evidence",
            }
            for index, record in enumerate(residuals):
                record_prefix = f"{prefix}.residuals[{index}]"
                if not isinstance(record, Mapping):
                    errors.append(f"{record_prefix}: expected object")
                    continue
                if set(record) != residual_fields:
                    errors.append(f"{record_prefix}: field population mismatch")
                gap_id = record.get("gap_id")
                if not isinstance(gap_id, str) or not gap_id:
                    errors.append(f"{record_prefix}.gap_id: required")
                    continue
                if gap_id in observed_residuals:
                    errors.append(f"{record_prefix}.gap_id: duplicate {gap_id}")
                    continue
                observed_residuals[gap_id] = record
                if record.get("severity") != "P0":
                    errors.append(f"{record_prefix}.severity: expected P0")
                evidence = record.get("evidence")
                if (
                    not isinstance(evidence, list)
                    or not evidence
                    or any(not isinstance(item, str) or not item.strip() for item in evidence)
                    or len(evidence) != len(set(evidence))
                ):
                    errors.append(
                        f"{record_prefix}.evidence: expected unique non-empty strings"
                    )
            if set(observed_residuals) != set(_POST_WAVE3_RESIDUALS):
                errors.append(f"{prefix}.residuals: gap population mismatch")
            for gap_id, (target_task, source_tasks) in _POST_WAVE3_RESIDUALS.items():
                record = observed_residuals.get(gap_id)
                if record is None:
                    continue
                record_prefix = f"{prefix}.residuals.{gap_id}"
                if record.get("target_task") != target_task:
                    errors.append(
                        f"{record_prefix}.target_task: expected {target_task}"
                    )
                observed_sources = record.get("source_tasks")
                if (
                    not isinstance(observed_sources, list)
                    or any(not isinstance(item, str) for item in observed_sources)
                    or len(observed_sources) != len(set(observed_sources))
                    or frozenset(observed_sources) != source_tasks
                ):
                    errors.append(f"{record_prefix}.source_tasks: population mismatch")

        provider = self.payload.get("provider_incident")
        if not isinstance(provider, Mapping):
            errors.append(f"{prefix}.provider_incident: expected object")
        else:
            if set(provider) != set(_POST_WAVE3_PROVIDER_INCIDENT):
                errors.append(f"{prefix}.provider_incident: field population mismatch")
            _require_sha256(
                errors,
                f"{prefix}.provider_incident.event_id",
                provider.get("event_id"),
            )
            _require_sha256(
                errors,
                f"{prefix}.provider_incident.log_sha256",
                provider.get("log_sha256"),
            )
            for field, expected in _POST_WAVE3_PROVIDER_INCIDENT.items():
                actual = provider.get(field)
                matches = (
                    actual is expected
                    if isinstance(expected, bool)
                    else type(actual) is int and actual == expected
                    if isinstance(expected, int)
                    else actual == expected
                )
                if not matches:
                    errors.append(
                        f"{prefix}.provider_incident.{field}: expected {expected!r}"
                    )

        disposition = self.payload.get("disposition")
        if not isinstance(disposition, Mapping):
            errors.append(f"{prefix}.disposition: expected object")
        else:
            if set(disposition) != set(_POST_WAVE3_DISPOSITION):
                errors.append(f"{prefix}.disposition: field population mismatch")
            for field, expected in _POST_WAVE3_DISPOSITION.items():
                actual = disposition.get(field)
                if isinstance(expected, bool):
                    matches = actual is expected
                elif isinstance(expected, list):
                    matches = isinstance(actual, list) and actual == expected
                else:
                    matches = actual == expected
                if not matches:
                    errors.append(
                        f"{prefix}.disposition.{field}: expected {expected!r}"
                    )
        return tuple(errors)


def _validate_exact_policy_object(
    errors: list[str],
    *,
    prefix: str,
    actual: Any,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(actual, Mapping):
        errors.append(f"{prefix}: expected object")
        return
    if set(actual) != set(expected):
        errors.append(f"{prefix}: field population mismatch")
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if isinstance(expected_value, bool):
            matches = actual_value is expected_value
        elif isinstance(expected_value, list):
            matches = isinstance(actual_value, list) and actual_value == expected_value
        else:
            matches = (
                type(actual_value) is type(expected_value)
                and actual_value == expected_value
            )
        if not matches:
            errors.append(f"{prefix}.{field}: expected {expected_value!r}")


@dataclass(frozen=True)
class FalseCompletionRecoveryReport:
    """Immutable evidence that green projections failed product acceptance."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> FalseCompletionRecoveryReport:
        return cls(dict(payload))

    @property
    def recovery_parent_head(self) -> str:
        source = self.payload.get("source", {})
        return (
            str(source.get("recovery_parent_head", ""))
            if isinstance(source, Mapping)
            else ""
        )

    @property
    def recovery_parent_tree(self) -> str:
        source = self.payload.get("source", {})
        return (
            str(source.get("recovery_parent_tree", ""))
            if isinstance(source, Mapping)
            else ""
        )

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        prefix = "false_completion_recovery"
        expected_fields = {
            "schema",
            "created_at",
            "board_namespace",
            "source",
            "false_completions",
            "failed_attempt",
            "fence",
            "disposition",
        }
        if set(self.payload) != expected_fields:
            errors.append(f"{prefix}: field population mismatch")
        if self.payload.get("schema") != FALSE_COMPLETION_RECOVERY_SCHEMA:
            errors.append(f"{prefix}.schema: unsupported schema")
        if self.payload.get("created_at") != _FALSE_COMPLETION_RECOVERY_CREATED_AT:
            errors.append(
                f"{prefix}.created_at: expected immutable UTC timestamp "
                f"{_FALSE_COMPLETION_RECOVERY_CREATED_AT}"
            )
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append(f"{prefix}.board_namespace: mismatch")
        _validate_exact_policy_object(
            errors,
            prefix=f"{prefix}.source",
            actual=self.payload.get("source"),
            expected=_FALSE_COMPLETION_RECOVERY_SOURCE,
        )
        _validate_exact_policy_object(
            errors,
            prefix=f"{prefix}.false_completions",
            actual=self.payload.get("false_completions"),
            expected=_FALSE_COMPLETION_RECORDS,
        )
        _validate_exact_policy_object(
            errors,
            prefix=f"{prefix}.failed_attempt",
            actual=self.payload.get("failed_attempt"),
            expected=_FALSE_COMPLETION_FAILED_ATTEMPT,
        )
        _validate_exact_policy_object(
            errors,
            prefix=f"{prefix}.fence",
            actual=self.payload.get("fence"),
            expected=_FALSE_COMPLETION_FENCE,
        )
        _validate_exact_policy_object(
            errors,
            prefix=f"{prefix}.disposition",
            actual=self.payload.get("disposition"),
            expected=_FALSE_COMPLETION_DISPOSITION,
        )
        for field in (
            "launch_base_head",
            "launch_base_tree",
            "recovery_parent_head",
            "recovery_parent_tree",
        ):
            source = self.payload.get("source", {})
            value = source.get(field) if isinstance(source, Mapping) else None
            _require_hex40(errors, f"{prefix}.source.{field}", value)
        for task_id, expected in _FALSE_COMPLETION_RECORDS.items():
            for field in (
                "implementation_commit",
                "implementation_tree",
                "merge_commit",
                "status_commit",
            ):
                _require_hex40(
                    errors,
                    f"{prefix}.false_completions.{task_id}.{field}",
                    expected[field],
                )
            _require_sha256(
                errors,
                f"{prefix}.false_completions.{task_id}.merge_receipt_sha256",
                expected["merge_receipt_sha256"],
            )
            snapshot = str(expected["merge_receipt_snapshot"])
            if (
                snapshot not in ARTIFACT_FILENAMES
                or not _is_safe_relative_path(snapshot)
            ):
                errors.append(
                    f"{prefix}.false_completions.{task_id}."
                    "merge_receipt_snapshot: unsafe or unprotected"
                )
            repair_task = str(expected["repair_task"])
            repair_shard = (
                int(hashlib.sha256(repair_task.encode()).hexdigest()[:8], 16) % 3
            )
            if expected["repair_strict_shard"] != repair_shard:
                errors.append(
                    f"{prefix}.false_completions.{task_id}."
                    "repair_strict_shard: repair-task hash mismatch"
                )
        _require_hex40(
            errors,
            f"{prefix}.failed_attempt.implementation_commit",
            _FALSE_COMPLETION_FAILED_ATTEMPT["implementation_commit"],
        )
        _require_hex40(
            errors,
            f"{prefix}.failed_attempt.implementation_tree",
            _FALSE_COMPLETION_FAILED_ATTEMPT["implementation_tree"],
        )
        _require_sha256(
            errors,
            f"{prefix}.failed_attempt.failed_event_id",
            _FALSE_COMPLETION_FAILED_ATTEMPT["failed_event_id"],
        )
        _require_sha256(
            errors,
            f"{prefix}.failed_attempt.failed_event_snapshot_sha256",
            _FALSE_COMPLETION_FAILED_ATTEMPT["failed_event_snapshot_sha256"],
        )
        failed_snapshot = str(
            _FALSE_COMPLETION_FAILED_ATTEMPT["failed_event_snapshot"]
        )
        if (
            failed_snapshot not in ARTIFACT_FILENAMES
            or not _is_safe_relative_path(failed_snapshot)
        ):
            errors.append(
                f"{prefix}.failed_attempt.failed_event_snapshot: "
                "unsafe or unprotected"
            )
        retry_task = str(_FALSE_COMPLETION_FAILED_ATTEMPT["task_id"])
        retry_shard = (
            int(hashlib.sha256(retry_task.encode()).hexdigest()[:8], 16) % 3
        )
        if _FALSE_COMPLETION_FAILED_ATTEMPT["retry_strict_shard"] != retry_shard:
            errors.append(
                f"{prefix}.failed_attempt.retry_strict_shard: task hash mismatch"
            )
        return tuple(errors)


@dataclass(frozen=True)
class ProviderFallbackPolicyAuthorization:
    """Prospective, source-bound authority for the narrow automatic fallback."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> ProviderFallbackPolicyAuthorization:
        return cls(dict(payload))

    @property
    def source_head(self) -> str:
        source = self.payload.get("authorization_source", {})
        return str(source.get("source_head", "")) if isinstance(source, Mapping) else ""

    @property
    def source_tree(self) -> str:
        source = self.payload.get("authorization_source", {})
        return str(source.get("source_tree", "")) if isinstance(source, Mapping) else ""

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        prefix = "provider_fallback_policy_authorization"
        expected_fields = {
            "schema",
            "created_at",
            "board_namespace",
            "authorization_source",
            "route",
            "ownership_contract",
            "bootstrap_route_guarantees",
            "ase3_019_completion_requirements",
            "external_docker_boundary",
            "denials",
            "historical_evidence",
        }
        if set(self.payload) != expected_fields:
            errors.append(f"{prefix}: field population mismatch")
        if self.payload.get("schema") != PROVIDER_FALLBACK_POLICY_AUTHORIZATION_SCHEMA:
            errors.append(f"{prefix}.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append(f"{prefix}.board_namespace: mismatch")
        created_at = self.payload.get("created_at")
        if (
            not isinstance(created_at, str)
            or _UTC_TIMESTAMP.fullmatch(created_at) is None
            or created_at != _PROVIDER_FALLBACK_AUTHORIZATION_CREATED_AT
        ):
            errors.append(
                f"{prefix}.created_at: expected immutable UTC timestamp "
                f"{_PROVIDER_FALLBACK_AUTHORIZATION_CREATED_AT}"
            )

        sections = (
            (
                "authorization_source",
                _PROVIDER_FALLBACK_AUTHORIZATION_SOURCE,
            ),
            ("route", _PROVIDER_FALLBACK_AUTHORIZATION_ROUTE),
            ("ownership_contract", _PROVIDER_FALLBACK_OWNERSHIP_CONTRACT),
            (
                "bootstrap_route_guarantees",
                _PROVIDER_FALLBACK_BOOTSTRAP_GUARANTEES,
            ),
            (
                "ase3_019_completion_requirements",
                _PROVIDER_FALLBACK_ASE3_019_REQUIREMENTS,
            ),
            ("external_docker_boundary", _PROVIDER_FALLBACK_DOCKER_BOUNDARY),
            ("denials", _PROVIDER_FALLBACK_DENIALS),
            ("historical_evidence", _PROVIDER_FALLBACK_HISTORICAL_EVIDENCE),
        )
        for field, expected in sections:
            _validate_exact_policy_object(
                errors,
                prefix=f"{prefix}.{field}",
                actual=self.payload.get(field),
                expected=expected,
            )

        source = self.payload.get("authorization_source", {})
        if isinstance(source, Mapping):
            _require_hex40(errors, f"{prefix}.authorization_source.source_head", source.get("source_head"))
            _require_hex40(errors, f"{prefix}.authorization_source.source_tree", source.get("source_tree"))
        boundary = self.payload.get("external_docker_boundary", {})
        if isinstance(boundary, Mapping):
            _require_sha256(errors, f"{prefix}.external_docker_boundary.image_id", boundary.get("image_id"))
        historical = self.payload.get("historical_evidence", {})
        if isinstance(historical, Mapping):
            _require_sha256(errors, f"{prefix}.historical_evidence.incident_event_id", historical.get("incident_event_id"))
            _require_sha256(errors, f"{prefix}.historical_evidence.incident_log_sha256", historical.get("incident_log_sha256"))
        return tuple(errors)


@dataclass(frozen=True)
class ConvergenceManifest:
    """Root binding for the bounded ASE3-000 evidence packet."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConvergenceManifest:
        return cls(dict(payload))

    def validate(self, baseline: CurrentMainBaseline) -> tuple[str, ...]:
        errors: list[str] = []
        if self.payload.get("schema") != CONVERGENCE_MANIFEST_SCHEMA:
            errors.append("convergence_manifest.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("convergence_manifest.board_namespace: mismatch")
        if self.payload.get("task_id") != "ASE3-000":
            errors.append("convergence_manifest.task_id: expected ASE3-000")
        if self.payload.get("goal_id") != "ASE3-G010":
            errors.append("convergence_manifest.goal_id: expected ASE3-G010")
        created_at = self.payload.get("created_at")
        if (
            not isinstance(created_at, str)
            or _UTC_TIMESTAMP.fullmatch(created_at) is None
            or created_at != CONVERGENCE_MANIFEST_CREATED_AT
        ):
            errors.append(
                "convergence_manifest.created_at: expected UTC timestamp "
                f"{CONVERGENCE_MANIFEST_CREATED_AT} for packet assembly"
            )
        if self.payload.get("integration_seed_commit") != baseline.integration_seed_commit:
            errors.append(
                "convergence_manifest.integration_seed_commit: baseline mismatch"
            )
        if self.payload.get("integration_seed_tree") != baseline.integration_seed_tree:
            errors.append("convergence_manifest.integration_seed_tree: baseline mismatch")
        if self.payload.get("historical_completion_authority") is not False:
            errors.append(
                "convergence_manifest.historical_completion_authority: must be false"
            )
        if self.payload.get("rescue_bulk_merge_allowed") is not False:
            errors.append("convergence_manifest.rescue_bulk_merge_allowed: must be false")
        components = self.payload.get("components")
        if not isinstance(components, Mapping):
            errors.append("convergence_manifest.components: expected object")
        else:
            if set(components) != set(ARTIFACT_FILENAMES):
                errors.append("convergence_manifest.components: population mismatch")
            for filename, digest in components.items():
                if not _is_safe_relative_path(str(filename)):
                    errors.append(f"convergence_manifest.components.{filename}: unsafe path")
                _require_sha256(errors, f"convergence_manifest.components.{filename}", digest)
        population = self.payload.get("population", {})
        if not isinstance(population, Mapping):
            errors.append("convergence_manifest.population: expected object")
        else:
            expected_population = {
                "rescue_commits": 36,
                "rescue_changed_paths": 35,
                "v2_tasks": 8,
                "historical_contradictions": 5,
                "v3_seed_tasks": 15,
                "v3_seed_goals": 9,
            }
            if set(population) != set(expected_population):
                errors.append("convergence_manifest.population: population mismatch")
            for key, value in expected_population.items():
                if population.get(key) != value:
                    errors.append(
                        f"convergence_manifest.population.{key}: expected {value}"
                    )
        completion_rules = self.payload.get("completion_rules", {})
        expected_completion_rules = {
            "historical_status_or_receipt_satisfies_v3": False,
            "branch_local_commit_satisfies_v3": False,
            "queue_drain_satisfies_goal_completion": False,
            "current_tree_acceptance_required": True,
            "forced_residual_scan_required": True,
        }
        if not isinstance(completion_rules, Mapping):
            errors.append("convergence_manifest.completion_rules: expected object")
        else:
            if set(completion_rules) != set(expected_completion_rules):
                errors.append(
                    "convergence_manifest.completion_rules: population mismatch"
                )
            for field, expected in expected_completion_rules.items():
                if completion_rules.get(field) is not expected:
                    errors.append(
                        f"convergence_manifest.completion_rules.{field}: expected {expected!r}"
                    )
        downstream_rules = self.payload.get("downstream_rules", {})
        expected_downstream_rules = {
            "required_ancestor": baseline.integration_seed_commit,
            "merge_target_branch": baseline.integration_branch,
            "rescue_disposition_required_before_use": True,
            "fresh_validation_receipt_required_per_task": True,
            "protected_source_checkout_may_be_modified": False,
        }
        if not isinstance(downstream_rules, Mapping):
            errors.append("convergence_manifest.downstream_rules: expected object")
        else:
            if set(downstream_rules) != set(expected_downstream_rules):
                errors.append("convergence_manifest.downstream_rules: population mismatch")
            for field, expected in expected_downstream_rules.items():
                actual = downstream_rules.get(field)
                matches = (
                    actual is expected
                    if isinstance(expected, bool)
                    else actual == expected
                )
                if not matches:
                    errors.append(
                        f"convergence_manifest.downstream_rules.{field}: expected {expected!r}"
                    )
        return tuple(errors)


@dataclass(frozen=True)
class ConvergenceValidationReport:
    """Machine-readable preflight result."""

    valid: bool
    errors: tuple[str, ...]
    checked_artifacts: tuple[str, ...]
    integration_seed_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONVERGENCE_REPORT_SCHEMA,
            "valid": self.valid,
            "errors": list(self.errors),
            "checked_artifacts": list(self.checked_artifacts),
            "integration_seed_commit": self.integration_seed_commit,
        }


def _parse_taskboard_metadata(text: str) -> dict[str, dict[str, str]]:
    """Parse the bounded Markdown metadata used by the bootstrap gate.

    The convergence validator is also executed directly by file path, where
    package-relative imports are unavailable.  Keep this parser deliberately
    small and reject duplicate task IDs or metadata keys instead of inheriting
    the runtime parser's last-value-wins behavior.
    """

    tasks: dict[str, dict[str, str]] = {}
    current_id = ""
    current_metadata: dict[str, str] = {}

    def flush() -> None:
        nonlocal current_id, current_metadata
        if not current_id:
            return
        if current_id in tasks:
            raise ValueError(f"duplicate task id: {current_id}")
        tasks[current_id] = dict(current_metadata)
        current_id = ""
        current_metadata = {}

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            header = line[3:].strip()
            header_parts = header.split(" ", 1)
            task_id = header_parts[0]
            if task_id.startswith("ASE3-"):
                current_id = task_id
                current_metadata[_TASK_TITLE_KEY] = (
                    header_parts[1].strip() if len(header_parts) == 2 else ""
                )
            continue
        if not current_id:
            continue
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key in current_metadata:
            raise ValueError(
                f"duplicate metadata key for {current_id}: {normalized_key}"
            )
        current_metadata[normalized_key] = value.strip()
    flush()
    return tasks


def _load_taskboard_metadata(taskboard_path: Path) -> dict[str, dict[str, str]]:
    """Read one regular nonsymlink board snapshot and reject malformed UTF-8."""

    raw = _read_regular_bytes(taskboard_path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{taskboard_path.name}: expected UTF-8 Markdown") from exc
    return _parse_taskboard_metadata(text)


def _parse_objective_metadata(text: str) -> dict[str, dict[str, str]]:
    """Parse bounded goal metadata while rejecting duplicate IDs and fields."""

    goals: dict[str, dict[str, str]] = {}
    current_id = ""
    current_metadata: dict[str, str] = {}

    def flush() -> None:
        nonlocal current_id, current_metadata
        if not current_id:
            return
        if current_id in goals:
            raise ValueError(f"duplicate goal id: {current_id}")
        goals[current_id] = dict(current_metadata)
        current_id = ""
        current_metadata = {}

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            header = line[3:].strip()
            header_parts = header.split(" ", 1)
            goal_id = header_parts[0]
            if goal_id.startswith("ASE3-G"):
                current_id = goal_id
                current_metadata[_TASK_TITLE_KEY] = (
                    header_parts[1].strip() if len(header_parts) == 2 else ""
                )
            continue
        if not current_id:
            continue
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key in current_metadata:
            raise ValueError(
                f"duplicate metadata key for {current_id}: {normalized_key}"
            )
        current_metadata[normalized_key] = value.strip()
    flush()
    return goals


def _taskboard_csv(metadata: Mapping[str, str], field: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in str(metadata.get(field, "")).split(",")
        if item.strip()
    )


def _task_contract_sha256(metadata: Mapping[str, str]) -> str:
    title = str(metadata.get(_TASK_TITLE_KEY, ""))
    fields = {
        str(key): str(value)
        for key, value in metadata.items()
        if key != _TASK_TITLE_KEY
    }
    encoded = json.dumps(
        {"title": title, "metadata": fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalize_identity_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _normalize_identity_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return re.sub(r"/+", "/", text).rstrip("/")


def _identity_sequence(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _canonical_task_cid_from_metadata(metadata: Mapping[str, str]) -> str:
    """Recompute the runtime task CID without importing package-relative code."""

    title = _normalize_identity_text(metadata.get(_TASK_TITLE_KEY, ""))
    outputs = sorted(
        {
            normalized
            for item in _identity_sequence(metadata.get("outputs", ""))
            if (normalized := _normalize_identity_path(item))
        }
    )
    acceptance = [
        normalized
        for item in _identity_sequence(metadata.get("acceptance", ""))
        if (normalized := _normalize_identity_text(item))
    ]
    evidence = sorted(
        {
            normalized
            for item in _identity_sequence(
                metadata.get("missing evidence") or metadata.get("evidence") or ""
            )
            if (normalized := _normalize_identity_text(item))
        }
    )
    evidence_outputs = sorted(
        {
            normalized
            for item in _identity_sequence(metadata.get("evidence outputs", ""))
            if (normalized := _normalize_identity_path(item))
        }
    )
    goal = _normalize_identity_text(
        metadata.get("goal id")
        or metadata.get("goal packet key")
        or metadata.get("goal")
        or ""
    )
    semantic_hint = _normalize_identity_text(
        metadata.get("semantic key")
        or metadata.get("bundle key")
        or metadata.get("work scope")
        or metadata.get("fingerprint")
        or ""
    )
    semantic = {
        key: value
        for key, value in {
            "title": title,
            "outputs": outputs,
            "acceptance": acceptance,
            "evidence": evidence,
            "evidence_outputs": evidence_outputs,
            "goal": goal,
            "semantic_hint": semantic_hint,
        }.items()
        if value
    }
    if not semantic:
        raise ValueError("task identity requires semantic work metadata")
    material = {"schema": _TASK_IDENTITY_SCHEMA, "semantic": semantic}
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    raw_cid = b"\x01\xa9\x02\x12\x20" + digest
    return "b" + base64.b32encode(raw_cid).decode("ascii").rstrip("=").lower()


def _validate_provider_attempt_reload_gate(
    *,
    tasks: Mapping[str, Mapping[str, str]],
    artifact_root: Path,
) -> list[str]:
    """Validate the initial noncanonical reload gate.

    ASE3-022 may transition to ``completed`` only after this module gains a
    strict validator for ``provider_attempt_daemon_reload_receipt.json`` and
    the convergence manifest binds that receipt's digest.  Until both changes
    land atomically with the protected taskboard transition, the only accepted
    state is the exact blocked, review-only gate declared below.
    """

    errors: list[str] = []
    prefix = "provider_attempt_reload_gate"
    reserved_receipts = (
        (PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME, "receipt"),
        (
            OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
            OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
        ),
    )
    for receipt_filename, receipt_label in reserved_receipts:
        receipt_path = artifact_root / receipt_filename
        try:
            receipt_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(
                f"{prefix}.{receipt_label}: unable to inspect reserved path: {exc}"
            )
        else:
            errors.append(
                f"{prefix}.{receipt_label}: present without a strict validator "
                "and convergence-manifest binding"
            )

    gate = tasks.get(_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID)
    if gate is None:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}: expected exactly one task"
        )
        return errors
    gate_status = gate.get("status", "todo").strip().lower()
    if gate_status == "completed":
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.status: "
            "completion requires a strict reload "
            "receipt validator and convergence-manifest binding"
        )
    if gate_status != "blocked":
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.status: expected blocked"
        )
    if (
        _task_contract_sha256(gate)
        != _PROVIDER_ATTEMPT_RELOAD_GATE_C1_CONTRACT_SHA256
    ):
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.contract_sha256: "
            "exact C1 incident/salvage/reload gate metadata and prose required"
        )
    if gate.get("completion", "manual").strip().lower() != "manual":
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.completion: "
            "expected manual"
        )

    expected_metadata = {
        "is schedulable": "false",
        "review only": "true",
        "canonical board task": "false",
        "blocked reason": _PROVIDER_ATTEMPT_RELOAD_GATE_BLOCKED_REASON,
    }
    for field, expected in expected_metadata.items():
        actual = gate.get(field)
        if actual != expected:
            errors.append(
                f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}."
                f"{field.replace(' ', '_')}: "
                f"expected {expected!r}"
            )
    if _taskboard_csv(gate, "depends on") != _PROVIDER_ATTEMPT_RELOAD_GATE_DEPENDENCIES:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.depends_on: "
            "expected exactly "
            + ",".join(_PROVIDER_ATTEMPT_RELOAD_GATE_DEPENDENCIES)
        )
    if "goal id" in gate:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.goal_id: must be absent"
        )
    if _taskboard_csv(gate, "outputs") != (
        PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
    ):
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.outputs: expected only "
            f"{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
        )
    if gate.get("predicted files") != (
        PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    ):
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.predicted_files: "
            "expected only "
            f"{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
        )

    refill_task = tasks.get("ASE3-021")
    if refill_task is None:
        errors.append(f"{prefix}.ASE3-021: expected exactly one task")
    elif _PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID not in _taskboard_csv(
        refill_task,
        "depends on",
    ):
        errors.append(f"{prefix}.ASE3-021.depends_on: missing ASE3-022")

    provider_task = tasks.get("ASE3-019")
    if provider_task is None:
        errors.append(f"{prefix}.ASE3-019: expected exactly one task")
    elif provider_task.get("status") != "todo":
        errors.append(
            f"{prefix}.ASE3-019.status: must remain todo until the tracked "
            "operator-salvage receipt is strictly validated and bound"
        )
    return errors


def _validate_provider_fallback_task_contract(
    *,
    tasks: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """Keep ASE3-019 aligned with the prospective fallback authorization."""

    errors: list[str] = []
    prefix = "provider_fallback_task_contract"
    task = tasks.get("ASE3-019")
    if task is None:
        errors.append(f"{prefix}.ASE3-019: expected exactly one task")
        return errors
    if task.get(_TASK_TITLE_KEY) != _ASE3_019_TITLE:
        errors.append(f"{prefix}.ASE3-019.title: exact title required")
    if _task_contract_sha256(task) != _ASE3_019_CONTRACT_SHA256:
        errors.append(
            f"{prefix}.ASE3-019.contract_sha256: exact metadata/prose required"
        )
    try:
        current_cid = _canonical_task_cid_from_metadata(task)
    except ValueError as exc:
        errors.append(f"{prefix}.ASE3-019.canonical_task_cid: {exc}")
    else:
        expected_cid = str(_FALSE_COMPLETION_FAILED_ATTEMPT["canonical_task_cid"])
        if current_cid != expected_cid:
            errors.append(
                f"{prefix}.ASE3-019.canonical_task_cid: expected {expected_cid}"
            )
    outputs = _taskboard_csv(task, "outputs")
    if outputs != _ASE3_019_REQUIRED_OUTPUTS:
        errors.append(
            f"{prefix}.ASE3-019.outputs: exact llm_router-owned route surface "
            "required"
        )
    if _taskboard_csv(task, "predicted files") != _ASE3_019_REQUIRED_OUTPUTS:
        errors.append(
            f"{prefix}.ASE3-019.predicted_files: exact llm_router-owned route "
            "surface required"
        )
    if task.get("validation") != _ASE3_019_REQUIRED_VALIDATION:
        errors.append(
            f"{prefix}.ASE3-019.validation: dedicated llm_router route contract "
            "test required"
        )
    if _taskboard_csv(task, "interfaces") != _ASE3_019_REQUIRED_INTERFACES:
        errors.append(
            f"{prefix}.ASE3-019.interfaces: immutable route-plan and typed-decision "
            "exports required"
        )
    if task.get("effects") != _ASE3_019_REQUIRED_EFFECTS:
        errors.append(
            f"{prefix}.ASE3-019.effects: exact automatic auth/quota fallback "
            "contract required"
        )
    if task.get("acceptance") != _ASE3_019_REQUIRED_ACCEPTANCE:
        errors.append(
            f"{prefix}.ASE3-019.acceptance: exact automatic auth/quota fallback "
            "contract required"
        )
    return errors


def _validate_false_completion_repair_tasks(
    *,
    tasks: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """Pin the two replacement tasks without rewriting historical receipts."""

    errors: list[str] = []
    prefix = "false_completion_repair_tasks"
    observed_outputs: dict[str, frozenset[str]] = {}
    for task_id, expected in _FALSE_COMPLETION_REPAIR_TASKS.items():
        task = tasks.get(task_id)
        if task is None:
            errors.append(f"{prefix}.{task_id}: expected exactly one task")
            continue
        if task.get(_TASK_TITLE_KEY) != expected["title"]:
            errors.append(f"{prefix}.{task_id}.title: exact title required")
        if _task_contract_sha256(task) != expected["contract_sha256"]:
            errors.append(
                f"{prefix}.{task_id}.contract_sha256: "
                "exact metadata/prose required"
            )
        for field, expected_value in {
            "status": "todo",
            "completion": "manual",
            "is schedulable": "true",
            "review only": "false",
            "priority": "P0",
            "canonical board task": "true",
            "goal id": expected["goal id"],
            "repairs task": expected["repairs task"],
        }.items():
            if task.get(field) != expected_value:
                errors.append(
                    f"{prefix}.{task_id}.{field.replace(' ', '_')}: "
                    f"expected {expected_value!r}"
                )
        for field in ("depends on", "outputs", "predicted files"):
            expected_items = (
                expected["outputs"] if field == "predicted files" else expected[field]
            )
            if _taskboard_csv(task, field) != expected_items:
                errors.append(
                    f"{prefix}.{task_id}.{field.replace(' ', '_')}: "
                    "exact population required"
                )
        if task.get("validation") != expected["validation"]:
            errors.append(f"{prefix}.{task_id}.validation: exact command required")
        evidence = task.get("evidence subset", "")
        if str(expected["evidence_anchor"]) not in evidence:
            errors.append(
                f"{prefix}.{task_id}.evidence_subset: recovery anchor required"
            )
        actual_shard = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 3
        if actual_shard != expected["strict_shard"]:
            errors.append(
                f"{prefix}.{task_id}.strict_shard: expected "
                f"{expected['strict_shard']}"
            )
        observed_outputs[task_id] = frozenset(_taskboard_csv(task, "outputs"))

    scheduler_outputs = observed_outputs.get("ASE3-023", frozenset())
    if scheduler_outputs & _ASE3_023_FORBIDDEN_OUTPUTS:
        errors.append(
            f"{prefix}.ASE3-023.outputs: provider/resolver conflict surface forbidden"
        )
    resolver_outputs = observed_outputs.get("ASE3-027", frozenset())
    provider_outputs = frozenset(_ASE3_019_REQUIRED_OUTPUTS)
    if resolver_outputs & provider_outputs or scheduler_outputs & provider_outputs:
        errors.append(f"{prefix}: repair/provider output overlap forbidden")
    if resolver_outputs & scheduler_outputs:
        errors.append(f"{prefix}: repair output overlap forbidden")
    return errors


def _validate_program_plan_expansion(
    *,
    tasks: Mapping[str, Mapping[str, str]],
    artifact_root: Path,
) -> list[str]:
    """Validate the canonical 27-task expansion and protected activation gate."""

    errors: list[str] = []
    prefix = "program_plan_expansion"
    expected_task_ids = {
        *_PROGRAM_CANONICAL_TASK_IDS,
        *_PROGRAM_NONCANONICAL_TASK_IDS,
    }
    if set(tasks) != expected_task_ids:
        errors.append(f"{prefix}.task_ids: exact canonical/noncanonical population required")

    observed_canonical = {
        task_id
        for task_id, metadata in tasks.items()
        if metadata.get("canonical board task", "true").strip().lower() != "false"
    }
    if observed_canonical != set(_PROGRAM_CANONICAL_TASK_IDS):
        errors.append(f"{prefix}.canonical_tasks: expected exact 27-task population")
    for task_id in _PROGRAM_NONCANONICAL_TASK_IDS:
        task = tasks.get(task_id)
        if task is not None and task.get("canonical board task") != "false":
            errors.append(f"{prefix}.{task_id}.canonical_board_task: expected false")

    for task_id, expected in _PROGRAM_EXPANSION_TASKS.items():
        task = tasks.get(task_id)
        if task is None:
            errors.append(f"{prefix}.{task_id}: expected exactly one task")
            continue
        if task.get(_TASK_TITLE_KEY) != expected["title"]:
            errors.append(f"{prefix}.{task_id}.title: exact title required")
        if _task_contract_sha256(task) != expected["contract_sha256"]:
            errors.append(
                f"{prefix}.{task_id}.contract_sha256: exact metadata/prose required"
            )
        for field, expected_value in {
            "status": "todo",
            "completion": "manual",
            "is schedulable": "true",
            "review only": "false",
            "priority": "P0",
            "canonical board task": "true",
            "goal id": expected["goal id"],
        }.items():
            if task.get(field) != expected_value:
                errors.append(
                    f"{prefix}.{task_id}.{field.replace(' ', '_')}: "
                    f"expected {expected_value!r}"
                )
        for field in ("depends on", "outputs", "predicted files"):
            expected_items = (
                expected["outputs"] if field == "predicted files" else expected[field]
            )
            if _taskboard_csv(task, field) != expected_items:
                errors.append(
                    f"{prefix}.{task_id}.{field.replace(' ', '_')}: "
                    "exact population required"
                )
        if task.get("validation") != expected["validation"]:
            errors.append(f"{prefix}.{task_id}.validation: exact command required")
        try:
            task_cid = _canonical_task_cid_from_metadata(task)
        except ValueError as exc:
            errors.append(f"{prefix}.{task_id}.canonical_task_cid: {exc}")
        else:
            if task_cid != expected["canonical_task_cid"]:
                errors.append(
                    f"{prefix}.{task_id}.canonical_task_cid: semantic identity drift"
                )

    identity_acceptance_path = (
        artifact_root / HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME
    )
    try:
        identity_acceptance_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        errors.append(
            f"{prefix}.ASE3-030.acceptance_receipt: unable to inspect reserved "
            f"path: {exc}"
        )
    else:
        errors.append(
            f"{prefix}.ASE3-030.acceptance_receipt: present without a strict "
            "schema validator and convergence-manifest binding"
        )
    identity_task = tasks.get("ASE3-030")
    if identity_task is not None and identity_task.get("status") == "completed":
        errors.append(
            f"{prefix}.ASE3-030.status: completion requires the strict reserved "
            "receipt validator and convergence-manifest binding"
        )

    for task_id, expected_cid in _PROGRAM_AMENDED_TASK_CIDS.items():
        task = tasks.get(task_id)
        if task is None:
            errors.append(f"{prefix}.{task_id}: expected exactly one amended task")
            continue
        if (
            _task_contract_sha256(task)
            != _PROGRAM_AMENDED_TASK_CONTRACT_SHA256S[task_id]
        ):
            errors.append(
                f"{prefix}.{task_id}.contract_sha256: exact amended "
                "metadata/prose required"
            )
        if task.get("status") != "todo":
            errors.append(f"{prefix}.{task_id}.status: expected 'todo'")
        if _taskboard_csv(task, "depends on") != _PROGRAM_AMENDED_TASK_DEPENDENCIES[
            task_id
        ]:
            errors.append(f"{prefix}.{task_id}.depends_on: exact expansion required")
        try:
            task_cid = _canonical_task_cid_from_metadata(task)
        except ValueError as exc:
            errors.append(f"{prefix}.{task_id}.canonical_task_cid: {exc}")
        else:
            if task_cid != expected_cid:
                errors.append(
                    f"{prefix}.{task_id}.canonical_task_cid: intentional amended "
                    "identity required"
                )
        searchable = " ".join(task.values())
        for requirement in _PROGRAM_AMENDED_TASK_REQUIREMENTS[task_id]:
            if requirement not in searchable:
                errors.append(
                    f"{prefix}.{task_id}.contract: missing {requirement!r}"
                )

    for task_id, expected_contract in (
        _PROGRAM_UNCHANGED_FUTURE_TASK_CONTRACT_SHA256S.items()
    ):
        task = tasks.get(task_id)
        if task is None:
            errors.append(f"{prefix}.{task_id}: expected exactly one future task")
            continue
        if _task_contract_sha256(task) != expected_contract:
            errors.append(
                f"{prefix}.{task_id}.contract_sha256: exact unchanged "
                "metadata/prose required"
            )
        if task.get("status") != "todo":
            errors.append(f"{prefix}.{task_id}.status: expected 'todo'")

    activation_path = artifact_root / PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME
    try:
        activation_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        errors.append(f"{prefix}.ASE3-026.receipt: unable to inspect: {exc}")
    else:
        errors.append(
            f"{prefix}.ASE3-026.receipt: present without strict validation and "
            "convergence-manifest binding"
        )
    activation = tasks.get(_PROTECTED_RUNTIME_ACTIVATION_TASK_ID)
    if activation is None:
        errors.append(f"{prefix}.ASE3-026: expected exactly one activation task")
    else:
        if (
            _task_contract_sha256(activation)
            != _PROTECTED_RUNTIME_ACTIVATION_CONTRACT_SHA256
        ):
            errors.append(
                f"{prefix}.ASE3-026.contract_sha256: exact blocked protected "
                "activation contract required"
            )
        for field, expected_value in {
            "status": "blocked",
            "completion": "manual",
            "is schedulable": "false",
            "review only": "true",
            "canonical board task": "true",
            "blocked reason": _PROTECTED_RUNTIME_ACTIVATION_BLOCKED_REASON,
            "goal id": "ASE3-G060",
        }.items():
            if activation.get(field) != expected_value:
                errors.append(
                    f"{prefix}.ASE3-026.{field.replace(' ', '_')}: "
                    f"expected {expected_value!r}"
                )
        try:
            activation_cid = _canonical_task_cid_from_metadata(activation)
        except ValueError as exc:
            errors.append(f"{prefix}.ASE3-026.canonical_task_cid: {exc}")
        else:
            if activation_cid != _PROTECTED_RUNTIME_ACTIVATION_TASK_CID:
                errors.append(f"{prefix}.ASE3-026.canonical_task_cid: mismatch")
        if PROTECTED_RUNTIME_ACTIVATION_RECEIPT_RELATIVE_PATH not in _taskboard_csv(
            activation,
            "outputs",
        ):
            errors.append(f"{prefix}.ASE3-026.outputs: activation receipt required")

    dependency_graph = {
        task_id: _taskboard_csv(metadata, "depends on")
        for task_id, metadata in tasks.items()
    }
    for task_id, dependencies in dependency_graph.items():
        unknown = sorted(set(dependencies) - set(tasks))
        if unknown:
            errors.append(
                f"{prefix}.{task_id}.depends_on: unknown dependencies "
                + ",".join(unknown)
            )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            errors.append(f"{prefix}.dependency_graph: cycle includes {task_id}")
            return
        visiting.add(task_id)
        for dependency in dependency_graph.get(task_id, ()):
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in sorted(dependency_graph):
        visit(task_id)

    task_order = tuple(tasks)
    required_order = (
        "ASE3-019",
        "ASE3-030",
        "ASE3-023",
        "ASE3-022",
        "ASE3-029",
        "ASE3-028",
        "ASE3-024",
        "ASE3-025",
    )
    try:
        positions = tuple(task_order.index(task_id) for task_id in required_order)
    except ValueError:
        pass  # Missing IDs are reported by the exact-population checks above.
    else:
        if positions != tuple(sorted(positions)):
            errors.append(
                f"{prefix}.task_order: exact hermetic/transition/layering chain "
                "required"
            )
    return errors


def _validate_program_scheduler_projection(
    *,
    repo_root: Path,
    tasks: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """Join the checked-in scheduler's groups/dependencies to the taskboard."""

    errors: list[str] = []
    prefix = "program_scheduler_projection"
    config_path = repo_root / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    try:
        config = _load_json(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"{prefix}: {exc}"]

    goals: Mapping[str, Mapping[str, str]] = {}
    objectives_path = repo_root / PROMPT_V3_OBJECTIVES_RELATIVE_PATH
    try:
        objectives_text = _read_regular_bytes(objectives_path).decode("utf-8")
        goals = _parse_objective_metadata(objectives_text)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"{prefix}.objectives: {exc}")

    plan_path = repo_root / PROMPT_V3_PLAN_RELATIVE_PATH
    try:
        plan_text = _read_regular_bytes(plan_path).decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"{prefix}.plan: {exc}")
    else:
        required_waves = (
            "Wave 3b identity:  ASE3-030",
            "Wave 3b adaptive:  ASE3-023",
            "Wave 3c gate:      ASE3-022",
            "Wave 3d:           ASE3-029",
            "Wave 3e:           ASE3-028",
            "Wave 3f:           ASE3-024",
            "Wave 3g:           ASE3-025",
            "Wave 3h:           ASE3-021",
            "Wave 3i:           ASE3-020",
            "Wave 4:            ASE3-008",
            "Wave 4b gate:      ASE3-026",
            "Wave 5:            ASE3-009",
            "Wave 6 (parallel): ASE3-010",
            "Wave 7:            ASE3-012",
            "Wave 8:            ASE3-013",
            "Wave 9:            ASE3-014",
        )
        positions = [plan_text.find(fragment) for fragment in required_waves]
        if any(position < 0 for position in positions) or positions != sorted(
            positions
        ):
            errors.append(f"{prefix}.plan.waves: exact ordered expansion required")
        if "`monitor_policy.canary_observation_seconds: 900`" not in plan_text:
            errors.append(
                f"{prefix}.plan.canary_observation_seconds: exact signed "
                "900-second policy required"
            )
        for field, fragment in {
            "hermetic_identity_acceptance_receipt": (
                HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH
            ),
            "hermetic_identity_acceptance_schema": (
                HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA
            ),
            "hermetic_identity_acceptance_fan_in": "Q→R→P→A",
        }.items():
            if fragment not in plan_text:
                errors.append(f"{prefix}.plan.{field}: exact protected join required")

    initial = config.get("initial_projection")
    if not isinstance(initial, Mapping):
        errors.append(f"{prefix}.initial_projection: expected object")
    else:
        if initial.get("task_count") != 27:
            errors.append(f"{prefix}.initial_projection.task_count: expected 27")
        if initial.get("canonical_task_ids") != list(_PROGRAM_CANONICAL_TASK_IDS):
            errors.append(
                f"{prefix}.initial_projection.canonical_task_ids: exact population "
                "required"
            )
        if initial.get("noncanonical_transition_task_ids") != ["ASE3-022"]:
            errors.append(
                f"{prefix}.initial_projection.noncanonical_transition_task_ids: "
                "expected ASE3-022"
            )

    groups = config.get("task_groups")
    if not isinstance(groups, Mapping):
        errors.append(f"{prefix}.task_groups: expected object")
    else:
        grouped: list[str] = []
        for goal_id, raw_task_ids in groups.items():
            if not isinstance(raw_task_ids, list):
                errors.append(f"{prefix}.task_groups.{goal_id}: expected array")
                continue
            for task_id in raw_task_ids:
                grouped.append(str(task_id))
                task = tasks.get(str(task_id))
                if task is None or task.get("goal id") != goal_id:
                    errors.append(
                        f"{prefix}.task_groups.{goal_id}: task/goal mismatch for "
                        f"{task_id}"
                    )
        if len(grouped) != len(set(grouped)):
            errors.append(f"{prefix}.task_groups: duplicate task membership")
        if set(grouped) != set(_PROGRAM_CANONICAL_TASK_IDS):
            errors.append(f"{prefix}.task_groups: exact 27-task population required")
        expected_goal_ids = {"ASE3-G000", *(str(goal_id) for goal_id in groups)}
        if set(goals) != expected_goal_ids:
            errors.append(f"{prefix}.objectives.goal_ids: exact population required")
        root_goal = goals.get("ASE3-G000")
        if root_goal is None or _taskboard_csv(
            root_goal,
            "producing tasks",
        ) != _PROGRAM_CANONICAL_TASK_IDS:
            errors.append(
                f"{prefix}.objectives.ASE3-G000.producing_tasks: exact canonical "
                "population required"
            )
        for goal_id, raw_task_ids in groups.items():
            goal = goals.get(str(goal_id))
            expected_producers = tuple(str(task_id) for task_id in raw_task_ids)
            if goal is None or _taskboard_csv(
                goal,
                "producing tasks",
            ) != expected_producers:
                errors.append(
                    f"{prefix}.objectives.{goal_id}.producing_tasks: task-group "
                    "mismatch"
                )

    dependencies = config.get("task_dependencies")
    if not isinstance(dependencies, Mapping):
        errors.append(f"{prefix}.task_dependencies: expected object")
    elif set(dependencies) != set(_PROGRAM_CANONICAL_TASK_IDS):
        errors.append(f"{prefix}.task_dependencies: exact key population required")
    else:
        for task_id in _PROGRAM_CANONICAL_TASK_IDS:
            if dependencies.get(task_id) != list(
                _taskboard_csv(tasks[task_id], "depends on")
            ):
                errors.append(
                    f"{prefix}.task_dependencies.{task_id}: taskboard mismatch"
                )

    expected_acceptance_prerequisites = {
        "ASE3-023": ["ASE3-030"],
        "ASE3-022": ["ASE3-030"],
    }
    if config.get("acceptance_prerequisites") != expected_acceptance_prerequisites:
        errors.append(
            f"{prefix}.acceptance_prerequisites: exact ASE3-030 fail-closed "
            "acceptance join required"
        )

    expected_identity_acceptance = {
        "task_id": "ASE3-030",
        "status": "reserved",
        "receipt_path": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        "receipt_schema": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA,
        "strict_validator_and_manifest_binding_required": True,
        "required_before_task_acceptance": ["ASE3-023", "ASE3-022"],
    }
    if config.get("protected_identity_acceptance") != expected_identity_acceptance:
        errors.append(
            f"{prefix}.protected_identity_acceptance: exact reserved ASE3-030 "
            "receipt contract required"
        )

    expected_activation = {
        "task_id": "ASE3-026",
        "status": "blocked",
        "receipt_path": PROTECTED_RUNTIME_ACTIVATION_RECEIPT_RELATIVE_PATH,
        "operator_review_required": True,
        "strict_validator_and_manifest_binding_required": True,
    }
    if config.get("protected_runtime_activation") != expected_activation:
        errors.append(f"{prefix}.protected_runtime_activation: exact gate required")
    if config.get("strict_task_sharding") is not True:
        errors.append(f"{prefix}.strict_task_sharding: must remain true before gate")
    if config.get("objective_refill_enabled") is not False:
        errors.append(
            f"{prefix}.objective_refill_enabled: must remain false before gate"
        )
    if config.get("codebase_refill_enabled") is not False:
        errors.append(f"{prefix}.codebase_refill_enabled: must remain false")
    refill = config.get("refill_policy")
    if not isinstance(refill, Mapping):
        errors.append(f"{prefix}.refill_policy: expected object")
    else:
        for field, expected in {
            "enable_after_task": "ASE3-026",
            "activation_task_id": "ASE3-026",
            "prompt_program_refill_enabled": False,
        }.items():
            if refill.get(field) != expected:
                errors.append(f"{prefix}.refill_policy.{field}: expected {expected!r}")
    monitor = config.get("monitor_policy")
    if not isinstance(monitor, Mapping):
        errors.append(f"{prefix}.monitor_policy: expected object")
    else:
        expected_monitor_policy = {
            "enabled": False,
            "detached": True,
            "activation_task_id": "ASE3-026",
            "heartbeat_seconds": 5,
            "stale_control_seconds": 30,
            "semantic_progress_seconds": 300,
            "max_recoveries_per_window": 3,
            "recovery_window_seconds": 1800,
            "canary_task_id": "ASE3-013",
            "canary_observation_seconds": 900,
            "continuous_health_required": True,
            "monotonic_elapsed_receipt_required": True,
            "prompt_may_override_observation_window": False,
            "running_requires_process_birth_lease_fence_and_heartbeat": True,
            "queue_drain_is_completion": False,
            "branch_local_completion_is_completion": False,
        }
        for field, expected in expected_monitor_policy.items():
            if monitor.get(field) != expected:
                errors.append(f"{prefix}.monitor_policy.{field}: expected {expected!r}")
    return errors


def _validate_merge_receipt_snapshot(
    *,
    task_id: str,
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    digest: str,
) -> list[str]:
    errors: list[str] = []
    prefix = f"false_completion_merge_receipt.{task_id}"
    expected_scalar = {
        "task_id": task_id,
        "commit_sha": record["implementation_commit"],
        "accepted": True,
        "acceptance_pending": False,
        "integrated": True,
        "merged": True,
        "status": "merged",
        "target_branch": _FALSE_COMPLETION_RECOVERY_SOURCE["branch"],
        "merge_commit": record["status_commit"],
        "target_commit": record["status_commit"],
    }
    for field, expected in expected_scalar.items():
        actual = payload.get(field)
        matches = actual is expected if isinstance(expected, bool) else actual == expected
        if not matches:
            errors.append(f"{prefix}.{field}: expected {expected!r}")
    if digest != record["merge_receipt_sha256"]:
        errors.append(f"{prefix}.sha256: recovery digest mismatch")

    merge_result = payload.get("merge_result")
    if not isinstance(merge_result, Mapping):
        errors.append(f"{prefix}.merge_result: expected object")
        return errors
    if merge_result.get("merged") is not True:
        errors.append(f"{prefix}.merge_result.merged: expected true")
    returncode = merge_result.get("returncode")
    if type(returncode) is not int or returncode != 0:
        errors.append(f"{prefix}.merge_result.returncode: expected integer zero")
    if merge_result.get("merge_commit") != record["merge_commit"]:
        errors.append(f"{prefix}.merge_result.merge_commit: mismatch")
    if merge_result.get("target_branch") != _FALSE_COMPLETION_RECOVERY_SOURCE["branch"]:
        errors.append(f"{prefix}.merge_result.target_branch: mismatch")

    proof = merge_result.get("integration_commit_proof")
    if not isinstance(proof, Mapping):
        errors.append(f"{prefix}.integration_commit_proof: expected object")
    else:
        expected_proof = {
            "implementation_commit": record["implementation_commit"],
            "integration_commit": record["merge_commit"],
            "integration_ref": record["merge_commit"],
            "target_branch": _FALSE_COMPLETION_RECOVERY_SOURCE["branch"],
            "passed": True,
            "reasons": [],
        }
        for field, expected in expected_proof.items():
            actual = proof.get(field)
            matches = (
                actual is expected
                if isinstance(expected, bool)
                else isinstance(actual, list) and actual == expected
                if isinstance(expected, list)
                else actual == expected
            )
            if not matches:
                errors.append(f"{prefix}.integration_commit_proof.{field}: mismatch")

    todo_result = merge_result.get("todo_update_result")
    if not isinstance(todo_result, Mapping):
        errors.append(f"{prefix}.todo_update_result: expected object")
        return errors
    for field, expected in {
        "task_id": task_id,
        "updated": True,
        "durable": True,
    }.items():
        actual = todo_result.get(field)
        matches = actual is expected if isinstance(expected, bool) else actual == expected
        if not matches:
            errors.append(f"{prefix}.todo_update_result.{field}: mismatch")
    commit_result = todo_result.get("commit_result")
    if not isinstance(commit_result, Mapping):
        errors.append(f"{prefix}.todo_update_result.commit_result: expected object")
    elif (
        commit_result.get("committed") is not True
        or commit_result.get("commit") != record["status_commit"]
        or commit_result.get("path") != PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
    ):
        errors.append(f"{prefix}.todo_update_result.commit_result: mismatch")
    completion_receipts = todo_result.get("completion_receipts")
    if not isinstance(completion_receipts, list) or len(completion_receipts) != 1:
        errors.append(f"{prefix}.completion_receipts: expected exactly one")
    else:
        member = completion_receipts[0]
        if not isinstance(member, Mapping):
            errors.append(f"{prefix}.completion_receipts[0]: expected object")
        else:
            expected_member = {
                "board_namespace": BOARD_NAMESPACE,
                "canonical_task_cid": record["canonical_task_cid"],
                "task_id": task_id,
                "status": "succeeded",
            }
            for field, expected in expected_member.items():
                if member.get(field) != expected:
                    errors.append(
                        f"{prefix}.completion_receipts[0].{field}: mismatch"
                    )
    postcondition = todo_result.get("protected_board_postcondition")
    if not isinstance(postcondition, Mapping):
        errors.append(f"{prefix}.protected_board_postcondition: expected object")
    else:
        for field in ("checked", "clean", "trusted"):
            if postcondition.get(field) is not True:
                errors.append(
                    f"{prefix}.protected_board_postcondition.{field}: expected true"
                )
        release_proof = postcondition.get("release_proof")
        if not isinstance(release_proof, Mapping):
            errors.append(
                f"{prefix}.protected_board_postcondition.release_proof: "
                "expected object"
            )
        else:
            for field in ("clean", "trusted"):
                if release_proof.get(field) is not True:
                    errors.append(
                        f"{prefix}.protected_board_postcondition.release_proof."
                        f"{field}: expected true"
                    )
    return errors


def _validate_failed_event_snapshot(
    *,
    payload: Mapping[str, Any],
    digest: str,
) -> list[str]:
    errors: list[str] = []
    prefix = "failed_validation_event.ASE3-019"
    expected = _FALSE_COMPLETION_FAILED_ATTEMPT
    if digest != expected["failed_event_snapshot_sha256"]:
        errors.append(f"{prefix}.sha256: recovery digest mismatch")
    body = dict(payload)
    claimed_event_id = str(body.pop("event_id", ""))
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        errors.append(f"{prefix}.event_id: noncanonical event: {exc}")
    else:
        computed_event_id = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if claimed_event_id != computed_event_id:
            errors.append(f"{prefix}.event_id: noncanonical identity")
        if claimed_event_id != expected["failed_event_id"]:
            errors.append(f"{prefix}.event_id: recovery identity mismatch")
    expected_event_fields = {
        "type": "failed_validation_worktree_preserved",
        "task_id": expected["task_id"],
        "board_namespace": BOARD_NAMESPACE,
        "canonical_task_cid": expected["canonical_task_cid"],
        "attempt": expected["attempt"],
        "implementation_commit": expected["implementation_commit"],
        "preserved_commit": expected["implementation_commit"],
        "rescue_branch": expected["rescue_branch"],
        "preserved": True,
    }
    for field, expected_value in expected_event_fields.items():
        actual = payload.get(field)
        matches = (
            actual is expected_value
            if isinstance(expected_value, bool)
            else actual == expected_value
        )
        if not matches:
            errors.append(f"{prefix}.{field}: expected {expected_value!r}")
    for field in ("merge_result", "merge_commit", "merge_dispatched"):
        if field in payload:
            errors.append(f"{prefix}.{field}: must be absent for a failed attempt")
    commit_result = payload.get("commit_result")
    if not isinstance(commit_result, Mapping):
        errors.append(f"{prefix}.commit_result: expected object")
    elif (
        commit_result.get("committed") is not True
        or commit_result.get("commit") != expected["implementation_commit"]
    ):
        errors.append(f"{prefix}.commit_result: candidate binding mismatch")
    validation = payload.get("validation_result")
    if not isinstance(validation, Mapping):
        errors.append(f"{prefix}.validation_result: expected object")
        return errors
    expected_validation = {
        "attempted": True,
        "passed": False,
        "authoritative": False,
        "completion_authoritative": False,
        "merge_eligible": False,
        "returncode": expected["validation_returncode"],
        "target_commit": _FALSE_COMPLETION_RECOVERY_SOURCE["launch_base_head"],
    }
    for field, expected_value in expected_validation.items():
        actual = validation.get(field)
        matches = (
            actual is expected_value
            if isinstance(expected_value, bool)
            else actual == expected_value
        )
        if not matches:
            errors.append(f"{prefix}.validation_result.{field}: mismatch")
    dag = validation.get("validation_dag_receipt")
    if not isinstance(dag, Mapping):
        errors.append(f"{prefix}.validation_dag_receipt: expected object")
    elif (
        dag.get("passed") is not False
        or dag.get("completion_authoritative") is not False
        or dag.get("repository_tree_id")
        != _FALSE_COMPLETION_RECOVERY_SOURCE["launch_base_head"]
    ):
        errors.append(f"{prefix}.validation_dag_receipt: failed gate mismatch")
    return errors


def _validate_attempt2_failed_event_snapshot(
    *,
    payload: Mapping[str, Any],
    digest: str,
) -> list[str]:
    """Bind the exhausted attempt-2 terminal event without granting completion."""

    errors: list[str] = []
    prefix = "self_host_seed_failure.ASE3-019.event_snapshot"
    if digest != _ASE3_019_ATTEMPT2_EVENT_SHA256:
        errors.append(f"{prefix}.sha256: exact attempt-2 event digest required")
    expected_event_order = (
        "prior_attempt_seeded",
        "implementation_started",
        "implementation_finished",
        "implementation_shutdown_reconciled",
    )
    expected_event_ids = (
        _ASE3_019_ATTEMPT2_SEED_EVENT_ID,
        _ASE3_019_ATTEMPT2_STARTED_EVENT_ID,
        _ASE3_019_ATTEMPT2_EVENT_ID,
        _ASE3_019_ATTEMPT2_SHUTDOWN_EVENT_ID,
    )
    expected_sequences = (128, 130, 134, 160)
    expected_types = (
        "implementation_prior_attempt_seeded",
        "implementation_started",
        "implementation_finished",
        "implementation_shutdown_reconciled",
    )
    expected_wrapper_fields = {
        "schema": ASE3_019_ATTEMPT2_EVENT_BUNDLE_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "task_id": "ASE3-019",
        "canonical_task_cid": _ASE3_019_ATTEMPT2_TASK_CID,
        "canonical_task_key": _ASE3_019_ATTEMPT2_TASK_KEY,
        "attempt": 2,
        "event_order": list(expected_event_order),
        "sequence_order": list(expected_sequences),
        "event_ids": list(expected_event_ids),
        "completion_authority": False,
    }
    for field, expected in expected_wrapper_fields.items():
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.{field}",
            actual=payload.get(field),
            expected=expected,
        )
    expected_wrapper_keys = {
        *expected_wrapper_fields,
        "stream_id",
        "snapshot_id",
        "events",
    }
    if set(payload) != expected_wrapper_keys:
        errors.append(f"{prefix}: exact bundle key population required")
    stream_id = payload.get("stream_id")
    snapshot_id = payload.get("snapshot_id")
    if not isinstance(stream_id, str) or not stream_id.startswith("event-log:sha256:"):
        errors.append(f"{prefix}.stream_id: exact event-log identity required")
    if (
        not isinstance(snapshot_id, str)
        or not snapshot_id.startswith("event-log-snapshot:sha256:")
    ):
        errors.append(f"{prefix}.snapshot_id: exact event-log snapshot required")
    events = payload.get("events")
    if not isinstance(events, Mapping):
        errors.append(f"{prefix}.events: expected object")
        return errors
    if tuple(events) != expected_event_order:
        errors.append(f"{prefix}.events: exact ordered population required")

    normalized_events: list[Mapping[str, Any]] = []
    for index, (event_name, expected_event_id, expected_sequence, expected_type) in (
        enumerate(
            zip(
                expected_event_order,
                expected_event_ids,
                expected_sequences,
                expected_types,
                strict=True,
            )
        )
    ):
        event = events.get(event_name)
        event_prefix = f"{prefix}.events.{event_name}"
        if not isinstance(event, Mapping):
            errors.append(f"{event_prefix}: expected object")
            continue
        normalized_events.append(event)
        event_body = dict(event)
        claimed_id = str(event_body.pop("event_id", ""))
        try:
            canonical = json.dumps(
                event_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            errors.append(f"{event_prefix}.event_id: noncanonical event: {exc}")
        else:
            recomputed_id = "sha256:" + hashlib.sha256(canonical).hexdigest()
            if claimed_id != recomputed_id or claimed_id != expected_event_id:
                errors.append(f"{event_prefix}.event_id: exact identity required")
        for field, expected in {
            "sequence": expected_sequence,
            "type": expected_type,
            "stream_id": stream_id,
            "snapshot_id": snapshot_id,
        }.items():
            _validate_exact_structure(
                errors,
                prefix=f"{event_prefix}.{field}",
                actual=event.get(field),
                expected=expected,
            )
        if index < 3:
            for field, expected in {
                "task_id": "ASE3-019",
                "canonical_task_cid": _ASE3_019_ATTEMPT2_TASK_CID,
                "canonical_task_key": _ASE3_019_ATTEMPT2_TASK_KEY,
                "attempt": 2,
            }.items():
                _validate_exact_structure(
                    errors,
                    prefix=f"{event_prefix}.{field}",
                    actual=event.get(field),
                    expected=expected,
                )
    if len(normalized_events) != 4:
        return errors

    seed_event, started_event, finished_event, shutdown_event = normalized_events
    for field, expected in {
        "applied": True,
        "baseline_ref": _ASE3_019_ATTEMPT2_LAUNCH["launch_head"],
        "branch": _ASE3_019_ATTEMPT2_BRANCH,
        "merge_base": _ASE3_019_ATTEMPT2_PRIOR_SEED["merge_base"],
        "no_change": False,
        "reason": "replayed_prior_delta",
        "seed_ref": _ASE3_019_ATTEMPT2_PRIOR_SEED["source_commit"],
        "replayed_root_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
        "scope_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
        "skipped_root_paths": [],
    }.items():
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.events.prior_attempt_seeded.{field}",
            actual=seed_event.get(field),
            expected=expected,
        )
    for field, expected in {
        "baseline_ref": _ASE3_019_ATTEMPT2_LAUNCH["launch_head"],
        "branch": _ASE3_019_ATTEMPT2_BRANCH,
        "execution_mode": "model-assisted",
    }.items():
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.events.implementation_started.{field}",
            actual=started_event.get(field),
            expected=expected,
        )
    command = started_event.get("command")
    if (
        not isinstance(command, list)
        or command[:3]
        != [
            "/home/barberb/.local/bin/python",
            "-m",
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        ]
        or "--agent-implementation-route-json" not in command
        or "--grok-failure-receipt-nonce" not in command
    ):
        errors.append(
            f"{prefix}.events.implementation_started.command: exact runner "
            "module and bound route arguments required"
        )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.events.implementation_shutdown_reconciled",
        actual={
            field: shutdown_event.get(field)
            for field in (
                "attempt",
                "attempt_recovery",
                "blocked",
                "task_id",
                "reason",
                "reconciled",
                "stale_lock_cleared",
            )
        },
        expected={
            "attempt": 0,
            "attempt_recovery": {},
            "blocked": False,
            "task_id": "ASE3-019",
            "reason": "already_quiesced",
            "reconciled": True,
            "stale_lock_cleared": False,
        },
    )

    payload = finished_event
    body = dict(payload)
    claimed_event_id = str(body.pop("event_id", ""))
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        errors.append(f"{prefix}.event_id: noncanonical event: {exc}")
    else:
        computed = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if claimed_event_id != computed:
            errors.append(f"{prefix}.event_id: noncanonical identity")
        if claimed_event_id != _ASE3_019_ATTEMPT2_EVENT_ID:
            errors.append(f"{prefix}.event_id: exact attempt-2 identity required")

    expected_event_fields = {
        "type": "implementation_finished",
        "task_id": "ASE3-019",
        "board_namespace": BOARD_NAMESPACE,
        "canonical_task_cid": _ASE3_019_ATTEMPT2_TASK_CID,
        "canonical_task_key": _ASE3_019_ATTEMPT2_TASK_KEY,
        "attempt": 2,
        "returncode": 2,
        "provider_dispatched": True,
        "attempt_consumed": True,
        "implementation_commit": "",
    }
    for field, expected in expected_event_fields.items():
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.{field}",
            actual=payload.get(field),
            expected=expected,
        )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.commit_result",
        actual=payload.get("commit_result"),
        expected={"committed": False},
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.validation_result",
        actual=payload.get("validation_result"),
        expected={
            "attempted": False,
            "passed": True,
            "reason": "not_run",
            "results": [],
            "returncode": 0,
        },
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.merge_result",
        actual=payload.get("merge_result"),
        expected={"merged": False, "reason": "not_attempted"},
    )

    workspace = payload.get("workspace_setup")
    if not isinstance(workspace, Mapping):
        errors.append(f"{prefix}.workspace_setup: expected object")
        return errors
    for field, expected in {
        "base_commit": _ASE3_019_ATTEMPT2_LAUNCH["launch_head"],
        "base_ref": _ASE3_019_ATTEMPT2_LAUNCH["branch"],
        "branch": _ASE3_019_ATTEMPT2_BRANCH,
        "cache_hit": False,
        "reused": False,
        "dependency_paths": [],
    }.items():
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.workspace_setup.{field}",
            actual=workspace.get(field),
            expected=expected,
        )
    seed = workspace.get("prior_attempt_seed")
    if not isinstance(seed, Mapping):
        errors.append(f"{prefix}.workspace_setup.prior_attempt_seed: expected object")
        return errors
    for field, expected in {
        "applied": True,
        "baseline_ref": _ASE3_019_ATTEMPT2_LAUNCH["launch_head"],
        "merge_base": _ASE3_019_ATTEMPT2_PRIOR_SEED["merge_base"],
        "no_change": False,
        "reason": "replayed_prior_delta",
        "seed_ref": _ASE3_019_ATTEMPT2_PRIOR_SEED["source_commit"],
        "replayed_root_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
        "scope_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
        "skipped_root_paths": [],
    }.items():
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.workspace_setup.prior_attempt_seed.{field}",
            actual=seed.get(field),
            expected=expected,
        )
    proposal_gate = seed.get("pre_dispatch_proposal_gate")
    if not isinstance(proposal_gate, Mapping):
        errors.append(
            f"{prefix}.workspace_setup.prior_attempt_seed."
            "pre_dispatch_proposal_gate: expected object"
        )
    else:
        for field, expected in {
            "accepted": True,
            "attempted": True,
            "changed_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
            "completion_authoritative": False,
            "proof_authoritative": False,
            "reason_codes": [],
            "repository_tree_id": _ASE3_019_ATTEMPT2_LAUNCH["launch_head"],
        }.items():
            _validate_exact_structure(
                errors,
                prefix=(
                    f"{prefix}.workspace_setup.prior_attempt_seed."
                    f"pre_dispatch_proposal_gate.{field}"
                ),
                actual=proposal_gate.get(field),
                expected=expected,
            )
    authority = seed.get("proposal_authority")
    if not isinstance(authority, Mapping):
        errors.append(
            f"{prefix}.workspace_setup.prior_attempt_seed."
            "proposal_authority: expected object"
        )
    else:
        for field, expected in {
            "ok": True,
            "task_id": "ASE3-019",
            "canonical_task_cid": _ASE3_019_ATTEMPT2_TASK_CID,
            "canonical_task_key": _ASE3_019_ATTEMPT2_TASK_KEY,
            "authorized_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
            "declared_scope_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
            "receipt_paths": list(_ASE3_019_ATTEMPT2_REPLAYED_PATHS),
            "dropped_protected_paths": [],
            "dropped_receipt_paths": [],
        }.items():
            _validate_exact_structure(
                errors,
                prefix=(
                    f"{prefix}.workspace_setup.prior_attempt_seed."
                    f"proposal_authority.{field}"
                ),
                actual=authority.get(field),
                expected=expected,
            )
    return errors


def _validate_attempt2_failed_log_snapshot(*, raw: bytes, digest: str) -> list[str]:
    """Validate the byte-exact redacted attempt-2 implementation-runner log."""

    errors: list[str] = []
    prefix = "self_host_seed_failure.ASE3-019.log_snapshot"
    if digest != _ASE3_019_ATTEMPT2_LOG_SHA256:
        errors.append(f"{prefix}.sha256: exact attempt-2 log digest required")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{prefix}: expected UTF-8 text")
        return errors
    required_fragments = (
        (
            "Task: ASE3-019 Seal signed provider authority, authentication lifecycle, "
            "and once-only fallback\n"
        ),
        f"Branch: {_ASE3_019_ATTEMPT2_BRANCH}\n",
        f"Baseline: {_ASE3_019_ATTEMPT2_LAUNCH['launch_head']}\n",
        " -m ipfs_accelerate_py.agent_supervisor.grok_cli_runner ",
        '"fallback_model_id":"gpt-5.6-terra"',
        '"fallback_reasoning_effort":"high"',
    )
    for fragment in required_fragments:
        if text.count(fragment) != 1:
            errors.append(f"{prefix}: exact launch binding is missing or duplicated")
    if not text.endswith(f"\n{_ASE3_019_ATTEMPT2_NORMALIZED_ERROR}\n"):
        errors.append(f"{prefix}.terminal_error: exact normalized error required")
    for forbidden in (
        "GROK_FAILURE_RECEIPT ",
        "GROK_ROUTE_OUTCOME ",
        "Authorization: Bearer ",
        '"access_token"',
        '"refresh_token"',
        '"client_secret"',
    ):
        if forbidden in text:
            errors.append(f"{prefix}: forbidden receipt or secret material present")
    return errors


def _validate_attempt2_self_host_incident(
    *,
    payload: Mapping[str, Any],
    digest: str,
    event_digest: str,
    log_digest: str,
) -> list[str]:
    """Validate the immutable C1 incident and its future no-provider salvage gate."""

    errors: list[str] = []
    prefix = "self_host_seed_failure.ASE3-019.incident"
    if digest != _ASE3_019_ATTEMPT2_INCIDENT_SHA256:
        errors.append(f"{prefix}.sha256: exact incident digest required")
    expected = {
        "schema": ASE3_019_ATTEMPT2_SELF_HOST_INCIDENT_SCHEMA,
        "created_at": _ASE3_019_ATTEMPT2_CREATED_AT,
        "board_namespace": BOARD_NAMESPACE,
        "task": {
            "task_id": "ASE3-019",
            "title": _ASE3_019_TITLE,
            "canonical_task_cid": _ASE3_019_ATTEMPT2_TASK_CID,
            "canonical_task_key": _ASE3_019_ATTEMPT2_TASK_KEY,
            "attempt": 2,
            "max_attempts": 2,
            "board_status": "todo",
            "runtime_status": "ready",
            "completion": "manual",
        },
        "launch": _ASE3_019_ATTEMPT2_LAUNCH,
        "prior_attempt_seed": _ASE3_019_ATTEMPT2_PRIOR_SEED,
        "terminal_failure": {
            "finished_event_id": _ASE3_019_ATTEMPT2_EVENT_ID,
            "event_snapshot": FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME,
            "event_snapshot_sha256": _ASE3_019_ATTEMPT2_EVENT_SHA256,
            "log_snapshot": FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME,
            "log_snapshot_sha256": _ASE3_019_ATTEMPT2_LOG_SHA256,
            "returncode": 2,
            "normalized_error": _ASE3_019_ATTEMPT2_NORMALIZED_ERROR,
            "implementation_runner_dispatched": True,
            "primary_provider_effect_dispatched": False,
            "fallback_provider_effect_dispatched": False,
            "provider_failure_receipt_emitted": False,
            "route_outcome_emitted": False,
            "attempt_consumed": True,
            "validation_attempted": False,
            "implementation_commit": "",
            "merge_dispatched": False,
        },
        "control_plane_provenance": {
            "classification": (
                "candidate_workspace_shadowed_accepted_provider_control_plane"
            ),
            "python_executable": "/home/barberb/.local/bin/python",
            "module_argv": [
                "-m",
                "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
            ],
            "subprocess_cwd": (
                "/home/barberb/lift_coding/.worktrees/prompt-self-improvement-v3/"
                "data/agent_supervisor/prompt_only_self_improvement_v3/live/"
                "worktrees/workspace-7fb9a620a3a4-75e6cfd1bfbf"
            ),
            "candidate_package_imported_from_candidate_cwd": True,
            "invocation_command_sha256": (
                "sha256:bc60c2bc61d4da36d516bc6b0a7ee92e889f792e746c7816271dc7b9e26ce49f"
            ),
            "accepted_generation_blobs": _ASE3_019_ATTEMPT2_ACCEPTED_BLOBS,
            "seeded_candidate_blobs": _ASE3_019_ATTEMPT2_CANDIDATE_BLOBS,
            "candidate_only_required_route_fields": [
                "fallback_implementer_identity",
                "reviewer_identity",
                "reviewer_provider",
            ],
            "provider_capacity_failure": False,
            "accepted_control_plane_required_for_salvage": True,
        },
        "attempt_accounting": {
            "attempts_by_task_id": 2,
            "attempts_by_canonical_task_cid": 2,
            "queue_attempt_count": 2,
            "consecutive_failures": 2,
            "selection_penalty": 200,
            "exhausted": True,
            "attempt_restoration_authorized": False,
            "attempt_counter_mutation_authorized": False,
            "runtime_state_mutation_authorized": False,
            "queue_history_mutation_authorized": False,
        },
        "fence": {
            "interrupted_at": "2026-08-08T17:37:31Z",
            "shutdown_signal": "SIGINT",
            "lane_statuses": ["stopped", "stopped", "stopped"],
            "lane_restart_counts": [0, 0, 0],
            "lane_1_shutdown_event_id": _ASE3_019_ATTEMPT2_SHUTDOWN_EVENT_ID,
            "lane_1_active_attempt_cleared": True,
            "zero_owned_processes": True,
            "zero_scoped_provider_containers": True,
            "interrupted_lane_0_and_2_candidates_preserved": True,
        },
        "operator_salvage_gate": {
            "receipt_filename": OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
            "receipt_present": False,
            "ase3_019_required_status": "todo",
            "reload_gate_task_id": "ASE3-022",
            "reload_gate_required_status": "blocked",
            "provider_dispatch_authorized": False,
            "accepted_control_plane_required": True,
            "completion_authority": False,
            "required_receipt_fields": list(
                _ASE3_019_OPERATOR_SALVAGE_REQUIRED_FIELDS
            ),
        },
    }
    _validate_exact_structure(
        errors,
        prefix=prefix,
        actual=payload,
        expected=expected,
    )
    terminal = payload.get("terminal_failure", {})
    if isinstance(terminal, Mapping):
        if terminal.get("event_snapshot_sha256") != event_digest:
            errors.append(f"{prefix}.terminal_failure.event_snapshot_sha256: mismatch")
        if terminal.get("log_snapshot_sha256") != log_digest:
            errors.append(f"{prefix}.terminal_failure.log_snapshot_sha256: mismatch")
    return errors


def _validate_false_completion_snapshots(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    digests: Mapping[str, str],
) -> list[str]:
    errors: list[str] = []
    for task_id, record in _FALSE_COMPLETION_RECORDS.items():
        filename = str(record["merge_receipt_snapshot"])
        errors.extend(
            _validate_merge_receipt_snapshot(
                task_id=task_id,
                record=record,
                payload=payloads[filename],
                digest=str(digests.get(filename, "")),
            )
        )
    failed_filename = str(
        _FALSE_COMPLETION_FAILED_ATTEMPT["failed_event_snapshot"]
    )
    errors.extend(
        _validate_failed_event_snapshot(
            payload=payloads[failed_filename],
            digest=str(digests.get(failed_filename, "")),
        )
    )
    return errors


def _validate_attempt2_incident_packet(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    raw_artifacts: Mapping[str, bytes],
    digests: Mapping[str, str],
) -> list[str]:
    event_digest = str(
        digests.get(FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME, "")
    )
    log_digest = str(
        digests.get(FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME, "")
    )
    errors = _validate_attempt2_failed_event_snapshot(
        payload=payloads[FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME],
        digest=event_digest,
    )
    errors.extend(
        _validate_attempt2_failed_log_snapshot(
            raw=raw_artifacts[FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME],
            digest=log_digest,
        )
    )
    errors.extend(
        _validate_attempt2_self_host_incident(
            payload=payloads[SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME],
            digest=str(
                digests.get(SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME, "")
            ),
            event_digest=event_digest,
            log_digest=log_digest,
        )
    )
    return errors


def _validate_repository_binding(
    *,
    repo_root: Path,
    baseline: CurrentMainBaseline,
    rescue: RescueDispositionReport,
    post_wave3: PostWave3ResidualReport,
    false_completion_recovery: FalseCompletionRecoveryReport,
    fallback_authorization: ProviderFallbackPolicyAuthorization,
) -> list[str]:
    errors: list[str] = []
    repo_root = repo_root.resolve()
    if not (repo_root / ".git").exists():
        # Linked worktrees use a .git file; ordinary clones use a directory.
        errors.append(f"repository_binding.repo_root: not a Git worktree: {repo_root}")
        return errors

    identity_sections = (
        ("upstream_main", baseline.payload.get("upstream_main", {}), "commit", "tree"),
        (
            "integration_seed",
            baseline.payload.get("integration_seed", {}),
            "commit",
            "tree",
        ),
        ("rescue_head", baseline.payload.get("rescue", {}), "head", "tree"),
        (
            "merge_base",
            baseline.payload.get("rescue", {}),
            "merge_base",
            "merge_base_tree",
        ),
    )
    for label, section, commit_field, tree_field in identity_sections:
        if not isinstance(section, Mapping):
            errors.append(f"repository_binding.{label}: baseline section unavailable")
            continue
        identity = str(section.get(commit_field, ""))
        expected_tree = str(section.get(tree_field, ""))
        result = _git(repo_root, "rev-parse", "--verify", f"{identity}^{{tree}}")
        if result.returncode != 0:
            errors.append(f"repository_binding.{label}: Git object unavailable")
        elif result.stdout.strip() != expected_tree:
            errors.append(f"repository_binding.{label}.tree: Git identity mismatch")

    integration = baseline.payload.get("integration_seed", {})
    expected_parent = (
        str(integration.get("parent", "")) if isinstance(integration, Mapping) else ""
    )
    parents = _git(
        repo_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        baseline.integration_seed_commit,
    )
    parent_fields = parents.stdout.strip().split()
    if parents.returncode != 0 or parent_fields[1:] != [expected_parent]:
        errors.append("repository_binding.integration_seed.parent: Git identity mismatch")

    actual_merge_base = _git(
        repo_root,
        "merge-base",
        baseline.upstream_main_commit,
        baseline.rescue_head,
    )
    if (
        actual_merge_base.returncode != 0
        or actual_merge_base.stdout.strip() != baseline.merge_base
    ):
        errors.append("repository_binding.merge_base: computed identity mismatch")

    divergence = _git(
        repo_root,
        "rev-list",
        "--left-right",
        "--count",
        f"{baseline.upstream_main_commit}...{baseline.rescue_head}",
    )
    rescue_payload = baseline.payload.get("rescue", {})
    try:
        current_main_ahead, rescue_ahead = (
            int(item) for item in divergence.stdout.strip().split()
        )
    except (TypeError, ValueError):
        current_main_ahead = rescue_ahead = -1
    if divergence.returncode != 0 or not isinstance(rescue_payload, Mapping):
        errors.append("repository_binding.rescue.divergence: unable to compute")
    elif (
        current_main_ahead != rescue_payload.get("current_main_ahead")
        or rescue_ahead != rescue_payload.get("rescue_ahead")
    ):
        errors.append("repository_binding.rescue.divergence: baseline mismatch")

    submodules = baseline.payload.get("submodules", ())
    if isinstance(submodules, Sequence) and not isinstance(submodules, (str, bytes)):
        for index, item in enumerate(submodules):
            if not isinstance(item, Mapping):
                continue
            relative = str(item.get("path", ""))
            expected_gitlink = str(item.get("gitlink_commit", ""))
            result = _git(
                repo_root,
                "ls-tree",
                baseline.integration_seed_commit,
                "--",
                relative,
            )
            match = re.fullmatch(
                rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(relative)}\n?",
                result.stdout,
            )
            if (
                result.returncode != 0
                or match is None
                or match.group(1) != expected_gitlink
            ):
                errors.append(
                    f"repository_binding.submodules[{index}].gitlink_commit: Git identity mismatch"
                )

    ancestor = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        baseline.integration_seed_commit,
        "HEAD",
    )
    if ancestor.returncode != 0:
        errors.append("repository_binding.integration_seed: not an ancestor of HEAD")

    commit_result = _git(
        repo_root,
        "rev-list",
        "--reverse",
        f"{baseline.merge_base}..{baseline.rescue_head}",
    )
    if commit_result.returncode != 0:
        errors.append("repository_binding.rescue_commits: unable to enumerate")
    else:
        actual_commits = tuple(line for line in commit_result.stdout.splitlines() if line)
        expected_commits = tuple(item.identity for item in rescue.commits)
        if actual_commits != expected_commits:
            errors.append("repository_binding.rescue_commits: manifest population mismatch")

    paths_result = _git(
        repo_root,
        "diff",
        "--name-only",
        baseline.merge_base,
        baseline.rescue_head,
    )
    if paths_result.returncode != 0:
        errors.append("repository_binding.rescue_paths: unable to enumerate")
    else:
        actual_paths = tuple(line for line in paths_result.stdout.splitlines() if line)
        expected_paths = tuple(item.identity for item in rescue.files)
        if actual_paths != expected_paths:
            errors.append("repository_binding.rescue_paths: manifest population mismatch")

    residual_tree = _git(
        repo_root,
        "rev-parse",
        "--verify",
        f"{post_wave3.repository_head}^{{tree}}",
    )
    if residual_tree.returncode != 0:
        errors.append("repository_binding.post_wave3.head: Git object unavailable")
    elif residual_tree.stdout.strip() != post_wave3.repository_tree:
        errors.append("repository_binding.post_wave3.tree: Git identity mismatch")

    residual_ancestor = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        post_wave3.repository_head,
        "HEAD",
    )
    if residual_ancestor.returncode != 0:
        errors.append("repository_binding.post_wave3.head: not an ancestor of HEAD")

    authorization_tree = _git(
        repo_root,
        "rev-parse",
        "--verify",
        f"{fallback_authorization.source_head}^{{tree}}",
    )
    if authorization_tree.returncode != 0:
        errors.append(
            "repository_binding.provider_fallback_authorization.source_head: "
            "Git object unavailable"
        )
    elif authorization_tree.stdout.strip() != fallback_authorization.source_tree:
        errors.append(
            "repository_binding.provider_fallback_authorization.source_tree: "
            "Git identity mismatch"
        )
    authorization_ancestor = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        fallback_authorization.source_head,
        "HEAD",
    )
    if authorization_ancestor.returncode != 0:
        errors.append(
            "repository_binding.provider_fallback_authorization.source_head: "
            "not an ancestor of HEAD"
        )

    for task_id in sorted(_POST_WAVE3_COMPLETED_TASKS):
        item = post_wave3.completed_task_evidence.get(task_id, {})
        if not isinstance(item, Mapping):
            continue
        identities = {
            field: str(item.get(field, ""))
            for field in (
                "implementation_commit",
                "merge_commit",
                "status_commit",
            )
        }
        for field, identity in identities.items():
            available = _git(
                repo_root,
                "rev-parse",
                "--verify",
                f"{identity}^{{commit}}",
            )
            if available.returncode != 0:
                errors.append(
                    f"repository_binding.post_wave3.{task_id}.{field}: "
                    "Git object unavailable"
                )
        ancestry_chain = (
            ("implementation_commit", "merge_commit"),
            ("merge_commit", "status_commit"),
        )
        for ancestor_field, descendant_field in ancestry_chain:
            ancestry = _git(
                repo_root,
                "merge-base",
                "--is-ancestor",
                identities[ancestor_field],
                identities[descendant_field],
            )
            if ancestry.returncode != 0:
                errors.append(
                    f"repository_binding.post_wave3.{task_id}.{ancestor_field}: "
                    f"not an ancestor of {descendant_field}"
                )
        status_ancestry = _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            identities["status_commit"],
            post_wave3.repository_head,
        )
        if status_ancestry.returncode != 0:
            errors.append(
                f"repository_binding.post_wave3.{task_id}.status_commit: "
                "not an ancestor of report head"
            )

    recovery_tree = _git(
        repo_root,
        "rev-parse",
        "--verify",
        f"{false_completion_recovery.recovery_parent_head}^{{tree}}",
    )
    if recovery_tree.returncode != 0:
        errors.append(
            "repository_binding.false_completion_recovery.parent_head: "
            "Git object unavailable"
        )
    elif recovery_tree.stdout.strip() != (
        false_completion_recovery.recovery_parent_tree
    ):
        errors.append(
            "repository_binding.false_completion_recovery.parent_tree: "
            "Git identity mismatch"
        )
    recovery_ancestor = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        false_completion_recovery.recovery_parent_head,
        "HEAD",
    )
    if recovery_ancestor.returncode != 0:
        errors.append(
            "repository_binding.false_completion_recovery.parent_head: "
            "not an ancestor of HEAD"
        )
    launch_tree = _git(
        repo_root,
        "rev-parse",
        "--verify",
        f"{_FALSE_COMPLETION_RECOVERY_SOURCE['launch_base_head']}^{{tree}}",
    )
    if launch_tree.returncode != 0:
        errors.append(
            "repository_binding.false_completion_recovery.launch_base_head: "
            "Git object unavailable"
        )
    elif launch_tree.stdout.strip() != _FALSE_COMPLETION_RECOVERY_SOURCE[
        "launch_base_tree"
    ]:
        errors.append(
            "repository_binding.false_completion_recovery.launch_base_tree: "
            "Git identity mismatch"
        )
    launch_ancestry = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        str(_FALSE_COMPLETION_RECOVERY_SOURCE["launch_base_head"]),
        false_completion_recovery.recovery_parent_head,
    )
    if launch_ancestry.returncode != 0:
        errors.append(
            "repository_binding.false_completion_recovery.launch_base_head: "
            "not an ancestor of recovery parent"
        )
    for relative_path, expected_blob in _FALSE_COMPLETION_RECOVERY_SOURCE[
        "protected_parent_blobs"
    ].items():
        observed_blob = _git(
            repo_root,
            "rev-parse",
            "--verify",
            f"{false_completion_recovery.recovery_parent_head}:{relative_path}",
        )
        if observed_blob.returncode != 0 or observed_blob.stdout.strip() != expected_blob:
            errors.append(
                "repository_binding.false_completion_recovery."
                f"protected_parent_blobs.{relative_path}: Git identity mismatch"
            )
    for task_id, record in _FALSE_COMPLETION_RECORDS.items():
        prefix = f"repository_binding.false_completion_recovery.{task_id}"
        for commit_field, tree_field in (
            ("implementation_commit", "implementation_tree"),
            ("merge_commit", None),
            ("status_commit", None),
        ):
            identity = str(record[commit_field])
            observed = _git(
                repo_root,
                "rev-parse",
                "--verify",
                f"{identity}^{{tree}}",
            )
            if observed.returncode != 0:
                errors.append(f"{prefix}.{commit_field}: Git object unavailable")
            elif tree_field is not None and observed.stdout.strip() != record[tree_field]:
                errors.append(f"{prefix}.{tree_field}: Git identity mismatch")
        implementation_parents = _git(
            repo_root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            str(record["implementation_commit"]),
        )
        if implementation_parents.stdout.strip().split() != [
            str(record["implementation_commit"]),
            str(_FALSE_COMPLETION_RECOVERY_SOURCE["launch_base_head"]),
        ]:
            errors.append(
                f"{prefix}.implementation_commit: expected exact launch-base parent"
            )
        integration_parent = (
            _FALSE_COMPLETION_RECOVERY_SOURCE["launch_base_head"]
            if task_id == "ASE3-006"
            else _FALSE_COMPLETION_RECORDS["ASE3-006"]["status_commit"]
        )
        merge_parents = _git(
            repo_root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            str(record["merge_commit"]),
        )
        if merge_parents.stdout.strip().split() != [
            str(record["merge_commit"]),
            str(integration_parent),
            str(record["implementation_commit"]),
        ]:
            errors.append(f"{prefix}.merge_commit: exact parent topology mismatch")
        status_parents = _git(
            repo_root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            str(record["status_commit"]),
        )
        if status_parents.stdout.strip().split() != [
            str(record["status_commit"]),
            str(record["merge_commit"]),
        ]:
            errors.append(f"{prefix}.status_commit: exact parent topology mismatch")
        historical_board = _git(
            repo_root,
            "show",
            f"{record['status_commit']}:{PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()}",
        )
        if historical_board.returncode != 0:
            errors.append(f"{prefix}.canonical_task_cid: historical board unavailable")
        else:
            try:
                historical_tasks = _parse_taskboard_metadata(historical_board.stdout)
                historical_task = historical_tasks[task_id]
                historical_cid = _canonical_task_cid_from_metadata(historical_task)
            except (KeyError, ValueError) as exc:
                errors.append(f"{prefix}.canonical_task_cid: {exc}")
            else:
                if historical_cid != record["canonical_task_cid"]:
                    errors.append(f"{prefix}.canonical_task_cid: historical mismatch")
    if false_completion_recovery.recovery_parent_head != _FALSE_COMPLETION_RECORDS[
        "ASE3-018"
    ]["status_commit"]:
        errors.append(
            "repository_binding.false_completion_recovery.parent_head: "
            "expected terminal ASE3-018 status commit"
        )
    failed_commit = str(_FALSE_COMPLETION_FAILED_ATTEMPT["implementation_commit"])
    failed_tree = _git(
        repo_root,
        "rev-parse",
        "--verify",
        f"{failed_commit}^{{tree}}",
    )
    if failed_tree.returncode != 0:
        errors.append(
            "repository_binding.false_completion_recovery.ASE3-019."
            "implementation_commit: Git object unavailable"
        )
    elif failed_tree.stdout.strip() != _FALSE_COMPLETION_FAILED_ATTEMPT[
        "implementation_tree"
    ]:
        errors.append(
            "repository_binding.false_completion_recovery.ASE3-019."
            "implementation_tree: Git identity mismatch"
        )
    failed_parents = _git(
        repo_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        failed_commit,
    )
    if failed_parents.stdout.strip().split() != [
        failed_commit,
        str(_FALSE_COMPLETION_RECOVERY_SOURCE["launch_base_head"]),
    ]:
        errors.append(
            "repository_binding.false_completion_recovery.ASE3-019."
            "implementation_commit: expected exact launch-base parent"
        )
    rescue_branch = str(_FALSE_COMPLETION_FAILED_ATTEMPT["rescue_branch"])
    rescue_candidates = (
        f"refs/heads/{rescue_branch}",
        f"refs/remotes/origin/{rescue_branch}",
    )
    rescue_targets: dict[str, str] = {}
    for rescue_ref in rescue_candidates:
        result = _git(repo_root, "rev-parse", "--verify", f"{rescue_ref}^{{commit}}")
        if result.returncode == 0:
            rescue_targets[rescue_ref] = result.stdout.strip()
    if not rescue_targets:
        errors.append(
            "repository_binding.false_completion_recovery.ASE3-019.rescue_branch: "
            "missing exact named local/origin rescue ref"
        )
    else:
        conflicting_refs = sorted(
            ref for ref, target in rescue_targets.items() if target != failed_commit
        )
        if conflicting_refs:
            errors.append(
                "repository_binding.false_completion_recovery.ASE3-019."
                "rescue_branch: exact named refs disagree with candidate: "
                + ", ".join(conflicting_refs)
            )
    for descendant_name, descendant in (
        ("recovery_parent", false_completion_recovery.recovery_parent_head),
        ("HEAD", "HEAD"),
    ):
        ancestry = _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            failed_commit,
            descendant,
        )
        if ancestry.returncode == 0:
            errors.append(
                "repository_binding.false_completion_recovery.ASE3-019."
                f"merge_dispatched: candidate is an ancestor of {descendant_name}"
            )
        elif ancestry.returncode != 1:
            errors.append(
                "repository_binding.false_completion_recovery.ASE3-019."
                f"merge_dispatched: unable to test {descendant_name} ancestry"
            )

    attempt2_prefix = "repository_binding.self_host_seed_failure.ASE3-019.attempt_2"
    launch_head = str(_ASE3_019_ATTEMPT2_LAUNCH["launch_head"])
    launch_tree = _git(repo_root, "rev-parse", "--verify", f"{launch_head}^{{tree}}")
    if launch_tree.returncode != 0:
        errors.append(f"{attempt2_prefix}.launch_head: Git object unavailable")
    elif launch_tree.stdout.strip() != _ASE3_019_ATTEMPT2_LAUNCH["launch_tree"]:
        errors.append(f"{attempt2_prefix}.launch_tree: Git identity mismatch")
    launch_ancestor = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        launch_head,
        "HEAD",
    )
    if launch_ancestor.returncode != 0:
        errors.append(f"{attempt2_prefix}.launch_head: not an ancestor of HEAD")

    attempt2_refs = (
        f"refs/heads/{_ASE3_019_ATTEMPT2_BRANCH}",
        f"refs/remotes/origin/{_ASE3_019_ATTEMPT2_BRANCH}",
    )
    attempt2_targets: dict[str, str] = {}
    for attempt2_ref in attempt2_refs:
        attempt2_target = _git(
            repo_root,
            "rev-parse",
            "--verify",
            f"{attempt2_ref}^{{commit}}",
        )
        if attempt2_target.returncode == 0:
            attempt2_targets[attempt2_ref] = attempt2_target.stdout.strip()
    if not attempt2_targets:
        errors.append(f"{attempt2_prefix}.attempt_2_branch: exact ref unavailable")
    else:
        mismatched_attempt2_refs = sorted(
            ref for ref, target in attempt2_targets.items() if target != launch_head
        )
        if mismatched_attempt2_refs:
            errors.append(
                f"{attempt2_prefix}.attempt_2_branch: exact refs disagree with "
                "launch head: " + ", ".join(mismatched_attempt2_refs)
            )

    source_commit = str(_ASE3_019_ATTEMPT2_PRIOR_SEED["source_commit"])
    source_tree = _git(
        repo_root,
        "rev-parse",
        "--verify",
        f"{source_commit}^{{tree}}",
    )
    if source_tree.returncode != 0:
        errors.append(f"{attempt2_prefix}.source_commit: Git object unavailable")
    elif source_tree.stdout.strip() != _ASE3_019_ATTEMPT2_PRIOR_SEED["source_tree"]:
        errors.append(f"{attempt2_prefix}.source_tree: Git identity mismatch")
    source_parents = _git(
        repo_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        source_commit,
    )
    if source_parents.stdout.strip().split() != [
        source_commit,
        str(_FALSE_COMPLETION_RECOVERY_SOURCE["launch_base_head"]),
    ]:
        errors.append(f"{attempt2_prefix}.source_commit: exact parent mismatch")

    delta = _git(
        repo_root,
        "diff",
        "--binary",
        "--full-index",
        str(_ASE3_019_ATTEMPT2_PRIOR_SEED["merge_base"]),
        source_commit,
    )
    if delta.returncode != 0:
        errors.append(f"{attempt2_prefix}.prior_delta: unable to compute")
    else:
        delta_digest = "sha256:" + hashlib.sha256(
            delta.stdout.encode("utf-8")
        ).hexdigest()
        if delta_digest != _ASE3_019_ATTEMPT2_PRIOR_SEED[
            "binary_full_index_delta_sha256"
        ]:
            errors.append(f"{attempt2_prefix}.prior_delta: exact digest mismatch")
    changed_paths = _git(
        repo_root,
        "diff",
        "--name-only",
        str(_ASE3_019_ATTEMPT2_PRIOR_SEED["merge_base"]),
        source_commit,
    )
    if changed_paths.returncode != 0:
        errors.append(f"{attempt2_prefix}.replayed_paths: unable to enumerate")
    elif tuple(changed_paths.stdout.splitlines()) != _ASE3_019_ATTEMPT2_REPLAYED_PATHS:
        errors.append(f"{attempt2_prefix}.replayed_paths: exact population mismatch")

    for generation, commit, expected_blobs in (
        ("accepted", launch_head, _ASE3_019_ATTEMPT2_ACCEPTED_BLOBS),
        ("candidate", source_commit, _ASE3_019_ATTEMPT2_CANDIDATE_BLOBS),
    ):
        for relative_path, expected_blob in expected_blobs.items():
            blob = _git(
                repo_root,
                "rev-parse",
                "--verify",
                f"{commit}:{relative_path}",
            )
            if blob.returncode != 0 or blob.stdout.strip() != expected_blob:
                errors.append(
                    f"{attempt2_prefix}.{generation}_control_plane."
                    f"{relative_path}: Git blob mismatch"
                )
    return errors


def validate_convergence_artifacts(
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    *,
    repo_root: Path | str | None = DEFAULT_REPOSITORY_ROOT,
    check_repository: bool = True,
    taskboard_path: Path | str | None = None,
) -> ConvergenceValidationReport:
    """Validate the entire ASE3-000 packet without trusting historical state."""

    root = Path(artifact_root)
    errors: list[str] = []
    checked: list[str] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    raw_artifacts: dict[str, bytes] = {}
    artifact_digests: dict[str, str] = {}
    try:
        root_status = root.lstat()
    except OSError as exc:
        return ConvergenceValidationReport(
            False,
            (f"artifact_root: {exc}",),
            (),
        )
    if root.is_symlink() or not stat.S_ISDIR(root_status.st_mode):
        return ConvergenceValidationReport(
            False,
            ("artifact_root: expected a directory, not a symlink",),
            (),
        )
    for filename in (*ARTIFACT_FILENAMES, MANIFEST_FILENAME):
        path = root / filename
        checked.append(filename)
        try:
            raw = _read_regular_bytes(
                path,
                maximum_bytes=_EVIDENCE_SNAPSHOT_BYTE_BOUNDS.get(
                    filename,
                    MAX_EVIDENCE_SNAPSHOT_BYTES,
                ),
            )
            raw_artifacts[filename] = raw
            if filename in (*JSON_ARTIFACT_FILENAMES, MANIFEST_FILENAME):
                payloads[filename] = _load_json_bytes(raw, name=filename)
            elif filename not in TEXT_ARTIFACT_FILENAMES:
                raise ValueError(f"{filename}: undeclared artifact encoding")
            artifact_digests[filename] = (
                "sha256:" + hashlib.sha256(raw).hexdigest()
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{filename}: {exc}")
    if errors:
        return ConvergenceValidationReport(False, tuple(errors), tuple(checked))

    baseline = CurrentMainBaseline.from_dict(payloads["current_main_baseline.json"])
    contradictions = HistoricalStateContradictionReport.from_dict(
        payloads["historical_state_contradictions.json"]
    )
    rescue = RescueDispositionReport.from_dict(
        payloads["rescue_artifact_dispositions.json"]
    )
    worktree = CleanIntegrationWorktreeReceipt.from_dict(
        payloads["clean_integration_worktree_receipt.json"]
    )
    post_wave3 = PostWave3ResidualReport.from_dict(
        payloads[POST_WAVE3_RESIDUAL_FILENAME]
    )
    false_completion_recovery = FalseCompletionRecoveryReport.from_dict(
        payloads[FALSE_COMPLETION_RECOVERY_FILENAME]
    )
    fallback_authorization = ProviderFallbackPolicyAuthorization.from_dict(
        payloads[PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME]
    )
    manifest = ConvergenceManifest.from_dict(payloads[MANIFEST_FILENAME])

    errors.extend(baseline.validate())
    errors.extend(contradictions.validate())
    errors.extend(rescue.validate(baseline))
    errors.extend(worktree.validate(baseline))
    errors.extend(post_wave3.validate())
    errors.extend(false_completion_recovery.validate())
    errors.extend(fallback_authorization.validate())
    errors.extend(manifest.validate(baseline))
    errors.extend(
        _validate_false_completion_snapshots(
            payloads=payloads,
            digests=artifact_digests,
        )
    )
    errors.extend(
        _validate_attempt2_incident_packet(
            payloads=payloads,
            raw_artifacts=raw_artifacts,
            digests=artifact_digests,
        )
    )
    board_path = (
        Path(taskboard_path)
        if taskboard_path is not None
        else Path(repo_root or DEFAULT_REPOSITORY_ROOT)
        / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    )
    try:
        board_tasks = _load_taskboard_metadata(board_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"taskboard_snapshot: {exc}")
    else:
        errors.extend(
            _validate_provider_attempt_reload_gate(
                tasks=board_tasks,
                artifact_root=root,
            )
        )
        errors.extend(_validate_provider_fallback_task_contract(tasks=board_tasks))
        errors.extend(_validate_false_completion_repair_tasks(tasks=board_tasks))
        errors.extend(
            _validate_program_plan_expansion(
                tasks=board_tasks,
                artifact_root=root,
            )
        )
        if taskboard_path is None and repo_root is not None:
            errors.extend(
                _validate_program_scheduler_projection(
                    repo_root=Path(repo_root),
                    tasks=board_tasks,
                )
            )

    components = manifest.payload.get("components", {})
    if isinstance(components, Mapping):
        for filename in ARTIFACT_FILENAMES:
            expected = components.get(filename)
            actual = artifact_digests.get(filename)
            if expected != actual:
                errors.append(
                    f"convergence_manifest.components.{filename}: digest mismatch"
                )

    if check_repository and repo_root is not None and not errors:
        errors.extend(
            _validate_repository_binding(
                repo_root=Path(repo_root),
                baseline=baseline,
                rescue=rescue,
                post_wave3=post_wave3,
                false_completion_recovery=false_completion_recovery,
                fallback_authorization=fallback_authorization,
            )
        )
    return ConvergenceValidationReport(
        valid=not errors,
        errors=tuple(errors),
        checked_artifacts=tuple(checked),
        integration_seed_commit=baseline.integration_seed_commit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="validate all checked-in ASE3-000 convergence artifacts",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="bounded convergence artifact directory",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT,
        help="Git worktree used for live object/population checks",
    )
    parser.add_argument(
        "--taskboard-path",
        type=Path,
        default=None,
        help="protected v3 taskboard; defaults below --repo-root",
    )
    parser.add_argument(
        "--no-repository-check",
        action="store_true",
        help="validate packet structure and digests without live Git checks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check_all:
        report = ConvergenceValidationReport(
            valid=False,
            errors=("--check-all is required",),
            checked_artifacts=(),
        )
    else:
        report = validate_convergence_artifacts(
            args.artifacts_root,
            repo_root=args.repo_root,
            check_repository=not args.no_repository_check,
            taskboard_path=args.taskboard_path,
        )
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess test
    raise SystemExit(main())
