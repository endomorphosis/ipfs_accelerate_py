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
import ast
import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

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
ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-convergence-manifest@2"
)
RELOAD_CONVERGENCE_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-convergence-manifest@3"
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
PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.provider-fallback-policy-authorization@2"
)
PROVIDER_FALLBACK_POLICY_REVIEW_V2_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.provider-fallback-policy-review@2"
)
LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.local-profile-lifecycle-root-pin@1"
)
LOCAL_PROFILE_LIFECYCLE_WITNESS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/local-profile-lifecycle-witness@1"
)
LOCAL_DEV_PROFILE_V5_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/local-dev-profile@5"
)
LOCAL_PROFILE_LIFECYCLE_ANCHOR_V3_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/local-profile-lifecycle-anchor@3"
)
LOCAL_PROFILE_ROOT_REGISTRY_V2_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/local-profile-root-registry@2"
)
LOCAL_PROFILE_DID_STATE_V1_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/local-profile-did-state@1"
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
OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME: Final = (
    "operator_acceptance_receipt_ase3_023_20260808.json"
)
OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME: Final = (
    "operator_acceptance_receipt_ase3_027_20260808.json"
)
HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME: Final = (
    "hermetic_control_plane_identity_acceptance_receipt.json"
)
HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.ase3-030-hermetic-identity-acceptance@1"
)
OPERATOR_SALVAGE_RECEIPT_019_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.ase3-019-operator-salvage@1"
)
OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.operator-repair-acceptance@1"
)
OPERATOR_ACCEPTANCE_RECEIPT_FILENAMES: Final = (
    OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
    HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME,
    OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME,
    OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME,
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
PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.provider-attempt-daemon-reload@1"
)
PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.provider-attempt-generation-birth@1"
)
PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_FILENAME: Final = (
    "provider_attempt_generation_birth_receipt.json"
)
PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_FILENAME
)
PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME: Final = (
    "protected_runtime_activation_receipt.json"
)
PROTECTED_RUNTIME_ACTIVATION_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME
)
PROTECTED_RUNTIME_ACTIVATION_AUTHORIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-runtime-activation-authorization@1"
)
PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_FILENAME: Final = (
    "protected_runtime_post_activation_observation_receipt.json"
)
PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_FILENAME
)
PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-runtime-post-activation-observation@1"
)
HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME
)
NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME: Final = (
    "native_dependency_launch_authorization.json"
)
NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME
)
NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "ase3-031-native-dependency-launch-authorization@1"
)
NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME: Final = (
    "sealed_native_dependency_acceptance_receipt.json"
)
NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME
)
NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.ase3-031-native-dependency-acceptance@1"
)
DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME: Final = (
    "duckdb_connection_policy_acceptance_receipt.json"
)
DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME
)
DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "ase3-032-duckdb-connection-policy-acceptance@1"
)
_CONVERGENCE_RELATIVE_ROOT: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
)
LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME: Final = (
    "local_profile_lifecycle_root_pin_20260808.json"
)
LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME: Final = (
    "local_operator_lifecycle_witness.json"
)
LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH: Final = (
    f"{_CONVERGENCE_RELATIVE_ROOT}/{LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME}"
)
LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH: Final = (
    f"{_CONVERGENCE_RELATIVE_ROOT}/{LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME}"
)
PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH: Final = (
    f"{_CONVERGENCE_RELATIVE_ROOT}/"
    f"{PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME}"
)
OPERATOR_ACCEPTANCE_RECEIPT_RELATIVE_PATHS: Final = tuple(
    f"{_CONVERGENCE_RELATIVE_ROOT}/{filename}"
    for filename in OPERATOR_ACCEPTANCE_RECEIPT_FILENAMES
)
_CONVERGENCE_MANIFEST_RELATIVE_PATH: Final = (
    f"{_CONVERGENCE_RELATIVE_ROOT}/{MANIFEST_FILENAME}"
)
SEQUENTIAL_ACCEPTANCE_PHASES: Final = (
    "Q",
    "R",
    "P019",
    "A019",
    "A030",
    "P031",
    "A031",
    "A032",
    "A023/027",
    "L",
)
SEQUENTIAL_ACCEPTANCE_TASK_IDS: Final = (
    "ASE3-019",
    "ASE3-030",
    "ASE3-031",
    "ASE3-032",
    "ASE3-023",
    "ASE3-027",
)
SEQUENTIAL_PHASE_PARENT: Final = {
    phase: SEQUENTIAL_ACCEPTANCE_PHASES[index - 1]
    for index, phase in enumerate(SEQUENTIAL_ACCEPTANCE_PHASES)
    if index
}
SEQUENTIAL_PHASE_STATUS_TRANSITIONS: Final = {
    "R": (),
    "P019": (),
    "A019": ("ASE3-019",),
    "A030": ("ASE3-030",),
    "P031": (),
    "A031": ("ASE3-031",),
    "A032": ("ASE3-032",),
    "A023/027": ("ASE3-023", "ASE3-027"),
    "L": ("ASE3-022",),
}
SEQUENTIAL_PHASE_RUNTIME_EFFECT_CLAIMS: Final = {
    "A019": False,
    "A030": False,
    "P031": False,
    "A031": True,
    "A032": False,
    "A023/027": True,
}
Q_TO_R_CHANGED_PATHS: Final = (LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH,)
R_TO_P019_CHANGED_PATHS: Final = (
    LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH,
    PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
    _CONVERGENCE_MANIFEST_RELATIVE_PATH,
)
P019_TO_A019_CHANGED_PATHS: Final = (
    f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_SALVAGE_RECEIPT_019_FILENAME}",
    _CONVERGENCE_MANIFEST_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
)
A019_TO_A030_CHANGED_PATHS: Final = (
    HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
    _CONVERGENCE_MANIFEST_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
)
A030_TO_P031_CHANGED_PATHS: Final = (
    NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
    _CONVERGENCE_MANIFEST_RELATIVE_PATH,
)
P031_TO_A031_CHANGED_PATHS: Final = (
    NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
    _CONVERGENCE_MANIFEST_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
)
A031_TO_A032_CHANGED_PATHS: Final = (
    DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
    _CONVERGENCE_MANIFEST_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
)
A032_TO_A023_027_CHANGED_PATHS: Final = (
    f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME}",
    f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME}",
    _CONVERGENCE_MANIFEST_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
)
A023_027_TO_L_CHANGED_PATHS: Final = (
    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
    _CONVERGENCE_MANIFEST_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
)
SEQUENTIAL_PHASE_CHANGED_PATHS: Final = {
    "R": Q_TO_R_CHANGED_PATHS,
    "P019": R_TO_P019_CHANGED_PATHS,
    "A019": P019_TO_A019_CHANGED_PATHS,
    "A030": A019_TO_A030_CHANGED_PATHS,
    "P031": A030_TO_P031_CHANGED_PATHS,
    "A031": P031_TO_A031_CHANGED_PATHS,
    "A032": A031_TO_A032_CHANGED_PATHS,
    "A023/027": A032_TO_A023_027_CHANGED_PATHS,
    "L": A023_027_TO_L_CHANGED_PATHS,
}
SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE: Final = {
    LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH: "R",
    LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH: "P019",
    f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_SALVAGE_RECEIPT_019_FILENAME}": "A019",
    HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH: "A030",
    NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH: "P031",
    NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH: "A031",
    DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH: "A032",
    f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME}": "A023/027",
    f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME}": "A023/027",
    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH: "L",
}
SEQUENTIAL_ACCEPTANCE_ARTIFACT_FILENAMES: Final = (
    OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
    HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME,
    NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME,
    NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME,
    DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME,
    OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME,
    OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME,
)
# Compatibility names remain read-only aliases; the validator below rejects
# the obsolete fan-in and uses the phase-indexed predicates above.
R_TO_P_CHANGED_PATHS: Final = R_TO_P019_CHANGED_PATHS
ACCEPTANCE_CHILD_CHANGED_PATHS: Final = P019_TO_A019_CHANGED_PATHS
RELOAD_CHILD_CHANGED_PATHS: Final = A023_027_TO_L_CHANGED_PATHS
_PROTECTED_PATHS: Final = (
    ".gitignore",
    PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH.as_posix(),
    PROMPT_V3_PLAN_RELATIVE_PATH.as_posix(),
    PROMPT_V3_OBJECTIVES_RELATIVE_PATH.as_posix(),
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
    "ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py",
    "test/api/test_agent_supervisor_prompt_v3_convergence.py",
    *(
        f"{_CONVERGENCE_RELATIVE_ROOT}/{name}"
        for name in (
            "current_main_baseline.json",
            "historical_state_contradictions.json",
            "rescue_artifact_dispositions.json",
            "clean_integration_worktree_receipt.json",
            MANIFEST_FILENAME,
            POST_WAVE3_RESIDUAL_FILENAME,
            FALSE_COMPLETION_RECOVERY_FILENAME,
            FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
            FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
            FAILED_VALIDATION_EVENT_019_FILENAME,
            FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME,
            FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME,
            SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME,
            PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME,
            LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME,
            LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME,
            OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
            HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME,
            NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME,
            NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME,
            DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME,
            OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME,
            OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME,
            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
            PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_FILENAME,
        )
    ),
)
DEFAULT_ARTIFACT_ROOT: Final = (
    DEFAULT_REPOSITORY_ROOT
    / "data"
    / "agent_supervisor"
    / "prompt_only_self_improvement_v3"
    / "convergence"
)
MAX_EVIDENCE_SNAPSHOT_BYTES: Final[int] = 1_048_576
MAX_OPERATOR_ACCEPTANCE_RECEIPT_BYTES: Final[int] = 256 * 1024
MAX_NATIVE_DEPENDENCY_AUTHORIZATION_BYTES: Final[int] = 128 * 1024
MAX_NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_BYTES: Final[int] = 256 * 1024
MAX_DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_BYTES: Final[int] = 256 * 1024
MAX_PROVIDER_ATTEMPT_RELOAD_RECEIPT_BYTES: Final[int] = 128 * 1024
MAX_PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_BYTES: Final[int] = 128 * 1024
MAX_PROVIDER_FALLBACK_AUTHORIZATION_BYTES: Final[int] = 128 * 1024
MAX_LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_BYTES: Final[int] = 32 * 1024
MAX_LOCAL_OPERATOR_LIFECYCLE_WITNESS_BYTES: Final[int] = 128 * 1024
_EVIDENCE_SNAPSHOT_BYTE_BOUNDS: Final = {
    PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME: (
        MAX_PROVIDER_FALLBACK_AUTHORIZATION_BYTES
    ),
    NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME: (
        MAX_NATIVE_DEPENDENCY_AUTHORIZATION_BYTES
    ),
    NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME: (
        MAX_NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_BYTES
    ),
    DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME: (
        MAX_DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_BYTES
    ),
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
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
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
    "ASE3-031",
    "ASE3-032",
    "ASE3-033",
)
_PROGRAM_NONCANONICAL_TASK_IDS: Final = (
    "ASE3-015",
    "ASE3-016",
    "ASE3-017",
    "ASE3-022",
)
_PROTECTED_TASK_BLOCK_SHA256S: Final = {
    "ASE3-019": "9573c8545b5bd981760e9a5255d6287492b2d882905cb248107f8237711a5fed",
    "ASE3-022": "e896f06b31aba2906eea41685f751349be4a266b0b5f1eb93f12acfe5dd3eb22",
    "ASE3-023": "355ba4cc330301df57386516b5bb4d4f9fbeae4fabea0272bedcd583df3c7521",
    "ASE3-027": "221f1cc85639b21931dc50497295117e44fd02530128c9e71a3cf25259543bdf",
}
_TRANSITION_CONSTRUCTION_OUTPUTS: Final = (
    "ipfs_accelerate_py/agent_supervisor/core/protected_acceptance_contracts.py",
    "ipfs_accelerate_py/agent_supervisor/merge/protected_acceptance_transition.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/"
    "protected_acceptance_transition.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/"
    "protected_acceptance_transition_cli.py",
    "test/api/test_agent_supervisor_prompt_v3_transition.py",
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "protected_acceptance_q_inventory.json",
    ".gitignore",
    PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH.as_posix(),
    PROMPT_V3_PLAN_RELATIVE_PATH.as_posix(),
    PROMPT_V3_OBJECTIVES_RELATIVE_PATH.as_posix(),
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
    "ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py",
    "test/api/test_agent_supervisor_prompt_v3_convergence.py",
)
_TRANSITION_Q_INVENTORY_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "protected_acceptance_q_inventory.json"
)
_TRANSITION_Q_CHANGED_PATHS: Final = (
    _TRANSITION_Q_INVENTORY_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
)
_TRANSITION_CONSTRUCTION_RESERVED_PATHS: Final = (
    # Tooling must already be integrated in Q's parent while ASE3-033 remains
    # todo. Only the Q inventory is reserved until the Q status transition.
    _TRANSITION_Q_INVENTORY_RELATIVE_PATH,
)
_TRANSITION_CONSTRUCTION_PUBLIC_APIS: Final = (
    "freeze_prompt_v3_product_provenance",
    "build_prompt_v3_root_pin",
    "initialize_prompt_v3_reviewer_after_root_pin",
    "build_prompt_v3_provider_authorization",
    "canonical_prompt_v3_review_bytes",
    "sign_prompt_v3_operator_artifact",
    "build_prompt_v3_phase_candidate",
    "run_prompt_v3_phase_evidence",
    "validate_prompt_v3_phase_candidate",
    "publish_prompt_v3_phase_candidate",
    "reject_prompt_v3_phase_candidate",
    "observe_prompt_v3_quiescence",
    "build_prompt_v3_reload_authorization",
    "consume_prompt_v3_a031_authorization",
    "append_prompt_v3_a031_failure_attempt",
    "reauthorize_prompt_v3_p031_attempt",
    "load_verified_prompt_v3_runtime_launch_authority",
)
_TRANSITION_CONSTRUCTION_PRE_Q_REVIEWS: Final = (
    "ASE3-019",
    "ASE3-030",
    "ASE3-031",
    "ASE3-032",
    "ASE3-023",
    "ASE3-027",
)
_TRANSITION_CONSTRUCTION_REQUIRED_PHASES: Final = (
    *SEQUENTIAL_ACCEPTANCE_PHASES[1:],
    "birth",
)
_TRANSITION_CONSTRUCTION_REQUIRED_TESTS: Final = (
    "test_q_inventory_rejects_every_future_pin_sentinel",
    "test_ase3_031_and_032_require_task_correct_replay_provenance",
    "test_every_pre_q_generation_reconstructs_exact_source_replay_integrated_commit_streams",
    "test_multi_commit_generation_chains_parents_and_final_commit_map_are_exact",
    "test_product_provenance_preserves_reviewed_modes_and_builder_artifacts_are_100644",
    "test_a019_binds_actual_provider_authorization_v2_digest",
    "test_birth_resolves_configured_target_ref_once_to_one_head",
    "test_every_phase_requires_fresh_authority_and_witness",
    "test_activation_is_impossible_before_valid_post_l_birth",
    "test_p031_is_a031_only_and_failure_reauthorization_is_bounded_append_only",
    "test_l_runtime_authorization_is_distinct_and_consumed_before_spawn",
    "test_runtime_authority_loader_returns_strict_dto_after_full_chain_only",
    "test_signing_adapter_uses_ascii_canonical_bytes_and_verified_ed25519_transcode",
    "test_signing_adapter_denies_repository_authority_rotation_revocation_and_ambient_key_races",
    "test_root_pin_builder_emits_only_exact_verified_public_artifact",
    "test_unheld_target_publication_uses_update_ref_cas_under_canonical_lease",
    "test_checked_out_target_publication_uses_hook_free_sign_free_ff_only_merge",
    "test_publication_failure_recovers_exact_pre_state_and_never_rejects_real_target_as_detached_only",
    "test_nested_agent_supervisor_core_is_not_gitignored",
    "test_public_artifact_key_and_directory_modes_fail_closed",
)
_TRANSITION_CONSTRUCTION_POLICY_SHA256: Final = (
    "sha256:0c4113e7cb01b477ba608579bebb783c385952a2a2e2f17b5f582ab8997c85ba"
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
            "sha256:7b71f4c298b8861687d1a03d0384b298430a50b2167fe0342c75fdb1f72fcd44"
        ),
        "canonical_task_cid": (
            "baguqeerauhje6jq6ejhsl5ocxk6vcnywg2pkurlkq5lftk3tgljnl6bjyouq"
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
        "requirements": (
            "connect_duckdb_with_policy",
            "every generated-board/planning-reachable connection",
            "formal_plan_compiler",
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
            "sha256:ea0eae512a86fc9e6098e329d58fcd0963c1706ec621b4da45b1cbdee6f08061"
        ),
        "canonical_task_cid": (
            "baguqeerapt2k36spo3uom3rhn5rucmvnlkqbk5k3nxznd2wetwznq4rg6h2a"
        ),
        "goal id": "ASE3-G020",
        "depends on": ("ASE3-022",),
        "outputs": (
            "ipfs_accelerate_py/agent_supervisor/contracts/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/contracts/authority.py",
            "ipfs_accelerate_py/agent_supervisor/contracts/execution.py",
            "ipfs_accelerate_py/agent_supervisor/contracts/provider_capacity.py",
            "ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py",
            "ipfs_accelerate_py/agent_supervisor/control/profile_authority.py",
            "ipfs_accelerate_py/agent_supervisor/control/plan_execution_store.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
            "ipfs_accelerate_py/llm_router.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_provider_auto.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor_runner.py",
            "test/api/test_agent_supervisor_contract_layering.py",
            "test/api/test_agent_supervisor_configured_board_scheduler.py",
            "test/api/test_agent_supervisor_implementation_daemon_runner.py",
            "test/api/test_llm_router_agent_supervisor_fallback_route.py",
            "test/api/test_agent_supervisor_control_plane_capsule_identity.py",
        ),
        "validation": (
            "python -m pytest test/api/test_agent_supervisor_contract_layering.py "
            "test/api/test_agent_supervisor_configured_board_scheduler.py "
            "test/api/test_agent_supervisor_implementation_daemon_runner.py "
            "test/api/test_agent_supervisor_implementation_supervisor_runner.py "
            "test/api/test_agent_supervisor_implementation_provider_receipts.py "
            "test/api/test_implementation_provider_auto.py "
            "test/api/test_agent_supervisor_entrypoint_contracts.py "
            "test/api/test_agent_supervisor_prompt_v3_parallelism.py "
            "test/api/test_agent_supervisor_prompt_v3_authority_hardening.py "
            "test/api/test_agent_supervisor_profile_resolver.py "
            "test/api/test_agent_supervisor_control_plane_capsule_identity.py "
            "test/api/test_llm_router_agent_implementation_route.py "
            "test/api/test_llm_router_agent_supervisor_fallback_route.py "
            "test/api/test_agent_supervisor_grok_quota_terra_gate.py -q"
        ),
        "requirements": (
            "roadmap supplies no expected import count",
            "canonical signed lifecycle-bound provider fallback authorization@2",
            (
                "test_agent_supervisor_implementation_daemon_runner.py::"
                "test_daemon_resolves_relative_worktree_root_for_runner_workspace"
            ),
            "canonical signed accepted route authorization and binding",
            "owned regular nonsymlinks at mode `0400`",
            "without deselection, verifier bypass, or route weakening",
            "an ambient registry, service locator, import hook",
            "control.provider_attempt_store",
            "control.profile_authority",
            "control.plan_execution_store",
            "AgentImplementationRoutePlan",
            "AgentImplementationFallbackDecision",
            "dispatch_authorized=False",
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
    "ASE3-031": {
        "title": (
            "Seal the reviewed DuckDB native extension for isolated supervisor "
            "launch"
        ),
        "contract_sha256": (
            "sha256:e749edab95f0f359cb39ab69cfc9a7858490600f4e73622a64e4f6f099eda7ff"
        ),
        "canonical_task_cid": (
            "baguqeeraxwgr3bpjg2efihsmv3id5fknng2oa5bapas3fdq4hwjklpa4lxdq"
        ),
        "goal id": "ASE3-G040",
        "depends on": ("ASE3-030",),
        "outputs": (
            "ipfs_accelerate_py/llm_router.py",
            "test/api/test_agent_supervisor_native_dependency_pin.py",
        ),
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_native_dependency_pin.py -q"
        ),
        "requirements": (
            "25fedf091dad928dad1f83c9f81a54c2d401eabe",
            "authorization_may_claim_launch_effect: false",
            "before any ASE3-023 capsule/native subprocess end-to-end runtime effect",
        ),
    },
    "ASE3-032": {
        "title": (
            "Enforce one configuration-locked DuckDB connection policy across "
            "supervisor state"
        ),
        "contract_sha256": (
            "sha256:b44b0a0c2853296def5276f1fb08219480d9b0bb2282f3b7d9587fafd1ec28be"
        ),
        "canonical_task_cid": (
            "baguqeerarapsgz3yexowvp6ppcroujqmfymfgsmrapanyv5ggclalkyxicva"
        ),
        "goal id": "ASE3-G040",
        "depends on": ("ASE3-031",),
        "outputs": (
            "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py",
            "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
            "test/api/test_agent_supervisor_duckdb_connection_policy.py",
        ),
        "validation": (
            "python -m pytest -q "
            "test/api/test_agent_supervisor_duckdb_connection_policy.py "
            "test/api/test_agent_supervisor_duckdb_state.py "
            "test/api/test_agent_supervisor_duckdb_task_source.py "
            "test/api/test_agent_supervisor_lease_coordination.py"
        ),
        "static validation commands json": (
            '["python -m ruff check --select E9,F63,F7,F82,I '
            "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py "
            "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py "
            "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py "
            'test/api/test_agent_supervisor_duckdb_connection_policy.py", '
            '"python -m py_compile '
            "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py "
            "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py "
            "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py "
            'test/api/test_agent_supervisor_duckdb_connection_policy.py", '
            '"git diff --check"]'
        ),
        "requirements": (
            "DUCKDB_CONNECTION_POLICY_SETTINGS",
            "connect_duckdb_with_policy",
            "schemas/tables/views/sequences/macros/custom types/indexes",
            "statically linked LOAD",
            "zero-`ATTACH` production-compaction",
            "bounded coverage, not a global claim",
        ),
    },
    "ASE3-033": {
        "title": (
            "Productionize protected transition construction, replay provenance, "
            "and phase-local authority"
        ),
        "contract_sha256": (
            "sha256:425d6ea677971447fa532a9f5fc4e7d94e719b0bc79d912177c77df1ef7f9a36"
        ),
        "canonical_task_cid": (
            "baguqeeraplpzmy3irdafe3aafaixj6vohug7vtxbikjnzccy7dxt4utt4zgq"
        ),
        "goal id": "ASE3-G055",
        "depends on": ("ASE3-000",),
        "outputs": _TRANSITION_CONSTRUCTION_OUTPUTS,
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_prompt_v3_transition.py "
            "test/api/test_agent_supervisor_prompt_v3_convergence.py -q"
        ),
        "static validation commands json": (
            '["python -m ruff check --select E9,F63,F7,F82,I '
            "ipfs_accelerate_py/agent_supervisor/core/"
            "protected_acceptance_contracts.py "
            "ipfs_accelerate_py/agent_supervisor/merge/"
            "protected_acceptance_transition.py "
            "ipfs_accelerate_py/agent_supervisor/entrypoints/"
            "protected_acceptance_transition.py "
            "ipfs_accelerate_py/agent_supervisor/entrypoints/"
            "protected_acceptance_transition_cli.py "
            "ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py "
            "test/api/test_agent_supervisor_prompt_v3_transition.py "
            'test/api/test_agent_supervisor_prompt_v3_convergence.py", '
            '"python -m py_compile '
            "ipfs_accelerate_py/agent_supervisor/core/"
            "protected_acceptance_contracts.py "
            "ipfs_accelerate_py/agent_supervisor/merge/"
            "protected_acceptance_transition.py "
            "ipfs_accelerate_py/agent_supervisor/entrypoints/"
            "protected_acceptance_transition.py "
            "ipfs_accelerate_py/agent_supervisor/entrypoints/"
            "protected_acceptance_transition_cli.py "
            "ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py "
            "test/api/test_agent_supervisor_prompt_v3_transition.py "
            'test/api/test_agent_supervisor_prompt_v3_convergence.py", '
            '"jq empty config/'
            'agent_supervisor_prompt_only_self_improvement_v3_scheduler.json", '
            '"git diff --check"]'
        ),
        "requirements": (
            "ipfs_accelerate_py.agent_supervisor.prompt-v3-product-generation@1",
            "Source and replay commits are independently reviewed nonancestors of Q",
            "integrated commits are ancestors of Q",
            "ASE3-019×2",
            "ASE3-023×3",
            "source=replay=integrated",
            (
                "git diff --no-ext-diff --no-textconv --no-renames --binary "
                "--full-index"
            ),
            "`ipfs_accelerate_py/llm_router.py` 100755",
            "strict ASCII receipt mappings",
            "standard-Base64 signature",
            "Reload the active profile/generation after signing",
            "AGENT_SUPERVISOR_LOCAL_PROFILE_KEY",
            "future-pin sentinels",
            "A019 binds the digest of the actual canonical provider authorization@2",
            "A031 acceptance preload",
            "at most three attempts",
            "failed attempts remain in Git",
            "scheduler/PlanRevisionStore before spawn",
            "load_verified_prompt_v3_runtime_launch_authority",
            "without importing the convergence validator",
            "never replay a provider effect",
            "git update-ref",
            "checked-out exact target",
            "`git merge --ff-only`",
            "exact pre/post ref/HEAD/tree/index/worktree",
            "rescue ref",
            "`O_NOFOLLOW`",
            "public artifacts are atomically renamed at 0400",
            "private keys remain 0600",
            "replace its exact unanchored `core` line with `/core`",
            "nested `agent_supervisor/core/protected_acceptance_contracts.py`",
            "configured target ref resolved once",
            "no protected runtime activates before matching birth",
            "Portable verification survives fresh clone plus pruning",
            "durable carrier is the sealed `prompt-v3-product-generation@1` record",
            "source/replay verify from sealed record fields only",
            "ambient ref carriers are forbidden",
            "Fresh clone plus pruning still verifies every nonancestor source/replay",
        ),
    },
}
_PROGRAM_AMENDED_TASK_CIDS: Final = {
    "ASE3-008": "baguqeerajytqcnamiixnkiekvawxnupxkdf2u2oeciswxhsrw3ylo5bjlr7q",
    "ASE3-009": "baguqeera7ly4s4ddus5vo5iyaobxuz5mmlmoi4g3ajcvcuycrpfihtzlbykq",
    "ASE3-012": "baguqeerazur2cegzialeuzvenwebarqg74hkysqi2fboswsewhv4mcs5tsgq",
    "ASE3-013": "baguqeerayddnpog5ef4uzdgh3ku67pm6nkm3kpyh75jjzc6oezr2nxiuxdva",
    "ASE3-020": "baguqeerauzjczeaxx4l36beiqr3q5qvzr6ltlascqkchgdts7nrgddunv4rq",
    "ASE3-021": "baguqeeraheqxu3slnx5dhc73yitbn2zdtrpsqtwfpesdblth72qz23djw3ya",
}
_PROGRAM_AMENDED_TASK_CONTRACT_SHA256S: Final = {
    "ASE3-008": (
        "sha256:a265387dfd68e8509ac9f1e63fd59aae2d67844cb29da083aa376b1bfed5378e"
    ),
    "ASE3-009": (
        "sha256:82e0a373cc1423b6b2aa9dd1d750cb5f44e8955c20f4b2e25b12bb44b7ab1e5f"
    ),
    "ASE3-012": (
        "sha256:3f36392550176ff016adc3bcbeefb0a7e2922c118997a6796eec66ebf7508477"
    ),
    "ASE3-013": (
        "sha256:1403cd1dc370787c31ab1c4943336c65e9403600f3ae614909c077fa1dbbf738"
    ),
    "ASE3-020": (
        "sha256:49df73850a85c9a2e19574afd640ffc106c402176ce6b51e40bdd832dd7cd525"
    ),
    "ASE3-021": (
        "sha256:0067a1c53aa3d471bcabd3551991eded661d8e29da23ff861edb7a88fdbd6484"
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
    "ASE3-008": ("ASE3-006", "ASE3-020", "ASE3-021"),
    "ASE3-009": ("ASE3-005", "ASE3-008", "ASE3-026"),
    "ASE3-012": ("ASE3-010", "ASE3-011"),
    "ASE3-013": ("ASE3-008", "ASE3-012", "ASE3-026"),
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
    "ASE3-008": (
        "ReviewedHostNamespaceReconciler",
        "configured-scheduler semantic-progress publication",
        (
            "one same-revision join of verified lifecycle and monitor births, "
            "leases, fences, fresh heartbeats, and monotonic event cursors"
        ),
        "unknown external-effect outcomes are adopted",
    ),
    "ASE3-009": ("ProductionServiceCompositionManifest", "ASE3-026"),
    "ASE3-012": (
        "black-box",
        "production composition CID",
        "PromptProductDuckDBConnectionAudit",
        "no prompt-product launch-reachable path can call raw `duckdb.connect`",
        "classified as legacy or proof-only",
    ),
    "ASE3-013": (
        "no preseeded objective or taskboard",
        "non-sentinel",
        "monitor_policy.canary_observation_seconds: 900",
        (
            "Client disconnect, monitor death, provider saturation, clock "
            "rollback, merge/refill stalls, and oscillation"
        ),
        "The 900-second clock begins only after the final recovery",
    ),
    "ASE3-020": (
        "ImmutableRunHistoryVector",
        "MonitorProgressCursorVector",
        "MonitorReadyEffectReservation",
        "UnknownOutcomeAdoptionReceipt",
        "persist UNKNOWN, prohibit replay",
        "actual supervisor/daemon parsers",
        "connect_duckdb_with_policy",
        "Every prompt-product-reachable run-registry and runtime-history connection",
    ),
    "ASE3-021": (
        "DurableRefillSagaCursor",
        "EVALUATING→APPEND_RESERVED→APPENDED→PLAN_INVALIDATED→RECOMPILED→DISPATCHED/ADOPTED",
        "phase-specific monitor deadlines",
        "all refill flags remain dormant",
    ),
}
_PROTECTED_RUNTIME_ACTIVATION_TASK_ID: Final = "ASE3-026"
_PROTECTED_RUNTIME_ACTIVATION_TASK_TITLE: Final = (
    "Authorize, activate, and observe the durable refill and autonomous monitor "
    "runtime"
)
_PROTECTED_RUNTIME_ACTIVATION_BLOCKED_REASON: Final = (
    "protected runtime activation receipt not yet accepted"
)
_PROTECTED_RUNTIME_ACTIVATION_CONTRACT_SHA256: Final = (
    "sha256:84d70803f2e42a6e96725b0a01db05a2673e63a68bac218d43bac09e835bde6d"
)
_PROTECTED_RUNTIME_ACTIVATION_TASK_CID: Final = (
    "baguqeerampybtjmxsa6zwz6eibyh6sa6agxik2f6kpsrfwz34jipcdul5aoa"
)
_PROTECTED_RUNTIME_ACTIVATION_DEPENDENCIES: Final = (
    "ASE3-008",
    "ASE3-020",
    "ASE3-021",
    "ASE3-025",
)
_PROTECTED_RUNTIME_ACTIVATION_REQUIREMENTS: Final = (
    "ProtectedRuntimeActivationAuthorization",
    "ProtectedRuntimePostActivationObservation",
    "authorization_effect_observed",
    "authorization alone never proves the effect ran",
    "ReviewedHostNamespaceReconciler",
)
_MONITOR_STRATEGY_OBJECTIVE_CONTRACT_SHA256S: Final = {
    "ASE3-G000": (
        "sha256:b28b76e0379c450c40e1865e687054448650fbfd6c3f6595785a11efbfca3d6f"
    ),
    "ASE3-G050": (
        "sha256:e3a3f34ac7693ec8cc99a1673a5c64a1dd845962f79beb662a8683937b77df93"
    ),
    "ASE3-G055": (
        "sha256:12cf39c777d1246127f569cfbc099e2e44ba1d4ee28fed1843f982aa2bfd9b7d"
    ),
    "ASE3-G060": (
        "sha256:b14fa7c4449e1106525236d60a156ee23b7b6e642897ea0a26a8022518342d29"
    ),
    "ASE3-G080": (
        "sha256:44256fe901e73323fa5eab9cc7a57a6c755703777212a12b368ec36984677658"
    ),
}
_NATIVE_DUCKDB_OBJECTIVE_CONTRACT_SHA256S: Final = {
    "ASE3-G040": (
        "sha256:eb85c1ea89c47c947dc3a3c7250bb1ca593853295ec93c0c0e5d0226653ba844"
    ),
    "ASE3-G070": (
        "sha256:5bd12c7f100336e81b239720b4499e2aa459f39fbacd52b3fc0c450f7c3be8a5"
    ),
}
_CONTRACT_LAYERING_OBJECTIVE_CONTRACT_SHA256S: Final = {
    "ASE3-G020": (
        "sha256:c8489e0556271d8ed3f7f372f4f6f132f4d8e9286545592e492f4c0a3e37e595"
    ),
}
_NATIVE_DUCKDB_GATE_CONFIG_SHA256S: Final = {
    "protected_native_dependency_launch_authorization": (
        "sha256:ebedb3eca63414a5d1deb856d41eda3ae3fbd2b8fff135cbfef979956ac2df96"
    ),
    "protected_native_dependency_acceptance": (
        "sha256:9dab4d24487f452b6a94e699ff6dac0f488a7555355aef253e7f7f25b9230d76"
    ),
    "protected_duckdb_connection_policy_acceptance": (
        "sha256:a161e61bcf3eea8c6fb3a1760c8d347a5b53fd6a7ababea5834a5eba668c93f5"
    ),
    "protected_native_duckdb_acceptance_sequence": (
        "sha256:b1e26e00f389672bebef1d8365b8a90244d7b6ac86ad6422ebbee6203c8d75e9"
    ),
}
_NATIVE_DUCKDB_ACCEPTANCE_SEQUENCE: Final = {
    "status": "reserved",
    "phases": [
        {
            "phase": "Q",
            "parent_phase": None,
            "task_ids": ["ASE3-033"],
            "changed_paths": list(_TRANSITION_Q_CHANGED_PATHS),
        },
        {
            "phase": "R",
            "parent_phase": "Q",
            "task_ids": [],
            "changed_paths": list(Q_TO_R_CHANGED_PATHS),
        },
        {
            "phase": "P019",
            "parent_phase": "R",
            "task_ids": [],
            "changed_paths": list(R_TO_P019_CHANGED_PATHS),
        },
        {
            "phase": "A019",
            "parent_phase": "P019",
            "task_ids": ["ASE3-019"],
            "changed_paths": list(P019_TO_A019_CHANGED_PATHS),
        },
        {
            "phase": "A030",
            "parent_phase": "A019",
            "task_ids": ["ASE3-030"],
            "changed_paths": list(A019_TO_A030_CHANGED_PATHS),
        },
        {
            "phase": "P031",
            "parent_phase": "A030",
            "task_ids": [],
            "changed_paths": list(A030_TO_P031_CHANGED_PATHS),
        },
        {
            "phase": "A031",
            "parent_phase": "P031",
            "task_ids": ["ASE3-031"],
            "changed_paths": list(P031_TO_A031_CHANGED_PATHS),
        },
        {
            "phase": "A032",
            "parent_phase": "A031",
            "task_ids": ["ASE3-032"],
            "changed_paths": list(A031_TO_A032_CHANGED_PATHS),
        },
        {
            "phase": "A023/027",
            "parent_phase": "A032",
            "task_ids": ["ASE3-023", "ASE3-027"],
            "changed_paths": list(A032_TO_A023_027_CHANGED_PATHS),
        },
        {
            "phase": "L",
            "parent_phase": "A023/027",
            "task_ids": ["ASE3-022"],
            "changed_paths": list(A023_027_TO_L_CHANGED_PATHS),
        },
    ],
    "root_pin_phase": "R",
    "provider_preparation_phase": "P019",
    "native_authorization_phase": "P031",
    "direct_single_parent_required": True,
    "exact_changed_paths_required": True,
    "manifest_parent_chain_required": True,
    "same_phase_status_dependencies_forbidden": True,
    "status_transition_requires_accepted_parent_phase": True,
    "signed_receipt_chronology_required": True,
    "parallel_phase_receipts_need_no_sibling_order": True,
    "pre_effect_authorization_only_phases": ["P031", "L"],
    "runtime_effect_receipt_phases": ["A031", "A023/027"],
    "post_launch_birth_receipt_path": (
        PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH
    ),
    "post_launch_birth_receipt_schema": (
        PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
    ),
    "post_launch_birth_receipt_required_after_phase": "L",
    "post_launch_birth_receipt_forbidden_through_phase": "L",
    "post_launch_birth_not_before_reload_authorization": True,
    "required_before_task_acceptance": ["ASE3-023", "ASE3-022"],
}
_CONTRACT_LAYERING_POLICY: Final = {
    "task_id": "ASE3-029",
    "task_contract_sha256": (
        "sha256:ea0eae512a86fc9e6098e329d58fcd0963c1706ec621b4da45b1cbdee6f08061"
    ),
    "canonical_task_cid": (
        "baguqeerapt2k36spo3uom3rhn5rucmvnlkqbk5k3nxznd2wetwznq4rg6h2a"
    ),
    "depends_on": ["ASE3-022"],
    "accepted_tree_inventory": {
        "source_task_id": "ASE3-023",
        "source_head_required": True,
        "source_tree_required": True,
        "analyzer_implementation_sha256_required": True,
        "normalized_edge_records_required": True,
        "inventory_sha256_required": True,
        "roadmap_fixed_edge_or_importer_count_allowed": False,
    },
    "neutral_contract_files": [
        "ipfs_accelerate_py/agent_supervisor/contracts/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/contracts/authority.py",
        "ipfs_accelerate_py/agent_supervisor/contracts/execution.py",
        "ipfs_accelerate_py/agent_supervisor/contracts/provider_capacity.py",
    ],
    "lower_effect_owners": {
        "provider_attempt_cas": (
            "ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py"
        ),
        "profile_key_lifecycle": (
            "ipfs_accelerate_py/agent_supervisor/control/profile_authority.py"
        ),
        "plan_store_transactions": (
            "ipfs_accelerate_py/agent_supervisor/control/plan_execution_store.py"
        ),
    },
    "compatibility_entrypoint_wrappers": [
        "ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
    ],
    "explicit_injection_roots": [
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_daemon_runner.py"
        ),
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_supervisor_runner.py"
        ),
    ],
    "ambient_effect_registry_allowed": False,
    "zero_runtime_or_todo_daemon_entrypoint_imports_required": True,
    "neutral_import_time_io_allowed": False,
    "capsule_security_critical_paths": [
        "ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
        "ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py",
        "ipfs_accelerate_py/agent_supervisor/control/profile_authority.py",
        "ipfs_accelerate_py/agent_supervisor/control/plan_execution_store.py",
    ],
    "protected_route_invariants": {
        "unchanged_types": [
            "AgentImplementationRoutePlan",
            "AgentImplementationFallbackDecision",
        ],
        "capacity_projection_dispatch_authorized": False,
        "independent_protected_review_required": True,
    },
    "scheduler_authorization_baseline": {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-policy-authorization@2"
        ),
        "signed_lifecycle_witness_required": True,
        "lifecycle_root_pin_required": True,
        "group_or_other_writable_allowed": False,
        "all_affected_suites_green_before_relocation": True,
    },
    "daemon_runner_authorization_baseline": {
        "test_path": (
            "test/api/test_agent_supervisor_implementation_daemon_runner.py"
        ),
        "stale_test_name": (
            "test_daemon_resolves_relative_worktree_root_for_runner_workspace"
        ),
        "ambient_only_route_fixture_allowed": False,
        "canonical_signed_accepted_route_authorization_and_binding_required": True,
        "accepted_public_artifact_mode": "0400",
        "owned_regular_nonsymlink_required": True,
        "private_signer_material_secure_mode_required": True,
        "complete_file_must_pass": True,
        "test_deselection_allowed": False,
        "route_verifier_bypass_allowed": False,
        "route_weakening_allowed": False,
    },
    "downstream_task_id": "ASE3-028",
    "downstream_requires_accepted_ase3_029": True,
}
_CONTRACT_LAYERING_POLICY_CONFIG_SHA256: Final = (
    "sha256:3a3df93ce151db0404a958bc226b1f32a82620d1fbf9540792521db74cea5326"
)
_PROTECTED_RUNTIME_ACTIVATION_CONFIG_SHA256: Final = (
    "sha256:f33ff19c7611fbfda288e5951515aa4feb12f1e2241866f84ee35a1e36c58d4b"
)
_REFILL_POLICY_CONFIG_SHA256: Final = (
    "sha256:722cf566d64764785aeb1a9f0e68c18a01b80ae82ed24cc081b4e8cc6d55dd3c"
)
_MONITOR_POLICY_CONFIG_SHA256: Final = (
    "sha256:9cbf35362cf2dab3f1c447dac79015ed0c37fe758117e10305ef28c55f6c4a4c"
)
_MONITOR_STRATEGY_PLAN_REQUIREMENTS: Final = (
    "ReviewedHostNamespaceReconciler",
    (
        "EVALUATING→APPEND_RESERVED→APPENDED→PLAN_INVALIDATED→RECOMPILED→"
        "DISPATCHED/ADOPTED"
    ),
    "900 uninterrupted healthy seconds after its final injected recovery",
)
_ASE3_026_PLAN_CONTAINING_HEADING: Final = (
    "## 9. Progress monitoring and deterministic recovery"
)
_ASE3_026_PLAN_SECTION_HEADING: Final = (
    "### 9.1 ASE3-026 protected activation authorization and observation"
)
_ASE3_026_PLAN_SECTION_END_HEADING: Final = "## 10. Implementation waves"
_ASE3_026_PLAN_SECTION_CONTRACT_SHA256: Final = (
    "sha256:23d6ab54a9f58b69c052b294113198ae9256b177287f099c3a9f7c647d5a6f78"
)
_TRANSITION_CONSTRUCTION_PLAN_CONTAINING_HEADING: Final = (
    "## 10. Implementation waves"
)
_TRANSITION_CONSTRUCTION_PLAN_SECTION_HEADING: Final = (
    "### 10.0 ASE3-033 protected transition construction and Q readiness"
)
_TRANSITION_CONSTRUCTION_PLAN_SECTION_END_HEADING: Final = (
    "### 10.1 ASE3-031/032 protected native DuckDB acceptance chain"
)
_TRANSITION_CONSTRUCTION_PLAN_SECTION_CONTRACT_SHA256: Final = (
    "sha256:ff57348d59948406517fb8aaf5006331f7379f756bba928a954838614ee617d4"
)
_NATIVE_DUCKDB_PLAN_CONTAINING_HEADING: Final = "## 10. Implementation waves"
_NATIVE_DUCKDB_PLAN_SECTION_HEADING: Final = (
    "### 10.1 ASE3-031/032 protected native DuckDB acceptance chain"
)
_NATIVE_DUCKDB_PLAN_SECTION_END_HEADING: Final = (
    "### 10.2 Existing repair, transition, and downstream ordering"
)
_NATIVE_DUCKDB_PLAN_SECTION_CONTRACT_SHA256: Final = (
    "sha256:1693ea6477822be1f65e60fe433a2011f50e0d028d2f7b20e0b1a91188debb86"
)
_CONTRACT_LAYERING_PLAN_CONTAINING_HEADING: Final = "## 10. Implementation waves"
_CONTRACT_LAYERING_PLAN_SECTION_HEADING: Final = (
    "### 10.3 ASE3-029 content-bound layering correction"
)
_CONTRACT_LAYERING_PLAN_SECTION_END_HEADING: Final = "## 11. Verification gates"
_CONTRACT_LAYERING_PLAN_SECTION_CONTRACT_SHA256: Final = (
    "sha256:ddeb7ba904712db080bbeda8effb6cd502b71d2db2a32e27eddf482f25264e72"
)
_CONTRACT_LAYERING_PLAN_OUTER_SECTION_CONTRACTS: Final = (
    (
        "audit_finding",
        "## 2. Audit finding and why v3 is required",
        "# Agent Supervisor Prompt-Only Self-Improvement v3 Plan",
        "## 3. Product contract",
        "sha256:19fd4824e6c96b169b30ebf31464ae5f8caa26597844458b51e4e253ea752d91",
    ),
    (
        "wave_ordering",
        "### 10.2 Existing repair, transition, and downstream ordering",
        "## 10. Implementation waves",
        "### 10.3 ASE3-029 content-bound layering correction",
        "sha256:2390412a3228a019556d37edba9f86f12a5f330f3fadac408f4c921c685f0dcb",
    ),
    (
        "verification_gates",
        "## 11. Verification gates",
        "# Agent Supervisor Prompt-Only Self-Improvement v3 Plan",
        "## 12. Rollout and rollback",
        "sha256:30b4475c5329217d15e557ffece17b528d7282894345d5e338f6fa44f788d755",
    ),
)
_CONTRACT_LAYERING_PLAN_REQUIREMENTS: Final = (
    "roadmap-fixed count",
    "canonical signed lifecycle-bound authorization@2",
    "control.provider_attempt_store",
    "control.profile_authority",
    "control.plan_execution_store",
    "ambient registries, service locators",
    "AgentImplementationRoutePlan",
    "AgentImplementationFallbackDecision",
    "dispatch_authorized=False",
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
    "primary_model_id": "grok-4.6",
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
_PROVIDER_FALLBACK_AUTHORIZATION_V2_OWNERSHIP_CONTRACT: Final = {
    "canonical_route_plan_owner": "ipfs_accelerate_py.llm_router",
    "typed_fallback_decision_owner": "ipfs_accelerate_py.llm_router",
    "duplicate_route_policy_or_failure_classification_outside_router_allowed": False,
}
_PROVIDER_FALLBACK_AUTHORIZATION_V2_BOOTSTRAP_GUARANTEES: Final = {
    "explicit_codex_review_conflict_denied": True,
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

# Acceptance phase constants are deliberately gathered here.  ASE3-027 and
# ASE3-023 source/integrated generations and final P-tree blob maps are frozen
# against Git.  ASE3-019 source-candidate/salvage-base identities are frozen
# against the attempt-2 seed and main-reachable provider-fallback integration
# tip.  Product-generation@1 source/clean-replay/integrated triples are frozen
# for ASE3-019/023/027/030/031/032.  ASE3-030 hermetic
# acceptance final values (generations, member blob/raw maps, capsule/archive
# digests, suite count) and ASE3-031/032 suite pins are frozen against Git and
# deterministic suite reports.  The reload generation still needs final
# identities.  Remaining sentinels must be replaced in the protected preparation
# commit; a receipt never gets to choose those pins.
#
# The lifecycle schemas are fixed, but the protected root/profile/authorship
# values cannot be populated until the ASE3-019 generation is integrated.
# These deliberately malformed sentinels keep every portable acceptance path
# closed without letting a receipt select its own lifecycle root or profile.
_FINAL_VALUE_PENDING_019: Final = "FILL_AFTER_ASE3_019_PRODUCT_INTEGRATION"
_FINAL_VALUE_PENDING_023: Final = "FILL_AFTER_ASE3_023_PRODUCT_INTEGRATION"
_FINAL_VALUE_PENDING_027_FINAL_BLOBS: Final = (
    "FILL_AFTER_ASE3_027_FINAL_P_TREE_BLOB_FREEZE"
)
_FINAL_VALUE_PENDING_030: Final = "FILL_AFTER_ASE3_030_PRODUCT_INTEGRATION"
_FINAL_VALUE_PENDING_031_ACCEPTANCE: Final = (
    "FILL_AFTER_ASE3_031_SIGNED_ACCEPTANCE_EVIDENCE"
)
_FINAL_VALUE_PENDING_032_ACCEPTANCE: Final = (
    "FILL_AFTER_ASE3_032_SIGNED_ACCEPTANCE_EVIDENCE"
)
_FINAL_VALUE_PENDING_RELOAD: Final = (
    "FILL_AFTER_A_AND_QUIESCENCE_OBSERVATION_FREEZE"
)
_FINAL_LIFECYCLE_ROOT_DID_PENDING: Final = (
    "did:key:z6Mktp3ogPs9QwXBnKEQrdMThdbuPPNKQXiAP7X7JwXVq1G7"
)
_FINAL_REVIEWER_DID_PENDING: Final = (
    "did:key:z6Mku1TT7TcoD2VksFwNmYGNpE1zprQMmXsT3tz39BzhVdsy"
)
_FINAL_REVIEWER_PROFILE_ID_PENDING: Final = (
    "78d545927196b5dad4c2c76b461927ec"
)
_FINAL_REVIEWER_PROFILE_CONTENT_ID_PENDING: Final = (
    "sha256:bb9681dbaa2e084bd0704675672133e20f4ddeaf2ad0c130b1034b109615ffec"
)
_FINAL_REVIEWER_LIFECYCLE_ANCHOR_ID_PENDING: Final = (
    "475715c3bc8d562132e5323dcf98c13dba1b5aaf8f34546c0c50d40d6f62b2d8"
)
_FINAL_REVIEWER_LIFECYCLE_ANCHOR_DIGEST_PENDING: Final = (
    "sha256:45ff0b7b60c71f8c682e7e32e9e6af1193a5f238fca846190089cf4ee8d41239"
)
_FINAL_REVIEWER_LIFECYCLE_GENERATION_PENDING: Final = 1

_ASE3_031_PRODUCT_IDENTITY: Final = {
    "commit": "25fedf091dad928dad1f83c9f81a54c2d401eabe",
    "parent": "35992cba2261714a0030dff9d58a7a52c31f1d80",
    "tree": "da9e18b507b9991935823dc10d4d7208a47f47f2",
    "binary_patch_sha256": (
        "sha256:90b45612588258bfa34559f84f253801179f44f9968d938698b8dc7de24186fd"
    ),
    "changed_paths": [
        "ipfs_accelerate_py/llm_router.py",
        "test/api/test_agent_supervisor_native_dependency_pin.py",
    ],
    "file_raw_sha256": {
        "ipfs_accelerate_py/llm_router.py": (
            "sha256:69bd7e48a0ffc13f7a868b0b3e9bf8a09506104c4a80e5a78146adabbb73beb4"
        ),
        "test/api/test_agent_supervisor_native_dependency_pin.py": (
            "sha256:429c86eeade5cb7f97eef359085c2efa2821b1df93c6565fc1219512046c59be"
        ),
    },
}
_ASE3_032_PRODUCT_IDENTITY: Final = {
    "commit": "9f1a3cb3c583924878293f9acd676a211106c2e7",
    "parent": "25fedf091dad928dad1f83c9f81a54c2d401eabe",
    "tree": "853191d0e00471bf41452801ae83b0a13b3607d5",
    "binary_patch_sha256": (
        "sha256:b05414db27f68f34634975ff248048c934714c973debcffe71839bdeb84ee124"
    ),
    "stable_patch_id": "161c8c765cb0407b2a55d9d8b03a3884f732179f",
    "ordered_file_hash_manifest_sha256": (
        "sha256:950491fbd586f8a8b5da38dd69b0c4241ae6856eb58e341dde9f346b0b3b3a5f"
    ),
    "changed_paths": [
        "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py",
        "test/api/test_agent_supervisor_duckdb_connection_policy.py",
    ],
    "file_raw_sha256": {
        "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py": (
            "sha256:1c312171d6c5b81003806cc8baddeeb5a41a6c33f2a2d5b1cac9f8f04843cb93"
        ),
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py": (
            "sha256:2e13d01fda9d0be2aa69704b0f3f325c69bfd3b07bc4f98820d99a123a8a7d46"
        ),
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py": (
            "sha256:40a8d96a85ca239b269581a525fab82014614e5d77234c13491a168cc79c3efe"
        ),
        "test/api/test_agent_supervisor_duckdb_connection_policy.py": (
            "sha256:637bad1d0655f0cc6f3e54caac7c84d740fb7a451d857153b24b55c4b10e0080"
        ),
    },
}
_FROZEN_PRODUCT_GIT_IDENTITIES: Final = {
    "25fedf091dad928dad1f83c9f81a54c2d401eabe": {
        "ipfs_accelerate_py/llm_router.py": {
            "mode": "100755",
            "blob": "db2a7d220acf681954d5311a50cc5a970c49573a",
        },
        "test/api/test_agent_supervisor_native_dependency_pin.py": {
            "mode": "100644",
            "blob": "234cf4b6b8d3de0ce7aea3786a553c9638e03f86",
        },
    },
    "9f1a3cb3c583924878293f9acd676a211106c2e7": {
        "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py": {
            "mode": "100644",
            "blob": "95c98fea1c1102d18f603de68459c9424a9ba1f3",
        },
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py": {
            "mode": "100644",
            "blob": "17ab907c3cd4a5f76b6a57c80272345997c422a7",
        },
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py": {
            "mode": "100644",
            "blob": "69cb74e9970e6a44fb16d9653bab0ac66d72b366",
        },
        "test/api/test_agent_supervisor_duckdb_connection_policy.py": {
            "mode": "100644",
            "blob": "808843204606cb6e90a1e6320b58405bec1621bc",
        },
    },
}
_ASE3_031_REVIEWED_DEPENDENCY_PIN: Final = {
    "schema": "ipfs_accelerate_py.agent_supervisor.native-dependency-pin@1",
    "dependency_id": (
        "sha256:bf982f675cc4c4fa212066d706cd387c9821b3b69f5f8cc7c07169bc347b88b5"
    ),
    "module_name": "_duckdb",
    "public_alias": "duckdb",
    "distribution_name": "duckdb",
    "distribution_version": "1.5.2",
    "engine_version": "v1.5.2",
    "extension_filename": "_duckdb.cpython-312-aarch64-linux-gnu.so",
    "python_cache_tag": "cpython-312",
    "python_soabi": "cpython-312-aarch64-linux-gnu",
    "platform_name": "linux",
    "platform_machine": "aarch64",
    "python_executable_sha256": (
        "sha256:1a301bb1763139d48ae638d97b11edf56de6cd185e1b054eae6dc28c271c0c5f"
    ),
    "payload_sha256": (
        "sha256:c378b8f61040764fdc904cf7c0643a005d547f491ab9303e6bd13c33aa353f2a"
    ),
    "size_bytes": 54_278_072,
    "elf_class_bits": 64,
    "elf_endianness": "little",
    "elf_ident_version": 1,
    "elf_osabi": 3,
    "elf_abi_version": 0,
    "elf_object_type": 3,
    "elf_machine": 183,
    "elf_object_version": 1,
    "elf_flags": 0,
    "elf_dt_needed": [
        "libdl.so.2",
        "libstdc++.so.6",
        "libm.so.6",
        "libgcc_s.so.1",
        "libpthread.so.0",
        "libc.so.6",
    ],
}
_ASE3_031_HOST_ABI_TRUST_BOUNDARY: Final = {
    "sealed_bytes": ["reviewed_python_executable", "reviewed__duckdb_payload"],
    "ordered_dt_needed_names_identity_bound": True,
    "system_loader_and_needed_library_bytes_sealed": False,
    "trusted_host_tcb": [
        "kernel",
        "default_system_elf_loader",
        "libdl.so.2",
        "libstdc++.so.6",
        "libm.so.6",
        "libgcc_s.so.1",
        "libpthread.so.0",
        "libc.so.6",
    ],
    "sanitized_parent_exec_required": True,
    "all_ld_environment_removed_before_exec": True,
    "python_side_ld_rejection_is_prevention": False,
    "fully_transitive_hermetic_claim_allowed": False,
}
_ASE3_032_CONNECTION_POLICY_SETTINGS: Final = [
    ["autoinstall_known_extensions", "false", False],
    ["autoload_known_extensions", "false", False],
    ["enable_external_access", "false", False],
    ["allow_unsigned_extensions", "false", False],
    ["lock_configuration", "true", True],
]
_ASE3_032_CONNECTION_TUNING_BOUNDS: Final = {
    "allowed_keys": ["memory_limit", "threads"],
    "threads_minimum": 1,
    "threads_maximum": 256,
    "memory_bytes_minimum": 1_000_000,
    "memory_bytes_maximum": 256_000_000,
    "default_memory_limit": "256MB",
}
_ASE3_032_CONNECTION_SITE_COUNTS: Final = {
    "merge_queue.initialize": 1,
    "merge_queue.legacy_import": 1,
    "merge_queue.operation": 1,
    "merge_resolver.initialize": 1,
    "merge_resolver.operation": 1,
    "duckdb_task_source.materialize": 1,
    "duckdb_task_source.snapshot": 1,
    "lease_coordinator.initialize": 1,
    "lease_coordinator.operation": 1,
    "lease_compaction.target_initialize": 1,
    "lease_compaction.source_read_only": 1,
    "lease_compaction.target_write": 1,
}
_ASE3_032_FOREIGN_CATALOG_CASES: Final = [
    "main_table",
    "main_view",
    "empty_schema",
    "foreign_schema_table",
    "foreign_schema_view",
    "sequence",
    "macro",
    "custom_type",
    "index",
    "check_constraint",
    "unique_constraint",
    "collation",
    "regular_column",
    "generated_column",
    "table_comment",
]

_LIFECYCLE_ROOT_PIN_REQUIRED_FIELDS: Final = (
    "schema",
    "board_namespace",
    "base_head",
    "base_tree",
    "root_identity_did",
    "pinned_at_ms",
    "pin_id",
)
_LOCAL_PROFILE_V5_REQUIRED_FIELDS: Final = (
    "schema",
    "repository_cid",
    "baseline_commit",
    "capabilities",
    "created_at",
    "profile_id",
    "identity_did",
    "revoked",
    "lifecycle_generation",
    "lifecycle_anchor_id",
    "lifecycle_root_path",
    "effect_bounds",
    "budget_cid",
    "resource_cid",
    "route_id",
    "reviewer_identity",
    "reviewer_provider",
    "fallback_provider_id",
    "fallback_model_id",
    "fallback_reasoning_effort",
)
_LOCAL_PROFILE_ANCHOR_V3_REQUIRED_FIELDS: Final = (
    "schema",
    "anchor_id",
    "generation",
    "status",
    "repository_cid",
    "profile_id",
    "profile_content_id",
    "identity_did",
    "did_state_id",
    "did_status",
    "previous_profile_id",
    "previous_profile_content_id",
    "previous_identity_did",
    "previous_anchor_digest",
    "updated_at_ns",
    "root_identity_did",
    "root_signature",
)
_LOCAL_PROFILE_DID_STATE_V1_REQUIRED_FIELDS: Final = (
    "schema",
    "identity_did",
    "status",
    "profile_path",
    "profile_id",
    "profile_content_id",
    "anchor_id",
    "generation",
    "previous_identity_did",
    "updated_at_ns",
    "root_identity_did",
    "root_signature",
    "state_id",
)
_LOCAL_PROFILE_REGISTRY_V2_REQUIRED_FIELDS: Final = (
    "schema",
    "profile_path",
    "lifecycle_root",
    "root_identity_did",
    "registry_id",
)
_LOCAL_OPERATOR_LIFECYCLE_WITNESS_REQUIRED_FIELDS: Final = (
    "schema",
    "board_namespace",
    "base_head",
    "base_tree",
    "observed_at_ms",
    "expires_at_ms",
    "nonce",
    "profile",
    "profile_content_id",
    "profile_signature",
    "anchor",
    "anchor_digest",
    "registry",
    "did_state",
    "did_state_digest",
    "root_identity_did",
    "active_key_signature",
    "root_signature",
    "witness_id",
)
_LOCAL_OPERATOR_LIFECYCLE_WITNESS_BODY_FIELDS: Final = (
    "schema",
    "board_namespace",
    "base_head",
    "base_tree",
    "observed_at_ms",
    "expires_at_ms",
    "nonce",
    "profile",
    "profile_content_id",
    "profile_signature",
    "anchor",
    "anchor_digest",
    "registry",
    "did_state",
    "did_state_digest",
    "root_identity_did",
)
_PROVIDER_FALLBACK_AUTHORIZATION_V1_REQUIRED_FIELDS: Final = (
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
)
_PROVIDER_FALLBACK_AUTHORIZATION_V2_REQUIRED_FIELDS: Final = (
    "schema",
    "board_namespace",
    "authorization_source",
    "route",
    "ownership_contract",
    "bootstrap_route_guarantees",
    "reviewer",
    "authority_bounds",
    "fallback_implementer_identity",
    "lifecycle_root_identity_did",
    "lifecycle_witness_nonce",
    "lifecycle_root_pin_path",
    "lifecycle_root_pin_sha256",
    "authorized_at_ms",
)
_PROVIDER_FALLBACK_AUTHORIZATION_V2_REVIEWER_FIELDS: Final = (
    "identity",
    "provider",
    "profile_id",
    "profile_content_id",
    "lifecycle_anchor_id",
    "generation",
    "witness_path",
    "witness_sha256",
    "signature",
)
_PROVIDER_FALLBACK_AUTHORIZATION_V2_AUTHORITY_BOUNDS_FIELDS: Final = (
    "repository_cid",
    "baseline_commit",
    "effects",
    "budget_cid",
    "resource_cid",
    "authority_cid",
)
_PROVIDER_FALLBACK_AUTHORIZATION_V2_EFFECTS: Final = (
    "edit",
    "isolated_worktree",
    "test",
)

_OPERATOR_REPAIR_ACCEPTANCE_REQUIRED_FIELDS: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "task",
    "recovery",
    "implementation",
    "acceptance_parent",
    "validation",
    "review",
    "denials",
)
_ACCEPTANCE_TASK_REQUIRED_FIELDS: Final = (
    "task_id",
    "canonical_task_cid",
    "goal_id",
    "repairs_task",
    "todo_contract_sha256",
    "completed_contract_sha256",
    "status_before",
    "status_after",
)
_ACCEPTANCE_RECOVERY_REQUIRED_FIELDS: Final = (
    "artifact",
    "pointer",
    "historical_completion_authority",
    "branch_local_completion_authority",
    "repair_required",
)
_ACCEPTANCE_IMPLEMENTATION_REQUIRED_FIELDS: Final = (
    "generations",
    "final_blobs",
)
_ACCEPTANCE_GENERATION_REQUIRED_FIELDS: Final = (
    "role",
    "source_commit",
    "source_parent",
    "source_tree",
    "integrated_commit",
    "integrated_parent",
    "integrated_tree",
    "binary_full_index_patch_sha256",
    "changed_paths",
)
_ACCEPTANCE_PARENT_REQUIRED_FIELDS: Final = (
    "head",
    "tree",
    "branch",
    "manifest_schema",
    "receipt_paths_absent",
    "task_statuses",
    "reload_gate_status",
)
_ACCEPTANCE_VALIDATION_REQUIRED_FIELDS: Final = (
    "command",
    "exit_code",
    "passed",
    "passed_count",
    "failed_count",
    "validated_head",
    "validated_tree",
)
_ACCEPTANCE_REVIEW_REQUIRED_FIELDS: Final = (
    "reviewer_identity",
    "reviewer_provider",
    "profile_id",
    "profile_content_id",
    "lifecycle_anchor_id",
    "lifecycle_anchor_digest",
    "lifecycle_generation",
    "lifecycle_witness_path",
    "lifecycle_witness_sha256",
    "lifecycle_witness_id",
    "lifecycle_witness_nonce",
    "lifecycle_root_pin_path",
    "lifecycle_root_pin_sha256",
    "lifecycle_root_identity_did",
    "fallback_authorization_id",
    "fallback_authorization_sha256",
    "implementer_identity",
    "implementer_provider",
    "algorithm",
    "signed_at",
    "signature",
)
_ACCEPTANCE_REVIEW_AUTHORITY_FIELDS: Final = (
    "reviewer_identity",
    "reviewer_provider",
    "profile_id",
    "profile_content_id",
    "lifecycle_anchor_id",
    "lifecycle_anchor_digest",
    "lifecycle_generation",
    "lifecycle_witness_path",
    "lifecycle_witness_sha256",
    "lifecycle_witness_id",
    "lifecycle_witness_nonce",
    "lifecycle_root_pin_path",
    "lifecycle_root_pin_sha256",
    "lifecycle_root_identity_did",
    "fallback_authorization_id",
    "fallback_authorization_sha256",
)
_LIFECYCLE_AUTHORITY_TIME_FIELDS: Final = (
    "lifecycle_witness_observed_at_ms",
    "lifecycle_witness_expires_at_ms",
    "fallback_authorized_at_ms",
)
_HERMETIC_IDENTITY_ACCEPTANCE_REQUIRED_FIELDS: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "task",
    "acceptance_parent",
    "provenance",
    "closure",
    "probe",
    "suite",
    "review",
    "denials",
)
_HERMETIC_PROVENANCE_REQUIRED_FIELDS: Final = (
    "generations",
    "final_blobs",
    "final_raw_sha256",
)
_HERMETIC_GENERATION_REQUIRED_FIELDS: Final = (
    "role",
    "source_commit",
    "source_parent",
    "source_tree",
    "replay_commit",
    "replay_parent",
    "replay_tree",
    "integrated_commit",
    "integrated_parent",
    "integrated_tree",
    "source_patch_sha256",
    "replay_patch_sha256",
    "integrated_patch_sha256",
    "changed_paths",
)
_HERMETIC_CLOSURE_REQUIRED_FIELDS: Final = (
    "manifest",
    "manifest_sha256",
    "capsule",
    "capsule_sha256",
    "archive",
    "members",
    "module_origins",
    "cid_vectors",
)
_HERMETIC_MANIFEST_REQUIRED_FIELDS: Final = (
    "schema",
    "source_head",
    "source_tree",
    "member_paths",
    "module_names",
    "cid_profile",
)
_HERMETIC_CAPSULE_REQUIRED_FIELDS: Final = (
    "schema",
    "manifest_sha256",
    "archive_sha256",
    "archive_root_sha256",
    "sealed_descriptor_sha256",
    "member_count",
)
_HERMETIC_ARCHIVE_REQUIRED_FIELDS: Final = (
    "schema",
    "format",
    "sha256",
    "root_sha256",
    "member_paths",
)
_HERMETIC_MEMBER_REQUIRED_FIELDS: Final = (
    "git_blob",
    "raw_sha256",
    "archive_member_sha256",
)
_HERMETIC_MODULE_ORIGIN_REQUIRED_FIELDS: Final = ("member_path", "origin")
_HERMETIC_PROBE_REQUIRED_FIELDS: Final = (
    "command",
    "environment",
    "exit_code",
    "isolated",
    "user_site_enabled",
    "pythonpath_present",
    "multiformats_imported",
    "repository_or_candidate_imported",
    "sealed_descriptor_only",
    "all_modules_imported",
    "all_module_origins_verified",
    "raw_cid_minted",
    "raw_cid_validated",
    "dag_json_cid_minted",
    "dag_json_cid_validated",
    "scheduler_or_provider_effect_started",
    "stdout_sha256",
    "stderr_sha256",
)
_HERMETIC_SUITE_REQUIRED_FIELDS: Final = (
    "command",
    "exit_code",
    "passed",
    "passed_count",
    "failed_count",
    "validated_head",
    "validated_tree",
    "report_sha256",
)
_SEQUENTIAL_ACCEPTANCE_PARENT_REQUIRED_FIELDS: Final = (
    "head",
    "tree",
    "branch",
    "phase",
    "manifest_schema",
    "prior_artifacts",
    "future_artifact_paths_absent",
    "task_statuses",
    "reload_gate_status",
)
_NATIVE_DEPENDENCY_TASK_REQUIRED_FIELDS: Final = (
    "task_id",
    "canonical_task_cid",
    "todo_contract_sha256",
    "completed_contract_sha256",
    "status_before",
    "status_after",
)
_NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_REQUIRED_FIELDS: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "phase",
    "task",
    "acceptance_parent",
    "authorization_id",
    "product",
    "native_pin",
    "host_abi_trust_boundary",
    "claims",
    "review",
    "denials",
)
_NATIVE_DEPENDENCY_AUTHORIZATION_CLAIMS: Final = {
    "pre_launch_authorization_only": True,
    "runtime_effect_started": False,
    "preload_observed": False,
    "sealed_fd_verified": False,
    "process_terminal_state_observed": False,
}
_NATIVE_DEPENDENCY_AUTHORIZATION_DENIALS: Final = {
    "authorization_creates_acceptance": False,
    "authorization_claims_launch_effect": False,
    "inspection_evidence_is_acceptance": False,
    "ambient_ld_environment_allowed": False,
    "python_late_rejection_counts_as_parent_sanitization": False,
    "receipt_selected_product_identity_allowed": False,
    "receipt_selected_native_pin_allowed": False,
    "self_review_allowed": False,
    "codex_or_openai_reviewer_allowed": False,
}
_NATIVE_DEPENDENCY_ACCEPTANCE_REQUIRED_FIELDS: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "phase",
    "task",
    "acceptance_parent",
    "launch_authorization",
    "product",
    "native_pin",
    "sealed_descriptor",
    "process_terminal",
    "preload_evidence",
    "host_abi_trust_boundary",
    "suite",
    "review",
    "denials",
)
_SEALED_NATIVE_DESCRIPTOR_REQUIRED_FIELDS: Final = (
    "schema",
    "descriptor",
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_nlink",
    "size_bytes",
    "payload_sha256",
    "seals",
)
_NATIVE_PROCESS_TERMINAL_REQUIRED_FIELDS: Final = (
    "terminal_sentinel_set_before_native_module_creation",
    "native_module_creation_started",
    "partial_initialization_retry_denied",
    "second_preload_attempt_denied",
    "terminal_returncode",
)
_NATIVE_PRELOAD_EVIDENCE_REQUIRED_FIELDS: Final = (
    "launch_schema",
    "accepted_authorization_id",
    "sealed_fd_verified_before_module_creation",
    "module_name",
    "public_alias",
    "distribution_version",
    "engine_version",
    "query_42_result",
    "parent_environment_sanitized_before_exec",
    "forbidden_parent_environment_names",
    "child_observed_forbidden_environment_names",
    "python_side_environment_rejection_triggered",
    "runtime_effect_started_at",
    "runtime_effect_started_after_authorization",
)
_NATIVE_DEPENDENCY_ACCEPTANCE_DENIALS: Final = {
    "unsealed_payload_allowed": False,
    "descriptor_substitution_allowed": False,
    "authorization_id_mismatch_allowed": False,
    "native_retry_after_partial_initialization_allowed": False,
    "ambient_ld_environment_allowed": False,
    "fully_transitive_hermetic_claim_allowed": False,
    "receipt_selected_product_identity_allowed": False,
    "receipt_selected_native_pin_allowed": False,
    "self_review_allowed": False,
    "codex_or_openai_reviewer_allowed": False,
}
_DUCKDB_POLICY_ACCEPTANCE_REQUIRED_FIELDS: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "phase",
    "task",
    "acceptance_parent",
    "product",
    "connection_birth_policy",
    "connection_sites",
    "external_byte_boundary",
    "catalog_seal",
    "legacy_migration",
    "compaction",
    "suite",
    "review",
    "denials",
)
_DUCKDB_CONNECTION_BIRTH_POLICY_REQUIRED_FIELDS: Final = (
    "settings_in_connect_call",
    "tuning_bounds",
    "lock_configuration_last",
    "returned_connection_exact_bool_tuple",
    "close_on_verification_failure",
    "caller_override_or_coercion_allowed",
)
_DUCKDB_EXTERNAL_BYTE_BOUNDARY: Final = {
    "install_or_fetch_allowed": False,
    "dynamic_external_extension_bytes_allowed": False,
    "external_filesystem_or_network_paths_allowed": False,
    "reviewed_statically_linked_load_allowed": True,
    "in_memory_attach_allowed": True,
    "compaction_attach_count": 0,
}
_DUCKDB_CATALOG_SEAL_REQUIRED_FIELDS: Final = (
    "path_independent_full_persistent_catalog_equality",
    "inventories",
    "foreign_catalog_cases_rejected",
    "source_bytes_unchanged_on_rejection",
    "temporary_files_cleaned_on_rejection",
)
_DUCKDB_LEGACY_MIGRATION_REQUIRED_FIELDS: Final = (
    "transactional_two_step_add_default_then_set_not_null",
    "populated_backfill_verified",
    "idempotent_reopen_verified",
    "mid_step_failure_rolls_back",
    "post_compaction_catalog_equal",
)
_DUCKDB_COMPACTION_REQUIRED_FIELDS: Final = (
    "attach_count",
    "source_read_only",
    "target_policy_initialized",
    "partial_copy_failure_preserves_authoritative_store",
    "atomic_replace_failure_preserves_authoritative_store",
    "foreign_catalog_rejection_preserves_source_bytes",
    "temporary_files_cleaned",
)
_DUCKDB_POLICY_ACCEPTANCE_DENIALS: Final = {
    "configuration_override_allowed": False,
    "boolean_integer_coercion_allowed": False,
    "foreign_persistent_catalog_drop_allowed": False,
    "one_step_not_null_default_migration_allowed": False,
    "compaction_attach_allowed": False,
    "receipt_selected_product_identity_allowed": False,
    "self_review_allowed": False,
    "codex_or_openai_reviewer_allowed": False,
}
_PROTECTED_ACCEPTANCE_SUITE_REQUIRED_FIELDS: Final = (
    "command",
    "exit_code",
    "passed",
    "passed_count",
    "failed_count",
    "validated_head",
    "validated_tree",
    "report_sha256",
    "required_test_functions",
)
_ASE3_031_REQUIRED_TEST_FUNCTIONS: Final = [
    "test_inspection_is_path_free_evidence_and_sealing_requires_acceptance",
    "test_source_inspection_is_stable_nofollow_evidence",
    "test_sealing_denies_wrong_pin_identity",
    "test_elf_parser_rejects_ambient_dynamic_loader_tags",
    "test_elf_parser_rejects_malformed_or_ambient_layout",
    "test_launch_json_and_descriptor_binding_are_strict",
    "test_unsealed_fd_and_descriptor_mutations_are_denied",
    "test_synthetic_preload_uses_exact_loader_alias_and_queries",
    "test_preload_denies_ambient_loader_environment_before_loader_creation",
    "test_preload_failure_makes_process_terminal_without_a_second_loader_call",
    "test_real_aarch64_duckdb_loads_from_sealed_fd_under_isolated_python",
]
_ASE3_032_REQUIRED_TEST_FUNCTIONS: Final = [
    "test_policy_is_atomic_verified_on_returned_connection_and_immutable",
    "test_caller_cannot_override_or_smuggle_connection_configuration",
    "test_policy_verification_rejects_integer_bool_spoof_and_closes",
    "test_policy_blocks_dynamic_extension_fetch_load_and_http_access",
    "test_reviewed_statically_linked_modules_do_not_cross_external_byte_boundary",
    "test_only_in_memory_attach_is_nonexternal",
    "test_policy_blocks_arbitrary_local_extension_load",
    "test_every_accepted_runtime_connection_site_uses_the_canonical_policy",
    "test_compaction_rejects_any_foreign_persistent_catalog_without_source_change",
    "test_populated_legacy_additive_schema_upgrades_idempotently_and_compacts",
    "test_legacy_additive_schema_failure_between_steps_rolls_back",
    "test_compaction_partial_copy_failure_keeps_authoritative_store",
    "test_compaction_atomic_replace_failure_keeps_authoritative_store",
]
_HERMETIC_CID_VECTORS: Final = (
    {
        "name": "empty-raw",
        "codec": "raw",
        "multicodec": 0x55,
        "multihash": "sha2-256",
        "input_sha256": (
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "cid": "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku",
        "minted": True,
        "validated": True,
    },
    {
        "name": "empty-dag-json",
        "codec": "dag-json",
        "multicodec": 0x0129,
        "multihash": "sha2-256",
        "input_sha256": (
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        ),
        "cid": "baguqeeraiqjw7i2vwntyuekgvulpp2det2kpwt6cd7tx5ayqybqpmhfk76fa",
        "minted": True,
        "validated": True,
    },
)
_HERMETIC_REQUIRED_MODULE_MEMBER_MAP: Final = {
    "ipfs_accelerate_py": "ipfs_accelerate_py/__init__.py",
    "ipfs_accelerate_py.agent_supervisor": (
        "ipfs_accelerate_py/agent_supervisor/__init__.py"
    ),
    "ipfs_accelerate_py.agent_supervisor.core": (
        "ipfs_accelerate_py/agent_supervisor/core/__init__.py"
    ),
    "ipfs_accelerate_py.agent_supervisor.core.multiformats_identity": (
        "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py"
    ),
    "ipfs_accelerate_py.llm_router": "ipfs_accelerate_py/llm_router.py",
    "ipfs_accelerate_py.utils": "ipfs_accelerate_py/utils/__init__.py",
    "ipfs_accelerate_py.utils.cid_utils": (
        "ipfs_accelerate_py/utils/cid_utils.py"
    ),
}
_HERMETIC_REQUIRED_MEMBER_PATHS: Final = frozenset(
    _HERMETIC_REQUIRED_MODULE_MEMBER_MAP.values()
)
_HERMETIC_HOSTILE_PROBE_ARGV: Final = (
    "python",
    "-I",
    "-c",
    (
        "import runpy;runpy.run_module("
        "'ipfs_accelerate_py.agent_supervisor.core.multiformats_identity',"
        "run_name='__main__')"
    ),
)
_HERMETIC_ACCEPTANCE_DENIALS: Final = {
    "machine_local_capsule_root_allowed": False,
    "user_site_allowed": False,
    "pythonpath_allowed": False,
    "optional_multiformats_required": False,
    "mutable_repository_or_candidate_import_allowed": False,
    "missing_member_allowed": False,
    "substituted_member_allowed": False,
    "extra_archive_member_allowed": False,
    "zip_shadow_allowed": False,
    "unverified_module_origin_allowed": False,
    "scheduler_or_provider_effect_before_probe_allowed": False,
    "receipt_selected_provenance_allowed": False,
    "convergence_manifest_digest_in_receipt_allowed": False,
}
_RELOAD_RECEIPT_REQUIRED_FIELDS: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "task",
    "acceptance_parent",
    "incident",
    "stopped_generation",
    "authorization",
    "review",
    "denials",
)
_RELOAD_TASK_REQUIRED_FIELDS: Final = (
    "task_id",
    "canonical_task_cid",
    "blocked_contract_sha256",
    "completed_contract_sha256",
    "status_before",
    "status_after",
)
_RELOAD_PARENT_REQUIRED_FIELDS: Final = (
    "head",
    "tree",
    "branch",
    "manifest_schema",
    "acceptance_receipts",
    "task_statuses",
)
_RELOAD_INCIDENT_REQUIRED_FIELDS: Final = (
    "attempt2_incident",
    "attempt2_incident_sha256",
    "operator_salvage_receipt",
    "operator_salvage_receipt_sha256",
    "accepted_control_plane_sha256",
)
_RELOAD_STOPPED_GENERATION_REQUIRED_FIELDS: Final = (
    "generation_id",
    "generation_number",
    "head",
    "tree",
    "scheduler_path",
    "scheduler_blob",
    "scheduler_raw_sha256",
    "daemon_path",
    "daemon_blob",
    "daemon_raw_sha256",
    "observed_owned_processes",
    "observed_scoped_provider_containers",
    "observed_inflight_attempts",
)
_RELOAD_AUTHORIZATION_REQUIRED_FIELDS: Final = (
    "source_head",
    "source_tree",
    "stopped_generation_id",
    "target_generation_id",
    "target_generation_number",
    "target_scheduler_blob",
    "target_daemon_blob",
    "lease_namespace",
    "lease_state_at_authorization",
    "required_cas_transition",
    "single_winner_required",
    "launch_only_after_l_validates",
    "post_launch_birth_receipt_required",
    "post_launch_birth_receipt_schema",
    "attempt_counters_unchanged",
    "queue_history_unchanged",
    "legacy_refill_unchanged",
    "runtime_effect_started",
)
_RELOAD_DENIALS: Final = {
    "receipt_selected_parent_allowed": False,
    "convergence_manifest_digest_in_receipt_allowed": False,
    "attempt_counter_mutation_authorized": False,
    "queue_history_mutation_authorized": False,
    "legacy_refill_enablement_authorized": False,
    "runtime_state_mutation_beyond_reload_authorized": False,
    "pre_l_lease_acquisition_allowed": False,
    "receipt_claims_new_generation_ran_allowed": False,
    "birth_evidence_in_reload_receipt_allowed": False,
    "self_review_allowed": False,
    "codex_or_openai_reviewer_allowed": False,
}
_PROVIDER_ATTEMPT_GENERATION_BIRTH_REQUIRED_FIELDS: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "phase",
    "reload_authorization",
    "generation",
    "process_birth",
    "review",
    "denials",
)
_PROVIDER_ATTEMPT_GENERATION_BIRTH_RELOAD_FIELDS: Final = (
    "path",
    "sha256",
    "head",
    "tree",
    "phase",
)
_PROVIDER_ATTEMPT_GENERATION_BIRTH_GENERATION_FIELDS: Final = (
    "generation_id",
    "generation_number",
)
_PROVIDER_ATTEMPT_GENERATION_BIRTH_PROCESS_FIELDS: Final = (
    "effect_started_at",
    "process_started_at",
    "runtime_effect_started",
)
_PROVIDER_ATTEMPT_GENERATION_BIRTH_DENIALS: Final = {
    "pre_l_effect_allowed": False,
    "receipt_selected_reload_authority_allowed": False,
    "generation_identity_substitution_allowed": False,
    "self_review_allowed": False,
    "codex_or_openai_reviewer_allowed": False,
}
_REPAIR_ACCEPTANCE_DENIALS: Final = {
    "historical_completion_authority": False,
    "branch_local_completion_authority": False,
    "self_review_allowed": False,
    "codex_or_openai_reviewer_allowed": False,
    "attempt_counter_mutation_authorized": False,
    "runtime_state_mutation_authorized": False,
}
_SALVAGE_ACCEPTANCE_DENIALS: Final = {
    "self_review_allowed": False,
    "codex_or_openai_reviewer_allowed": False,
    "arbitrary_failure_fallback_allowed": False,
    "post_effect_fallback_allowed": False,
    "attempt_counter_mutation_authorized": False,
    "provider_capacity_attempt_restoration_allowed": False,
    "objective_refill_authorized": False,
    "codebase_refill_authorized": False,
}
_ACCEPTANCE_TASK_CONTRACTS: Final = {
    "ASE3-019": {
        "filename": OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
        "schema": OPERATOR_SALVAGE_RECEIPT_019_SCHEMA,
        "canonical_task_cid": _ASE3_019_ATTEMPT2_TASK_CID,
        "goal_id": "ASE3-G020",
        "repairs_task": "ASE3-019-attempt-2",
        "todo_contract_sha256": _ASE3_019_CONTRACT_SHA256,
        "completed_contract_sha256": (
            "sha256:1be44352e66949dcf7789ea22e67c5d821e6d93b47177f81476bd318737e041c"
        ),
    },
    "ASE3-030": {
        "filename": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME,
        "schema": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA,
        "canonical_task_cid": (
            "baguqeeraixg3vmaaqjjzelv2eh5hhib3y57ezmtrp2uq5aufnvuakmjnov6q"
        ),
        "goal_id": "ASE3-G040",
        "repairs_task": "ASE3-019-control-plane-capsule",
        "todo_contract_sha256": (
            "sha256:fe06816d222c538150df4f2c67773e722233c2d0cf4ad0199ae9968e11e52263"
        ),
        "completed_contract_sha256": (
            "sha256:987d74c4304897722a00a614f161938c0ac0803569c11f09b7e63ebad9a2a935"
        ),
    },
    "ASE3-023": {
        "filename": OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME,
        "schema": OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
        "canonical_task_cid": (
            "baguqeerazljo4lkewfr3e6obxky2dydxuqzrczxwz74sjzfz32bwdvf4qvla"
        ),
        "goal_id": "ASE3-G040",
        "repairs_task": "ASE3-006",
        "todo_contract_sha256": (
            "sha256:c13240a72521f3f7f71b39e5d404daa5825581b1606e707a5dad8e693af73f25"
        ),
        "completed_contract_sha256": (
            "sha256:cb8ee6d5381dbbf8c1c46523d106e2bdc665fe8fcddc2f86b0248082eeb5d477"
        ),
    },
    "ASE3-027": {
        "filename": OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME,
        "schema": OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
        "canonical_task_cid": (
            "baguqeerarq7rlvae2dqoqdctzibxe5fuvte4mxptjcn45ri75c4hp742lb6q"
        ),
        "goal_id": "ASE3-G020",
        "repairs_task": "ASE3-018",
        "todo_contract_sha256": (
            "sha256:69853f7f6174a9bd118b4fca13d5ba8e897e962def801d7fb012d9e4969f7d8c"
        ),
        "completed_contract_sha256": (
            "sha256:49ac9f26bef2d2b71d8afe9e45a06a500902f79091f1b125731e41ecbe4cdadd"
        ),
    },
}
_SEQUENTIAL_TASK_CONTRACTS: Final = {
    **_ACCEPTANCE_TASK_CONTRACTS,
    "ASE3-031": {
        "filename": NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME,
        "schema": NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA,
        "canonical_task_cid": (
            "baguqeeraxwgr3bpjg2efihsmv3id5fknng2oa5bapas3fdq4hwjklpa4lxdq"
        ),
        "todo_contract_sha256": (
            "sha256:e749edab95f0f359cb39ab69cfc9a7858490600f4e73622a64e4f6f099eda7ff"
        ),
        "completed_contract_sha256": (
            "sha256:4508ec83e0ee46240fee9a9b2566733842e90bbed8cc73acf84f1319c8ad141f"
        ),
    },
    "ASE3-032": {
        "filename": DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME,
        "schema": DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
        "canonical_task_cid": (
            "baguqeerarapsgz3yexowvp6ppcroujqmfymfgsmrapanyv5ggclalkyxicva"
        ),
        "todo_contract_sha256": (
            "sha256:b44b0a0c2853296def5276f1fb08219480d9b0bb2282f3b7d9587fafd1ec28be"
        ),
        "completed_contract_sha256": (
            "sha256:0fdea6170e4e336a27c9a61f3eefaefdcc8fc2a77ae5433a57a8e52a22d8cd75"
        ),
    },
}
_ACCEPTANCE_REVIEWER_FINAL_VALUES: Final = {
    "reviewer_identity": _FINAL_REVIEWER_DID_PENDING,
    "profile_id": _FINAL_REVIEWER_PROFILE_ID_PENDING,
    "profile_content_id": _FINAL_REVIEWER_PROFILE_CONTENT_ID_PENDING,
    "lifecycle_anchor_id": _FINAL_REVIEWER_LIFECYCLE_ANCHOR_ID_PENDING,
    "lifecycle_anchor_digest": (
        _FINAL_REVIEWER_LIFECYCLE_ANCHOR_DIGEST_PENDING
    ),
    "lifecycle_generation": _FINAL_REVIEWER_LIFECYCLE_GENERATION_PENDING,
}
_ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES: Final = {
    "ASE3-019": {
        "ready": True,
        "pending": None,
        # Attempt-2 incident seed (independent non-ancestor of Q parents).
        "source_candidate": {
            "source_commit": "eb68ff2a20e0719388f60ffef1f5bfcb90b79263",
            "source_tree": "695e2d6f07bc1c48bdc34ebb490342444de2cbef",
        },
        # Main-reachable integrated tip of the provider-fallback product seal.
        "salvage_base": {
            "head": "1bd07b7261c86e4cf8301b34dbaf9728fc6e7818",
            "tree": "e20cf0acc9fe5d17ac439b9c9ddff6332810e583",
            "branch": "agent/prompt-self-improvement-v3",
        },
        "generations": (),
        "final_blobs": {},
        "validation_passed_count": 160,
    },
    "ASE3-023": {
        "ready": True,
        "pending": None,
        "generations": (
            {
                "role": "product-salvage",
                "source_commit": "b072068dcf8b954d6ec454d89a014a8a80b6d2ef",
                "source_parent": "c5da756756064869dedab4aff17a0d17d8549488",
                "source_tree": "8567257c0d101198b5b2e7fde47031c603ab5e09",
                "integrated_commit": "dee1c4f09f01bf8131a6b675eccce68e7aacb34d",
                "integrated_parent": "8f613252c2ff1460e6f2b551a2a8600a2d3ee519",
                "integrated_tree": "cb0ad281d6e89bea8cce24abe8e6a0bfc3117610",
                "binary_full_index_patch_sha256": (
                    "sha256:ba3000aab09784b9830eec492442da29168a03462e04a6659909e495490500ad"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
                    "test/api/test_agent_supervisor_configured_board_scheduler.py",
                    "test/api/test_agent_supervisor_implementation_supervisor_runner.py",
                    "test/api/test_agent_supervisor_prompt_v3_parallelism.py",
                ),
            },
            {
                "role": "capsule-identity",
                "source_commit": "cb7df87b0b1b96915af01d07fe7b1d3e991dfdc9",
                "source_parent": "b072068dcf8b954d6ec454d89a014a8a80b6d2ef",
                "source_tree": "30e9ddf36737f7c0667ec60fb07e90130fd5b899",
                "integrated_commit": "bd1cd48d4379b30f1584f8967921345216166f56",
                "integrated_parent": "dee1c4f09f01bf8131a6b675eccce68e7aacb34d",
                "integrated_tree": "e0d9a530d559a24a1f5785d2ed250d878571e242",
                "binary_full_index_patch_sha256": (
                    "sha256:ef27f4a69ab3743e0b60eec43e2cae866f3d9cbe18b7b9307cbcb68b0807f892"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
                    "test/api/test_agent_supervisor_configured_board_scheduler.py",
                    "test/api/test_agent_supervisor_prompt_v3_parallelism.py",
                ),
            },
            {
                "role": "recovery-barrier",
                "source_commit": "7abfc4ca768dd5086600bb30815256c79aaace74",
                "source_parent": "cb7df87b0b1b96915af01d07fe7b1d3e991dfdc9",
                "source_tree": "79ed78e765eada55762bae63bdfdf81264111f8a",
                "integrated_commit": "a43b2ce74816ac9226f6319b92425d0b002b6be6",
                "integrated_parent": "bd1cd48d4379b30f1584f8967921345216166f56",
                "integrated_tree": "bbb94ffe87c3b582e40b1052ba5b9dc1ca8b4c40",
                "binary_full_index_patch_sha256": (
                    "sha256:0dce5226c7e543997cf55c1115622cc150a8261ca68f1d69252600ffd8fc3bb3"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
                    "test/api/test_agent_supervisor_configured_board_scheduler.py",
                    "test/api/test_agent_supervisor_implementation_supervisor_runner.py",
                    "test/api/test_agent_supervisor_prompt_v3_parallelism.py",
                ),
            },
        ),
        "final_blobs": {
            "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py": (
                "4af712560cc96b1a2014002c45ee53acc4d114bf"
            ),
            "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py": (
                "d47251c96e6882e39d8dc4bb22f22d193d2852ba"
            ),
            "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py": (
                "1e98703b8661979037ec409bcd6aee7cdf6e7fe6"
            ),
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py": (
                "217f6aba6e85a5fefe56f24aefc4848bd2ec41f0"
            ),
            "test/api/test_agent_supervisor_prompt_v3_parallelism.py": (
                "8cbfdc6f63e0db87cfef2a99ea7210b077a2ff90"
            ),
            "test/api/test_agent_supervisor_configured_board_scheduler.py": (
                "24f8bbf3d23bfef37d13c8246aef15f74eeb42e8"
            ),
            "test/api/test_agent_supervisor_implementation_supervisor_runner.py": (
                "903f076214fd9cc077869bf1cd2837828d1b2643"
            ),
        },
        "validation_passed_count": 110,
    },
    "ASE3-027": {
        "ready": True,
        "pending": None,
        "generations": (
            {
                "role": "product",
                "source_commit": "aaf7d722a0c23f5a047b38708f6290631848e06b",
                "source_parent": "e6f8e4a7771907372fc93b0f35cfde30170c2b2a",
                "source_tree": "323f14dbcd9b15b09046cd3a481eb8588a6ede2a",
                "integrated_commit": "6a0047436f9515281127c17913132a23cecfe56c",
                "integrated_parent": "0321fd148bf7c5dc6e91251d119dc25f853e546f",
                "integrated_tree": "ec240271c204fc8befcddef6b7ca2bcad124dc3b",
                "binary_full_index_patch_sha256": (
                    "sha256:b2f8be2a8126e5302d1a02f627dc1def01c54899181ec5ade59cfa22c2649062"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py",
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py",
                    "test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py",
                ),
            },
            {
                "role": "test-contract-correction",
                "source_commit": "bd93ae76c277ae8761cd2abe0df79685d0b1b8ef",
                "source_parent": "aaf7d722a0c23f5a047b38708f6290631848e06b",
                "source_tree": "043677ccb5216204b3142bb8e2b7f71d4ca74bd9",
                "integrated_commit": "d32415e4308a8462e96b4d04f807338f0a2d8b53",
                "integrated_parent": "6a0047436f9515281127c17913132a23cecfe56c",
                "integrated_tree": "87191ce65498a637c7b9500d72d434cadb8efbef",
                "binary_full_index_patch_sha256": (
                    "sha256:e64ab06bc28e13ae08591708f634a855bc752208a8bfbaf7164d704386f0d9fd"
                ),
                "changed_paths": (
                    "test/api/test_agent_supervisor_inference_runtime.py",
                    "test/api/test_agent_supervisor_prompt_v3_resolution.py",
                ),
            },
        ),
        "final_blobs": {
            "ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py": (
                "61cddc9adabb431fbf2aa98a300072d88be8088b"
            ),
            "ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py": (
                "4671f417029bf7f9a3f7b578b9db65c3633f4242"
            ),
            "test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py": (
                "d2736d0af243995977f8a10050ec89fab4dc9785"
            ),
            "test/api/test_agent_supervisor_prompt_v3_resolution.py": (
                "5b8d0087ec4f92e7e3f7f942a71d378fa3d37a3f"
            ),
            "test/api/test_agent_supervisor_inference_runtime.py": (
                "dde3e01a95ef430d87d7e879c579c7d4d6fbac1d"
            ),
        },
        "validation_passed_count": 174,
    },
}
_HERMETIC_IDENTITY_FINAL_VALUES: Final = {
    "ready": True,
    "pending": None,
    "generations": (
            {
                "role": "hermetic-cid-seal",
                "source_commit": "fd2fb0b42e60ed6f9e03ccfef175b0cdd9ba9c2b",
                "source_parent": "c5da756756064869dedab4aff17a0d17d8549488",
                "source_tree": "22101ffb8eb2568d5f3c457e9664bba014e1e8ee",
                "replay_commit": "f97cad9607f16e71c7b1383e55b624f5149def71",
                "replay_parent": "c5da756756064869dedab4aff17a0d17d8549488",
                "replay_tree": "22101ffb8eb2568d5f3c457e9664bba014e1e8ee",
                "integrated_commit": "8ef29834e9af7629a621d583ec43bf37f136b10e",
                "integrated_parent": "32face22bc17eb0b76f09fed2186ef799075b110",
                "integrated_tree": "46be4b75f0a5ab9be06cf3b85a75691157fba09e",
                "source_patch_sha256": (
                    "sha256:863e754bc88c6743b5313548724bbbe5b741263bff1ff143dc5b38aa4f898d16"
                ),
                "replay_patch_sha256": (
                    "sha256:863e754bc88c6743b5313548724bbbe5b741263bff1ff143dc5b38aa4f898d16"
                ),
                "integrated_patch_sha256": (
                    "sha256:863e754bc88c6743b5313548724bbbe5b741263bff1ff143dc5b38aa4f898d16"
                ),
                "binary_full_index_patch_sha256": (
                    "sha256:863e754bc88c6743b5313548724bbbe5b741263bff1ff143dc5b38aa4f898d16"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
                    "ipfs_accelerate_py/llm_router.py",
                    "ipfs_accelerate_py/utils/cid_utils.py",
                    "test/api/test_agent_supervisor_hermetic_cid_capsule.py",
                ),
            },
            {
                "role": "hermetic-cid-close",
                "source_commit": "35992cba2261714a0030dff9d58a7a52c31f1d80",
                "source_parent": "fd2fb0b42e60ed6f9e03ccfef175b0cdd9ba9c2b",
                "source_tree": "d91e3cb65806c3dcd4068e10e5f5fe5362a60c6a",
                "replay_commit": "eb9944cf7c214403531f375f971756f5acc6766b",
                "replay_parent": "fd2fb0b42e60ed6f9e03ccfef175b0cdd9ba9c2b",
                "replay_tree": "d91e3cb65806c3dcd4068e10e5f5fe5362a60c6a",
                "integrated_commit": "3740b4bc2c31a945748bb9cd9861a37a54abd6aa",
                "integrated_parent": "8ef29834e9af7629a621d583ec43bf37f136b10e",
                "integrated_tree": "d1956cb171afbe24225ba1c65920c4192b4a8177",
                "source_patch_sha256": (
                    "sha256:7666046a183a681f098e75090331172277f1cef988068120bae00c44e6907c26"
                ),
                "replay_patch_sha256": (
                    "sha256:7666046a183a681f098e75090331172277f1cef988068120bae00c44e6907c26"
                ),
                "integrated_patch_sha256": (
                    "sha256:7666046a183a681f098e75090331172277f1cef988068120bae00c44e6907c26"
                ),
                "binary_full_index_patch_sha256": (
                    "sha256:7666046a183a681f098e75090331172277f1cef988068120bae00c44e6907c26"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
                    "ipfs_accelerate_py/llm_router.py",
                    "ipfs_accelerate_py/utils/cid_utils.py",
                    "test/api/test_agent_supervisor_control_plane.py",
                    "test/api/test_agent_supervisor_control_plane_capsule_identity.py",
                    "test/api/test_llm_router_agent_implementation_route.py",
                ),
            },
    ),
    "final_blobs": {
            "ipfs_accelerate_py/__init__.py": (
                "4b21d326f200dab5927c7be628438a182bf0f612"
            ),
            "ipfs_accelerate_py/agent_supervisor/__init__.py": (
                "346c809c0457f0d612d378672abbdb0324de1f47"
            ),
            "ipfs_accelerate_py/agent_supervisor/core/__init__.py": (
                "24322042c103d710c215d52e53a6947a836e7ad9"
            ),
            "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py": (
                "fdc9a1ef5f93814d38dec4692336b44b48623c70"
            ),
            "ipfs_accelerate_py/llm_router.py": (
                "c569d03e80368319d2f26f5acff89f31d683f4f8"
            ),
            "ipfs_accelerate_py/utils/__init__.py": (
                "4bb5af77be27aa8fb0b50618b0c05c57904bee49"
            ),
            "ipfs_accelerate_py/utils/cid_utils.py": (
                "5d520ac3ee4191b132117ef28bce9ac2b0af16e2"
            ),
        },
    "final_raw_sha256": {
            "ipfs_accelerate_py/__init__.py": (
                "sha256:0bb676dde293ed70132b5b5c7df5b3154639dc9ca36996967cf64da32cf41958"
            ),
            "ipfs_accelerate_py/agent_supervisor/__init__.py": (
                "sha256:df692cbf44bf2eee9fb6f113bc2220a3f93d386082c95dfae7097b6a3a2dd50d"
            ),
            "ipfs_accelerate_py/agent_supervisor/core/__init__.py": (
                "sha256:8f63664c0f36cffc31902ee60919f14280ef47c81d8c5a67fd18c13476a32f7a"
            ),
            "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py": (
                "sha256:e8a2b5beae30ac0a40dcc1095a9241cfd80597f55f319828e2c844aeee1aa1ce"
            ),
            "ipfs_accelerate_py/llm_router.py": (
                "sha256:0494c54c25d4df4144b47ed1382685a381dc8650264b8155f51be6eafe9c80b8"
            ),
            "ipfs_accelerate_py/utils/__init__.py": (
                "sha256:07c143316f3fb9d40d5ee1d0f6c584948ba49438b059bcdb6d868e0f81e3d3e3"
            ),
            "ipfs_accelerate_py/utils/cid_utils.py": (
                "sha256:e7f6e94532c19ae22781fef967715f28c6825ffe4721bf52a52b647dce55e139"
            ),
        },
    "member_paths": (
            "ipfs_accelerate_py/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/core/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
            "ipfs_accelerate_py/llm_router.py",
            "ipfs_accelerate_py/utils/__init__.py",
            "ipfs_accelerate_py/utils/cid_utils.py",
        ),
    "module_origins": {
            "ipfs_accelerate_py": {
                "member_path": "ipfs_accelerate_py/__init__.py",
                "origin": "capsule://sealed/ipfs_accelerate_py/__init__.py",
            },
            "ipfs_accelerate_py.agent_supervisor": {
                "member_path": "ipfs_accelerate_py/agent_supervisor/__init__.py",
                "origin": "capsule://sealed/ipfs_accelerate_py/agent_supervisor/__init__.py",
            },
            "ipfs_accelerate_py.agent_supervisor.core": {
                "member_path": "ipfs_accelerate_py/agent_supervisor/core/__init__.py",
                "origin": "capsule://sealed/ipfs_accelerate_py/agent_supervisor/core/__init__.py",
            },
            "ipfs_accelerate_py.agent_supervisor.core.multiformats_identity": {
                "member_path": "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
                "origin": "capsule://sealed/ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
            },
            "ipfs_accelerate_py.llm_router": {
                "member_path": "ipfs_accelerate_py/llm_router.py",
                "origin": "capsule://sealed/ipfs_accelerate_py/llm_router.py",
            },
            "ipfs_accelerate_py.utils": {
                "member_path": "ipfs_accelerate_py/utils/__init__.py",
                "origin": "capsule://sealed/ipfs_accelerate_py/utils/__init__.py",
            },
            "ipfs_accelerate_py.utils.cid_utils": {
                "member_path": "ipfs_accelerate_py/utils/cid_utils.py",
                "origin": "capsule://sealed/ipfs_accelerate_py/utils/cid_utils.py",
            },
        },
    "manifest_sha256": (
        "sha256:01c4622824b0fbc57b4abd0dc9a9839bb852ed2c4dd793a26acd9ee21ccd6f61"
    ),
    "capsule_sha256": (
        "sha256:821f5dd3d1634ffeae06f9916e29572898e685c4123114a45ad4cc68f4e4a5d6"
    ),
    "archive_sha256": (
        "sha256:acc3dd0ed4f754e206c7508626c6dfb2f69e57b732f427eb18f4994a11cd5023"
    ),
    "archive_root_sha256": (
        "sha256:c4e8926743b7b3c5f5b6f2283cde93051eccaa55b33d407742c9a760aa0c2f57"
    ),
    "sealed_descriptor_sha256": (
        "sha256:53f0c97055e168c81c8d8693baadb2c1f5467709e9de04b95087768ec595172c"
    ),
    "probe_command": _HERMETIC_HOSTILE_PROBE_ARGV,
    "suite_passed_count": 108,
    "suite_report_sha256": (
        "sha256:7d934821dcb386e22c19e0704ca0261cdbbf354557e6f02fad56298c2162b0dd"
    ),
}
_NATIVE_DEPENDENCY_ACCEPTANCE_FINAL_VALUES: Final = {
    "ready": True,
    "pending": None,
    "passed_count": 46,
    "report_sha256": (
        "sha256:3f87da558705cc3d9197e9489ffd4349adea0c6a849e99e29342864f7f53d7e8"
    ),
}
_DUCKDB_POLICY_ACCEPTANCE_FINAL_VALUES: Final = {
    "ready": True,
    "pending": None,
    "passed_count": 51,
    "report_sha256": (
        "sha256:2863198d96699a5767cae0568892c9e5f6bf0d73bf84d03878f430695defd96d"
    ),
}

_PRODUCT_GENERATION_FINAL_VALUES: Final = {
    "ASE3-023": {
        "ready": True,
        "pending": None,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "prompt-v3-product-generation@1"
        ),
        "generations": (
            {
                "role": "product-salvage",
                "source_commit": "b072068dcf8b954d6ec454d89a014a8a80b6d2ef",
                "source_parent": "c5da756756064869dedab4aff17a0d17d8549488",
                "source_tree": "8567257c0d101198b5b2e7fde47031c603ab5e09",
                "replay_commit": "de5d7dcb49503b4bcc648a0aeacbcd0598b6e787",
                "replay_parent": "c5da756756064869dedab4aff17a0d17d8549488",
                "replay_tree": "8567257c0d101198b5b2e7fde47031c603ab5e09",
                "integrated_commit": "dee1c4f09f01bf8131a6b675eccce68e7aacb34d",
                "integrated_parent": "8f613252c2ff1460e6f2b551a2a8600a2d3ee519",
                "integrated_tree": "cb0ad281d6e89bea8cce24abe8e6a0bfc3117610",
                "source_patch_sha256": (
                    "sha256:ba3000aab09784b9830eec492442da29168a03462e04a6659909e495490500ad"
                ),
                "replay_patch_sha256": (
                    "sha256:ba3000aab09784b9830eec492442da29168a03462e04a6659909e495490500ad"
                ),
                "integrated_patch_sha256": (
                    "sha256:ba3000aab09784b9830eec492442da29168a03462e04a6659909e495490500ad"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
                    "test/api/test_agent_supervisor_configured_board_scheduler.py",
                    "test/api/test_agent_supervisor_implementation_supervisor_runner.py",
                    "test/api/test_agent_supervisor_prompt_v3_parallelism.py",
                ),
            },
            {
                "role": "capsule-identity",
                "source_commit": "cb7df87b0b1b96915af01d07fe7b1d3e991dfdc9",
                "source_parent": "b072068dcf8b954d6ec454d89a014a8a80b6d2ef",
                "source_tree": "30e9ddf36737f7c0667ec60fb07e90130fd5b899",
                "replay_commit": "13fb5479f9d5b18d604e6fafe028a45dacef7c80",
                "replay_parent": "b072068dcf8b954d6ec454d89a014a8a80b6d2ef",
                "replay_tree": "30e9ddf36737f7c0667ec60fb07e90130fd5b899",
                "integrated_commit": "bd1cd48d4379b30f1584f8967921345216166f56",
                "integrated_parent": "dee1c4f09f01bf8131a6b675eccce68e7aacb34d",
                "integrated_tree": "e0d9a530d559a24a1f5785d2ed250d878571e242",
                "source_patch_sha256": (
                    "sha256:ef27f4a69ab3743e0b60eec43e2cae866f3d9cbe18b7b9307cbcb68b0807f892"
                ),
                "replay_patch_sha256": (
                    "sha256:ef27f4a69ab3743e0b60eec43e2cae866f3d9cbe18b7b9307cbcb68b0807f892"
                ),
                "integrated_patch_sha256": (
                    "sha256:ef27f4a69ab3743e0b60eec43e2cae866f3d9cbe18b7b9307cbcb68b0807f892"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
                    "test/api/test_agent_supervisor_configured_board_scheduler.py",
                    "test/api/test_agent_supervisor_prompt_v3_parallelism.py",
                ),
            },
            {
                "role": "recovery-barrier",
                "source_commit": "7abfc4ca768dd5086600bb30815256c79aaace74",
                "source_parent": "cb7df87b0b1b96915af01d07fe7b1d3e991dfdc9",
                "source_tree": "79ed78e765eada55762bae63bdfdf81264111f8a",
                "replay_commit": "6e93027b6f3b5d5488e3e438d1d6bd89286bed17",
                "replay_parent": "cb7df87b0b1b96915af01d07fe7b1d3e991dfdc9",
                "replay_tree": "79ed78e765eada55762bae63bdfdf81264111f8a",
                "integrated_commit": "a43b2ce74816ac9226f6319b92425d0b002b6be6",
                "integrated_parent": "bd1cd48d4379b30f1584f8967921345216166f56",
                "integrated_tree": "bbb94ffe87c3b582e40b1052ba5b9dc1ca8b4c40",
                "source_patch_sha256": (
                    "sha256:0dce5226c7e543997cf55c1115622cc150a8261ca68f1d69252600ffd8fc3bb3"
                ),
                "replay_patch_sha256": (
                    "sha256:0dce5226c7e543997cf55c1115622cc150a8261ca68f1d69252600ffd8fc3bb3"
                ),
                "integrated_patch_sha256": (
                    "sha256:0dce5226c7e543997cf55c1115622cc150a8261ca68f1d69252600ffd8fc3bb3"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
                    "test/api/test_agent_supervisor_configured_board_scheduler.py",
                    "test/api/test_agent_supervisor_implementation_supervisor_runner.py",
                    "test/api/test_agent_supervisor_prompt_v3_parallelism.py",
                ),
            },
        ),
    },
    "ASE3-027": {
        "ready": True,
        "pending": None,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "prompt-v3-product-generation@1"
        ),
        "generations": (
            {
                "role": "product",
                "source_commit": "aaf7d722a0c23f5a047b38708f6290631848e06b",
                "source_parent": "e6f8e4a7771907372fc93b0f35cfde30170c2b2a",
                "source_tree": "323f14dbcd9b15b09046cd3a481eb8588a6ede2a",
                "replay_commit": "33154678bbdaf8e4a845bdb84be8d4b9937cad0e",
                "replay_parent": "e6f8e4a7771907372fc93b0f35cfde30170c2b2a",
                "replay_tree": "323f14dbcd9b15b09046cd3a481eb8588a6ede2a",
                "integrated_commit": "6a0047436f9515281127c17913132a23cecfe56c",
                "integrated_parent": "0321fd148bf7c5dc6e91251d119dc25f853e546f",
                "integrated_tree": "ec240271c204fc8befcddef6b7ca2bcad124dc3b",
                "source_patch_sha256": (
                    "sha256:b2f8be2a8126e5302d1a02f627dc1def01c54899181ec5ade59cfa22c2649062"
                ),
                "replay_patch_sha256": (
                    "sha256:b2f8be2a8126e5302d1a02f627dc1def01c54899181ec5ade59cfa22c2649062"
                ),
                "integrated_patch_sha256": (
                    "sha256:b2f8be2a8126e5302d1a02f627dc1def01c54899181ec5ade59cfa22c2649062"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py",
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py",
                    "test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py",
                ),
            },
            {
                "role": "test-contract-correction",
                "source_commit": "bd93ae76c277ae8761cd2abe0df79685d0b1b8ef",
                "source_parent": "aaf7d722a0c23f5a047b38708f6290631848e06b",
                "source_tree": "043677ccb5216204b3142bb8e2b7f71d4ca74bd9",
                "replay_commit": "54969e2987430bb6cc52770de809d0c9540b614b",
                "replay_parent": "aaf7d722a0c23f5a047b38708f6290631848e06b",
                "replay_tree": "043677ccb5216204b3142bb8e2b7f71d4ca74bd9",
                "integrated_commit": "d32415e4308a8462e96b4d04f807338f0a2d8b53",
                "integrated_parent": "6a0047436f9515281127c17913132a23cecfe56c",
                "integrated_tree": "87191ce65498a637c7b9500d72d434cadb8efbef",
                "source_patch_sha256": (
                    "sha256:e64ab06bc28e13ae08591708f634a855bc752208a8bfbaf7164d704386f0d9fd"
                ),
                "replay_patch_sha256": (
                    "sha256:e64ab06bc28e13ae08591708f634a855bc752208a8bfbaf7164d704386f0d9fd"
                ),
                "integrated_patch_sha256": (
                    "sha256:e64ab06bc28e13ae08591708f634a855bc752208a8bfbaf7164d704386f0d9fd"
                ),
                "changed_paths": (
                    "test/api/test_agent_supervisor_inference_runtime.py",
                    "test/api/test_agent_supervisor_prompt_v3_resolution.py",
                ),
            },
        ),
    },
    "ASE3-030": {
        "ready": True,
        "pending": None,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "prompt-v3-product-generation@1"
        ),
        "generations": (
            {
                "role": "hermetic-cid-seal",
                "source_commit": "fd2fb0b42e60ed6f9e03ccfef175b0cdd9ba9c2b",
                "source_parent": "c5da756756064869dedab4aff17a0d17d8549488",
                "source_tree": "22101ffb8eb2568d5f3c457e9664bba014e1e8ee",
                "replay_commit": "f97cad9607f16e71c7b1383e55b624f5149def71",
                "replay_parent": "c5da756756064869dedab4aff17a0d17d8549488",
                "replay_tree": "22101ffb8eb2568d5f3c457e9664bba014e1e8ee",
                "integrated_commit": "8ef29834e9af7629a621d583ec43bf37f136b10e",
                "integrated_parent": "32face22bc17eb0b76f09fed2186ef799075b110",
                "integrated_tree": "46be4b75f0a5ab9be06cf3b85a75691157fba09e",
                "source_patch_sha256": (
                    "sha256:863e754bc88c6743b5313548724bbbe5b741263bff1ff143dc5b38aa4f898d16"
                ),
                "replay_patch_sha256": (
                    "sha256:863e754bc88c6743b5313548724bbbe5b741263bff1ff143dc5b38aa4f898d16"
                ),
                "integrated_patch_sha256": (
                    "sha256:863e754bc88c6743b5313548724bbbe5b741263bff1ff143dc5b38aa4f898d16"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
                    "ipfs_accelerate_py/llm_router.py",
                    "ipfs_accelerate_py/utils/cid_utils.py",
                    "test/api/test_agent_supervisor_hermetic_cid_capsule.py",
                ),
            },
            {
                "role": "hermetic-cid-close",
                "source_commit": "35992cba2261714a0030dff9d58a7a52c31f1d80",
                "source_parent": "fd2fb0b42e60ed6f9e03ccfef175b0cdd9ba9c2b",
                "source_tree": "d91e3cb65806c3dcd4068e10e5f5fe5362a60c6a",
                "replay_commit": "eb9944cf7c214403531f375f971756f5acc6766b",
                "replay_parent": "fd2fb0b42e60ed6f9e03ccfef175b0cdd9ba9c2b",
                "replay_tree": "d91e3cb65806c3dcd4068e10e5f5fe5362a60c6a",
                "integrated_commit": "3740b4bc2c31a945748bb9cd9861a37a54abd6aa",
                "integrated_parent": "8ef29834e9af7629a621d583ec43bf37f136b10e",
                "integrated_tree": "d1956cb171afbe24225ba1c65920c4192b4a8177",
                "source_patch_sha256": (
                    "sha256:7666046a183a681f098e75090331172277f1cef988068120bae00c44e6907c26"
                ),
                "replay_patch_sha256": (
                    "sha256:7666046a183a681f098e75090331172277f1cef988068120bae00c44e6907c26"
                ),
                "integrated_patch_sha256": (
                    "sha256:7666046a183a681f098e75090331172277f1cef988068120bae00c44e6907c26"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
                    "ipfs_accelerate_py/llm_router.py",
                    "ipfs_accelerate_py/utils/cid_utils.py",
                    "test/api/test_agent_supervisor_control_plane.py",
                    "test/api/test_agent_supervisor_control_plane_capsule_identity.py",
                    "test/api/test_llm_router_agent_implementation_route.py",
                ),
            },
        ),
    },
    "ASE3-031": {
        "ready": True,
        "pending": None,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "prompt-v3-product-generation@1"
        ),
        "generations": (
            {
                "role": "native-duckdb-dependency",
                "source_commit": "25fedf091dad928dad1f83c9f81a54c2d401eabe",
                "source_parent": "35992cba2261714a0030dff9d58a7a52c31f1d80",
                "source_tree": "da9e18b507b9991935823dc10d4d7208a47f47f2",
                "replay_commit": "c9333d7934c5aa19f1888d6d41aa863d1dfc6b85",
                "replay_parent": "35992cba2261714a0030dff9d58a7a52c31f1d80",
                "replay_tree": "da9e18b507b9991935823dc10d4d7208a47f47f2",
                "integrated_commit": "1a419e525699e74254a450c0137264d8dd60ea00",
                "integrated_parent": "3740b4bc2c31a945748bb9cd9861a37a54abd6aa",
                "integrated_tree": "06ea3c5444bd190996ad591b5fc82cad2443dcc4",
                "source_patch_sha256": (
                    "sha256:3daaf796199701b25e7e231f0bbd3ce5e0c1b487912e699d7f11473d02a784a2"
                ),
                "replay_patch_sha256": (
                    "sha256:3daaf796199701b25e7e231f0bbd3ce5e0c1b487912e699d7f11473d02a784a2"
                ),
                "integrated_patch_sha256": (
                    "sha256:3daaf796199701b25e7e231f0bbd3ce5e0c1b487912e699d7f11473d02a784a2"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/llm_router.py",
                    "test/api/test_agent_supervisor_native_dependency_pin.py",
                ),
            },
        ),
    },
    "ASE3-032": {
        "ready": True,
        "pending": None,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "prompt-v3-product-generation@1"
        ),
        "generations": (
            {
                "role": "duckdb-connection-policy",
                "source_commit": "9f1a3cb3c583924878293f9acd676a211106c2e7",
                "source_parent": "25fedf091dad928dad1f83c9f81a54c2d401eabe",
                "source_tree": "853191d0e00471bf41452801ae83b0a13b3607d5",
                "replay_commit": "d57a5b47f1ce107b3357839bfd1e0b7120048785",
                "replay_parent": "25fedf091dad928dad1f83c9f81a54c2d401eabe",
                "replay_tree": "853191d0e00471bf41452801ae83b0a13b3607d5",
                "integrated_commit": "8f613252c2ff1460e6f2b551a2a8600a2d3ee519",
                "integrated_parent": "1a419e525699e74254a450c0137264d8dd60ea00",
                "integrated_tree": "f7160446af32a9affeef554d319c443b6449271a",
                "source_patch_sha256": (
                    "sha256:b93ffbe2a20ffbc62287e1f21f291a34a2ea84d5cecde5eee4f07677c128b0fb"
                ),
                "replay_patch_sha256": (
                    "sha256:b93ffbe2a20ffbc62287e1f21f291a34a2ea84d5cecde5eee4f07677c128b0fb"
                ),
                "integrated_patch_sha256": (
                    "sha256:b93ffbe2a20ffbc62287e1f21f291a34a2ea84d5cecde5eee4f07677c128b0fb"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
                    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
                    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py",
                    "test/api/test_agent_supervisor_duckdb_connection_policy.py",
                ),
            },
        ),
    },
    "ASE3-019": {
        "ready": True,
        "pending": None,
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "prompt-v3-product-generation@1"
        ),
        "generations": (
            {
                "role": "provider-fallback",
                "source_commit": "f49bf853f7ceae64ba3e2379db3d709e50077734",
                "source_parent": "e6f8e4a7771907372fc93b0f35cfde30170c2b2a",
                "source_tree": "2f1e39f99df98b0bdfd707f9cd7de5f6ca4e871f",
                "replay_commit": "d07eaab3690242df371b8cc2f0641f156ce6a166",
                "replay_parent": "e6f8e4a7771907372fc93b0f35cfde30170c2b2a",
                "replay_tree": "2f1e39f99df98b0bdfd707f9cd7de5f6ca4e871f",
                "integrated_commit": "1bd07b7261c86e4cf8301b34dbaf9728fc6e7818",
                "integrated_parent": "20ea872b958c44e1af9b07312594809b0986d535",
                "integrated_tree": "e20cf0acc9fe5d17ac439b9c9ddff6332810e583",
                "source_patch_sha256": (
                    "sha256:fc0b76720a68faf82e36bdddc1c612769dd7e268042bc20fcab2744856a55180"
                ),
                "replay_patch_sha256": (
                    "sha256:fc0b76720a68faf82e36bdddc1c612769dd7e268042bc20fcab2744856a55180"
                ),
                "integrated_patch_sha256": (
                    "sha256:fc0b76720a68faf82e36bdddc1c612769dd7e268042bc20fcab2744856a55180"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
                    "ipfs_accelerate_py/llm_router.py",
                    "test/api/test_agent_supervisor_prompt_v3_authority_hardening.py",
                    "test/api/test_llm_router_agent_supervisor_fallback_route.py",
                ),
            },
            {
                "role": "attempt-2-product",
                "source_commit": "eb68ff2a20e0719388f60ffef1f5bfcb90b79263",
                "source_parent": "0c40afb32f9b95ca54d73b18e06a4a2c193469f7",
                "source_tree": "695e2d6f07bc1c48bdc34ebb490342444de2cbef",
                "replay_commit": "798847e8f527c21452888e3ddc1d1daafdebfe27",
                "replay_parent": "0c40afb32f9b95ca54d73b18e06a4a2c193469f7",
                "replay_tree": "695e2d6f07bc1c48bdc34ebb490342444de2cbef",
                "integrated_commit": "10db367e5ba2d2ece22058ed88d658587867ea28",
                "integrated_parent": "0c40afb32f9b95ca54d73b18e06a4a2c193469f7",
                "integrated_tree": "695e2d6f07bc1c48bdc34ebb490342444de2cbef",
                "source_patch_sha256": (
                    "sha256:0dca974830907318ccc8b056e2fd190773b608082b91458bdce9b9393c904403"
                ),
                "replay_patch_sha256": (
                    "sha256:0dca974830907318ccc8b056e2fd190773b608082b91458bdce9b9393c904403"
                ),
                "integrated_patch_sha256": (
                    "sha256:0dca974830907318ccc8b056e2fd190773b608082b91458bdce9b9393c904403"
                ),
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
                    "ipfs_accelerate_py/llm_router.py",
                    "test/api/test_agent_supervisor_prompt_v3_authority_hardening.py",
                    "test/api/test_llm_router_agent_supervisor_fallback_route.py",
                ),
            },
        ),
    },
}

_RELOAD_FINAL_VALUES: Final = {
    "ready": False,
    "pending": _FINAL_VALUE_PENDING_RELOAD,
    "stopped_generation_id": _FINAL_VALUE_PENDING_RELOAD,
    "stopped_generation_number": -1,
    "target_generation_id": _FINAL_VALUE_PENDING_RELOAD,
    "scheduler_blob": _FINAL_VALUE_PENDING_RELOAD,
    "scheduler_raw_sha256": _FINAL_VALUE_PENDING_RELOAD,
    "daemon_blob": _FINAL_VALUE_PENDING_RELOAD,
    "daemon_raw_sha256": _FINAL_VALUE_PENDING_RELOAD,
}
_RELOAD_TASK_CONTRACT: Final = {
    "task_id": "ASE3-022",
    "canonical_task_cid": (
        "baguqeeradovptvx4kagyourgywrwno4kehcajrsyzduqzu7sr2bcd6vtn3na"
    ),
    "blocked_contract_sha256": _PROVIDER_ATTEMPT_RELOAD_GATE_C1_CONTRACT_SHA256,
    "completed_contract_sha256": (
        "sha256:21535aad10d951add2781d705e996c502a579466ba3bc7674f27a5f95e68b16d"
    ),
    "status_before": "blocked",
    "status_after": "completed",
}
_RELOAD_SCHEDULER_PATH: Final = (
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py"
)
_RELOAD_DAEMON_PATH: Final = (
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"
)
_ASE3_019_ACCEPTED_CONTROL_PLANE: Final = {
    "schema": (
        "ipfs_accelerate_py.agent_supervisor."
        "operator-accepted-control-plane-contract@1"
    ),
    "canonical_route_owner": "ipfs_accelerate_py.llm_router",
    "route_id": _PROVIDER_FALLBACK_AUTHORIZATION_ROUTE["route_id"],
    "primary_provider_id": "grok_cli",
    "primary_model_id": "grok-4.6",
    "fallback_provider_id": "codex",
    "fallback_model_id": "gpt-5.6-terra",
    "fallback_reasoning_effort": "high",
    "allowed_trigger_classes": [
        "grok_authentication_unavailable",
        "grok_hard_quota_exhausted",
    ],
    "public_api": {
        "route_plan_type": "AgentImplementationRoutePlan",
        "fallback_decision_type": "AgentImplementationFallbackDecision",
        "capacity_projection_api": (
            "project_agent_implementation_route_capacity"
        ),
        "control_plane_pin_type": "AgentImplementationControlPlanePin",
        "sealed_control_plane_type": (
            "AgentImplementationSealedControlPlane"
        ),
        "source_generation_api": (
            "agent_implementation_control_plane_source_generation"
        ),
        "materialize_api": (
            "materialize_agent_implementation_control_plane_capsule"
        ),
        "build_pin_api": "build_agent_implementation_control_plane_pin",
        "seal_api": "seal_agent_implementation_control_plane_capsule",
        "verify_sealed_api": (
            "verify_agent_implementation_sealed_control_plane"
        ),
        "pin_schema": (
            "ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2"
        ),
        "manifest_schema": (
            "ipfs_accelerate_py.agent_supervisor.materialized-control-plane@1"
        ),
        "terminal_outcome_field": "accepted_control_plane",
    },
    "portable_acceptance_evidence": {
        "source_head_required": True,
        "source_tree_required": True,
        "package_module_blob_manifest_required": True,
        "sealed_control_plane_digest_required": True,
        "isolated_argv_origin_proof_required": True,
        "candidate_workspace_identity_required": True,
        "shadow_regression_receipt_required": True,
        "machine_local_capsule_path_forbidden": True,
        "machine_local_memfd_path_forbidden": True,
    },
    "immutable_generation_capsule_required": True,
    "current_profile_rechecked_at_effect": True,
    "native_signed_hard_quota_required": True,
    "durable_cas_states": ["reserved", "effect_started", "terminal"],
    "crash_adopts_winning_effect_receipt": True,
    "same_logical_attempt": True,
    "docker_runtime": "runc",
    "docker_image_id": _PROVIDER_FALLBACK_DOCKER_BOUNDARY["image_id"],
    "attempt_counter_mutation_authorized": False,
    "provider_capacity_attempt_restoration_allowed": False,
}
_ACCEPTANCE_MANIFEST_REQUIRED_FIELDS: Final = (
    "phase",
    "parent_phase",
    "parent_head",
    "parent_tree",
    "parent_manifest_sha256",
    "artifacts",
    "task_statuses",
    "reload_gate_status",
    "pre_launch_authorization_only",
    "runtime_effect_claimed",
)
_RELOAD_MANIFEST_REQUIRED_FIELDS: Final = (
    "phase",
    "acceptance_head",
    "acceptance_tree",
    "receipt",
    "task",
    "accepted_task_statuses",
    "reload_gate_completed",
    "launch_authorization_only",
    "post_launch_birth_receipt_required",
    "post_launch_birth_receipt_schema",
)
_CONVERGENCE_MANIFEST_V1_TOP_LEVEL_FIELDS: Final = (
    "schema",
    "board_namespace",
    "task_id",
    "goal_id",
    "created_at",
    "integration_seed_commit",
    "integration_seed_tree",
    "historical_completion_authority",
    "rescue_bulk_merge_allowed",
    "components",
    "population",
    "completion_rules",
    "downstream_rules",
)
_CONVERGENCE_MANIFEST_V2_TOP_LEVEL_FIELDS: Final = (
    *_CONVERGENCE_MANIFEST_V1_TOP_LEVEL_FIELDS,
    "acceptance",
)
_CONVERGENCE_MANIFEST_V3_TOP_LEVEL_FIELDS: Final = (
    *_CONVERGENCE_MANIFEST_V2_TOP_LEVEL_FIELDS,
    "reload",
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
        int(status.st_mode),
        int(status.st_nlink),
        int(status.st_uid),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _directory_snapshot(status: os.stat_result) -> tuple[int, ...]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_mode),
        int(status.st_nlink),
        int(status.st_uid),
    )


def _lexical_absolute_path(path: Path) -> Path:
    if ".." in path.parts:
        raise ValueError(f"{path.name}: parent traversal is forbidden")
    return Path(os.path.abspath(os.fspath(path)))


def _open_nofollow_parent(path: Path) -> tuple[int, tuple[tuple[int, ...], ...]]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ValueError(f"{path.name}: no-follow directory reads are unavailable")
    flags = (
        os.O_RDONLY
        | directory_flag
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    descriptors: list[int] = []
    try:
        parent_descriptor = os.open(path.anchor, flags)
        descriptors.append(parent_descriptor)
        identities = [_directory_snapshot(os.fstat(parent_descriptor))]
        for component in path.parts[1:-1]:
            child_descriptor = os.open(
                component,
                flags,
                dir_fd=parent_descriptor,
            )
            descriptors.append(child_descriptor)
            parent_descriptor = child_descriptor
            identities.append(_directory_snapshot(os.fstat(parent_descriptor)))
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ValueError(f"{path.name}: path contains a symlink or non-directory") from exc
    for descriptor in descriptors[:-1]:
        os.close(descriptor)
    return descriptors[-1], tuple(identities)


@dataclass(frozen=True)
class _RegularFileSnapshot:
    raw: bytes
    path: Path
    uid: int
    mode: int


def _read_regular_snapshot(
    path: Path,
    *,
    maximum_bytes: int = MAX_EVIDENCE_SNAPSHOT_BYTES,
) -> _RegularFileSnapshot:
    """Read one bounded, single-link, stable evidence-file snapshot."""

    if maximum_bytes < 0:
        raise ValueError(f"{path.name}: invalid evidence snapshot byte bound")
    lexical = _lexical_absolute_path(path)
    initial = lexical.lstat()
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"{path.name}: expected a regular nonsymlink file")
    if initial.st_nlink != 1:
        raise ValueError(f"{path.name}: expected a single-link evidence file")
    if initial.st_size > maximum_bytes:
        raise ValueError(
            f"{path.name}: exceeds {maximum_bytes}-byte evidence snapshot bound"
        )

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError(f"{path.name}: no-follow evidence reads are unavailable")
    parent_descriptor, parent_identities = _open_nofollow_parent(lexical)
    try:
        descriptor = os.open(
            lexical.name,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | nofollow,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise ValueError(f"{path.name}: expected a regular nonsymlink file") from exc
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
        final_path = os.stat(
            lexical.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        final_lexical_path = lexical.lstat()
        final_parent_descriptor, final_parent_identities = _open_nofollow_parent(
            lexical
        )
        os.close(final_parent_descriptor)
        payload = b"".join(chunks)
        if (
            len(payload) != opened.st_size
            or _file_snapshot(final_descriptor) != _file_snapshot(opened)
            or _file_snapshot(final_path) != _file_snapshot(opened)
            or _file_snapshot(final_lexical_path) != _file_snapshot(opened)
            or final_parent_identities != parent_identities
        ):
            raise ValueError(
                f"{path.name}: evidence file changed during bounded read"
            )
        return _RegularFileSnapshot(
            raw=payload,
            path=lexical,
            uid=int(opened.st_uid),
            mode=stat.S_IMODE(opened.st_mode),
        )
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int = MAX_EVIDENCE_SNAPSHOT_BYTES,
) -> bytes:
    return _read_regular_snapshot(path, maximum_bytes=maximum_bytes).raw


def _require_authority_file_snapshot(
    snapshot: _RegularFileSnapshot,
    *,
    repository_root: Path | None = None,
    expected_relative_path: str | None = None,
) -> None:
    if snapshot.uid not in {0, os.geteuid()}:
        raise ValueError(f"{snapshot.path.name}: authority file owner mismatch")
    if snapshot.mode & 0o022:
        raise ValueError(
            f"{snapshot.path.name}: authority file is group-or-other writable"
        )
    if (repository_root is None) != (expected_relative_path is None):
        raise ValueError(
            f"{snapshot.path.name}: repository authority path is incomplete"
        )
    if repository_root is not None and expected_relative_path is not None:
        expected = _lexical_absolute_path(repository_root) / Path(
            expected_relative_path
        )
        if snapshot.path != expected:
            raise ValueError(
                f"{snapshot.path.name}: authority file must use its lexical "
                "repository path"
            )


def _load_json_bytes(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: expected UTF-8 JSON") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{name}: non-finite JSON constant {value!r}")
            ),
        )
    except RecursionError as exc:
        raise ValueError(f"{name}: JSON nesting exceeds parser bound") from exc
    if not isinstance(payload, Mapping):
        # The document is JSON but its value violates this object's schema.
        raise ValueError(f"{name}: root must be a JSON object")  # noqa: TRY004
    return payload


def _load_json(path: Path) -> Mapping[str, Any]:
    return _load_json_bytes(_read_regular_bytes(path), name=path.name)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Match the canonical JSON projection used by the ASE3-019 lifecycle API."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _utc_timestamp_to_ms(value: Any) -> int | None:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(_read_regular_bytes(path)).hexdigest()


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _validate_hermetic_probe_argv(
    errors: list[str],
    *,
    prefix: str,
    value: Any,
) -> None:
    if not isinstance(value, list) or not value or any(
        not isinstance(argument, str) or not argument for argument in value
    ):
        errors.append(f"{prefix}: expected nonempty string argv")
        return
    if tuple(value) != _HERMETIC_HOSTILE_PROBE_ARGV:
        errors.append(f"{prefix}: exact reviewed hostile-probe argv required")
    if value[0] != "python" or "/" in value[0] or "\\" in value[0]:
        errors.append(f"{prefix}[0]: fixed portable interpreter token required")
    if len(value) < 2 or value[1] != "-I" or value.count("-I") != 1:
        errors.append(f"{prefix}: exactly one positional -I is required")
    machine_local_markers = (
        "file://",
        "/dev/fd/",
        "/home/",
        "/proc/",
        "/private/",
        "/tmp/",
        "/Users/",
        "capsule_root",
    )
    for index, argument in enumerate(value):
        if (
            argument.startswith(("/", "~", "\\"))
            or re.search(r"(?:^|[\"'])(?:[A-Za-z]:[\\/]|/)", argument)
            is not None
            or any(marker in argument for marker in machine_local_markers)
        ):
            errors.append(
                f"{prefix}[{index}]: absolute or machine-local argv value forbidden"
            )


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


def _git_bytes(
    repo_root: Path,
    *args: str,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )


def _git_replacement_object_errors(repo_root: Path, *, prefix: str) -> list[str]:
    """Fail closed when local replacement refs could rewrite audited history."""

    replacements = _git(
        repo_root,
        "--no-replace-objects",
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    )
    if replacements.returncode != 0:
        return [f"{prefix}: unable to prove replacement-object absence"]
    if replacements.stdout.splitlines():
        return [f"{prefix}: Git replacement objects are forbidden"]
    return []


_DETERMINISTIC_GIT_DIFF_FLAGS: Final = (
    "--no-ext-diff",
    "--no-textconv",
    "--no-renames",
    "--no-color",
    "--diff-algorithm=myers",
    "--indent-heuristic",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "--unified=3",
)


def _git_diff_names(
    repo_root: Path,
    parent: str,
    child: str,
) -> subprocess.CompletedProcess[str]:
    return _git(
        repo_root,
        "diff",
        *_DETERMINISTIC_GIT_DIFF_FLAGS,
        "--name-only",
        parent,
        child,
    )


def _git_diff_patch(
    repo_root: Path,
    parent: str,
    child: str,
) -> subprocess.CompletedProcess[bytes]:
    return _git_bytes(
        repo_root,
        "diff",
        *_DETERMINISTIC_GIT_DIFF_FLAGS,
        "--binary",
        "--full-index",
        parent,
        child,
    )


def _is_unpopulated_final_value(value: Any) -> bool:
    """True when a sealed final still carries a FILL_AFTER / pending sentinel."""

    if value is None:
        return True
    if type(value) is int and value < 0:
        return True
    if isinstance(value, str) and "FILL_AFTER" in value:
        return True
    return False


def _phase_artifact_path_set() -> frozenset[str]:
    """Paths whose edits are owned by protected phase commits only."""

    paths: set[str] = {
        PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
        _CONVERGENCE_MANIFEST_RELATIVE_PATH,
        PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
    }
    for phase_paths in SEQUENTIAL_PHASE_CHANGED_PATHS.values():
        paths.update(phase_paths)
    paths.update(SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE)
    return frozenset(paths)


def _is_phase_neutral_changed_paths(paths: Sequence[str]) -> bool:
    """True when a commit only touches non-phase code (freezes/composition)."""

    if not paths:
        return True
    artifacts = _phase_artifact_path_set()
    return all(path not in artifacts for path in paths)


def _first_parent_of(repo_root: Path, commit: str) -> str | None:
    lineage = _git(repo_root, "rev-list", "--parents", "-n", "1", commit)
    if lineage.returncode != 0:
        return None
    parts = lineage.stdout.strip().split()
    if len(parts) < 2 or parts[0] != commit or _HEX40.fullmatch(parts[1]) is None:
        return None
    return parts[1]


def _validate_exact_direct_child(
    *,
    repo_root: Path,
    parent: str,
    child: str,
    expected_paths: Sequence[str],
    prefix: str,
) -> list[str]:
    """Require the child phase delta; allow code-only freeze intermediates.

    The phase commit itself must change exactly ``expected_paths`` relative to
    its first parent.  Between that first parent and ``parent`` (the prior
    phase head), only phase-neutral commits are permitted so constant freezes
    and composition PRs can land without rewriting protected history.
    """

    errors: list[str] = []
    expected = tuple(sorted(expected_paths))
    current = child
    saw_phase_delta = False
    for _ in range(64):
        if current == parent:
            if not saw_phase_delta:
                errors.append(f"{prefix}.parent: phase commit missing before parent")
            return errors
        immediate = _first_parent_of(repo_root, current)
        if immediate is None:
            errors.append(f"{prefix}.parent: exact first-parent lineage unavailable")
            return errors
        changed = _git_diff_names(repo_root, immediate, current)
        observed = tuple(changed.stdout.splitlines())
        if changed.returncode != 0:
            errors.append(f"{prefix}.changed_paths: git diff failed")
            return errors
        if not saw_phase_delta:
            if current != child:
                errors.append(f"{prefix}.parent: phase commit must be the child tip")
            if observed != expected:
                errors.append(
                    f"{prefix}.changed_paths: expected exact deterministic population "
                    + ",".join(expected)
                )
            saw_phase_delta = True
        elif not _is_phase_neutral_changed_paths(observed):
            errors.append(
                f"{prefix}.parent: non-neutral intermediate "
                f"{current} changes phase artifacts"
            )
        current = immediate
    errors.append(f"{prefix}.parent: prior phase head not reached")
    return errors


def _discover_sequential_phase_heads(
    *,
    repo_root: Path,
    head: str,
    through_phase: str,
) -> tuple[dict[str, str], list[str]]:
    """Map Q..through_phase heads, skipping phase-neutral freeze commits."""

    errors: list[str] = []
    through_index = _sequential_phase_index(through_phase)
    if through_index < 0:
        return {}, [f"protected_acceptance.discovery.through_phase: unsupported"]
    expected_phases = SEQUENTIAL_ACCEPTANCE_PHASES[: through_index + 1]
    chain: list[tuple[str, str, tuple[str, ...]]] = []
    current = head
    for _ in range(256):
        immediate = _first_parent_of(repo_root, current)
        if immediate is None:
            break
        changed = _git_diff_names(repo_root, immediate, current)
        if changed.returncode != 0:
            return {}, ["protected_acceptance.discovery: git diff failed"]
        paths = tuple(changed.stdout.splitlines())
        chain.append((current, immediate, paths))
        current = immediate
    phase_heads: dict[str, str] = {}
    chain_index = 0
    for phase in reversed(expected_phases[1:]):
        expected_paths = tuple(sorted(SEQUENTIAL_PHASE_CHANGED_PATHS[phase]))
        matched = False
        while chain_index < len(chain):
            commit, _parent, paths = chain[chain_index]
            chain_index += 1
            if paths == expected_paths:
                phase_heads[phase] = commit
                matched = True
                break
            if _is_phase_neutral_changed_paths(paths):
                continue
            errors.append(
                "protected_acceptance.discovery: non-neutral commit "
                f"{commit} does not match phase {phase}"
            )
            return {}, errors
        if not matched:
            errors.append(
                f"protected_acceptance.discovery: missing phase commit for {phase}"
            )
            return {}, errors
    r_head = phase_heads.get("R")
    if not isinstance(r_head, str):
        errors.append("protected_acceptance.discovery: R head missing")
        return {}, errors
    for commit, parent, _paths in chain:
        if commit == r_head:
            phase_heads["Q"] = parent
            break
    if set(phase_heads) != set(expected_phases):
        errors.append(
            "protected_acceptance.discovery: exact contiguous phase population required"
        )
        return {}, errors
    return phase_heads, errors


def _validate_git_regular_modes(
    *, repo_root: Path, head: str, paths: Sequence[str], prefix: str
) -> list[str]:
    errors: list[str] = []
    for relative_path in paths:
        entry = _git(repo_root, "ls-tree", head, "--", relative_path)
        fields = entry.stdout.rstrip("\n").split(maxsplit=3)
        if (
            entry.returncode != 0
            or len(fields) != 4
            or fields[0] != "100644"
            or fields[1] != "blob"
            or fields[3] != relative_path
        ):
            errors.append(f"{prefix}.{relative_path}: exact 100644 Git mode required")
    return errors


def _protected_paths_for_phase(phase: str) -> frozenset[str]:
    future = {
        *SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE,
        PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH,
    }
    present = set(_PROTECTED_PATHS) - future
    if _sequential_phase_index(phase) >= 0:
        present.update(_sequential_artifacts_after(phase))
    return frozenset(present)


def _validate_protected_file_authority(
    *,
    repo_root: Path,
    phase: str,
    head: str = "HEAD",
) -> list[str]:
    """Require exact protected population and filesystem/Git file authority."""

    errors: list[str] = []
    prefix = "protected_authority"
    config_path = repo_root / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    try:
        config_snapshot = _read_regular_snapshot(config_path)
        _require_authority_file_snapshot(
            config_snapshot,
            repository_root=repo_root,
            expected_relative_path=PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH.as_posix(),
        )
        config_payload = _load_json_bytes(
            config_snapshot.raw,
            name=PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH.name,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{prefix}.scheduler_config: {exc}")
    else:
        protected_paths = config_payload.get("protected_paths")
        if protected_paths != list(_PROTECTED_PATHS):
            errors.append(f"{prefix}.scheduler_config.protected_paths: exact order required")

    expected_present = _protected_paths_for_phase(phase)
    for relative_path in _PROTECTED_PATHS:
        path = repo_root / relative_path
        should_exist = relative_path in expected_present
        try:
            snapshot = _read_regular_snapshot(path)
        except FileNotFoundError:
            if should_exist:
                errors.append(f"{prefix}.{relative_path}: required protected file absent")
            continue
        except (OSError, ValueError) as exc:
            errors.append(f"{prefix}.{relative_path}: {exc}")
            continue
        if not should_exist:
            errors.append(f"{prefix}.{relative_path}: protected future file present early")
            continue
        try:
            _require_authority_file_snapshot(
                snapshot,
                repository_root=repo_root,
                expected_relative_path=relative_path,
            )
        except ValueError as exc:
            errors.append(f"{prefix}.{relative_path}: {exc}")

        tree_entry = _git(repo_root, "ls-tree", head, "--", relative_path)
        fields = tree_entry.stdout.rstrip("\n").split(maxsplit=3)
        if (
            tree_entry.returncode != 0
            or len(fields) != 4
            or fields[1] != "blob"
            or fields[3] != relative_path
            or fields[0] not in {"100644", "100755"}
        ):
            errors.append(f"{prefix}.{relative_path}: exact regular Git mode required")
            continue
        executable = bool(snapshot.mode & 0o111)
        if executable is not (fields[0] == "100755"):
            errors.append(f"{prefix}.{relative_path}: filesystem/Git mode mismatch")
    return errors


def _receipt_forbidden_binding_paths(payload: Any) -> tuple[str, ...]:
    """Find circular/current-machine authority fields anywhere in a receipt."""

    forbidden: list[str] = []
    circular_keys = {
        "acceptance_head",
        "reload_head",
        "convergence_manifest_sha256",
        "convergence_manifest_digest",
    }
    machine_local_keys = {
        "capsule_root",
        "runner_path",
        "descriptor_path",
        "executable_path",
        "memfd_path",
    }

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key)
                child = f"{path}.{key}" if path else key
                if key in circular_keys or key in machine_local_keys:
                    forbidden.append(child)
                walk(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload, "")
    return tuple(forbidden)


def _require_exact_keys(
    errors: list[str],
    *,
    prefix: str,
    value: Any,
    expected: Sequence[str],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{prefix}: expected object")
        return None
    if set(value) != set(expected):
        errors.append(f"{prefix}: exact key population required")
    return value


def _require_bounded_string(
    errors: list[str],
    *,
    prefix: str,
    value: Any,
    maximum: int = 4096,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value)
        or len(value.encode("utf-8")) > maximum
    ):
        qualifier = "possibly-empty" if allow_empty else "nonempty"
        errors.append(
            f"{prefix}: expected {qualifier} UTF-8 string bounded to {maximum} bytes"
        )
        return ""
    return value


def _require_exact_integer(
    errors: list[str],
    *,
    prefix: str,
    value: Any,
    minimum: int = 0,
    maximum: int = 1_000_000,
) -> int | None:
    if type(value) is not int or not minimum <= value <= maximum:
        errors.append(
            f"{prefix}: expected integer in inclusive range {minimum}..{maximum}"
        )
        return None
    return value


def _require_positive_finite_number(
    errors: list[str],
    *,
    prefix: str,
    value: Any,
) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        errors.append(f"{prefix}: expected positive finite JSON number")
        return None
    return float(value)


def _require_trimmed_string(
    errors: list[str],
    *,
    prefix: str,
    value: Any,
    maximum: int = 4096,
    allow_empty: bool = False,
) -> str:
    text = _require_bounded_string(
        errors,
        prefix=prefix,
        value=value,
        maximum=maximum,
        allow_empty=allow_empty,
    )
    if text and (text != text.strip() or any(char in text for char in "\x00\r\n")):
        errors.append(f"{prefix}: expected trimmed single-line text")
        return ""
    return text


def _require_sorted_unique_string_array(
    errors: list[str],
    *,
    prefix: str,
    value: Any,
    maximum_items: int = 128,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not 0 < len(value) <= maximum_items:
        errors.append(
            f"{prefix}: expected nonempty array bounded to {maximum_items} items"
        )
        return ()
    observed = tuple(
        _require_trimmed_string(
            errors,
            prefix=f"{prefix}[{index}]",
            value=item,
        )
        for index, item in enumerate(value)
    )
    if observed != tuple(sorted(set(observed))):
        errors.append(f"{prefix}: expected sorted unique string population")
    return observed


def _require_exact_string_array(
    errors: list[str],
    *,
    prefix: str,
    value: Any,
    maximum_items: int,
    safe_paths: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not 0 < len(value) <= maximum_items:
        errors.append(
            f"{prefix}: expected nonempty array bounded to {maximum_items} items"
        )
        return ()
    observed: list[str] = []
    for index, item in enumerate(value):
        text = _require_bounded_string(
            errors,
            prefix=f"{prefix}[{index}]",
            value=item,
        )
        if safe_paths and text and not _is_safe_relative_path(text):
            errors.append(f"{prefix}[{index}]: unsafe relative path")
        observed.append(text)
    if len(set(observed)) != len(observed):
        errors.append(f"{prefix}: duplicate entries forbidden")
    return tuple(observed)


def _base58btc_decode(value: str) -> bytes:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    indexes = {character: index for index, character in enumerate(alphabet)}
    if not value:
        raise ValueError("empty base58btc payload")
    accumulator = 0
    for character in value:
        try:
            digit = indexes[character]
        except KeyError as exc:
            raise ValueError("invalid base58btc character") from exc
        accumulator = accumulator * 58 + digit
    body = (
        accumulator.to_bytes((accumulator.bit_length() + 7) // 8, "big")
        if accumulator
        else b""
    )
    leading_zeroes = len(value) - len(value.lstrip("1"))
    return (b"\x00" * leading_zeroes) + body


def _ed25519_public_key_from_did_key(value: Any) -> bytes:
    if not isinstance(value, str) or not value.startswith("did:key:z"):
        raise ValueError("reviewer identity must be an Ed25519 did:key:z identity")
    decoded = _base58btc_decode(value.removeprefix("did:key:z"))
    if len(decoded) != 34 or decoded[:2] != b"\xed\x01":
        raise ValueError("reviewer did:key must contain one Ed25519 public key")
    return decoded[2:]


def _verify_standard_ed25519_signature(
    errors: list[str],
    *,
    prefix: str,
    signer_identity_did: Any,
    signature_token: Any,
    message: bytes,
) -> None:
    """Verify one canonical standard-base64 Ed25519 signature."""

    try:
        public_key = _ed25519_public_key_from_did_key(signer_identity_did)
    except ValueError as exc:
        errors.append(f"{prefix}.signer: {exc}")
        return
    if not isinstance(signature_token, str) or not signature_token:
        errors.append(f"{prefix}: expected standard-base64 Ed25519 signature")
        return
    try:
        signature = base64.b64decode(signature_token, validate=True)
    except (binascii.Error, ValueError) as exc:
        errors.append(f"{prefix}: invalid standard base64: {exc}")
        return
    if (
        len(signature) != 64
        or base64.b64encode(signature).decode("ascii") != signature_token
    ):
        errors.append(f"{prefix}: noncanonical Ed25519 signature")
        return
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (InvalidSignature, ValueError):
        errors.append(f"{prefix}: cryptographic verification failed")


@dataclass(frozen=True)
class LocalProfileLifecycleRootPinSnapshot:
    """One fixed-root pin loaded from bounded duplicate-safe bytes."""

    payload: Mapping[str, Any]
    raw: bytes
    sha256: str

    @property
    def root_identity_did(self) -> str:
        return str(self.payload.get("root_identity_did", ""))


def validate_local_profile_lifecycle_root_pin(
    payload: Mapping[str, Any],
    *,
    expected_root_identity_did: str | None = None,
    expected_base_head: str | None = None,
    expected_base_tree: str | None = None,
) -> tuple[str, ...]:
    """Validate the protected root pin without accepting a receipt-selected root."""

    errors: list[str] = []
    prefix = "local_profile_lifecycle_root_pin"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_LIFECYCLE_ROOT_PIN_REQUIRED_FIELDS,
    )
    expected_root = (
        _FINAL_LIFECYCLE_ROOT_DID_PENDING
        if expected_root_identity_did is None
        else expected_root_identity_did
    )
    if payload.get("schema") != LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    _require_hex40(errors, f"{prefix}.base_head", payload.get("base_head"))
    _require_hex40(errors, f"{prefix}.base_tree", payload.get("base_tree"))
    if expected_base_head is not None and payload.get("base_head") != expected_base_head:
        errors.append(f"{prefix}.base_head: exact Q phase head required")
    if expected_base_tree is not None and payload.get("base_tree") != expected_base_tree:
        errors.append(f"{prefix}.base_tree: exact Q phase tree required")
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.pinned_at_ms",
        value=payload.get("pinned_at_ms"),
        minimum=1,
        maximum=10**16,
    )
    pin_without_id = dict(payload)
    pin_id = pin_without_id.pop("pin_id", None)
    _require_sha256(errors, f"{prefix}.pin_id", pin_id)
    if pin_id != _canonical_sha256(pin_without_id):
        errors.append(f"{prefix}.pin_id: canonical root-pin identity mismatch")
    # FILL_AFTER sentinels keep the portable path closed.  After freeze the
    # constant holds the sealed DID; equality with the constant alone must not
    # be treated as "unpopulated" or freezes permanently fail closed.
    if (
        isinstance(expected_root, str)
        and "FILL_AFTER" in expected_root
    ):
        errors.append(f"{prefix}.root_identity_did: final root pin is not populated")
    elif payload.get("root_identity_did") != expected_root:
        errors.append(f"{prefix}.root_identity_did: fixed root mismatch")
    try:
        _ed25519_public_key_from_did_key(payload.get("root_identity_did"))
    except ValueError as exc:
        errors.append(f"{prefix}.root_identity_did: {exc}")
    return tuple(errors)


def load_local_profile_lifecycle_root_pin(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> LocalProfileLifecycleRootPinSnapshot:
    pin_path = Path(path)
    if pin_path.name != LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME:
        raise ValueError("local-profile lifecycle root-pin filename mismatch")
    file_snapshot = _read_regular_snapshot(
        pin_path,
        maximum_bytes=MAX_LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_BYTES,
    )
    _require_authority_file_snapshot(
        file_snapshot,
        repository_root=(
            Path(repository_root) if repository_root is not None else None
        ),
        expected_relative_path=(
            LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
            if repository_root is not None
            else None
        ),
    )
    raw = file_snapshot.raw
    payload = _load_json_bytes(raw, name=pin_path.name)
    if set(payload) != set(_LIFECYCLE_ROOT_PIN_REQUIRED_FIELDS):
        raise ValueError("local-profile lifecycle root pin requires exact fields")
    return LocalProfileLifecycleRootPinSnapshot(
        payload=payload,
        raw=raw,
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _validate_local_dev_profile_v5(
    errors: list[str],
    *,
    profile: Any,
    profile_content_id: Any,
    profile_signature: Any,
    expected_final_values: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    prefix = "local_operator_lifecycle_witness.profile"
    record = _require_exact_keys(
        errors,
        prefix=prefix,
        value=profile,
        expected=_LOCAL_PROFILE_V5_REQUIRED_FIELDS,
    )
    if record is None:
        return None
    if record.get("schema") != LOCAL_DEV_PROFILE_V5_SCHEMA:
        errors.append(f"{prefix}.schema: expected local-dev-profile@5")
    _require_trimmed_string(
        errors,
        prefix=f"{prefix}.repository_cid",
        value=record.get("repository_cid"),
    )
    _require_hex40(errors, f"{prefix}.baseline_commit", record.get("baseline_commit"))
    capabilities = _require_sorted_unique_string_array(
        errors,
        prefix=f"{prefix}.capabilities",
        value=record.get("capabilities"),
    )
    _require_positive_finite_number(
        errors,
        prefix=f"{prefix}.created_at",
        value=record.get("created_at"),
    )
    for field in (
        "profile_id",
        "lifecycle_root_path",
        "budget_cid",
        "resource_cid",
        "route_id",
        "reviewer_identity",
        "reviewer_provider",
        "fallback_provider_id",
        "fallback_model_id",
        "fallback_reasoning_effort",
    ):
        _require_trimmed_string(
            errors,
            prefix=f"{prefix}.{field}",
            value=record.get(field),
        )
    try:
        _ed25519_public_key_from_did_key(record.get("identity_did"))
    except ValueError as exc:
        errors.append(f"{prefix}.identity_did: {exc}")
    if record.get("revoked") is not False:
        errors.append(f"{prefix}.revoked: active non-revoked profile required")
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.lifecycle_generation",
        value=record.get("lifecycle_generation"),
        minimum=1,
    )
    anchor_id = record.get("lifecycle_anchor_id")
    if not isinstance(anchor_id, str) or _HEX64.fullmatch(anchor_id) is None:
        errors.append(f"{prefix}.lifecycle_anchor_id: expected lowercase 64-hex")
    effect_bounds = _require_sorted_unique_string_array(
        errors,
        prefix=f"{prefix}.effect_bounds",
        value=record.get("effect_bounds"),
    )
    if not set(effect_bounds).issubset(capabilities):
        errors.append(f"{prefix}.effect_bounds: must be a capability subset")
    exact_policy = {
        "route_id": _PROVIDER_FALLBACK_AUTHORIZATION_ROUTE["route_id"],
        "reviewer_provider": "local_operator",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
    }
    for field, expected in exact_policy.items():
        if record.get(field) != expected:
            errors.append(f"{prefix}.{field}: expected {expected!r}")
    if record.get("reviewer_identity") != record.get("identity_did"):
        errors.append(f"{prefix}.reviewer_identity: active profile DID mismatch")

    computed_content_id = _canonical_sha256(record)
    _require_sha256(errors, f"{prefix}_content_id", profile_content_id)
    if profile_content_id != computed_content_id:
        errors.append(f"{prefix}_content_id: canonical profile digest mismatch")
    _verify_standard_ed25519_signature(
        errors,
        prefix=f"{prefix}_signature",
        signer_identity_did=record.get("identity_did"),
        signature_token=profile_signature,
        message=_canonical_json_bytes(record),
    )

    final_checks = {
        "identity_did": "reviewer_identity",
        "profile_id": "profile_id",
        "lifecycle_anchor_id": "lifecycle_anchor_id",
        "lifecycle_generation": "lifecycle_generation",
    }
    for profile_field, final_field in final_checks.items():
        expected = expected_final_values.get(final_field)
        if _is_unpopulated_final_value(expected):
            errors.append(f"{prefix}.{profile_field}: final pin is not populated")
        elif record.get(profile_field) != expected:
            errors.append(f"{prefix}.{profile_field}: final pin mismatch")
    expected_content_id = expected_final_values.get("profile_content_id")
    if _is_unpopulated_final_value(expected_content_id):
        errors.append(f"{prefix}_content_id: final pin is not populated")
    elif profile_content_id != expected_content_id:
        errors.append(f"{prefix}_content_id: final pin mismatch")
    return record


def _validate_local_profile_anchor_v3(
    errors: list[str],
    *,
    anchor: Any,
    anchor_digest: Any,
    root_identity_did: str,
) -> Mapping[str, Any] | None:
    prefix = "local_operator_lifecycle_witness.anchor"
    record = _require_exact_keys(
        errors,
        prefix=prefix,
        value=anchor,
        expected=_LOCAL_PROFILE_ANCHOR_V3_REQUIRED_FIELDS,
    )
    if record is None:
        return None
    if record.get("schema") != LOCAL_PROFILE_LIFECYCLE_ANCHOR_V3_SCHEMA:
        errors.append(f"{prefix}.schema: expected lifecycle anchor@3")
    anchor_id = record.get("anchor_id")
    if not isinstance(anchor_id, str) or _HEX64.fullmatch(anchor_id) is None:
        errors.append(f"{prefix}.anchor_id: expected lowercase 64-hex")
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.generation",
        value=record.get("generation"),
        minimum=1,
    )
    if record.get("status") != "active":
        errors.append(f"{prefix}.status: expected 'active'")
    if record.get("did_status") != "active":
        errors.append(f"{prefix}.did_status: expected 'active'")
    for field in ("repository_cid", "profile_id"):
        _require_trimmed_string(
            errors,
            prefix=f"{prefix}.{field}",
            value=record.get(field),
        )
    _require_sha256(errors, f"{prefix}.did_state_id", record.get("did_state_id"))
    _require_sha256(errors, f"{prefix}.profile_content_id", record.get("profile_content_id"))
    for field in (
        "identity_did",
        "root_identity_did",
    ):
        try:
            _ed25519_public_key_from_did_key(record.get(field))
        except ValueError as exc:
            errors.append(f"{prefix}.{field}: {exc}")
    for field in (
        "previous_profile_id",
        "previous_profile_content_id",
        "previous_identity_did",
        "previous_anchor_digest",
    ):
        _require_trimmed_string(
            errors,
            prefix=f"{prefix}.{field}",
            value=record.get(field),
            allow_empty=True,
        )
    for field in ("previous_profile_content_id", "previous_anchor_digest"):
        value = record.get(field)
        if value not in {"", None}:
            _require_sha256(errors, f"{prefix}.{field}", value)
    previous_did = record.get("previous_identity_did")
    if previous_did not in {"", None}:
        try:
            _ed25519_public_key_from_did_key(previous_did)
        except ValueError as exc:
            errors.append(f"{prefix}.previous_identity_did: {exc}")
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.updated_at_ns",
        value=record.get("updated_at_ns"),
        minimum=1,
        maximum=10**21,
    )
    generation = record.get("generation")
    if type(generation) is int:
        prior_values = {
            field: record.get(field)
            for field in (
                "previous_profile_id",
                "previous_profile_content_id",
                "previous_identity_did",
                "previous_anchor_digest",
            )
        }
        if generation == 1 and any(value != "" for value in prior_values.values()):
            errors.append(f"{prefix}: generation 1 must not claim predecessor state")
        elif generation > 1:
            if not prior_values["previous_profile_id"]:
                errors.append(f"{prefix}.previous_profile_id: required after generation 1")
            for field in (
                "previous_profile_content_id",
                "previous_anchor_digest",
            ):
                _require_sha256(errors, f"{prefix}.{field}", prior_values[field])
            if not prior_values["previous_identity_did"]:
                errors.append(
                    f"{prefix}.previous_identity_did: required after generation 1"
                )
    if record.get("root_identity_did") != root_identity_did:
        errors.append(f"{prefix}.root_identity_did: root-pin mismatch")
    unsigned = dict(record)
    signature = unsigned.pop("root_signature", None)
    _verify_standard_ed25519_signature(
        errors,
        prefix=f"{prefix}.root_signature",
        signer_identity_did=root_identity_did,
        signature_token=signature,
        message=_canonical_json_bytes(unsigned),
    )
    _require_sha256(errors, "local_operator_lifecycle_witness.anchor_digest", anchor_digest)
    computed_digest = _canonical_sha256(record)
    if anchor_digest != computed_digest:
        errors.append(
            "local_operator_lifecycle_witness.anchor_digest: canonical anchor "
            "digest mismatch"
        )
    return record


def _validate_local_profile_did_state_v1(
    errors: list[str],
    *,
    did_state: Any,
    did_state_digest: Any,
    root_identity_did: str,
) -> Mapping[str, Any] | None:
    prefix = "local_operator_lifecycle_witness.did_state"
    record = _require_exact_keys(
        errors,
        prefix=prefix,
        value=did_state,
        expected=_LOCAL_PROFILE_DID_STATE_V1_REQUIRED_FIELDS,
    )
    if record is None:
        return None
    if record.get("schema") != LOCAL_PROFILE_DID_STATE_V1_SCHEMA:
        errors.append(f"{prefix}.schema: expected DID-state@1")
    if record.get("status") != "active":
        errors.append(f"{prefix}.status: expected 'active'")
    for field in ("profile_path", "profile_id"):
        _require_trimmed_string(
            errors,
            prefix=f"{prefix}.{field}",
            value=record.get(field),
        )
    anchor_id = record.get("anchor_id")
    if not isinstance(anchor_id, str) or _HEX64.fullmatch(anchor_id) is None:
        errors.append(f"{prefix}.anchor_id: expected lowercase 64-hex")
    _require_sha256(errors, f"{prefix}.profile_content_id", record.get("profile_content_id"))
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.generation",
        value=record.get("generation"),
        minimum=1,
    )
    _require_trimmed_string(
        errors,
        prefix=f"{prefix}.previous_identity_did",
        value=record.get("previous_identity_did"),
        allow_empty=True,
    )
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.updated_at_ns",
        value=record.get("updated_at_ns"),
        minimum=1,
        maximum=10**21,
    )
    for field in ("identity_did", "root_identity_did"):
        try:
            _ed25519_public_key_from_did_key(record.get(field))
        except ValueError as exc:
            errors.append(f"{prefix}.{field}: {exc}")
    previous_did = record.get("previous_identity_did")
    if previous_did not in {"", None}:
        try:
            _ed25519_public_key_from_did_key(previous_did)
        except ValueError as exc:
            errors.append(f"{prefix}.previous_identity_did: {exc}")
    if record.get("root_identity_did") != root_identity_did:
        errors.append(f"{prefix}.root_identity_did: root-pin mismatch")

    unsigned = dict(record)
    state_id = unsigned.pop("state_id", None)
    signature = unsigned.pop("root_signature", None)
    _verify_standard_ed25519_signature(
        errors,
        prefix=f"{prefix}.root_signature",
        signer_identity_did=root_identity_did,
        signature_token=signature,
        message=_canonical_json_bytes(unsigned),
    )
    signed_state = dict(unsigned)
    signed_state["root_signature"] = signature
    _require_sha256(errors, f"{prefix}.state_id", state_id)
    if state_id != _canonical_sha256(signed_state):
        errors.append(f"{prefix}.state_id: canonical DID-state identity mismatch")
    _require_sha256(
        errors,
        "local_operator_lifecycle_witness.did_state_digest",
        did_state_digest,
    )
    if did_state_digest != _canonical_sha256(record):
        errors.append(
            "local_operator_lifecycle_witness.did_state_digest: canonical "
            "DID-state digest mismatch"
        )
    return record


def _validate_local_profile_registry_v2(
    errors: list[str],
    *,
    registry: Any,
    root_identity_did: str,
) -> Mapping[str, Any] | None:
    prefix = "local_operator_lifecycle_witness.registry"
    record = _require_exact_keys(
        errors,
        prefix=prefix,
        value=registry,
        expected=_LOCAL_PROFILE_REGISTRY_V2_REQUIRED_FIELDS,
    )
    if record is None:
        return None
    if record.get("schema") != LOCAL_PROFILE_ROOT_REGISTRY_V2_SCHEMA:
        errors.append(f"{prefix}.schema: expected root registry@2")
    for field in ("profile_path", "lifecycle_root"):
        _require_trimmed_string(
            errors,
            prefix=f"{prefix}.{field}",
            value=record.get(field),
        )
    try:
        _ed25519_public_key_from_did_key(record.get("root_identity_did"))
    except ValueError as exc:
        errors.append(f"{prefix}.root_identity_did: {exc}")
    if record.get("root_identity_did") != root_identity_did:
        errors.append(f"{prefix}.root_identity_did: root-pin mismatch")
    unsigned = dict(record)
    registry_id = unsigned.pop("registry_id", None)
    _require_sha256(errors, f"{prefix}.registry_id", registry_id)
    if registry_id != _canonical_sha256(unsigned):
        errors.append(f"{prefix}.registry_id: canonical registry identity mismatch")
    return record


@dataclass(frozen=True)
class LocalOperatorLifecycleWitnessSnapshot:
    """One bounded local-operator lifecycle witness and its raw digest."""

    payload: Mapping[str, Any]
    raw: bytes
    sha256: str

    @property
    def witness_id(self) -> str:
        return str(self.payload.get("witness_id", ""))

    @property
    def reviewer_identity(self) -> str:
        profile = self.payload.get("profile")
        return str(profile.get("identity_did", "")) if isinstance(profile, Mapping) else ""


def load_local_operator_lifecycle_witness(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> LocalOperatorLifecycleWitnessSnapshot:
    witness_path = Path(path)
    if witness_path.name != LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME:
        raise ValueError("local-operator lifecycle witness filename mismatch")
    file_snapshot = _read_regular_snapshot(
        witness_path,
        maximum_bytes=MAX_LOCAL_OPERATOR_LIFECYCLE_WITNESS_BYTES,
    )
    _require_authority_file_snapshot(
        file_snapshot,
        repository_root=(
            Path(repository_root) if repository_root is not None else None
        ),
        expected_relative_path=(
            LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
            if repository_root is not None
            else None
        ),
    )
    raw = file_snapshot.raw
    payload = _load_json_bytes(raw, name=witness_path.name)
    if set(payload) != set(_LOCAL_OPERATOR_LIFECYCLE_WITNESS_REQUIRED_FIELDS):
        raise ValueError("local-operator lifecycle witness requires exact fields")
    return LocalOperatorLifecycleWitnessSnapshot(
        payload=payload,
        raw=raw,
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def validate_local_operator_lifecycle_witness(
    payload: Mapping[str, Any],
    *,
    root_identity_did: str,
    expected_base_head: str | None = None,
    expected_base_tree: str | None = None,
    reference_time_ms: int | None = None,
    earliest_observed_at_ms: int | None = None,
    expected_final_values: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Verify one portable witness without consulting the current wall clock."""

    errors: list[str] = []
    prefix = "local_operator_lifecycle_witness"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_LOCAL_OPERATOR_LIFECYCLE_WITNESS_REQUIRED_FIELDS,
    )
    if payload.get("schema") != LOCAL_PROFILE_LIFECYCLE_WITNESS_SCHEMA:
        errors.append(f"{prefix}.schema: expected lifecycle witness@1")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    _require_hex40(errors, f"{prefix}.base_head", payload.get("base_head"))
    _require_hex40(errors, f"{prefix}.base_tree", payload.get("base_tree"))
    if expected_base_head is not None and payload.get("base_head") != expected_base_head:
        errors.append(f"{prefix}.base_head: signed base mismatch")
    if expected_base_tree is not None and payload.get("base_tree") != expected_base_tree:
        errors.append(f"{prefix}.base_tree: signed base mismatch")
    observed = _require_exact_integer(
        errors,
        prefix=f"{prefix}.observed_at_ms",
        value=payload.get("observed_at_ms"),
        minimum=1,
        maximum=10**16,
    )
    expires = _require_exact_integer(
        errors,
        prefix=f"{prefix}.expires_at_ms",
        value=payload.get("expires_at_ms"),
        minimum=1,
        maximum=10**16,
    )
    if observed is not None and expires is not None:
        if observed >= expires:
            errors.append(f"{prefix}: observed_at_ms must precede expires_at_ms")
        if expires - observed > 600_000:
            errors.append(f"{prefix}: witness lifetime exceeds 600000ms")
        if earliest_observed_at_ms is not None and observed < earliest_observed_at_ms:
            errors.append(f"{prefix}.observed_at_ms: predates root-pin commit")
        if reference_time_ms is not None and not observed <= reference_time_ms <= expires:
            errors.append(f"{prefix}: deterministic reference time is outside witness")
        if reference_time_ms is not None and reference_time_ms - observed > 600_000:
            errors.append(f"{prefix}: witness exceeds deterministic maximum age")
    _require_trimmed_string(
        errors,
        prefix=f"{prefix}.nonce",
        value=payload.get("nonce"),
        maximum=512,
    )
    try:
        _ed25519_public_key_from_did_key(payload.get("root_identity_did"))
    except ValueError as exc:
        errors.append(f"{prefix}.root_identity_did: {exc}")
    if payload.get("root_identity_did") != root_identity_did:
        errors.append(f"{prefix}.root_identity_did: fixed root-pin mismatch")

    final_values = (
        _ACCEPTANCE_REVIEWER_FINAL_VALUES
        if expected_final_values is None
        else expected_final_values
    )
    profile = _validate_local_dev_profile_v5(
        errors,
        profile=payload.get("profile"),
        profile_content_id=payload.get("profile_content_id"),
        profile_signature=payload.get("profile_signature"),
        expected_final_values=final_values,
    )
    anchor = _validate_local_profile_anchor_v3(
        errors,
        anchor=payload.get("anchor"),
        anchor_digest=payload.get("anchor_digest"),
        root_identity_did=root_identity_did,
    )
    did_state = _validate_local_profile_did_state_v1(
        errors,
        did_state=payload.get("did_state"),
        did_state_digest=payload.get("did_state_digest"),
        root_identity_did=root_identity_did,
    )
    registry = _validate_local_profile_registry_v2(
        errors,
        registry=payload.get("registry"),
        root_identity_did=root_identity_did,
    )
    if profile is not None and profile.get("baseline_commit") != payload.get(
        "base_head"
    ):
        errors.append(f"{prefix}: profile baseline does not match signed base")
    if profile is not None and anchor is not None:
        equalities = (
            ("repository_cid", "repository_cid"),
            ("profile_id", "profile_id"),
            ("identity_did", "identity_did"),
            ("lifecycle_generation", "generation"),
            ("lifecycle_anchor_id", "anchor_id"),
        )
        for profile_field, anchor_field in equalities:
            if profile.get(profile_field) != anchor.get(anchor_field):
                errors.append(
                    f"{prefix}: profile.{profile_field} != anchor.{anchor_field}"
                )
        if payload.get("profile_content_id") != anchor.get("profile_content_id"):
            errors.append(f"{prefix}: profile content does not match anchor")
    if profile is not None and did_state is not None:
        equalities = (
            ("profile_id", "profile_id"),
            ("identity_did", "identity_did"),
            ("lifecycle_generation", "generation"),
            ("lifecycle_anchor_id", "anchor_id"),
        )
        for profile_field, state_field in equalities:
            if profile.get(profile_field) != did_state.get(state_field):
                errors.append(
                    f"{prefix}: profile.{profile_field} != did_state.{state_field}"
                )
        if payload.get("profile_content_id") != did_state.get("profile_content_id"):
            errors.append(f"{prefix}: profile content does not match DID state")
    if anchor is not None and did_state is not None:
        if anchor.get("did_state_id") != did_state.get("state_id"):
            errors.append(f"{prefix}: anchor DID-state identity mismatch")
        if anchor.get("did_status") != did_state.get("status"):
            errors.append(f"{prefix}: anchor DID status mismatch")
    if (
        profile is not None
        and registry is not None
        and profile.get("lifecycle_root_path") != registry.get("lifecycle_root")
    ):
        errors.append(f"{prefix}: profile lifecycle root != registry root")
    if (
        did_state is not None
        and registry is not None
        and did_state.get("profile_path") != registry.get("profile_path")
    ):
        errors.append(f"{prefix}: DID-state profile path != registry profile path")
    if anchor is not None and registry is not None:
        profile_path = registry.get("profile_path")
        expected_anchor_id = (
            hashlib.sha256(profile_path.encode("utf-8")).hexdigest()
            if isinstance(profile_path, str)
            else ""
        )
        if anchor.get("anchor_id") != expected_anchor_id:
            errors.append(f"{prefix}: registry profile path does not derive anchor ID")

    expected_anchor_digest = final_values.get("lifecycle_anchor_digest")
    if _is_unpopulated_final_value(expected_anchor_digest):
        errors.append(f"{prefix}.anchor_digest: final pin is not populated")
    elif payload.get("anchor_digest") != expected_anchor_digest:
        errors.append(f"{prefix}.anchor_digest: final pin mismatch")

    body = {
        field: payload.get(field)
        for field in _LOCAL_OPERATOR_LIFECYCLE_WITNESS_BODY_FIELDS
    }
    profile_identity = profile.get("identity_did") if profile is not None else None
    _verify_standard_ed25519_signature(
        errors,
        prefix=f"{prefix}.active_key_signature",
        signer_identity_did=profile_identity,
        signature_token=payload.get("active_key_signature"),
        message=_canonical_json_bytes(body),
    )
    root_signed = dict(body)
    root_signed["active_key_signature"] = payload.get("active_key_signature")
    _verify_standard_ed25519_signature(
        errors,
        prefix=f"{prefix}.root_signature",
        signer_identity_did=root_identity_did,
        signature_token=payload.get("root_signature"),
        message=_canonical_json_bytes(root_signed),
    )
    witness_without_id = dict(payload)
    witness_id = witness_without_id.pop("witness_id", None)
    _require_sha256(errors, f"{prefix}.witness_id", witness_id)
    if witness_id != _canonical_sha256(witness_without_id):
        errors.append(f"{prefix}.witness_id: canonical witness identity mismatch")
    return tuple(errors)


def canonical_operator_acceptance_review_bytes(
    payload: Mapping[str, Any],
) -> bytes:
    """Canonical receipt bytes covered by the operator review signature.

    The only excluded value is ``review.signature``.  In particular, no
    receipt-selected digest or partial projection can narrow the signed
    authority surface.
    """

    review = payload.get("review")
    if not isinstance(review, Mapping) or "signature" not in review:
        raise ValueError("review.signature is required")
    unsigned = dict(payload)
    unsigned_review = dict(review)
    unsigned_review.pop("signature")
    unsigned["review"] = unsigned_review
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def validate_operator_acceptance_signature(
    payload: Mapping[str, Any],
    *,
    expected_authority: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Verify a non-self review bound to P's lifecycle witness and auth @2."""

    errors: list[str] = []
    prefix = "operator_acceptance.review"
    review = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload.get("review"),
        expected=_ACCEPTANCE_REVIEW_REQUIRED_FIELDS,
    )
    if review is None:
        return tuple(errors)
    reviewer = _require_bounded_string(
        errors,
        prefix=f"{prefix}.reviewer_identity",
        value=review.get("reviewer_identity"),
        maximum=256,
    )
    reviewer_provider = _require_bounded_string(
        errors,
        prefix=f"{prefix}.reviewer_provider",
        value=review.get("reviewer_provider"),
        maximum=64,
    )
    for field in (
        "profile_id",
        "lifecycle_witness_id",
        "lifecycle_witness_nonce",
    ):
        _require_trimmed_string(
            errors,
            prefix=f"{prefix}.{field}",
            value=review.get(field),
        )
    for field in (
        "profile_content_id",
        "lifecycle_anchor_digest",
        "lifecycle_witness_sha256",
        "lifecycle_root_pin_sha256",
        "fallback_authorization_id",
        "fallback_authorization_sha256",
    ):
        _require_sha256(errors, f"{prefix}.{field}", review.get(field))
    anchor_id = review.get("lifecycle_anchor_id")
    if not isinstance(anchor_id, str) or _HEX64.fullmatch(anchor_id) is None:
        errors.append(f"{prefix}.lifecycle_anchor_id: expected lowercase 64-hex")
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.lifecycle_generation",
        value=review.get("lifecycle_generation"),
        minimum=1,
    )
    if review.get("lifecycle_witness_path") != (
        LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    ):
        errors.append(f"{prefix}.lifecycle_witness_path: protected path mismatch")
    if review.get("lifecycle_root_pin_path") != (
        LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    ):
        errors.append(f"{prefix}.lifecycle_root_pin_path: protected path mismatch")
    try:
        _ed25519_public_key_from_did_key(review.get("lifecycle_root_identity_did"))
    except ValueError as exc:
        errors.append(f"{prefix}.lifecycle_root_identity_did: {exc}")
    implementer = _require_bounded_string(
        errors,
        prefix=f"{prefix}.implementer_identity",
        value=review.get("implementer_identity"),
        maximum=256,
    )
    implementer_provider = _require_bounded_string(
        errors,
        prefix=f"{prefix}.implementer_provider",
        value=review.get("implementer_provider"),
        maximum=64,
    )
    if reviewer_provider != "local_operator":
        errors.append(f"{prefix}.reviewer_provider: expected 'local_operator'")
    if reviewer_provider.casefold() in {"codex", "openai"}:
        errors.append(f"{prefix}.reviewer_provider: Codex/OpenAI review is denied")
    if reviewer and implementer and reviewer.casefold() == implementer.casefold():
        errors.append(f"{prefix}: self-review is denied")
    if (
        reviewer_provider
        and implementer_provider
        and reviewer_provider.casefold() == implementer_provider.casefold()
    ):
        errors.append(f"{prefix}: reviewer and implementer providers must differ")
    if review.get("algorithm") != "Ed25519":
        errors.append(f"{prefix}.algorithm: expected 'Ed25519'")
    signed_at = review.get("signed_at")
    if not isinstance(signed_at, str) or _UTC_TIMESTAMP.fullmatch(signed_at) is None:
        errors.append(f"{prefix}.signed_at: expected UTC timestamp")
    if payload.get("created_at") != signed_at:
        errors.append(f"{prefix}.signed_at: must equal receipt created_at")

    if expected_authority is None:
        errors.append(f"{prefix}: verified lifecycle authority is required")
    else:
        authority = {
            field: review.get(field) for field in _ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
        }
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.authority",
            actual=authority,
            expected={
                field: expected_authority.get(field)
                for field in _ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
            },
        )
        authority_times: dict[str, int | None] = {}
        for field in _LIFECYCLE_AUTHORITY_TIME_FIELDS:
            authority_times[field] = _require_exact_integer(
                errors,
                prefix=f"{prefix}.authority.{field}",
                value=expected_authority.get(field),
                minimum=1,
                maximum=10**16,
            )
        observed_at_ms = authority_times["lifecycle_witness_observed_at_ms"]
        expires_at_ms = authority_times["lifecycle_witness_expires_at_ms"]
        authorized_at_ms = authority_times["fallback_authorized_at_ms"]
        if (
            observed_at_ms is not None
            and expires_at_ms is not None
            and observed_at_ms >= expires_at_ms
        ):
            errors.append(
                f"{prefix}.authority: witness observed_at_ms must precede "
                "expires_at_ms"
            )
        if (
            observed_at_ms is not None
            and expires_at_ms is not None
            and authorized_at_ms is not None
            and not observed_at_ms <= authorized_at_ms <= expires_at_ms
        ):
            errors.append(
                f"{prefix}.authority.fallback_authorized_at_ms: "
                "outside witness validity"
            )
        signed_at_ms = _utc_timestamp_to_ms(signed_at)
        if signed_at_ms is None and isinstance(signed_at, str):
            errors.append(f"{prefix}.signed_at: invalid UTC calendar timestamp")
        if (
            signed_at_ms is not None
            and observed_at_ms is not None
            and expires_at_ms is not None
            and not observed_at_ms <= signed_at_ms <= expires_at_ms
        ):
            errors.append(f"{prefix}.signed_at: outside witness validity")
        if (
            signed_at_ms is not None
            and authorized_at_ms is not None
            and signed_at_ms < authorized_at_ms
        ):
            errors.append(
                f"{prefix}.signed_at: predates fallback authorization"
            )

    try:
        public_key = _ed25519_public_key_from_did_key(reviewer)
    except ValueError as exc:
        errors.append(f"{prefix}.reviewer_identity: {exc}")
        return tuple(errors)
    signature_token = review.get("signature")
    if not isinstance(signature_token, str) or not signature_token.startswith(
        "ed25519:"
    ):
        errors.append(f"{prefix}.signature: expected ed25519:<base64url>")
        return tuple(errors)
    encoded_signature = signature_token.removeprefix("ed25519:")
    try:
        signature = base64.b64decode(
            encoded_signature + ("=" * (-len(encoded_signature) % 4)),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        errors.append(f"{prefix}.signature: invalid base64url encoding: {exc}")
        return tuple(errors)
    if (
        len(signature) != 64
        or base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
        != encoded_signature
    ):
        errors.append(f"{prefix}.signature: noncanonical Ed25519 signature")
        return tuple(errors)
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            canonical_operator_acceptance_review_bytes(payload),
        )
    except (InvalidSignature, ValueError):
        errors.append(f"{prefix}.signature: cryptographic verification failed")
    return tuple(errors)


def _sequential_phase_index(phase: str) -> int:
    try:
        return SEQUENTIAL_ACCEPTANCE_PHASES.index(phase)
    except ValueError:
        return -1


def _sequential_task_statuses_after(phase: str) -> dict[str, str]:
    """Return the only task-state projection admitted after one phase."""

    phase_index = _sequential_phase_index(phase)
    if phase_index < 0:
        return {}
    statuses = {task_id: "todo" for task_id in SEQUENTIAL_ACCEPTANCE_TASK_IDS}
    statuses["ASE3-022"] = "blocked"
    for observed_phase in SEQUENTIAL_ACCEPTANCE_PHASES[1 : phase_index + 1]:
        for task_id in SEQUENTIAL_PHASE_STATUS_TRANSITIONS[observed_phase]:
            statuses[task_id] = "completed"
    return statuses


def _sequential_artifacts_after(phase: str) -> tuple[str, ...]:
    phase_index = _sequential_phase_index(phase)
    if phase_index < 0:
        return ()
    accepted = [
        path
        for path, introduction in (
            SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE.items()
        )
        if _sequential_phase_index(introduction) <= phase_index
    ]
    if phase_index >= _sequential_phase_index("P019"):
        accepted.append(PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH)
    return tuple(sorted(accepted))


def _sequential_future_artifacts_after(phase: str) -> tuple[str, ...]:
    accepted = set(_sequential_artifacts_after(phase))
    future = [
        path
        for path in SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE
        if path not in accepted
    ]
    future.append(PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH)
    return tuple(sorted(future))


def _validate_sequential_acceptance_parent(
    *,
    payload: Any,
    expected_phase: str,
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    parent = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_SEQUENTIAL_ACCEPTANCE_PARENT_REQUIRED_FIELDS,
    )
    if parent is None:
        return errors
    _require_hex40(errors, f"{prefix}.head", parent.get("head"))
    _require_hex40(errors, f"{prefix}.tree", parent.get("tree"))
    if parent.get("branch") != "agent/prompt-self-improvement-v3":
        errors.append(f"{prefix}.branch: integration branch mismatch")
    if parent.get("phase") != expected_phase:
        errors.append(f"{prefix}.phase: expected {expected_phase!r}")
    expected_schema = (
        CONVERGENCE_MANIFEST_SCHEMA
        if expected_phase in {"Q", "R", "P019"}
        else ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
    )
    if parent.get("manifest_schema") != expected_schema:
        errors.append(f"{prefix}.manifest_schema: phase schema mismatch")
    prior = parent.get("prior_artifacts")
    expected_prior = _sequential_artifacts_after(expected_phase)
    if not isinstance(prior, Mapping) or set(prior) != set(expected_prior):
        errors.append(f"{prefix}.prior_artifacts: exact phase population required")
    elif isinstance(prior, Mapping):
        for path in expected_prior:
            _require_sha256(errors, f"{prefix}.prior_artifacts.{path}", prior.get(path))
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.future_artifact_paths_absent",
        actual=parent.get("future_artifact_paths_absent"),
        expected=list(_sequential_future_artifacts_after(expected_phase)),
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.task_statuses",
        actual=parent.get("task_statuses"),
        expected=_sequential_task_statuses_after(expected_phase),
    )
    if parent.get("reload_gate_status") != "blocked":
        errors.append(f"{prefix}.reload_gate_status: expected 'blocked'")
    return errors


def _validate_native_task(
    *,
    payload: Any,
    task_id: str,
    authorization_only: bool,
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    task = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_NATIVE_DEPENDENCY_TASK_REQUIRED_FIELDS,
    )
    if task is None:
        return errors
    expected = _SEQUENTIAL_TASK_CONTRACTS[task_id]
    _validate_exact_structure(
        errors,
        prefix=prefix,
        actual=task,
        expected={
            "task_id": task_id,
            "canonical_task_cid": expected["canonical_task_cid"],
            "todo_contract_sha256": expected["todo_contract_sha256"],
            "completed_contract_sha256": expected["completed_contract_sha256"],
            "status_before": "todo",
            "status_after": "todo" if authorization_only else "completed",
        },
    )
    return errors


def native_dependency_launch_authorization_id(
    payload: Mapping[str, Any],
) -> str:
    """Derive P031 identity without a self-selected ID or signature cycle."""

    body = dict(payload)
    body.pop("authorization_id", None)
    review = body.get("review")
    if isinstance(review, Mapping):
        unsigned_review = dict(review)
        unsigned_review.pop("signature", None)
        body["review"] = unsigned_review
    return _canonical_sha256(body)


def _validate_frozen_product_identity(
    *,
    actual: Any,
    expected: Mapping[str, Any],
    prefix: str,
    repo_root: Path | str | None,
    acceptance_parent_head: str,
    acceptance_parent_tree: str = "",
    require_acceptance_parent: bool = False,
    required_test_functions: Sequence[str] = (),
    required_connection_sites: Mapping[str, int] | None = None,
) -> list[str]:
    errors: list[str] = []
    _validate_exact_structure(errors, prefix=prefix, actual=actual, expected=expected)
    if repo_root is None:
        if require_acceptance_parent:
            errors.append(f"{prefix}.current_tree: repository is required")
        return errors
    if not isinstance(actual, Mapping):
        return errors
    repo = Path(repo_root)

    def git_text(*args: str) -> subprocess.CompletedProcess[str]:
        return _git(repo, "--no-replace-objects", *args)

    def git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
        return _git_bytes(repo, "--no-replace-objects", *args)

    commit = str(expected["commit"])
    parent = str(expected["parent"])
    frozen_git_identities = _FROZEN_PRODUCT_GIT_IDENTITIES.get(commit, {})
    if require_acceptance_parent:
        if (
            _HEX40.fullmatch(acceptance_parent_head) is None
            or _HEX40.fullmatch(acceptance_parent_tree) is None
        ):
            errors.append(
                f"{prefix}.current_tree: exact acceptance-parent head/tree required"
            )
        parent_tree = git_text(
            "rev-parse",
            "--verify",
            f"{acceptance_parent_head}^{{tree}}",
        )
        if (
            parent_tree.returncode != 0
            or parent_tree.stdout.strip() != acceptance_parent_tree
        ):
            errors.append(
                f"{prefix}.acceptance_parent_tree: head/tree identity mismatch"
            )
    tree = git_text("rev-parse", "--verify", f"{commit}^{{tree}}")
    if tree.returncode != 0 or tree.stdout.strip() != expected["tree"]:
        errors.append(f"{prefix}.tree: Git object mismatch")
    lineage = git_text("rev-list", "--parents", "-n", "1", commit)
    if lineage.returncode != 0 or lineage.stdout.strip().split() != [commit, parent]:
        errors.append(f"{prefix}.parent: exact product parent required")
    names = git_text(
        "diff",
        *_DETERMINISTIC_GIT_DIFF_FLAGS,
        "--name-only",
        parent,
        commit,
    )
    if names.returncode != 0 or names.stdout.splitlines() != sorted(
        expected["changed_paths"]
    ):
        errors.append(f"{prefix}.changed_paths: Git patch population mismatch")
    patch = git_bytes(
        "diff",
        *_DETERMINISTIC_GIT_DIFF_FLAGS,
        "--binary",
        parent,
        commit,
    )
    patch_sha = "sha256:" + hashlib.sha256(patch.stdout).hexdigest()
    if patch.returncode != 0 or patch_sha != expected["binary_patch_sha256"]:
        errors.append(f"{prefix}.binary_patch_sha256: Git bytes mismatch")
    expected_patch_id = expected.get("stable_patch_id")
    if expected_patch_id is not None and patch.returncode == 0:
        patch_id = subprocess.run(
            ["git", "patch-id", "--stable"],
            cwd=repo,
            input=patch.stdout,
            check=False,
            capture_output=True,
        )
        observed_patch_id = (
            patch_id.stdout.decode("ascii", errors="strict").split()[0]
            if patch_id.returncode == 0 and patch_id.stdout.split()
            else ""
        )
        if observed_patch_id != expected_patch_id:
            errors.append(f"{prefix}.stable_patch_id: Git patch-id mismatch")
    changed_paths = list(expected["changed_paths"])
    expected_file_hashes = expected["file_raw_sha256"]
    if set(expected_file_hashes) != set(changed_paths):
        errors.append(f"{prefix}.file_raw_sha256: exact changed-path population required")
    if set(frozen_git_identities) != set(changed_paths):
        errors.append(f"{prefix}.git_identity: exact changed-path population required")
    observed_file_hashes: dict[str, str] = {}
    frozen_blobs: dict[str, bytes] = {}
    acceptance_parent_blobs: dict[str, bytes] = {}
    for relative_path in changed_paths:
        expected_sha = expected_file_hashes.get(relative_path, "")
        blob = git_bytes("show", f"{commit}:{relative_path}")
        observed = "sha256:" + hashlib.sha256(blob.stdout).hexdigest()
        observed_file_hashes[relative_path] = observed
        if blob.returncode != 0 or observed != expected_sha:
            errors.append(f"{prefix}.file_raw_sha256.{relative_path}: Git bytes mismatch")
        if blob.returncode == 0 and len(blob.stdout) <= MAX_EVIDENCE_SNAPSHOT_BYTES:
            frozen_blobs[relative_path] = blob.stdout
        expected_git_identity = frozen_git_identities.get(relative_path, {})
        frozen_entry = git_text("ls-tree", commit, "--", relative_path)
        frozen_fields = frozen_entry.stdout.rstrip("\n").split(maxsplit=3)
        frozen_blob_id = (
            frozen_fields[2]
            if frozen_entry.returncode == 0
            and len(frozen_fields) == 4
            and frozen_fields[0] == expected_git_identity.get("mode")
            and frozen_fields[1] == "blob"
            and frozen_fields[2] == expected_git_identity.get("blob")
            and frozen_fields[3] == relative_path
            else ""
        )
        if not frozen_blob_id:
            errors.append(
                f"{prefix}.reviewed_tree.{relative_path}: "
                "exact regular frozen Git blob required"
            )
        if acceptance_parent_head:
            current_entry = git_text(
                "ls-tree",
                acceptance_parent_head,
                "--",
                relative_path,
            )
            current_fields = current_entry.stdout.rstrip("\n").split(maxsplit=3)
            if (
                current_entry.returncode != 0
                or len(current_fields) != 4
                or current_fields[0] != expected_git_identity.get("mode")
                or current_fields[1] != "blob"
                or current_fields[2] != expected_git_identity.get("blob")
                or current_fields[3] != relative_path
            ):
                errors.append(
                    f"{prefix}.acceptance_parent_tree.{relative_path}: "
                    "exact frozen Git blob and mode required"
                )
            current_blob = git_bytes(
                "show",
                f"{acceptance_parent_head}:{relative_path}",
            )
            current_sha = "sha256:" + hashlib.sha256(current_blob.stdout).hexdigest()
            if current_blob.returncode != 0 or current_sha != expected_sha:
                errors.append(
                    f"{prefix}.acceptance_parent_raw_sha256.{relative_path}: "
                    "frozen reviewed bytes required"
                )
            if (
                current_blob.returncode == 0
                and len(current_blob.stdout) <= MAX_EVIDENCE_SNAPSHOT_BYTES
            ):
                acceptance_parent_blobs[relative_path] = current_blob.stdout
    expected_manifest_sha = expected.get("ordered_file_hash_manifest_sha256")
    if expected_manifest_sha is not None:
        manifest_bytes = b"".join(
            (
                observed_file_hashes.get(relative_path, "sha256:").removeprefix(
                    "sha256:"
                )
                + "  "
                + relative_path
                + "\n"
            ).encode("utf-8")
            for relative_path in expected["changed_paths"]
        )
        observed_manifest_sha = (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        )
        if observed_manifest_sha != expected_manifest_sha:
            errors.append(
                f"{prefix}.ordered_file_hash_manifest_sha256: Git bytes mismatch"
            )
    if acceptance_parent_head:
        ancestor = git_text(
            "merge-base",
            "--is-ancestor",
            commit,
            acceptance_parent_head,
        )
        if ancestor.returncode != 0:
            errors.append(f"{prefix}.commit: not accepted in parent history")
    if required_test_functions:
        test_paths = [
            path
            for path in changed_paths
            if str(path).startswith("test/") and str(path).endswith(".py")
        ]
        if (
            len(test_paths) != 1
            or test_paths[0] not in frozen_blobs
            or test_paths[0] not in acceptance_parent_blobs
        ):
            errors.append(f"{prefix}.test_ast: exact frozen test blob unavailable")
        else:
            try:
                frozen_test_tree = ast.parse(
                    frozen_blobs[test_paths[0]],
                    filename=f"{commit}:{test_paths[0]}",
                )
                test_tree = ast.parse(
                    acceptance_parent_blobs[test_paths[0]],
                    filename=f"{acceptance_parent_head}:{test_paths[0]}",
                )
            except (SyntaxError, ValueError, TypeError) as exc:
                errors.append(f"{prefix}.test_ast: invalid Python AST: {exc}")
            else:
                if ast.dump(test_tree, include_attributes=False) != ast.dump(
                    frozen_test_tree,
                    include_attributes=False,
                ):
                    errors.append(
                        f"{prefix}.test_ast: exact frozen reviewed AST required"
                    )
                observed_tests = [
                    node.name
                    for node in test_tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                ]
                if observed_tests != list(required_test_functions):
                    errors.append(
                        f"{prefix}.test_ast: exact reviewed test functions required"
                    )
                if required_connection_sites is not None:
                    site_tests = [
                        node
                        for node in test_tree.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name
                        == "test_every_accepted_runtime_connection_site_uses_the_canonical_policy"
                    ]
                    observed_sites: list[object] = []
                    if len(site_tests) == 1:
                        for node in ast.walk(site_tests[0]):
                            if (
                                isinstance(node, ast.Compare)
                                and len(node.ops) == 1
                                and isinstance(node.ops[0], ast.Eq)
                                and len(node.comparators) == 1
                                and isinstance(node.left, ast.Call)
                                and isinstance(node.left.func, ast.Name)
                                and node.left.func.id == "Counter"
                                and len(node.left.args) == 1
                                and not node.left.keywords
                                and isinstance(node.left.args[0], ast.GeneratorExp)
                                and isinstance(node.comparators[0], ast.Call)
                                and isinstance(node.comparators[0].func, ast.Name)
                                and node.comparators[0].func.id == "Counter"
                                and len(node.comparators[0].args) == 1
                                and not node.comparators[0].keywords
                                and isinstance(node.comparators[0].args[0], ast.Dict)
                            ):
                                try:
                                    observed_sites.append(
                                        ast.literal_eval(node.comparators[0].args[0])
                                    )
                                except (ValueError, TypeError, SyntaxError):
                                    observed_sites.append(None)
                    if observed_sites != [dict(required_connection_sites)]:
                        errors.append(
                            f"{prefix}.site_ast: exact reviewed connection-site "
                            "Counter required"
                        )
    return errors


def _validate_protected_acceptance_suite(
    *,
    payload: Any,
    task_id: str,
    parent: Mapping[str, Any],
    required_tests: Sequence[str],
    final_values: Mapping[str, Any],
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    suite = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_PROTECTED_ACCEPTANCE_SUITE_REQUIRED_FIELDS,
    )
    if suite is None:
        return errors
    if not final_values.get("ready"):
        errors.append(
            f"{prefix}: final signed suite values are not populated "
            f"({final_values.get('pending')})"
        )
        return errors
    expected = {
        "command": _PROGRAM_EXPANSION_TASKS[task_id]["validation"],
        "exit_code": 0,
        "passed": True,
        "passed_count": final_values["passed_count"],
        "failed_count": 0,
        "validated_head": parent.get("head"),
        "validated_tree": parent.get("tree"),
        "report_sha256": final_values["report_sha256"],
        "required_test_functions": list(required_tests),
    }
    _validate_exact_structure(errors, prefix=prefix, actual=suite, expected=expected)
    return errors


def validate_native_dependency_launch_authorization(
    payload: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
    lifecycle_authority: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate signed P031 authority while proving that it claims no effect."""

    errors: list[str] = []
    prefix = "native_dependency.P031"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_REQUIRED_FIELDS,
    )
    if payload.get("schema") != NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    if payload.get("phase") != "P031":
        errors.append(f"{prefix}.phase: exact P031 required")
    created_at = payload.get("created_at")
    if _utc_timestamp_to_ms(created_at) is None:
        errors.append(f"{prefix}.created_at: valid UTC timestamp required")
    errors.extend(
        _validate_native_task(
            payload=payload.get("task"),
            task_id="ASE3-031",
            authorization_only=True,
            prefix=f"{prefix}.task",
        )
    )
    parent = payload.get("acceptance_parent")
    errors.extend(
        _validate_sequential_acceptance_parent(
            payload=parent,
            expected_phase="A030",
            prefix=f"{prefix}.acceptance_parent",
        )
    )
    errors.extend(
        _validate_frozen_product_identity(
            actual=payload.get("product"),
            expected=_ASE3_031_PRODUCT_IDENTITY,
            prefix=f"{prefix}.product",
            repo_root=repo_root,
            acceptance_parent_head=(
                str(parent.get("head", "")) if isinstance(parent, Mapping) else ""
            ),
            acceptance_parent_tree=(
                str(parent.get("tree", "")) if isinstance(parent, Mapping) else ""
            ),
            require_acceptance_parent=True,
            required_test_functions=_ASE3_031_REQUIRED_TEST_FUNCTIONS,
        )
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.native_pin",
        actual=payload.get("native_pin"),
        expected=_ASE3_031_REVIEWED_DEPENDENCY_PIN,
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.host_abi_trust_boundary",
        actual=payload.get("host_abi_trust_boundary"),
        expected=_ASE3_031_HOST_ABI_TRUST_BOUNDARY,
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.claims",
        actual=payload.get("claims"),
        expected=_NATIVE_DEPENDENCY_AUTHORIZATION_CLAIMS,
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.denials",
        actual=payload.get("denials"),
        expected=_NATIVE_DEPENDENCY_AUTHORIZATION_DENIALS,
    )
    observed_id = payload.get("authorization_id")
    _require_sha256(errors, f"{prefix}.authorization_id", observed_id)
    if observed_id != native_dependency_launch_authorization_id(payload):
        errors.append(f"{prefix}.authorization_id: canonical identity mismatch")
    errors.extend(
        validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
    )
    return tuple(errors)


def validate_native_dependency_acceptance_receipt(
    payload: Mapping[str, Any],
    *,
    launch_authorization: Mapping[str, Any] | None = None,
    launch_authorization_raw: bytes | None = None,
    repo_root: Path | str | None = None,
    lifecycle_authority: Mapping[str, Any] | None = None,
    final_values: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate A031 sealed-fd, terminal-process, preload, and ABI evidence."""

    errors: list[str] = []
    prefix = "native_dependency.A031"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_NATIVE_DEPENDENCY_ACCEPTANCE_REQUIRED_FIELDS,
    )
    if payload.get("schema") != NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    if payload.get("phase") != "A031":
        errors.append(f"{prefix}.phase: exact A031 required")
    created_at_ms = _utc_timestamp_to_ms(payload.get("created_at"))
    if created_at_ms is None:
        errors.append(f"{prefix}.created_at: valid UTC timestamp required")
    errors.extend(
        _validate_native_task(
            payload=payload.get("task"),
            task_id="ASE3-031",
            authorization_only=False,
            prefix=f"{prefix}.task",
        )
    )
    parent = payload.get("acceptance_parent")
    errors.extend(
        _validate_sequential_acceptance_parent(
            payload=parent,
            expected_phase="P031",
            prefix=f"{prefix}.acceptance_parent",
        )
    )
    parent_head = str(parent.get("head", "")) if isinstance(parent, Mapping) else ""
    errors.extend(
        _validate_frozen_product_identity(
            actual=payload.get("product"),
            expected=_ASE3_031_PRODUCT_IDENTITY,
            prefix=f"{prefix}.product",
            repo_root=repo_root,
            acceptance_parent_head=parent_head,
            acceptance_parent_tree=(
                str(parent.get("tree", "")) if isinstance(parent, Mapping) else ""
            ),
            require_acceptance_parent=True,
            required_test_functions=_ASE3_031_REQUIRED_TEST_FUNCTIONS,
        )
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.native_pin",
        actual=payload.get("native_pin"),
        expected=_ASE3_031_REVIEWED_DEPENDENCY_PIN,
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.host_abi_trust_boundary",
        actual=payload.get("host_abi_trust_boundary"),
        expected=_ASE3_031_HOST_ABI_TRUST_BOUNDARY,
    )
    launch = _require_exact_keys(
        errors,
        prefix=f"{prefix}.launch_authorization",
        value=payload.get("launch_authorization"),
        expected=("path", "sha256", "authorization_id", "phase"),
    )
    expected_authorization_id = (
        launch_authorization.get("authorization_id")
        if isinstance(launch_authorization, Mapping)
        else None
    )
    expected_authorization_sha = (
        "sha256:" + hashlib.sha256(launch_authorization_raw).hexdigest()
        if isinstance(launch_authorization_raw, bytes)
        else None
    )
    if launch is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.launch_authorization",
            actual=launch,
            expected={
                "path": NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
                "sha256": expected_authorization_sha,
                "authorization_id": expected_authorization_id,
                "phase": "P031",
            },
        )
    if launch_authorization is None or launch_authorization_raw is None:
        errors.append(f"{prefix}.launch_authorization: signed P031 bytes required")
    else:
        errors.extend(
            f"{prefix}.launch_authorization.{error}"
            for error in validate_native_dependency_launch_authorization(
                launch_authorization,
                repo_root=repo_root,
                lifecycle_authority=lifecycle_authority,
            )
        )
    descriptor = _require_exact_keys(
        errors,
        prefix=f"{prefix}.sealed_descriptor",
        value=payload.get("sealed_descriptor"),
        expected=_SEALED_NATIVE_DESCRIPTOR_REQUIRED_FIELDS,
    )
    if descriptor is not None:
        expected_descriptor_values = {
            "schema": "ipfs_accelerate_py.agent_supervisor.native-dependency-descriptor@1",
            "st_mode": stat.S_IFREG | 0o500,
            "st_nlink": 1,
            "size_bytes": _ASE3_031_REVIEWED_DEPENDENCY_PIN["size_bytes"],
            "payload_sha256": _ASE3_031_REVIEWED_DEPENDENCY_PIN["payload_sha256"],
            "seals": 15,
        }
        for field, expected_value in expected_descriptor_values.items():
            if descriptor.get(field) != expected_value or type(descriptor.get(field)) is not type(expected_value):
                errors.append(f"{prefix}.sealed_descriptor.{field}: exact sealed value required")
        for field, minimum in (("descriptor", 3), ("st_dev", 0), ("st_ino", 1), ("st_uid", 0)):
            _require_exact_integer(
                errors,
                prefix=f"{prefix}.sealed_descriptor.{field}",
                value=descriptor.get(field),
                minimum=minimum,
            )
    terminal = _require_exact_keys(
        errors,
        prefix=f"{prefix}.process_terminal",
        value=payload.get("process_terminal"),
        expected=_NATIVE_PROCESS_TERMINAL_REQUIRED_FIELDS,
    )
    if terminal is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.process_terminal",
            actual=terminal,
            expected={
                "terminal_sentinel_set_before_native_module_creation": True,
                "native_module_creation_started": True,
                "partial_initialization_retry_denied": True,
                "second_preload_attempt_denied": True,
                "terminal_returncode": 0,
            },
        )
    preload = _require_exact_keys(
        errors,
        prefix=f"{prefix}.preload_evidence",
        value=payload.get("preload_evidence"),
        expected=_NATIVE_PRELOAD_EVIDENCE_REQUIRED_FIELDS,
    )
    if preload is not None:
        runtime_effect_started_at = preload.get("runtime_effect_started_at")
        runtime_effect_started_at_ms = _utc_timestamp_to_ms(
            runtime_effect_started_at
        )
        if runtime_effect_started_at_ms is None:
            errors.append(
                f"{prefix}.preload_evidence.runtime_effect_started_at: "
                "valid UTC timestamp required"
            )
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.preload_evidence",
            actual=preload,
            expected={
                "launch_schema": "ipfs_accelerate_py.agent_supervisor.native-dependency-launch@1",
                "accepted_authorization_id": expected_authorization_id,
                "sealed_fd_verified_before_module_creation": True,
                "module_name": "_duckdb",
                "public_alias": "duckdb",
                "distribution_version": "1.5.2",
                "engine_version": "v1.5.2",
                "query_42_result": 42,
                "parent_environment_sanitized_before_exec": True,
                "forbidden_parent_environment_names": [
                    "GLIBC_TUNABLES",
                    "LD_AUDIT",
                    "LD_DEBUG",
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                    "PYTHONHOME",
                    "PYTHONPATH",
                ],
                "child_observed_forbidden_environment_names": [],
                "python_side_environment_rejection_triggered": False,
                "runtime_effect_started_at": runtime_effect_started_at,
                "runtime_effect_started_after_authorization": True,
            },
        )
        authorization_created_at_ms = (
            _utc_timestamp_to_ms(launch_authorization.get("created_at"))
            if isinstance(launch_authorization, Mapping)
            else None
        )
        if (
            authorization_created_at_ms is not None
            and runtime_effect_started_at_ms is not None
            and runtime_effect_started_at_ms < authorization_created_at_ms
        ):
            errors.append(
                f"{prefix}.preload_evidence.runtime_effect_started_at: "
                "predates signed P031 authorization"
            )
        if (
            runtime_effect_started_at_ms is not None
            and created_at_ms is not None
            and runtime_effect_started_at_ms > created_at_ms
        ):
            errors.append(
                f"{prefix}.preload_evidence.runtime_effect_started_at: "
                "must not follow signed A031 receipt"
            )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.denials",
        actual=payload.get("denials"),
        expected=_NATIVE_DEPENDENCY_ACCEPTANCE_DENIALS,
    )
    errors.extend(
        _validate_protected_acceptance_suite(
            payload=payload.get("suite"),
            task_id="ASE3-031",
            parent=parent if isinstance(parent, Mapping) else {},
            required_tests=_ASE3_031_REQUIRED_TEST_FUNCTIONS,
            final_values=final_values or _NATIVE_DEPENDENCY_ACCEPTANCE_FINAL_VALUES,
            prefix=f"{prefix}.suite",
        )
    )
    errors.extend(
        validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
    )
    return tuple(errors)


def validate_duckdb_connection_policy_acceptance_receipt(
    payload: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
    lifecycle_authority: Mapping[str, Any] | None = None,
    final_values: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate A032 policy, catalog, migration, compaction, and site proof."""

    errors: list[str] = []
    prefix = "duckdb_connection_policy.A032"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_DUCKDB_POLICY_ACCEPTANCE_REQUIRED_FIELDS,
    )
    if payload.get("schema") != DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    if payload.get("phase") != "A032":
        errors.append(f"{prefix}.phase: exact A032 required")
    if _utc_timestamp_to_ms(payload.get("created_at")) is None:
        errors.append(f"{prefix}.created_at: valid UTC timestamp required")
    errors.extend(
        _validate_native_task(
            payload=payload.get("task"),
            task_id="ASE3-032",
            authorization_only=False,
            prefix=f"{prefix}.task",
        )
    )
    parent = payload.get("acceptance_parent")
    errors.extend(
        _validate_sequential_acceptance_parent(
            payload=parent,
            expected_phase="A031",
            prefix=f"{prefix}.acceptance_parent",
        )
    )
    parent_head = str(parent.get("head", "")) if isinstance(parent, Mapping) else ""
    errors.extend(
        _validate_frozen_product_identity(
            actual=payload.get("product"),
            expected=_ASE3_032_PRODUCT_IDENTITY,
            prefix=f"{prefix}.product",
            repo_root=repo_root,
            acceptance_parent_head=parent_head,
            acceptance_parent_tree=(
                str(parent.get("tree", "")) if isinstance(parent, Mapping) else ""
            ),
            require_acceptance_parent=True,
            required_test_functions=_ASE3_032_REQUIRED_TEST_FUNCTIONS,
            required_connection_sites=_ASE3_032_CONNECTION_SITE_COUNTS,
        )
    )
    policy = _require_exact_keys(
        errors,
        prefix=f"{prefix}.connection_birth_policy",
        value=payload.get("connection_birth_policy"),
        expected=_DUCKDB_CONNECTION_BIRTH_POLICY_REQUIRED_FIELDS,
    )
    if policy is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.connection_birth_policy",
            actual=policy,
            expected={
                "settings_in_connect_call": _ASE3_032_CONNECTION_POLICY_SETTINGS,
                "tuning_bounds": _ASE3_032_CONNECTION_TUNING_BOUNDS,
                "lock_configuration_last": True,
                "returned_connection_exact_bool_tuple": [False, False, False, False, True],
                "close_on_verification_failure": True,
                "caller_override_or_coercion_allowed": False,
            },
        )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.connection_sites",
        actual=payload.get("connection_sites"),
        expected=_ASE3_032_CONNECTION_SITE_COUNTS,
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.external_byte_boundary",
        actual=payload.get("external_byte_boundary"),
        expected=_DUCKDB_EXTERNAL_BYTE_BOUNDARY,
    )
    catalog = _require_exact_keys(
        errors,
        prefix=f"{prefix}.catalog_seal",
        value=payload.get("catalog_seal"),
        expected=_DUCKDB_CATALOG_SEAL_REQUIRED_FIELDS,
    )
    if catalog is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.catalog_seal",
            actual=catalog,
            expected={
                "path_independent_full_persistent_catalog_equality": True,
                "inventories": [
                    "databases",
                    "schemas",
                    "tables",
                    "views",
                    "sequences",
                    "macros_and_functions",
                    "types",
                    "indexes",
                    "constraints",
                    "columns",
                ],
                "foreign_catalog_cases_rejected": _ASE3_032_FOREIGN_CATALOG_CASES,
                "source_bytes_unchanged_on_rejection": True,
                "temporary_files_cleaned_on_rejection": True,
            },
        )
    migration = _require_exact_keys(
        errors,
        prefix=f"{prefix}.legacy_migration",
        value=payload.get("legacy_migration"),
        expected=_DUCKDB_LEGACY_MIGRATION_REQUIRED_FIELDS,
    )
    if migration is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.legacy_migration",
            actual=migration,
            expected={field: True for field in _DUCKDB_LEGACY_MIGRATION_REQUIRED_FIELDS},
        )
    compaction = _require_exact_keys(
        errors,
        prefix=f"{prefix}.compaction",
        value=payload.get("compaction"),
        expected=_DUCKDB_COMPACTION_REQUIRED_FIELDS,
    )
    if compaction is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.compaction",
            actual=compaction,
            expected={
                "attach_count": 0,
                "source_read_only": True,
                "target_policy_initialized": True,
                "partial_copy_failure_preserves_authoritative_store": True,
                "atomic_replace_failure_preserves_authoritative_store": True,
                "foreign_catalog_rejection_preserves_source_bytes": True,
                "temporary_files_cleaned": True,
            },
        )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.denials",
        actual=payload.get("denials"),
        expected=_DUCKDB_POLICY_ACCEPTANCE_DENIALS,
    )
    errors.extend(
        _validate_protected_acceptance_suite(
            payload=payload.get("suite"),
            task_id="ASE3-032",
            parent=parent if isinstance(parent, Mapping) else {},
            required_tests=_ASE3_032_REQUIRED_TEST_FUNCTIONS,
            final_values=final_values or _DUCKDB_POLICY_ACCEPTANCE_FINAL_VALUES,
            prefix=f"{prefix}.suite",
        )
    )
    errors.extend(
        validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
    )
    return tuple(errors)


@dataclass(frozen=True)
class OperatorAcceptanceReceiptSnapshot:
    """One bounded, stable, duplicate-key-safe acceptance receipt snapshot."""

    filename: str
    payload: Mapping[str, Any]
    sha256: str
    raw: bytes


def _load_sequential_acceptance_artifact(
    path: Path | str,
    *,
    expected_filename: str,
    expected_relative_path: str,
    expected_schema: str,
    required_fields: Sequence[str],
    maximum_bytes: int,
    repository_root: Path | str | None = None,
) -> OperatorAcceptanceReceiptSnapshot:
    """Load one phase artifact through the protected no-follow authority path."""

    artifact_path = Path(path)
    if artifact_path.name != expected_filename:
        raise ValueError(f"{expected_filename}: filename mismatch")
    snapshot = _read_regular_snapshot(
        artifact_path,
        maximum_bytes=maximum_bytes,
    )
    _require_authority_file_snapshot(
        snapshot,
        repository_root=(
            Path(repository_root) if repository_root is not None else None
        ),
        expected_relative_path=(
            expected_relative_path if repository_root is not None else None
        ),
    )
    payload = _load_json_bytes(snapshot.raw, name=expected_filename)
    if set(payload) != set(required_fields):
        raise ValueError(f"{expected_filename}: exact top-level fields required")
    if payload.get("schema") != expected_schema:
        raise ValueError(f"{expected_filename}: schema mismatch")
    return OperatorAcceptanceReceiptSnapshot(
        filename=expected_filename,
        payload=payload,
        sha256="sha256:" + hashlib.sha256(snapshot.raw).hexdigest(),
        raw=snapshot.raw,
    )


def load_native_dependency_launch_authorization(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> OperatorAcceptanceReceiptSnapshot:
    """Load bounded signed P031 authority without following links."""

    return _load_sequential_acceptance_artifact(
        path,
        expected_filename=NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME,
        expected_relative_path=NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
        expected_schema=NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA,
        required_fields=_NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_REQUIRED_FIELDS,
        maximum_bytes=MAX_NATIVE_DEPENDENCY_AUTHORIZATION_BYTES,
        repository_root=repository_root,
    )


def load_native_dependency_acceptance_receipt(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> OperatorAcceptanceReceiptSnapshot:
    """Load bounded A031 evidence without following links."""

    return _load_sequential_acceptance_artifact(
        path,
        expected_filename=NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME,
        expected_relative_path=NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        expected_schema=NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA,
        required_fields=_NATIVE_DEPENDENCY_ACCEPTANCE_REQUIRED_FIELDS,
        maximum_bytes=MAX_NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_BYTES,
        repository_root=repository_root,
    )


def load_duckdb_connection_policy_acceptance_receipt(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> OperatorAcceptanceReceiptSnapshot:
    """Load bounded A032 evidence without following links."""

    return _load_sequential_acceptance_artifact(
        path,
        expected_filename=DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME,
        expected_relative_path=DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        expected_schema=DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
        required_fields=_DUCKDB_POLICY_ACCEPTANCE_REQUIRED_FIELDS,
        maximum_bytes=MAX_DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_BYTES,
        repository_root=repository_root,
    )


def load_operator_acceptance_receipt(
    path: Path | str,
    *,
    task_id: str,
    repository_root: Path | str | None = None,
) -> OperatorAcceptanceReceiptSnapshot:
    """Load an eventual operator receipt without following or trusting links."""

    receipt_path = Path(path)
    expected = _ACCEPTANCE_TASK_CONTRACTS.get(task_id)
    if expected is None:
        raise ValueError(f"unsupported operator acceptance task: {task_id}")
    if receipt_path.name != expected["filename"]:
        raise ValueError(f"{task_id}: receipt filename mismatch")
    file_snapshot = _read_regular_snapshot(
        receipt_path,
        maximum_bytes=MAX_OPERATOR_ACCEPTANCE_RECEIPT_BYTES,
    )
    _require_authority_file_snapshot(
        file_snapshot,
        repository_root=(
            Path(repository_root) if repository_root is not None else None
        ),
        expected_relative_path=(
            f"{_CONVERGENCE_RELATIVE_ROOT}/{expected['filename']}"
            if repository_root is not None
            else None
        ),
    )
    raw = file_snapshot.raw
    payload = _load_json_bytes(raw, name=receipt_path.name)
    required_fields = (
        _ASE3_019_OPERATOR_SALVAGE_REQUIRED_FIELDS
        if task_id == "ASE3-019"
        else (
            _HERMETIC_IDENTITY_ACCEPTANCE_REQUIRED_FIELDS
            if task_id == "ASE3-030"
            else _OPERATOR_REPAIR_ACCEPTANCE_REQUIRED_FIELDS
        )
    )
    if set(payload) != set(required_fields):
        raise ValueError(f"{task_id}: exact top-level receipt fields required")
    if payload.get("schema") != expected["schema"]:
        raise ValueError(f"{task_id}: receipt schema mismatch")
    task = payload.get("task")
    if not isinstance(task, Mapping) or task.get("task_id") != task_id:
        raise ValueError(f"{task_id}: receipt task identity mismatch")
    return OperatorAcceptanceReceiptSnapshot(
        filename=receipt_path.name,
        payload=payload,
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        raw=raw,
    )


def _validate_acceptance_task(
    *,
    payload: Mapping[str, Any],
    task_id: str,
) -> list[str]:
    errors: list[str] = []
    prefix = f"operator_acceptance.{task_id}.task"
    task = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload.get("task"),
        expected=_ACCEPTANCE_TASK_REQUIRED_FIELDS,
    )
    if task is None:
        return errors
    expected = _ACCEPTANCE_TASK_CONTRACTS[task_id]
    expected_values = {
        "task_id": task_id,
        "canonical_task_cid": expected["canonical_task_cid"],
        "goal_id": expected["goal_id"],
        "repairs_task": expected["repairs_task"],
        "todo_contract_sha256": expected["todo_contract_sha256"],
        "completed_contract_sha256": expected["completed_contract_sha256"],
        "status_before": "todo",
        "status_after": "completed",
    }
    _validate_exact_structure(
        errors,
        prefix=prefix,
        actual=task,
        expected=expected_values,
    )
    return errors


def _validate_acceptance_parent(
    *,
    payload: Mapping[str, Any],
    prefix: str,
    expected_phase: str,
) -> list[str]:
    errors: list[str] = []
    parent = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_ACCEPTANCE_PARENT_REQUIRED_FIELDS,
    )
    if parent is None:
        return errors
    _require_hex40(errors, f"{prefix}.head", parent.get("head"))
    _require_hex40(errors, f"{prefix}.tree", parent.get("tree"))
    if parent.get("branch") != "agent/prompt-self-improvement-v3":
        errors.append(f"{prefix}.branch: integration branch mismatch")
    expected_manifest_schema = (
        CONVERGENCE_MANIFEST_SCHEMA
        if expected_phase in {"Q", "R", "P019"}
        else ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
    )
    if parent.get("manifest_schema") != expected_manifest_schema:
        errors.append(f"{prefix}.manifest_schema: phase schema mismatch")
    expected_absent = list(_sequential_future_artifacts_after(expected_phase))
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.receipt_paths_absent",
        actual=parent.get("receipt_paths_absent"),
        expected=expected_absent,
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.task_statuses",
        actual=parent.get("task_statuses"),
        expected=_sequential_task_statuses_after(expected_phase),
    )
    if parent.get("reload_gate_status") != "blocked":
        errors.append(f"{prefix}.reload_gate_status: expected 'blocked'")
    return errors


def _validate_acceptance_validation(
    *,
    payload: Mapping[str, Any],
    task_id: str,
    acceptance_parent: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    prefix = f"operator_acceptance.{task_id}.validation"
    validation = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_ACCEPTANCE_VALIDATION_REQUIRED_FIELDS,
    )
    if validation is None:
        return errors
    expected_command = (
        _ASE3_019_REQUIRED_VALIDATION
        if task_id == "ASE3-019"
        else str(_FALSE_COMPLETION_REPAIR_TASKS[task_id]["validation"])
    )
    if validation.get("command") != expected_command:
        errors.append(f"{prefix}.command: exact declared command required")
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.exit_code",
        value=validation.get("exit_code"),
        minimum=0,
        maximum=0,
    )
    if validation.get("passed") is not True:
        errors.append(f"{prefix}.passed: expected true")
    passed_count = _require_exact_integer(
        errors,
        prefix=f"{prefix}.passed_count",
        value=validation.get("passed_count"),
        minimum=1,
    )
    _require_exact_integer(
        errors,
        prefix=f"{prefix}.failed_count",
        value=validation.get("failed_count"),
        minimum=0,
        maximum=0,
    )
    final_values = _ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[task_id]
    if not final_values["ready"]:
        errors.append(
            f"{prefix}: final {task_id} product validation values are not populated"
        )
    elif passed_count != final_values["validation_passed_count"]:
        errors.append(f"{prefix}.passed_count: exact accepted count mismatch")
    if validation.get("validated_head") != acceptance_parent.get("head"):
        errors.append(f"{prefix}.validated_head: acceptance parent mismatch")
    if validation.get("validated_tree") != acceptance_parent.get("tree"):
        errors.append(f"{prefix}.validated_tree: acceptance parent mismatch")
    return errors


def validate_git_generation_provenance(
    *,
    repo_root: Path | str,
    generation: Mapping[str, Any],
    acceptance_parent_head: str,
    prefix: str = "operator_acceptance.implementation.generation",
) -> tuple[str, ...]:
    """Reconstruct one source/integrated patch generation from Git objects."""

    errors: list[str] = []
    repo = Path(repo_root)
    record = _require_exact_keys(
        errors,
        prefix=prefix,
        value=generation,
        expected=_ACCEPTANCE_GENERATION_REQUIRED_FIELDS,
    )
    if record is None:
        return tuple(errors)
    _require_bounded_string(
        errors,
        prefix=f"{prefix}.role",
        value=record.get("role"),
        maximum=128,
    )
    for field in (
        "source_commit",
        "source_parent",
        "source_tree",
        "integrated_commit",
        "integrated_parent",
        "integrated_tree",
    ):
        _require_hex40(errors, f"{prefix}.{field}", record.get(field))
    _require_sha256(
        errors,
        f"{prefix}.binary_full_index_patch_sha256",
        record.get("binary_full_index_patch_sha256"),
    )
    paths = _require_exact_string_array(
        errors,
        prefix=f"{prefix}.changed_paths",
        value=record.get("changed_paths"),
        maximum_items=64,
        safe_paths=True,
    )
    if errors:
        return tuple(errors)

    for kind in ("source", "integrated"):
        commit = str(record[f"{kind}_commit"])
        parent = str(record[f"{kind}_parent"])
        tree = str(record[f"{kind}_tree"])
        identity = _git(repo, "rev-list", "--parents", "-n", "1", commit)
        if identity.returncode != 0 or identity.stdout.strip().split() != [
            commit,
            parent,
        ]:
            errors.append(f"{prefix}.{kind}_commit: exact single parent mismatch")
        actual_tree = _git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")
        if actual_tree.returncode != 0 or actual_tree.stdout.strip() != tree:
            errors.append(f"{prefix}.{kind}_tree: Git tree mismatch")
        changed = _git_diff_names(repo, parent, commit)
        if changed.returncode != 0 or tuple(changed.stdout.splitlines()) != paths:
            errors.append(f"{prefix}.{kind}_changed_paths: exact population mismatch")

    expected_patch = str(record["binary_full_index_patch_sha256"])
    for kind in ("source", "integrated"):
        patch = _git_diff_patch(
            repo,
            str(record[f"{kind}_parent"]),
            str(record[f"{kind}_commit"]),
        )
        digest = ""
        if patch.returncode == 0:
            digest = "sha256:" + hashlib.sha256(patch.stdout).hexdigest()
        if patch.returncode != 0 or digest != expected_patch:
            errors.append(f"{prefix}.{kind}_patch: exact binary patch mismatch")

    source_ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        str(record["source_commit"]),
        acceptance_parent_head,
    )
    if source_ancestor.returncode != 1:
        errors.append(f"{prefix}.source_commit: must not be integration ancestor")
    integrated_ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        str(record["integrated_commit"]),
        acceptance_parent_head,
    )
    if integrated_ancestor.returncode != 0:
        errors.append(f"{prefix}.integrated_commit: must be integration ancestor")
    return tuple(errors)


def validate_hermetic_generation_provenance(
    *,
    repo_root: Path | str,
    generation: Mapping[str, Any],
    acceptance_parent_head: str,
    prefix: str = "operator_acceptance.ASE3-030.provenance.generation",
) -> tuple[str, ...]:
    """Reconstruct source, replay, and integrated ASE3-030 generations."""

    errors: list[str] = []
    repo = Path(repo_root)
    record = _require_exact_keys(
        errors,
        prefix=prefix,
        value=generation,
        expected=_HERMETIC_GENERATION_REQUIRED_FIELDS,
    )
    if record is None:
        return tuple(errors)
    _require_trimmed_string(
        errors,
        prefix=f"{prefix}.role",
        value=record.get("role"),
        maximum=128,
    )
    reconstructed_patches: dict[str, bytes] = {}
    for kind in ("source", "replay", "integrated"):
        for suffix in ("commit", "parent", "tree"):
            _require_hex40(
                errors,
                f"{prefix}.{kind}_{suffix}",
                record.get(f"{kind}_{suffix}"),
            )
        _require_sha256(
            errors,
            f"{prefix}.{kind}_patch_sha256",
            record.get(f"{kind}_patch_sha256"),
        )
    paths = _require_exact_string_array(
        errors,
        prefix=f"{prefix}.changed_paths",
        value=record.get("changed_paths"),
        maximum_items=128,
        safe_paths=True,
    )
    if paths != tuple(sorted(paths)):
        errors.append(f"{prefix}.changed_paths: sorted population required")
    if errors:
        return tuple(errors)

    for kind in ("source", "replay", "integrated"):
        commit = str(record[f"{kind}_commit"])
        parent = str(record[f"{kind}_parent"])
        tree = str(record[f"{kind}_tree"])
        lineage = _git(repo, "rev-list", "--parents", "-n", "1", commit)
        if lineage.returncode != 0 or lineage.stdout.strip().split() != [
            commit,
            parent,
        ]:
            errors.append(f"{prefix}.{kind}_commit: exact single parent mismatch")
        actual_tree = _git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")
        if actual_tree.returncode != 0 or actual_tree.stdout.strip() != tree:
            errors.append(f"{prefix}.{kind}_tree: Git tree mismatch")
        changed = _git_diff_names(repo, parent, commit)
        if changed.returncode != 0 or tuple(changed.stdout.splitlines()) != paths:
            errors.append(f"{prefix}.{kind}_changed_paths: exact population mismatch")
        patch = _git_diff_patch(repo, parent, commit)
        if patch.returncode == 0:
            reconstructed_patches[kind] = patch.stdout
        digest = (
            "sha256:" + hashlib.sha256(patch.stdout).hexdigest()
            if patch.returncode == 0
            else ""
        )
        if digest != record[f"{kind}_patch_sha256"]:
            errors.append(f"{prefix}.{kind}_patch: exact binary patch mismatch")
        ancestry = _git(
            repo,
            "merge-base",
            "--is-ancestor",
            commit,
            acceptance_parent_head,
        )
        expected_returncode = 0 if kind == "integrated" else 1
        if ancestry.returncode != expected_returncode:
            relation = "must be" if kind == "integrated" else "must not be"
            errors.append(
                f"{prefix}.{kind}_commit: {relation} an acceptance-parent ancestor"
            )
    recorded_patch_digests = {
        str(record[f"{kind}_patch_sha256"])
        for kind in ("source", "replay", "integrated")
    }
    if len(recorded_patch_digests) != 1:
        errors.append(
            f"{prefix}: source/replay/integrated patch digests must be identical"
        )
    if len(reconstructed_patches) == 3 and not (
        reconstructed_patches["source"]
        == reconstructed_patches["replay"]
        == reconstructed_patches["integrated"]
    ):
        errors.append(
            f"{prefix}: source/replay/integrated full-index binary patches "
            "must be byte-identical"
        )
    return tuple(errors)


def _validate_hermetic_closure(
    *,
    payload: Mapping[str, Any],
    acceptance_parent_head: str,
    acceptance_parent_tree: str,
    final_values: Mapping[str, Any],
    repo_root: Path | None,
) -> list[str]:
    errors: list[str] = []
    prefix = "operator_acceptance.ASE3-030.closure"
    required_module_names = tuple(sorted(_HERMETIC_REQUIRED_MODULE_MEMBER_MAP))
    required_member_paths = tuple(
        sorted(_HERMETIC_REQUIRED_MODULE_MEMBER_MAP.values())
    )
    required_module_origins = {
        module_name: {
            "member_path": member_path,
            "origin": f"capsule://sealed/{member_path}",
        }
        for module_name, member_path in sorted(
            _HERMETIC_REQUIRED_MODULE_MEMBER_MAP.items()
        )
    }
    closure = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_HERMETIC_CLOSURE_REQUIRED_FIELDS,
    )
    if closure is None:
        return errors

    manifest = _require_exact_keys(
        errors,
        prefix=f"{prefix}.manifest",
        value=closure.get("manifest"),
        expected=_HERMETIC_MANIFEST_REQUIRED_FIELDS,
    )
    members = closure.get("members")
    origins = closure.get("module_origins")
    member_paths: tuple[str, ...] = ()
    module_names: tuple[str, ...] = ()
    if not isinstance(members, Mapping) or not members:
        errors.append(f"{prefix}.members: expected nonempty exact object")
        members = {}
    else:
        member_paths = tuple(str(path) for path in members)
        if member_paths != tuple(sorted(member_paths)):
            errors.append(f"{prefix}.members: paths must be sorted")
        if any(not _is_safe_relative_path(path) for path in member_paths):
            errors.append(f"{prefix}.members: unsafe member path")
        if member_paths != required_member_paths:
            errors.append(
                f"{prefix}.members: exact reviewed dependency closure required"
            )
        for path, value in members.items():
            member = _require_exact_keys(
                errors,
                prefix=f"{prefix}.members.{path}",
                value=value,
                expected=_HERMETIC_MEMBER_REQUIRED_FIELDS,
            )
            if member is None:
                continue
            _require_hex40(
                errors,
                f"{prefix}.members.{path}.git_blob",
                member.get("git_blob"),
            )
            for field in ("raw_sha256", "archive_member_sha256"):
                _require_sha256(
                    errors,
                    f"{prefix}.members.{path}.{field}",
                    member.get(field),
                )
            if member.get("archive_member_sha256") != member.get("raw_sha256"):
                errors.append(
                    f"{prefix}.members.{path}: stored archive member/raw mismatch"
                )
        expected_blobs = dict(final_values.get("final_blobs", {}))
        expected_raw = dict(final_values.get("final_raw_sha256", {}))
        if set(expected_blobs) != set(required_member_paths) or set(
            expected_raw
        ) != set(required_member_paths):
            errors.append(
                f"{prefix}.members: frozen maps must use exact reviewed population"
            )
        if set(member_paths) != set(expected_blobs) or set(member_paths) != set(
            expected_raw
        ):
            errors.append(f"{prefix}.members: exact frozen blob/raw population required")
        for path, member in members.items():
            if not isinstance(member, Mapping):
                continue
            if member.get("git_blob") != expected_blobs.get(path):
                errors.append(f"{prefix}.members.{path}.git_blob: frozen mismatch")
            if member.get("raw_sha256") != expected_raw.get(path):
                errors.append(f"{prefix}.members.{path}.raw_sha256: frozen mismatch")

    if not isinstance(origins, Mapping) or not origins:
        errors.append(f"{prefix}.module_origins: expected nonempty exact object")
        origins = {}
    else:
        module_names = tuple(str(name) for name in origins)
        if module_names != tuple(sorted(module_names)):
            errors.append(f"{prefix}.module_origins: modules must be sorted")
        if module_names != required_module_names:
            errors.append(
                f"{prefix}.module_origins: exact reviewed module population required"
            )
        origin_paths: list[str] = []
        for module_name, value in origins.items():
            origin = _require_exact_keys(
                errors,
                prefix=f"{prefix}.module_origins.{module_name}",
                value=value,
                expected=_HERMETIC_MODULE_ORIGIN_REQUIRED_FIELDS,
            )
            if origin is None:
                continue
            path = origin.get("member_path")
            if isinstance(path, str):
                origin_paths.append(path)
            if path not in members:
                errors.append(
                    f"{prefix}.module_origins.{module_name}: unknown member path"
                )
            if origin.get("origin") != f"capsule://sealed/{path}":
                errors.append(
                    f"{prefix}.module_origins.{module_name}: sealed origin required"
                )
        if len(origin_paths) != len(set(origin_paths)):
            errors.append(f"{prefix}.module_origins: shadowed member origin forbidden")
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.module_origins.reviewed_map",
            actual=origins,
            expected=required_module_origins,
        )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.frozen_module_origins",
        actual=final_values.get("module_origins"),
        expected=required_module_origins,
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.frozen_member_paths",
        actual=list(final_values.get("member_paths", ())),
        expected=list(required_member_paths),
    )

    if manifest is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.manifest.identity",
            actual={
                "schema": manifest.get("schema"),
                "source_head": manifest.get("source_head"),
                "source_tree": manifest.get("source_tree"),
                "member_paths": manifest.get("member_paths"),
                "module_names": manifest.get("module_names"),
                "cid_profile": manifest.get("cid_profile"),
            },
            expected={
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "control-plane-dependency-manifest@1"
                ),
                "source_head": acceptance_parent_head,
                "source_tree": acceptance_parent_tree,
                "member_paths": list(member_paths),
                "module_names": list(module_names),
                "cid_profile": "cidv1-base32-lower-raw+dag-json-sha2-256",
            },
        )
        expected_manifest_sha = _canonical_sha256(manifest)
        if closure.get("manifest_sha256") != expected_manifest_sha:
            errors.append(f"{prefix}.manifest_sha256: deterministic digest mismatch")
    _require_sha256(errors, f"{prefix}.manifest_sha256", closure.get("manifest_sha256"))

    archive = _require_exact_keys(
        errors,
        prefix=f"{prefix}.archive",
        value=closure.get("archive"),
        expected=_HERMETIC_ARCHIVE_REQUIRED_FIELDS,
    )
    if archive is not None:
        archive_root = _canonical_sha256(
            {"member_paths": list(member_paths), "members": dict(members)}
        )
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.archive.contract",
            actual=archive,
            expected={
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "deterministic-control-plane-archive@1"
                ),
                "format": "zip-stored-sorted-v1",
                "sha256": final_values.get("archive_sha256"),
                "root_sha256": archive_root,
                "member_paths": list(member_paths),
            },
        )
    capsule = _require_exact_keys(
        errors,
        prefix=f"{prefix}.capsule",
        value=closure.get("capsule"),
        expected=_HERMETIC_CAPSULE_REQUIRED_FIELDS,
    )
    if capsule is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.capsule.contract",
            actual=capsule,
            expected={
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "sealed-control-plane-capsule@1"
                ),
                "manifest_sha256": closure.get("manifest_sha256"),
                "archive_sha256": final_values.get("archive_sha256"),
                "archive_root_sha256": (
                    archive.get("root_sha256") if archive is not None else ""
                ),
                "sealed_descriptor_sha256": final_values.get(
                    "sealed_descriptor_sha256"
                ),
                "member_count": len(member_paths),
            },
        )
        if closure.get("capsule_sha256") != _canonical_sha256(capsule):
            errors.append(f"{prefix}.capsule_sha256: deterministic digest mismatch")
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.cid_vectors",
        actual=closure.get("cid_vectors"),
        expected=list(_HERMETIC_CID_VECTORS),
    )

    pinned = {
        "member_paths": list(member_paths),
        "module_origins": dict(origins),
        "manifest_sha256": closure.get("manifest_sha256"),
        "capsule_sha256": closure.get("capsule_sha256"),
        "archive_sha256": archive.get("sha256") if archive is not None else "",
        "archive_root_sha256": (
            archive.get("root_sha256") if archive is not None else ""
        ),
        "sealed_descriptor_sha256": (
            capsule.get("sealed_descriptor_sha256") if capsule is not None else ""
        ),
    }
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.frozen",
        actual=pinned,
        expected={
            "member_paths": list(final_values.get("member_paths", ())),
            "module_origins": dict(final_values.get("module_origins", {})),
            "manifest_sha256": final_values.get("manifest_sha256"),
            "capsule_sha256": final_values.get("capsule_sha256"),
            "archive_sha256": final_values.get("archive_sha256"),
            "archive_root_sha256": final_values.get("archive_root_sha256"),
            "sealed_descriptor_sha256": final_values.get(
                "sealed_descriptor_sha256"
            ),
        },
    )
    if repo_root is not None:
        for path, value in members.items():
            if not isinstance(value, Mapping):
                continue
            blob = _git(
                repo_root,
                "rev-parse",
                "--verify",
                f"{acceptance_parent_head}:{path}",
            )
            raw = _git_bytes(repo_root, "show", f"{acceptance_parent_head}:{path}")
            raw_sha = (
                "sha256:" + hashlib.sha256(raw.stdout).hexdigest()
                if raw.returncode == 0
                else ""
            )
            if blob.returncode != 0 or blob.stdout.strip() != value.get("git_blob"):
                errors.append(f"{prefix}.members.{path}.git_blob: Git mismatch")
            if raw_sha != value.get("raw_sha256"):
                errors.append(f"{prefix}.members.{path}.raw_sha256: Git mismatch")
    return errors


def validate_hermetic_identity_acceptance_receipt(
    payload: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
    lifecycle_authority: Mapping[str, Any] | None = None,
    frozen_values: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate the exact signed ASE3-030 dependency-closure receipt."""

    errors: list[str] = []
    prefix = "operator_acceptance.ASE3-030"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_HERMETIC_IDENTITY_ACCEPTANCE_REQUIRED_FIELDS,
    )
    if payload.get("schema") != HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or _UTC_TIMESTAMP.fullmatch(created_at) is None:
        errors.append(f"{prefix}.created_at: expected UTC timestamp")
    for forbidden_path in _receipt_forbidden_binding_paths(payload):
        errors.append(f"{prefix}.{forbidden_path}: forbidden receipt authority field")
    errors.extend(_validate_acceptance_task(payload=payload, task_id="ASE3-030"))
    parent = payload.get("acceptance_parent")
    errors.extend(
        _validate_acceptance_parent(
            payload=parent if isinstance(parent, Mapping) else {},
            prefix=f"{prefix}.acceptance_parent",
            expected_phase="A019",
        )
    )
    parent_head = str(parent.get("head", "")) if isinstance(parent, Mapping) else ""
    parent_tree = str(parent.get("tree", "")) if isinstance(parent, Mapping) else ""

    expected = _HERMETIC_IDENTITY_FINAL_VALUES if frozen_values is None else frozen_values
    if expected.get("ready") is not True:
        errors.append(
            f"{prefix}.provenance: final product values are not populated "
            f"({expected.get('pending', _FINAL_VALUE_PENDING_030)})"
        )
    else:
        if not expected.get("generations"):
            errors.append(f"{prefix}.provenance: frozen generations must be nonempty")
        final_blobs = expected.get("final_blobs")
        final_raw = expected.get("final_raw_sha256")
        if (
            not isinstance(final_blobs, Mapping)
            or not final_blobs
            or not isinstance(final_raw, Mapping)
            or set(final_blobs) != set(final_raw)
        ):
            errors.append(f"{prefix}.provenance: frozen full blob/raw maps required")
        if not expected.get("member_paths"):
            errors.append(f"{prefix}.closure: frozen inventory required")
        if tuple(expected.get("probe_command", ())) != (
            _HERMETIC_HOSTILE_PROBE_ARGV
        ):
            errors.append(
                f"{prefix}.probe: independently frozen hostile-probe argv mismatch"
            )
        if type(expected.get("suite_passed_count")) is not int or expected.get(
            "suite_passed_count"
        ) < 1:
            errors.append(f"{prefix}.suite: positive frozen passed count required")
    provenance = _require_exact_keys(
        errors,
        prefix=f"{prefix}.provenance",
        value=payload.get("provenance"),
        expected=_HERMETIC_PROVENANCE_REQUIRED_FIELDS,
    )
    if provenance is not None:
        generations = provenance.get("generations")
        expected_generations = expected.get("generations", ())
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.provenance.generations",
            actual=generations,
            expected=list(expected_generations),
        )
        if isinstance(generations, list):
            if not generations:
                errors.append(f"{prefix}.provenance.generations: nonempty required")
            for index, generation in enumerate(generations):
                if isinstance(generation, Mapping):
                    patch_digests = tuple(
                        generation.get(f"{kind}_patch_sha256")
                        for kind in ("source", "replay", "integrated")
                    )
                    if not (
                        patch_digests[0]
                        == patch_digests[1]
                        == patch_digests[2]
                    ):
                        errors.append(
                            f"{prefix}.provenance.generations[{index}]: "
                            "source/replay/integrated patch digests must be identical"
                        )
                    if repo_root is not None:
                        errors.extend(
                            validate_hermetic_generation_provenance(
                                repo_root=repo_root,
                                generation=generation,
                                acceptance_parent_head=parent_head,
                                prefix=(
                                    f"{prefix}.provenance.generations[{index}]"
                                ),
                            )
                        )
        for field in ("final_blobs", "final_raw_sha256"):
            _validate_exact_structure(
                errors,
                prefix=f"{prefix}.provenance.{field}",
                actual=provenance.get(field),
                expected=dict(expected.get(field, {})),
            )
        final_blobs = provenance.get("final_blobs")
        final_raw = provenance.get("final_raw_sha256")
        if isinstance(final_blobs, Mapping) and isinstance(final_raw, Mapping):
            if set(final_blobs) != set(final_raw):
                errors.append(f"{prefix}.provenance: full blob/raw maps must match")
            if repo_root is not None:
                for path, expected_blob in final_blobs.items():
                    blob = _git(
                        Path(repo_root),
                        "rev-parse",
                        "--verify",
                        f"{parent_head}:{path}",
                    )
                    raw = _git_bytes(Path(repo_root), "show", f"{parent_head}:{path}")
                    raw_sha = (
                        "sha256:" + hashlib.sha256(raw.stdout).hexdigest()
                        if raw.returncode == 0
                        else ""
                    )
                    if blob.returncode != 0 or blob.stdout.strip() != expected_blob:
                        errors.append(
                            f"{prefix}.provenance.final_blobs.{path}: Git mismatch"
                        )
                    if raw_sha != final_raw.get(path):
                        errors.append(
                            f"{prefix}.provenance.final_raw_sha256.{path}: Git mismatch"
                        )
    errors.extend(
        _validate_hermetic_closure(
            payload=(
                payload.get("closure")
                if isinstance(payload.get("closure"), Mapping)
                else {}
            ),
            acceptance_parent_head=parent_head,
            acceptance_parent_tree=parent_tree,
            final_values=expected,
            repo_root=Path(repo_root) if repo_root is not None else None,
        )
    )

    probe = _require_exact_keys(
        errors,
        prefix=f"{prefix}.probe",
        value=payload.get("probe"),
        expected=_HERMETIC_PROBE_REQUIRED_FIELDS,
    )
    if probe is not None:
        _validate_hermetic_probe_argv(
            errors,
            prefix=f"{prefix}.probe.command",
            value=probe.get("command"),
        )
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.probe.contract",
            actual=probe,
            expected={
                "command": list(_HERMETIC_HOSTILE_PROBE_ARGV),
                "environment": {"PYTHONNOUSERSITE": "1", "PYTHONPATH": None},
                "exit_code": 0,
                "isolated": True,
                "user_site_enabled": False,
                "pythonpath_present": False,
                "multiformats_imported": False,
                "repository_or_candidate_imported": False,
                "sealed_descriptor_only": True,
                "all_modules_imported": True,
                "all_module_origins_verified": True,
                "raw_cid_minted": True,
                "raw_cid_validated": True,
                "dag_json_cid_minted": True,
                "dag_json_cid_validated": True,
                "scheduler_or_provider_effect_started": False,
                "stdout_sha256": probe.get("stdout_sha256"),
                "stderr_sha256": probe.get("stderr_sha256"),
            },
        )
        _require_sha256(errors, f"{prefix}.probe.stdout_sha256", probe.get("stdout_sha256"))
        _require_sha256(errors, f"{prefix}.probe.stderr_sha256", probe.get("stderr_sha256"))

    suite = _require_exact_keys(
        errors,
        prefix=f"{prefix}.suite",
        value=payload.get("suite"),
        expected=_HERMETIC_SUITE_REQUIRED_FIELDS,
    )
    if suite is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.suite.contract",
            actual=suite,
            expected={
                "command": _PROGRAM_EXPANSION_TASKS["ASE3-030"]["validation"],
                "exit_code": 0,
                "passed": True,
                "passed_count": expected.get("suite_passed_count"),
                "failed_count": 0,
                "validated_head": parent_head,
                "validated_tree": parent_tree,
                "report_sha256": expected.get("suite_report_sha256"),
            },
        )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.denials",
        actual=payload.get("denials"),
        expected=_HERMETIC_ACCEPTANCE_DENIALS,
    )
    errors.extend(
        validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
    )
    return tuple(errors)


def load_provider_attempt_reload_receipt(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> OperatorAcceptanceReceiptSnapshot:
    """Load the bounded protected ASE3-022 reload receipt."""

    receipt_path = Path(path)
    if receipt_path.name != PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME:
        raise ValueError("provider-attempt reload receipt filename mismatch")
    file_snapshot = _read_regular_snapshot(
        receipt_path,
        maximum_bytes=MAX_PROVIDER_ATTEMPT_RELOAD_RECEIPT_BYTES,
    )
    _require_authority_file_snapshot(
        file_snapshot,
        repository_root=(
            Path(repository_root) if repository_root is not None else None
        ),
        expected_relative_path=(
            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
            if repository_root is not None
            else None
        ),
    )
    payload = _load_json_bytes(file_snapshot.raw, name=receipt_path.name)
    if set(payload) != set(_RELOAD_RECEIPT_REQUIRED_FIELDS):
        raise ValueError("provider-attempt reload receipt requires exact fields")
    if payload.get("schema") != PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_SCHEMA:
        raise ValueError("provider-attempt reload receipt schema mismatch")
    return OperatorAcceptanceReceiptSnapshot(
        filename=receipt_path.name,
        payload=payload,
        sha256="sha256:" + hashlib.sha256(file_snapshot.raw).hexdigest(),
        raw=file_snapshot.raw,
    )


def load_provider_attempt_generation_birth_receipt(
    path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> OperatorAcceptanceReceiptSnapshot:
    """Load the separate post-L birth receipt through the no-follow path."""

    return _load_sequential_acceptance_artifact(
        path,
        expected_filename=PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_FILENAME,
        expected_relative_path=(
            PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH
        ),
        expected_schema=PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA,
        required_fields=_PROVIDER_ATTEMPT_GENERATION_BIRTH_REQUIRED_FIELDS,
        maximum_bytes=MAX_PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_BYTES,
        repository_root=repository_root,
    )


def validate_provider_attempt_reload_receipt(
    payload: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
    lifecycle_authority: Mapping[str, Any] | None = None,
    frozen_values: Mapping[str, Any] | None = None,
    accepted_control_plane: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate one signed receipt bound to A, never to its own L commit."""

    errors: list[str] = []
    prefix = "provider_attempt_reload"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_RELOAD_RECEIPT_REQUIRED_FIELDS,
    )
    if payload.get("schema") != PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or _UTC_TIMESTAMP.fullmatch(created_at) is None:
        errors.append(f"{prefix}.created_at: expected UTC timestamp")
    for forbidden_path in _receipt_forbidden_binding_paths(payload):
        errors.append(f"{prefix}.{forbidden_path}: forbidden receipt authority field")
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.task",
        actual=payload.get("task"),
        expected=_RELOAD_TASK_CONTRACT,
    )

    parent = _require_exact_keys(
        errors,
        prefix=f"{prefix}.acceptance_parent",
        value=payload.get("acceptance_parent"),
        expected=_RELOAD_PARENT_REQUIRED_FIELDS,
    )
    parent_head = ""
    parent_tree = ""
    receipt_bindings: Mapping[str, Any] = {}
    if parent is not None:
        parent_head = str(parent.get("head", ""))
        parent_tree = str(parent.get("tree", ""))
        _require_hex40(errors, f"{prefix}.acceptance_parent.head", parent_head)
        _require_hex40(errors, f"{prefix}.acceptance_parent.tree", parent_tree)
        if parent.get("branch") != "agent/prompt-self-improvement-v3":
            errors.append(f"{prefix}.acceptance_parent.branch: mismatch")
        if parent.get("manifest_schema") != ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA:
            errors.append(f"{prefix}.acceptance_parent.manifest_schema: @2 required")
        raw_bindings = parent.get("acceptance_receipts")
        if not isinstance(raw_bindings, Mapping) or set(raw_bindings) != set(
            SEQUENTIAL_ACCEPTANCE_ARTIFACT_FILENAMES
        ):
            errors.append(
                f"{prefix}.acceptance_parent.acceptance_receipts: exact population required"
            )
        else:
            receipt_bindings = raw_bindings
            for filename, digest in raw_bindings.items():
                _require_sha256(
                    errors,
                    f"{prefix}.acceptance_parent.acceptance_receipts.{filename}",
                    digest,
                )
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.acceptance_parent.task_statuses",
            actual=parent.get("task_statuses"),
            expected={
                **{
                    task_id: "completed"
                    for task_id in SEQUENTIAL_ACCEPTANCE_TASK_IDS
                },
                "ASE3-022": "blocked",
            },
        )

    expected = _RELOAD_FINAL_VALUES if frozen_values is None else frozen_values
    if expected.get("ready") is not True:
        errors.append(
            f"{prefix}: final reload values are not populated "
            f"({expected.get('pending', _FINAL_VALUE_PENDING_RELOAD)})"
        )
    incident = _require_exact_keys(
        errors,
        prefix=f"{prefix}.incident",
        value=payload.get("incident"),
        expected=_RELOAD_INCIDENT_REQUIRED_FIELDS,
    )
    accepted_control_plane_sha256 = ""
    if accepted_control_plane is None:
        errors.append(
            f"{prefix}.incident.accepted_control_plane_sha256: "
            "committed signed ASE3-019 object is required"
        )
    else:
        errors.extend(validate_ase3_019_accepted_control_plane(accepted_control_plane))
        accepted_control_plane_sha256 = _canonical_sha256(accepted_control_plane)
    if incident is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.incident",
            actual=incident,
            expected={
                "attempt2_incident": SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME,
                "attempt2_incident_sha256": _ASE3_019_ATTEMPT2_INCIDENT_SHA256,
                "operator_salvage_receipt": OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
                "operator_salvage_receipt_sha256": receipt_bindings.get(
                    OPERATOR_SALVAGE_RECEIPT_019_FILENAME
                ),
                "accepted_control_plane_sha256": accepted_control_plane_sha256,
            },
        )

    stopped = _require_exact_keys(
        errors,
        prefix=f"{prefix}.stopped_generation",
        value=payload.get("stopped_generation"),
        expected=_RELOAD_STOPPED_GENERATION_REQUIRED_FIELDS,
    )
    if stopped is not None:
        _require_exact_integer(
            errors,
            prefix=f"{prefix}.stopped_generation.generation_number",
            value=stopped.get("generation_number"),
            minimum=0,
            maximum=2**63 - 2,
        )
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.stopped_generation.contract",
            actual=stopped,
            expected={
                "generation_id": expected.get("stopped_generation_id"),
                "generation_number": expected.get("stopped_generation_number"),
                "head": parent_head,
                "tree": parent_tree,
                "scheduler_path": _RELOAD_SCHEDULER_PATH,
                "scheduler_blob": expected.get("scheduler_blob"),
                "scheduler_raw_sha256": expected.get("scheduler_raw_sha256"),
                "daemon_path": _RELOAD_DAEMON_PATH,
                "daemon_blob": expected.get("daemon_blob"),
                "daemon_raw_sha256": expected.get("daemon_raw_sha256"),
                "observed_owned_processes": 0,
                "observed_scoped_provider_containers": 0,
                "observed_inflight_attempts": 0,
            },
        )
        generation_without_id = dict(stopped)
        generation_id = generation_without_id.pop("generation_id", None)
        if generation_id != _canonical_sha256(generation_without_id):
            errors.append(
                f"{prefix}.stopped_generation.generation_id: deterministic mismatch"
            )
        if repo_root is not None:
            for label, path in (
                ("scheduler", _RELOAD_SCHEDULER_PATH),
                ("daemon", _RELOAD_DAEMON_PATH),
            ):
                blob = _git(
                    Path(repo_root),
                    "rev-parse",
                    "--verify",
                    f"{parent_head}:{path}",
                )
                raw = _git_bytes(Path(repo_root), "show", f"{parent_head}:{path}")
                raw_sha = (
                    "sha256:" + hashlib.sha256(raw.stdout).hexdigest()
                    if raw.returncode == 0
                    else ""
                )
                if blob.returncode != 0 or blob.stdout.strip() != stopped.get(
                    f"{label}_blob"
                ):
                    errors.append(
                        f"{prefix}.stopped_generation.{label}_blob: Git mismatch"
                    )
                if raw_sha != stopped.get(f"{label}_raw_sha256"):
                    errors.append(
                        f"{prefix}.stopped_generation.{label}_raw_sha256: Git mismatch"
                    )

    authorization = _require_exact_keys(
        errors,
        prefix=f"{prefix}.authorization",
        value=payload.get("authorization"),
        expected=_RELOAD_AUTHORIZATION_REQUIRED_FIELDS,
    )
    if authorization is not None:
        target_number = expected.get("stopped_generation_number")
        expected_target_number = (
            target_number + 1 if type(target_number) is int else None
        )
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.authorization.contract",
            actual=authorization,
            expected={
                "source_head": parent_head,
                "source_tree": parent_tree,
                "stopped_generation_id": expected.get("stopped_generation_id"),
                "target_generation_id": expected.get("target_generation_id"),
                "target_generation_number": expected_target_number,
                "target_scheduler_blob": expected.get("scheduler_blob"),
                "target_daemon_blob": expected.get("daemon_blob"),
                "lease_namespace": BOARD_NAMESPACE,
                "lease_state_at_authorization": "unclaimed",
                "required_cas_transition": "unclaimed_to_reserved",
                "single_winner_required": True,
                "launch_only_after_l_validates": True,
                "post_launch_birth_receipt_required": True,
                "post_launch_birth_receipt_schema": (
                    PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
                ),
                "attempt_counters_unchanged": True,
                "queue_history_unchanged": True,
                "legacy_refill_unchanged": True,
                "runtime_effect_started": False,
            },
        )
        _require_sha256(
            errors,
            f"{prefix}.authorization.target_generation_id",
            authorization.get("target_generation_id"),
        )
        _require_exact_integer(
            errors,
            prefix=f"{prefix}.authorization.target_generation_number",
            value=authorization.get("target_generation_number"),
            minimum=1,
            maximum=2**63 - 1,
        )
        target_identity = {
            "source_head": parent_head,
            "source_tree": parent_tree,
            "generation_number": expected_target_number,
            "scheduler_blob": expected.get("scheduler_blob"),
            "daemon_blob": expected.get("daemon_blob"),
        }
        if authorization.get("target_generation_id") != _canonical_sha256(
            target_identity
        ):
            errors.append(
                f"{prefix}.authorization.target_generation_id: "
                "deterministic old-plus-one mismatch"
            )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.denials",
        actual=payload.get("denials"),
        expected=_RELOAD_DENIALS,
    )
    errors.extend(
        validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
    )
    return tuple(errors)


def validate_provider_attempt_generation_birth_receipt(
    payload: Mapping[str, Any],
    *,
    birth_receipt_raw: bytes,
    birth_head: str,
    phase_heads: Mapping[str, str],
    repo_root: Path | str,
    lifecycle_authority: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate a signed process birth at or after its committed L authority."""

    errors: list[str] = []
    prefix = "provider_attempt_generation_birth"
    repository = Path(repo_root)
    if tuple(phase_heads) != SEQUENTIAL_ACCEPTANCE_PHASES:
        errors.append(f"{prefix}.phase_heads: exact Q-through-L population required")
    errors.extend(
        f"{prefix}.{error}"
        for error in validate_protected_acceptance_sequence(
            repo_root=repository,
            phase_heads=phase_heads,
            through_phase="L",
        )
    )
    reload_head = phase_heads.get("L", "")
    current_head = _git(
        repository,
        "--no-replace-objects",
        "rev-parse",
        "--verify",
        "HEAD",
    )
    if current_head.returncode != 0 or current_head.stdout.strip() != birth_head:
        errors.append(f"{prefix}.birth_head: checked-out exact birth head required")
    parent_result = _git(
        repository,
        "--no-replace-objects",
        "rev-parse",
        "--verify",
        f"{birth_head}^",
    )
    if parent_result.returncode != 0 or parent_result.stdout.strip() != reload_head:
        errors.append(f"{prefix}.birth_head: exact direct L child required")
    changed = _git(
        repository,
        "--no-replace-objects",
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        reload_head,
        birth_head,
    )
    if (
        changed.returncode != 0
        or tuple(changed.stdout.splitlines())
        != (PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH,)
    ):
        errors.append(f"{prefix}.birth_head: exact birth-only changed path required")
    committed_birth = _git_bytes(
        repository,
        "--no-replace-objects",
        "show",
        f"{birth_head}:{PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH}",
    )
    if committed_birth.returncode != 0:
        errors.append(f"{prefix}.committed_bytes: birth receipt unavailable")
        committed_birth_raw = b""
    else:
        committed_birth_raw = committed_birth.stdout
        if committed_birth_raw != birth_receipt_raw:
            errors.append(f"{prefix}.committed_bytes: exact Git birth bytes required")
    if len(committed_birth_raw) > MAX_PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_BYTES:
        errors.append(f"{prefix}.committed_bytes: birth receipt exceeds byte bound")
    try:
        committed_birth_payload = _load_json_bytes(
            committed_birth_raw,
            name=PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_FILENAME,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{prefix}.committed_bytes: {exc}")
        committed_birth_payload = {}
    try:
        supplied_birth_bytes = _canonical_json_bytes(payload)
        committed_birth_bytes = _canonical_json_bytes(committed_birth_payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"{prefix}.committed_bytes: noncanonical payload: {exc}")
    else:
        if supplied_birth_bytes != committed_birth_bytes:
            errors.append(f"{prefix}.committed_bytes: supplied payload/raw mismatch")
    payload = committed_birth_payload
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_PROVIDER_ATTEMPT_GENERATION_BIRTH_REQUIRED_FIELDS,
    )
    if payload.get("schema") != PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    if payload.get("phase") != "post-L":
        errors.append(f"{prefix}.phase: exact post-L required")

    _require_hex40(errors, f"{prefix}.reload_authorization.head", reload_head)
    reload_tree_result = _git(
        repository,
        "--no-replace-objects",
        "rev-parse",
        "--verify",
        f"{reload_head}^{{tree}}",
    )
    reload_tree = (
        reload_tree_result.stdout.strip()
        if reload_tree_result.returncode == 0
        else ""
    )
    if _HEX40.fullmatch(reload_tree) is None:
        errors.append(
            f"{prefix}.reload_authorization.tree: committed L tree required"
        )
    birth_parent_tree = _git(
        repository,
        "--no-replace-objects",
        "rev-parse",
        "--verify",
        f"{birth_head}^^{{tree}}",
    )
    if (
        birth_parent_tree.returncode != 0
        or birth_parent_tree.stdout.strip() != reload_tree
    ):
        errors.append(
            f"{prefix}.birth_head: exact validated L parent tree required"
        )
    birth_at_l = _git(
        repository,
        "--no-replace-objects",
        "cat-file",
        "-e",
        f"{reload_head}:{PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH}",
    )
    if birth_at_l.returncode == 0:
        errors.append(f"{prefix}.reload_authorization: birth receipt present in L")
    committed_reload = _git_bytes(
        repository,
        "--no-replace-objects",
        "show",
        f"{reload_head}:{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}",
    )
    if committed_reload.returncode != 0:
        errors.append(
            f"{prefix}.reload_authorization: committed L receipt unavailable"
        )
        committed_reload_raw = b""
    else:
        committed_reload_raw = committed_reload.stdout
    if len(committed_reload_raw) > MAX_PROVIDER_ATTEMPT_RELOAD_RECEIPT_BYTES:
        errors.append(
            f"{prefix}.reload_authorization: committed L receipt exceeds byte bound"
        )
    try:
        reload_receipt = _load_json_bytes(
            committed_reload_raw,
            name=PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{prefix}.reload_authorization: {exc}")
        reload_receipt = {}
    _require_exact_keys(
        errors,
        prefix=f"{prefix}.reload_authorization.committed_receipt",
        value=reload_receipt,
        expected=_RELOAD_RECEIPT_REQUIRED_FIELDS,
    )
    if reload_receipt.get("schema") != PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_SCHEMA:
        errors.append(
            f"{prefix}.reload_authorization.committed_receipt: schema mismatch"
        )

    live_reload_path = (
        repository / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    )
    try:
        live_reload = _read_regular_snapshot(
            live_reload_path,
            maximum_bytes=MAX_PROVIDER_ATTEMPT_RELOAD_RECEIPT_BYTES,
        )
        _require_authority_file_snapshot(
            live_reload,
            repository_root=repository,
            expected_relative_path=(
                PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
            ),
        )
    except (OSError, ValueError) as exc:
        errors.append(f"{prefix}.reload_authorization.live_receipt: {exc}")
    else:
        if live_reload.raw != committed_reload_raw:
            errors.append(
                f"{prefix}.reload_authorization.live_receipt: "
                "exact committed L bytes required"
            )

    committed_l_paths = tuple(
        dict.fromkeys(
            (
                *_sequential_artifacts_after("L"),
                _CONVERGENCE_MANIFEST_RELATIVE_PATH,
                PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
            )
        )
    )
    for relative_path in committed_l_paths:
        filename = PurePosixPath(relative_path).name
        try:
            live_snapshot = _read_regular_snapshot(
                repository / relative_path,
                maximum_bytes=_EVIDENCE_SNAPSHOT_BYTE_BOUNDS.get(
                    filename,
                    MAX_EVIDENCE_SNAPSHOT_BYTES,
                ),
            )
            _require_authority_file_snapshot(
                live_snapshot,
                repository_root=repository,
                expected_relative_path=relative_path,
            )
        except (OSError, ValueError) as exc:
            errors.append(
                f"{prefix}.validated_L_packet.{relative_path}.live_bytes: {exc}"
            )
            continue
        committed_snapshot = _git_bytes(
            repository,
            "--no-replace-objects",
            "show",
            f"{reload_head}:{relative_path}",
        )
        if (
            committed_snapshot.returncode != 0
            or committed_snapshot.stdout != live_snapshot.raw
        ):
            errors.append(
                f"{prefix}.validated_L_packet.{relative_path}.live_bytes: "
                "exact committed L bytes required"
            )

    committed_manifest = _git_bytes(
        repository,
        "--no-replace-objects",
        "show",
        f"{reload_head}:{_CONVERGENCE_MANIFEST_RELATIVE_PATH}",
    )
    committed_taskboard = _git_bytes(
        repository,
        "--no-replace-objects",
        "show",
        f"{reload_head}:{PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()}",
    )
    committed_authorization = _git_bytes(
        repository,
        "--no-replace-objects",
        "show",
        f"{reload_head}:{PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH}",
    )
    committed_l_inputs = (
        ("manifest", committed_manifest),
        ("taskboard", committed_taskboard),
        ("fallback_authorization", committed_authorization),
    )
    for label, result in committed_l_inputs:
        if result.returncode != 0:
            errors.append(
                f"{prefix}.validated_L_packet.{label}: committed bytes unavailable"
            )
    if all(result.returncode == 0 for _, result in committed_l_inputs):
        try:
            manifest_payload = _load_json_bytes(
                committed_manifest.stdout,
                name=f"L-{MANIFEST_FILENAME}",
            )
            authorization_payload = _load_json_bytes(
                committed_authorization.stdout,
                name=f"L-{PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME}",
            )
            committed_fallback_authorization = (
                ProviderFallbackPolicyAuthorization.from_dict(
                    authorization_payload
                )
            )
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}.validated_L_packet: {exc}")
        else:
            root_identity_did = (
                lifecycle_authority.get("lifecycle_root_identity_did")
                if isinstance(lifecycle_authority, Mapping)
                else None
            )
            l_packet_errors, _ = _validate_sequential_phase_packet(
                phase="L",
                artifact_root=repository / _CONVERGENCE_RELATIVE_ROOT,
                manifest=ConvergenceManifest.from_dict(manifest_payload),
                repo_root=repository,
                fallback_authorization=committed_fallback_authorization,
                fallback_authorization_raw=committed_authorization.stdout,
                manifest_raw=committed_manifest.stdout,
                taskboard_raw=committed_taskboard.stdout,
                expected_root_identity_did=(
                    str(root_identity_did)
                    if isinstance(root_identity_did, str)
                    else None
                ),
                phase_head_override=reload_head,
            )
            errors.extend(
                f"{prefix}.validated_L_packet.{error}"
                for error in l_packet_errors
            )

    reload_binding = _require_exact_keys(
        errors,
        prefix=f"{prefix}.reload_authorization",
        value=payload.get("reload_authorization"),
        expected=_PROVIDER_ATTEMPT_GENERATION_BIRTH_RELOAD_FIELDS,
    )
    if reload_binding is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.reload_authorization",
            actual=reload_binding,
            expected={
                "path": PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
                "sha256": (
                    "sha256:" + hashlib.sha256(committed_reload_raw).hexdigest()
                ),
                "head": reload_head,
                "tree": reload_tree,
                "phase": "L",
            },
        )

    generation = _require_exact_keys(
        errors,
        prefix=f"{prefix}.generation",
        value=payload.get("generation"),
        expected=_PROVIDER_ATTEMPT_GENERATION_BIRTH_GENERATION_FIELDS,
    )
    reload_authorization = reload_receipt.get("authorization")
    if generation is not None:
        expected_generation_id = (
            reload_authorization.get("target_generation_id")
            if isinstance(reload_authorization, Mapping)
            else None
        )
        expected_generation_number = (
            reload_authorization.get("target_generation_number")
            if isinstance(reload_authorization, Mapping)
            else None
        )
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.generation",
            actual=generation,
            expected={
                "generation_id": expected_generation_id,
                "generation_number": expected_generation_number,
            },
        )
        _require_sha256(
            errors,
            f"{prefix}.generation.generation_id",
            generation.get("generation_id"),
        )
        _require_exact_integer(
            errors,
            prefix=f"{prefix}.generation.generation_number",
            value=generation.get("generation_number"),
            minimum=1,
            maximum=2**63 - 1,
        )

    created_at_ms = _utc_timestamp_to_ms(payload.get("created_at"))
    if created_at_ms is None:
        errors.append(f"{prefix}.created_at: valid UTC timestamp required")
    review = payload.get("review")
    signed_at_ms = (
        _utc_timestamp_to_ms(review.get("signed_at"))
        if isinstance(review, Mapping)
        else None
    )
    if signed_at_ms is None:
        errors.append(f"{prefix}.review.signed_at: valid UTC timestamp required")

    process_birth = _require_exact_keys(
        errors,
        prefix=f"{prefix}.process_birth",
        value=payload.get("process_birth"),
        expected=_PROVIDER_ATTEMPT_GENERATION_BIRTH_PROCESS_FIELDS,
    )
    effect_started_at_ms: int | None = None
    process_started_at_ms: int | None = None
    if process_birth is not None:
        effect_started_at_ms = _utc_timestamp_to_ms(
            process_birth.get("effect_started_at")
        )
        process_started_at_ms = _utc_timestamp_to_ms(
            process_birth.get("process_started_at")
        )
        if effect_started_at_ms is None:
            errors.append(
                f"{prefix}.process_birth.effect_started_at: "
                "valid UTC timestamp required"
            )
        if process_started_at_ms is None:
            errors.append(
                f"{prefix}.process_birth.process_started_at: "
                "valid UTC timestamp required"
            )
        if process_birth.get("runtime_effect_started") is not True:
            errors.append(
                f"{prefix}.process_birth.runtime_effect_started: expected True"
            )
        if (
            effect_started_at_ms is not None
            and process_started_at_ms is not None
            and effect_started_at_ms > process_started_at_ms
        ):
            errors.append(
                f"{prefix}.process_birth: effect start must not follow process birth"
            )
        if (
            process_started_at_ms is not None
            and signed_at_ms is not None
            and process_started_at_ms > signed_at_ms
        ):
            errors.append(
                f"{prefix}.process_birth: process birth must not follow signed receipt"
            )

    reload_review = reload_receipt.get("review")
    reload_times = (
        _utc_timestamp_to_ms(reload_receipt.get("created_at")),
        (
            _utc_timestamp_to_ms(reload_review.get("signed_at"))
            if isinstance(reload_review, Mapping)
            else None
        ),
    )
    if any(value is None for value in reload_times):
        errors.append(
            f"{prefix}.reload_authorization: valid signed L time required"
        )
    birth_times = (
        created_at_ms,
        signed_at_ms,
        effect_started_at_ms,
        process_started_at_ms,
    )
    if (
        not any(value is None for value in (*reload_times, *birth_times))
        and max(reload_times) > min(birth_times)  # type: ignore[arg-type]
    ):
        errors.append(
            f"{prefix}.chronology: post-L birth predates signed L authority"
        )

    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.denials",
        actual=payload.get("denials"),
        expected=_PROVIDER_ATTEMPT_GENERATION_BIRTH_DENIALS,
    )
    errors.extend(
        f"{prefix}.{error}"
        for error in validate_operator_acceptance_signature(
            reload_receipt,
            expected_authority=lifecycle_authority,
        )
    )
    errors.extend(
        validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
    )
    return tuple(errors)


def _validate_acceptance_implementation(
    *,
    payload: Mapping[str, Any],
    task_id: str,
    acceptance_parent_head: str,
    repo_root: Path | None,
) -> list[str]:
    errors: list[str] = []
    prefix = f"operator_acceptance.{task_id}.implementation"
    implementation = _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_ACCEPTANCE_IMPLEMENTATION_REQUIRED_FIELDS,
    )
    if implementation is None:
        return errors
    final_values = _ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[task_id]
    if not final_values["ready"]:
        errors.append(
            f"{prefix}: final product values are not populated "
            f"({final_values['pending']})"
        )
        return errors
    generations = implementation.get("generations")
    expected_generations = final_values["generations"]
    if not isinstance(generations, list) or len(generations) != len(
        expected_generations
    ):
        errors.append(f"{prefix}.generations: exact population required")
        generations = []
    for index, expected_generation in enumerate(expected_generations):
        if index >= len(generations):
            break
        generation = generations[index]
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.generations[{index}]",
            actual=generation,
            expected={
                key: list(value) if key == "changed_paths" else value
                for key, value in expected_generation.items()
            },
        )
        if repo_root is not None and isinstance(generation, Mapping):
            errors.extend(
                validate_git_generation_provenance(
                    repo_root=repo_root,
                    generation=generation,
                    acceptance_parent_head=acceptance_parent_head,
                    prefix=f"{prefix}.generations[{index}]",
                )
            )
    final_blobs = implementation.get("final_blobs")
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.final_blobs",
        actual=final_blobs,
        expected=final_values["final_blobs"],
    )
    if repo_root is not None and isinstance(final_blobs, Mapping):
        for relative_path, expected_blob in final_values["final_blobs"].items():
            blob = _git(
                repo_root,
                "rev-parse",
                "--verify",
                f"{acceptance_parent_head}:{relative_path}",
            )
            if blob.returncode != 0 or blob.stdout.strip() != expected_blob:
                errors.append(f"{prefix}.final_blobs.{relative_path}: Git blob mismatch")
    return errors


def validate_operator_repair_acceptance_receipt(
    payload: Mapping[str, Any],
    *,
    task_id: str,
    repo_root: Path | str | None = None,
    lifecycle_authority: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate one strict @1 ASE3-023 or ASE3-027 repair receipt."""

    if task_id not in {"ASE3-023", "ASE3-027"}:
        return (f"operator_acceptance.{task_id}: unsupported repair task",)
    errors: list[str] = []
    prefix = f"operator_acceptance.{task_id}"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_OPERATOR_REPAIR_ACCEPTANCE_REQUIRED_FIELDS,
    )
    if payload.get("schema") != OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    for forbidden_path in _receipt_forbidden_binding_paths(payload):
        errors.append(f"{prefix}.{forbidden_path}: forbidden receipt authority field")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or _UTC_TIMESTAMP.fullmatch(created_at) is None:
        errors.append(f"{prefix}.created_at: expected UTC timestamp")
    errors.extend(_validate_acceptance_task(payload=payload, task_id=task_id))

    recovery_prefix = f"{prefix}.recovery"
    recovery = _require_exact_keys(
        errors,
        prefix=recovery_prefix,
        value=payload.get("recovery"),
        expected=_ACCEPTANCE_RECOVERY_REQUIRED_FIELDS,
    )
    if recovery is not None:
        evidence_anchor = str(
            _FALSE_COMPLETION_REPAIR_TASKS[task_id]["evidence_anchor"]
        )
        artifact, pointer = evidence_anchor.split("#", 1)
        _validate_exact_structure(
            errors,
            prefix=recovery_prefix,
            actual=recovery,
            expected={
                "artifact": artifact,
                "pointer": pointer,
                "historical_completion_authority": False,
                "branch_local_completion_authority": False,
                "repair_required": True,
            },
        )

    parent = payload.get("acceptance_parent")
    errors.extend(
        _validate_acceptance_parent(
            payload=parent if isinstance(parent, Mapping) else {},
            prefix=f"{prefix}.acceptance_parent",
            expected_phase="A032",
        )
    )
    parent_head = str(parent.get("head", "")) if isinstance(parent, Mapping) else ""
    errors.extend(
        _validate_acceptance_implementation(
            payload=(
                payload.get("implementation")
                if isinstance(payload.get("implementation"), Mapping)
                else {}
            ),
            task_id=task_id,
            acceptance_parent_head=parent_head,
            repo_root=Path(repo_root) if repo_root is not None else None,
        )
    )
    errors.extend(
        _validate_acceptance_validation(
            payload=(
                payload.get("validation")
                if isinstance(payload.get("validation"), Mapping)
                else {}
            ),
            task_id=task_id,
            acceptance_parent=parent if isinstance(parent, Mapping) else {},
        )
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.denials",
        actual=payload.get("denials"),
        expected=_REPAIR_ACCEPTANCE_DENIALS,
    )
    errors.extend(
        validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
    )
    return tuple(errors)


_SALVAGE_INCIDENT_REQUIRED_FIELDS: Final = (
    "artifact",
    "artifact_sha256",
    "attempt",
    "event_snapshot",
    "event_snapshot_sha256",
    "attempts_exhausted",
    "attempt_counter_mutation_authorized",
)
_SALVAGE_AUTHORITY_REQUIRED_FIELDS: Final = (
    "authorization_artifact",
    "authorization_artifact_sha256",
    "prospective_only",
    "route_id",
    "canonical_route_owner",
)
_SALVAGE_SOURCE_CANDIDATE_REQUIRED_FIELDS: Final = (
    "branch",
    "source_attempt",
    "source_commit",
    "source_tree",
    "replayed_paths",
    "candidate_blobs",
)
_SALVAGE_BASE_REQUIRED_FIELDS: Final = ("head", "tree", "branch")
_SALVAGE_MERGE_REQUIRED_FIELDS: Final = (
    "acceptance_parent_head",
    "acceptance_parent_tree",
    "source_commits_are_acceptance_parent_ancestors",
    "integrated_commits_are_acceptance_parent_ancestors",
)


def validate_ase3_019_accepted_control_plane(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """Pin the accepted router/effect/accounting boundary for ASE3-019."""

    errors: list[str] = []
    for machine_local_field in (
        "runner_path",
        "capsule_root",
        "descriptor",
        "executable_path",
    ):
        if machine_local_field in payload:
            errors.append(
                "operator_acceptance.ASE3-019.accepted_control_plane."
                f"{machine_local_field}: machine-local evidence is forbidden"
            )
    _validate_exact_structure(
        errors,
        prefix="operator_acceptance.ASE3-019.accepted_control_plane",
        actual=payload,
        expected=_ASE3_019_ACCEPTED_CONTROL_PLANE,
    )
    return tuple(errors)


def validate_operator_salvage_receipt_019(
    payload: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
    lifecycle_authority: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate the strict eventual @1 ASE3-019 operator salvage receipt."""

    errors: list[str] = []
    prefix = "operator_acceptance.ASE3-019"
    _require_exact_keys(
        errors,
        prefix=prefix,
        value=payload,
        expected=_ASE3_019_OPERATOR_SALVAGE_REQUIRED_FIELDS,
    )
    if payload.get("schema") != OPERATOR_SALVAGE_RECEIPT_019_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    for forbidden_path in _receipt_forbidden_binding_paths(payload):
        errors.append(f"{prefix}.{forbidden_path}: forbidden receipt authority field")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or _UTC_TIMESTAMP.fullmatch(created_at) is None:
        errors.append(f"{prefix}.created_at: expected UTC timestamp")
    errors.extend(_validate_acceptance_task(payload=payload, task_id="ASE3-019"))

    incident = _require_exact_keys(
        errors,
        prefix=f"{prefix}.incident",
        value=payload.get("incident"),
        expected=_SALVAGE_INCIDENT_REQUIRED_FIELDS,
    )
    if incident is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.incident",
            actual=incident,
            expected={
                "artifact": SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME,
                "artifact_sha256": _ASE3_019_ATTEMPT2_INCIDENT_SHA256,
                "attempt": 2,
                "event_snapshot": FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME,
                "event_snapshot_sha256": _ASE3_019_ATTEMPT2_EVENT_SHA256,
                "attempts_exhausted": True,
                "attempt_counter_mutation_authorized": False,
            },
        )
    authority = _require_exact_keys(
        errors,
        prefix=f"{prefix}.authority",
        value=payload.get("authority"),
        expected=_SALVAGE_AUTHORITY_REQUIRED_FIELDS,
    )
    if authority is not None:
        _validate_exact_structure(
            errors,
            prefix=f"{prefix}.authority",
            actual=authority,
            expected={
                "authorization_artifact": (
                    PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME
                ),
                "authorization_artifact_sha256": (
                    "sha256:1f2c354ae473d2ea7007d75b2a839df69127d866f3850d184d46e5ff87739e94"
                ),
                "prospective_only": True,
                "route_id": _PROVIDER_FALLBACK_AUTHORIZATION_ROUTE["route_id"],
                "canonical_route_owner": "ipfs_accelerate_py.llm_router",
            },
        )
    source_candidate = _require_exact_keys(
        errors,
        prefix=f"{prefix}.source_candidate",
        value=payload.get("source_candidate"),
        expected=_SALVAGE_SOURCE_CANDIDATE_REQUIRED_FIELDS,
    )
    if source_candidate is not None:
        if source_candidate.get("branch") != _ASE3_019_ATTEMPT2_BRANCH:
            errors.append(f"{prefix}.source_candidate.branch: incident branch mismatch")
        for field in ("source_commit", "source_tree"):
            _require_hex40(
                errors,
                f"{prefix}.source_candidate.{field}",
                source_candidate.get(field),
            )
        _require_exact_integer(
            errors,
            prefix=f"{prefix}.source_candidate.source_attempt",
            value=source_candidate.get("source_attempt"),
            minimum=2,
            maximum=2,
        )
        replayed_paths = _require_exact_string_array(
            errors,
            prefix=f"{prefix}.source_candidate.replayed_paths",
            value=source_candidate.get("replayed_paths"),
            maximum_items=32,
            safe_paths=True,
        )
        if replayed_paths != _ASE3_019_ATTEMPT2_REPLAYED_PATHS:
            errors.append(
                f"{prefix}.source_candidate.replayed_paths: incident population mismatch"
            )
        candidate_blobs = source_candidate.get("candidate_blobs")
        if (
            not isinstance(candidate_blobs, Mapping)
            or not candidate_blobs
            or len(candidate_blobs) > 32
        ):
            errors.append(f"{prefix}.source_candidate.candidate_blobs: expected object")
        else:
            for relative_path, blob in candidate_blobs.items():
                if not isinstance(relative_path, str) or not _is_safe_relative_path(
                    relative_path
                ):
                    errors.append(
                        f"{prefix}.source_candidate.candidate_blobs: unsafe path"
                    )
                _require_hex40(
                    errors,
                    f"{prefix}.source_candidate.candidate_blobs.{relative_path}",
                    blob,
                )
            _validate_exact_structure(
                errors,
                prefix=f"{prefix}.source_candidate.candidate_blobs",
                actual=candidate_blobs,
                expected=_ASE3_019_ATTEMPT2_CANDIDATE_BLOBS,
            )
        final_019 = _ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES["ASE3-019"]
        if final_019["ready"]:
            for field, expected_value in final_019["source_candidate"].items():
                if source_candidate.get(field) != expected_value:
                    errors.append(
                        f"{prefix}.source_candidate.{field}: final value mismatch"
                    )
    salvage_base = _require_exact_keys(
        errors,
        prefix=f"{prefix}.salvage_base",
        value=payload.get("salvage_base"),
        expected=_SALVAGE_BASE_REQUIRED_FIELDS,
    )
    if salvage_base is not None:
        _require_hex40(errors, f"{prefix}.salvage_base.head", salvage_base.get("head"))
        _require_hex40(errors, f"{prefix}.salvage_base.tree", salvage_base.get("tree"))
        _require_bounded_string(
            errors,
            prefix=f"{prefix}.salvage_base.branch",
            value=salvage_base.get("branch"),
            maximum=256,
        )
        final_019 = _ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES["ASE3-019"]
        if final_019["ready"]:
            _validate_exact_structure(
                errors,
                prefix=f"{prefix}.salvage_base",
                actual=salvage_base,
                expected=final_019["salvage_base"],
            )
    merge = _require_exact_keys(
        errors,
        prefix=f"{prefix}.merge",
        value=payload.get("merge"),
        expected=_SALVAGE_MERGE_REQUIRED_FIELDS,
    )
    parent_head = ""
    if merge is not None:
        parent_head = str(merge.get("acceptance_parent_head", ""))
        _require_hex40(errors, f"{prefix}.merge.acceptance_parent_head", parent_head)
        _require_hex40(
            errors,
            f"{prefix}.merge.acceptance_parent_tree",
            merge.get("acceptance_parent_tree"),
        )
        if merge.get("source_commits_are_acceptance_parent_ancestors") is not False:
            errors.append(f"{prefix}.merge: source commits must be non-ancestors")
        if merge.get("integrated_commits_are_acceptance_parent_ancestors") is not True:
            errors.append(f"{prefix}.merge: integrated commits must be ancestors")
    errors.extend(
        _validate_acceptance_implementation(
            payload=(
                payload.get("implementation")
                if isinstance(payload.get("implementation"), Mapping)
                else {}
            ),
            task_id="ASE3-019",
            acceptance_parent_head=parent_head,
            repo_root=Path(repo_root) if repo_root is not None else None,
        )
    )
    errors.extend(
        _validate_acceptance_validation(
            payload=(
                payload.get("validation")
                if isinstance(payload.get("validation"), Mapping)
                else {}
            ),
            task_id="ASE3-019",
            acceptance_parent={
                "head": parent_head,
                "tree": merge.get("acceptance_parent_tree", "")
                if merge is not None
                else "",
            },
        )
    )
    errors.extend(
        validate_ase3_019_accepted_control_plane(
            payload.get("accepted_control_plane")
            if isinstance(payload.get("accepted_control_plane"), Mapping)
            else {}
        )
    )
    _validate_exact_structure(
        errors,
        prefix=f"{prefix}.denials",
        actual=payload.get("denials"),
        expected=_SALVAGE_ACCEPTANCE_DENIALS,
    )
    errors.extend(
        validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
    )
    return tuple(errors)


def _status_only_acceptance_board(parent_raw: bytes) -> bytes:
    raise ValueError(
        "obsolete atomic A fan-in is forbidden; use a phase-owned sequential edit"
    )


def _obsolete_status_only_acceptance_board(parent_raw: bytes) -> bytes:
    """Historical implementation retained only as non-dispatched audit context."""

    try:
        text = parent_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("preparation taskboard is not UTF-8") from exc
    target_ids = set(_ACCEPTANCE_TASK_CONTRACTS)
    replaced: set[str] = set()
    current_task = ""
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            current_task = line[3:].strip().split(" ", 1)[0]
        if current_task in target_ids and line.rstrip("\r\n") == "- Status: todo":
            newline = line[len(line.rstrip("\r\n")) :]
            line = f"- Status: completed{newline}"
            replaced.add(current_task)
        output.append(line)
    if replaced != target_ids:
        raise ValueError("preparation taskboard lacks exact todo status lines")
    return "".join(output).encode("utf-8")


def _status_only_sequential_phase_board(parent_raw: bytes, phase: str) -> bytes:
    """Apply only the task status edits owned by one protected phase."""

    if phase not in SEQUENTIAL_PHASE_STATUS_TRANSITIONS:
        raise ValueError(f"unknown sequential acceptance phase: {phase}")
    try:
        text = parent_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("parent taskboard is not UTF-8") from exc
    parent_phase = SEQUENTIAL_PHASE_PARENT[phase]
    expected_parent = _sequential_task_statuses_after(parent_phase)
    transitions = set(SEQUENTIAL_PHASE_STATUS_TRANSITIONS[phase])
    replaced: set[str] = set()
    current_task = ""
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            current_task = line[3:].strip().split(" ", 1)[0]
        expected_status = expected_parent.get(current_task)
        if (
            current_task in transitions
            and expected_status is not None
            and line.rstrip("\r\n") == f"- Status: {expected_status}"
        ):
            newline = line[len(line.rstrip("\r\n")) :]
            line = f"- Status: completed{newline}"
            replaced.add(current_task)
        output.append(line)
    if replaced != transitions:
        raise ValueError(
            f"parent taskboard lacks exact {phase} status lines"
        )
    result = "".join(output).encode("utf-8")
    parsed = _parse_taskboard_metadata(result.decode("utf-8"))
    expected_child = _sequential_task_statuses_after(phase)
    observed = {
        task_id: parsed.get(task_id, {}).get("status", "")
        for task_id in expected_child
    }
    if observed != expected_child:
        raise ValueError(f"{phase} taskboard has a non-phase status transition")
    return result


def _phase_manifest_artifacts(
    *,
    repo_root: Path,
    head: str,
    phase: str,
) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    for relative_path in SEQUENTIAL_PHASE_CHANGED_PATHS[phase]:
        if relative_path in {
            _CONVERGENCE_MANIFEST_RELATIVE_PATH,
            PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
        }:
            continue
        blob = _git_bytes(repo_root, "show", f"{head}:{relative_path}")
        if blob.returncode != 0 or len(blob.stdout) > MAX_EVIDENCE_SNAPSHOT_BYTES:
            errors.append(
                f"protected_acceptance.{phase}.artifacts.{relative_path}: "
                "Git blob unavailable or oversized"
            )
            continue
        artifacts[relative_path] = (
            "sha256:" + hashlib.sha256(blob.stdout).hexdigest()
        )
    return artifacts, errors


def validate_sequential_acceptance_child_transition(
    *,
    repo_root: Path | str,
    phase: str,
    child_head: str,
    parent_head: str,
    parent_tree: str,
    consumed_child_blobs: Mapping[str, bytes] | None = None,
) -> tuple[str, ...]:
    """Validate one exact child in Q→R→P019→…→L without phase skipping."""

    errors: list[str] = []
    prefix = f"protected_acceptance.{phase}"
    repo = Path(repo_root)
    if phase not in SEQUENTIAL_PHASE_PARENT:
        return (f"{prefix}: unsupported child phase",)
    for label, value in (
        ("child_head", child_head),
        ("parent_head", parent_head),
        ("parent_tree", parent_tree),
    ):
        if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
            errors.append(f"{prefix}.{label}: expected lowercase 40-hex")
    if errors:
        return tuple(errors)
    errors.extend(
        _git_replacement_object_errors(
            repo,
            prefix=f"{prefix}.replacement_objects",
        )
    )
    changed_paths = SEQUENTIAL_PHASE_CHANGED_PATHS[phase]
    errors.extend(
        _validate_exact_direct_child(
            repo_root=repo,
            parent=parent_head,
            child=child_head,
            expected_paths=changed_paths,
            prefix=f"{prefix}.direct_child",
        )
    )
    errors.extend(
        _validate_git_regular_modes(
            repo_root=repo,
            head=child_head,
            paths=changed_paths,
            prefix=f"{prefix}.changed_modes",
        )
    )
    observed_parent_tree = _git(
        repo,
        "rev-parse",
        "--verify",
        f"{parent_head}^{{tree}}",
    )
    if (
        observed_parent_tree.returncode != 0
        or observed_parent_tree.stdout.strip() != parent_tree
    ):
        errors.append(f"{prefix}.parent_tree: Git tree mismatch")
    strict_parent_tree = _git(
        repo,
        "--no-replace-objects",
        "rev-parse",
        "--verify",
        f"{parent_head}^{{tree}}",
    )
    if (
        strict_parent_tree.returncode != 0
        or strict_parent_tree.stdout.strip() != parent_tree
    ):
        errors.append(f"{prefix}.parent_tree: replacement-free Git tree mismatch")

    if phase == "L":
        committed_reload = _git_bytes(
            repo,
            "--no-replace-objects",
            "show",
            f"{child_head}:{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}",
        )
        if committed_reload.returncode != 0:
            errors.append(f"{prefix}.reload_receipt: committed bytes unavailable")
        else:
            try:
                reload_payload = _load_json_bytes(
                    committed_reload.stdout,
                    name=PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
                )
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{prefix}.reload_receipt: {exc}")
            else:
                reload_parent = reload_payload.get("acceptance_parent")
                observed_reload_head = (
                    reload_parent.get("head")
                    if isinstance(reload_parent, Mapping)
                    else None
                )
                observed_reload_tree = (
                    reload_parent.get("tree")
                    if isinstance(reload_parent, Mapping)
                    else None
                )
                if (
                    observed_reload_head != parent_head
                    or observed_reload_tree != parent_tree
                ):
                    errors.append(
                        f"{prefix}.reload_receipt.acceptance_parent: "
                        "exact A023/027 head/tree required"
                    )

    parent_phase = SEQUENTIAL_PHASE_PARENT[phase]
    parent_index = _sequential_phase_index(parent_phase)
    child_index = _sequential_phase_index(phase)
    for relative_path, introduction in (
        SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE.items()
    ):
        introduction_index = _sequential_phase_index(introduction)
        for label, head, phase_index in (
            ("parent", parent_head, parent_index),
            ("child", child_head, child_index),
        ):
            should_exist = introduction_index <= phase_index
            observed = _git(repo, "cat-file", "-e", f"{head}:{relative_path}")
            if (observed.returncode == 0) is not should_exist:
                expectation = "present" if should_exist else "absent"
                errors.append(
                    f"{prefix}.{label}.{relative_path}: expected {expectation}"
                )
    for label, head in (("parent", parent_head), ("child", child_head)):
        for always_path in (
            PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
            _CONVERGENCE_MANIFEST_RELATIVE_PATH,
        ):
            if _git(repo, "cat-file", "-e", f"{head}:{always_path}").returncode != 0:
                errors.append(f"{prefix}.{label}.{always_path}: expected present")
        if _git(
            repo,
            "cat-file",
            "-e",
            f"{head}:{PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH}",
        ).returncode == 0:
            errors.append(
                f"{prefix}.{label}: post-L birth receipt is forbidden"
            )
    present_child_paths = [
        *(_sequential_artifacts_after(phase)),
        _CONVERGENCE_MANIFEST_RELATIVE_PATH,
    ]
    errors.extend(
        _validate_git_regular_modes(
            repo_root=repo,
            head=child_head,
            paths=tuple(sorted(set(present_child_paths))),
            prefix=f"{prefix}.cumulative_modes",
        )
    )

    board_path = PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
    parent_board = _git_bytes(repo, "show", f"{parent_head}:{board_path}")
    child_board = _git_bytes(repo, "show", f"{child_head}:{board_path}")
    if parent_board.returncode != 0 or child_board.returncode != 0:
        errors.append(f"{prefix}.taskboard: Git blobs unavailable")
    else:
        try:
            expected_board = _status_only_sequential_phase_board(
                parent_board.stdout,
                phase,
            )
        except ValueError as exc:
            errors.append(f"{prefix}.taskboard: {exc}")
        else:
            if child_board.stdout != expected_board:
                errors.append(
                    f"{prefix}.taskboard: only phase-owned status edits allowed"
                )

    parent_manifest_raw = _git_bytes(
        repo,
        "show",
        f"{parent_head}:{_CONVERGENCE_MANIFEST_RELATIVE_PATH}",
    )
    child_manifest_raw = _git_bytes(
        repo,
        "show",
        f"{child_head}:{_CONVERGENCE_MANIFEST_RELATIVE_PATH}",
    )
    if parent_manifest_raw.returncode != 0 or child_manifest_raw.returncode != 0:
        errors.append(f"{prefix}.manifest: Git blobs unavailable")
    else:
        try:
            parent_manifest = _load_json_bytes(
                parent_manifest_raw.stdout,
                name=f"{parent_phase}-{MANIFEST_FILENAME}",
            )
            child_manifest = _load_json_bytes(
                child_manifest_raw.stdout,
                name=f"{phase}-{MANIFEST_FILENAME}",
            )
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}.manifest: {exc}")
        else:
            if phase == "R":
                if child_manifest_raw.stdout != parent_manifest_raw.stdout:
                    errors.append(f"{prefix}.manifest: Q bytes must remain unchanged")
            elif phase == "P019":
                if child_manifest.get("schema") != CONVERGENCE_MANIFEST_SCHEMA:
                    errors.append(f"{prefix}.manifest: preparation @1 required")
                if "acceptance" in child_manifest or "reload" in child_manifest:
                    errors.append(f"{prefix}.manifest: effect phases are forbidden")
                authorization = _git_bytes(
                    repo,
                    "show",
                    f"{child_head}:{PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH}",
                )
                expected_digest = (
                    "sha256:" + hashlib.sha256(authorization.stdout).hexdigest()
                )
                if authorization.returncode != 0:
                    errors.append(
                        f"{prefix}.authorization: Git blob unavailable"
                    )
                else:
                    try:
                        authorization_payload = _load_json_bytes(
                            authorization.stdout,
                            name=f"P019-{PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME}",
                        )
                    except (ValueError, json.JSONDecodeError) as exc:
                        errors.append(f"{prefix}.authorization: {exc}")
                    else:
                        if authorization_payload.get("schema") != (
                            PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA
                        ):
                            errors.append(
                                f"{prefix}.authorization.schema: exact @2 required"
                            )
                parent_components = parent_manifest.get("components")
                if not isinstance(parent_components, Mapping):
                    errors.append(
                        f"{prefix}.parent_manifest.components: expected object"
                    )
                else:
                    expected_manifest = dict(parent_manifest)
                    expected_components = dict(parent_components)
                    expected_components[
                        PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME
                    ] = expected_digest
                    expected_manifest["components"] = expected_components
                    _validate_exact_structure(
                        errors,
                        prefix=f"{prefix}.manifest_transformation",
                        actual=child_manifest,
                        expected=expected_manifest,
                    )
            elif phase == "L":
                expected_manifest = dict(parent_manifest)
                expected_manifest["schema"] = RELOAD_CONVERGENCE_MANIFEST_SCHEMA
                expected_manifest["created_at"] = child_manifest.get("created_at")
                expected_manifest["reload"] = child_manifest.get("reload")
                _validate_exact_structure(
                    errors,
                    prefix=f"{prefix}.manifest_transformation",
                    actual=child_manifest,
                    expected=expected_manifest,
                )
                artifacts, artifact_errors = _phase_manifest_artifacts(
                    repo_root=repo,
                    head=child_head,
                    phase=phase,
                )
                errors.extend(artifact_errors)
                reload_binding = child_manifest.get("reload")
                expected_reload = {
                    "phase": "provider_attempt_daemon_reload",
                    "acceptance_head": parent_head,
                    "acceptance_tree": parent_tree,
                    "receipt": {
                        PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME: artifacts.get(
                            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
                        )
                    },
                    "task": _RELOAD_TASK_CONTRACT,
                    "accepted_task_statuses": {
                        task_id: "completed"
                        for task_id in SEQUENTIAL_ACCEPTANCE_TASK_IDS
                    },
                    "reload_gate_completed": True,
                    "launch_authorization_only": True,
                    "post_launch_birth_receipt_required": True,
                    "post_launch_birth_receipt_schema": (
                        PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
                    ),
                }
                _validate_exact_structure(
                    errors,
                    prefix=f"{prefix}.manifest.reload",
                    actual=reload_binding,
                    expected=expected_reload,
                )
            else:
                expected_manifest = dict(parent_manifest)
                expected_manifest["schema"] = ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
                expected_manifest["created_at"] = child_manifest.get("created_at")
                expected_manifest["acceptance"] = child_manifest.get("acceptance")
                _validate_exact_structure(
                    errors,
                    prefix=f"{prefix}.manifest_transformation",
                    actual=child_manifest,
                    expected=expected_manifest,
                )
                artifacts, artifact_errors = _phase_manifest_artifacts(
                    repo_root=repo,
                    head=child_head,
                    phase=phase,
                )
                errors.extend(artifact_errors)
                expected_acceptance = {
                    "phase": phase,
                    "parent_phase": parent_phase,
                    "parent_head": parent_head,
                    "parent_tree": parent_tree,
                    "parent_manifest_sha256": (
                        "sha256:"
                        + hashlib.sha256(parent_manifest_raw.stdout).hexdigest()
                    ),
                    "artifacts": artifacts,
                    "task_statuses": _sequential_task_statuses_after(phase),
                    "reload_gate_status": "blocked",
                    "pre_launch_authorization_only": phase == "P031",
                    "runtime_effect_claimed": (
                        SEQUENTIAL_PHASE_RUNTIME_EFFECT_CLAIMS[phase]
                    ),
                }
                _validate_exact_structure(
                    errors,
                    prefix=f"{prefix}.manifest.acceptance",
                    actual=child_manifest.get("acceptance"),
                    expected=expected_acceptance,
                )

    if consumed_child_blobs is not None:
        if set(consumed_child_blobs) != set(changed_paths):
            errors.append(f"{prefix}.consumed_blobs: exact changed-path population required")
        for relative_path in changed_paths:
            raw = consumed_child_blobs.get(relative_path)
            committed = _git_bytes(repo, "show", f"{child_head}:{relative_path}")
            if not isinstance(raw, bytes) or committed.returncode != 0 or raw != committed.stdout:
                errors.append(f"{prefix}.consumed_blobs.{relative_path}: Git bytes mismatch")
    return tuple(errors)


def validate_protected_acceptance_sequence(
    *,
    repo_root: Path | str,
    phase_heads: Mapping[str, str],
    through_phase: str = "L",
) -> tuple[str, ...]:
    """Validate the exact contiguous protected child chain through one phase."""

    errors: list[str] = []
    prefix = "protected_acceptance.sequence"
    through_index = _sequential_phase_index(through_phase)
    if through_index < 0:
        return (f"{prefix}.through_phase: unsupported phase",)
    expected_phases = SEQUENTIAL_ACCEPTANCE_PHASES[: through_index + 1]
    if set(phase_heads) != set(expected_phases):
        errors.append(f"{prefix}.phase_heads: exact contiguous population required")
        return tuple(errors)
    repo = Path(repo_root)
    for phase in expected_phases:
        head = phase_heads.get(phase)
        if not isinstance(head, str) or _HEX40.fullmatch(head) is None:
            errors.append(f"{prefix}.{phase}: expected lowercase 40-hex")
    if errors:
        return tuple(errors)
    q_head = phase_heads["Q"]
    errors.extend(
        _git_replacement_object_errors(
            repo,
            prefix=f"{prefix}.replacement_objects",
        )
    )
    q_manifest_raw = _git_bytes(
        repo,
        "show",
        f"{q_head}:{_CONVERGENCE_MANIFEST_RELATIVE_PATH}",
    )
    q_authorization_raw = _git_bytes(
        repo,
        "show",
        f"{q_head}:{PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH}",
    )
    if (
        q_manifest_raw.returncode != 0
        or len(q_manifest_raw.stdout) > MAX_EVIDENCE_SNAPSHOT_BYTES
    ):
        errors.append(f"{prefix}.Q.manifest: unavailable or oversized")
    else:
        try:
            q_manifest = _load_json_bytes(
                q_manifest_raw.stdout,
                name=f"Q-{MANIFEST_FILENAME}",
            )
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}.Q.manifest: {exc}")
        else:
            if q_manifest.get("schema") != CONVERGENCE_MANIFEST_SCHEMA:
                errors.append(f"{prefix}.Q.manifest.schema: exact @1 required")
            if "acceptance" in q_manifest or "reload" in q_manifest:
                errors.append(f"{prefix}.Q.manifest: effect phases are forbidden")
            components = q_manifest.get("components")
            expected_authorization_digest = (
                "sha256:" + hashlib.sha256(q_authorization_raw.stdout).hexdigest()
            )
            if (
                q_authorization_raw.returncode != 0
                or not isinstance(components, Mapping)
                or components.get(PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME)
                != expected_authorization_digest
            ):
                errors.append(
                    f"{prefix}.Q.manifest.components: Q authorization mismatch"
                )
    if (
        q_authorization_raw.returncode != 0
        or len(q_authorization_raw.stdout)
        > MAX_PROVIDER_FALLBACK_AUTHORIZATION_BYTES
    ):
        errors.append(f"{prefix}.Q.authorization: unavailable or oversized")
    else:
        try:
            q_authorization = _load_json_bytes(
                q_authorization_raw.stdout,
                name=f"Q-{PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME}",
            )
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}.Q.authorization: {exc}")
        else:
            if (
                q_authorization.get("schema")
                != PROVIDER_FALLBACK_POLICY_AUTHORIZATION_SCHEMA
            ):
                errors.append(f"{prefix}.Q.authorization.schema: exact @1 required")
    errors.extend(
        _validate_git_regular_modes(
            repo_root=repo,
            head=q_head,
            paths=(
                PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
                PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
                _CONVERGENCE_MANIFEST_RELATIVE_PATH,
            ),
            prefix=f"{prefix}.Q.modes",
        )
    )
    q_board = _git_bytes(
        repo,
        "show",
        f"{q_head}:{PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()}",
    )
    if q_board.returncode != 0:
        errors.append(f"{prefix}.Q.taskboard: unavailable")
    else:
        try:
            q_tasks = _parse_taskboard_metadata(q_board.stdout.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            errors.append(f"{prefix}.Q.taskboard: {exc}")
        else:
            observed = {
                task_id: q_tasks.get(task_id, {}).get("status", "")
                for task_id in _sequential_task_statuses_after("Q")
            }
            if observed != _sequential_task_statuses_after("Q"):
                errors.append(f"{prefix}.Q.task_statuses: exact dormant state required")
    for path in SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE:
        if _git(repo, "cat-file", "-e", f"{q_head}:{path}").returncode == 0:
            errors.append(f"{prefix}.Q.{path}: reserved artifact must be absent")
    for index in range(1, len(expected_phases)):
        phase = expected_phases[index]
        parent_phase = expected_phases[index - 1]
        parent_head = phase_heads[parent_phase]
        parent_tree_result = _git(
            repo,
            "rev-parse",
            "--verify",
            f"{parent_head}^{{tree}}",
        )
        parent_tree = parent_tree_result.stdout.strip()
        if parent_tree_result.returncode != 0 or _HEX40.fullmatch(parent_tree) is None:
            errors.append(f"{prefix}.{parent_phase}.tree: unavailable")
            continue
        errors.extend(
            validate_sequential_acceptance_child_transition(
                repo_root=repo,
                phase=phase,
                child_head=phase_heads[phase],
                parent_head=parent_head,
                parent_tree=parent_tree,
            )
        )
    return tuple(errors)
def _status_only_reload_board(parent_raw: bytes) -> bytes:
    try:
        text = parent_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("acceptance taskboard is not UTF-8") from exc
    current_task = ""
    replacements = 0
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            current_task = line[3:].strip().split(" ", 1)[0]
        if current_task == "ASE3-022" and line.rstrip("\r\n") == "- Status: blocked":
            newline = line[len(line.rstrip("\r\n")) :]
            line = f"- Status: completed{newline}"
            replacements += 1
        output.append(line)
    if replacements != 1:
        raise ValueError("acceptance taskboard lacks exact ASE3-022 blocked status")
    return "".join(output).encode("utf-8")


def _validate_qrp_transition(
    *,
    repo_root: Path,
    q_head: str,
    q_tree: str,
    r_head: str,
    p_head: str,
) -> list[str]:
    """Validate exact final-prep Q, root-pin R, and authority-prep P."""

    errors: list[str] = []
    prefix = "operator_acceptance.transition"
    errors.extend(
        _validate_exact_direct_child(
            repo_root=repo_root,
            parent=q_head,
            child=r_head,
            expected_paths=Q_TO_R_CHANGED_PATHS,
            prefix=f"{prefix}.Q_to_R",
        )
    )
    errors.extend(
        _validate_git_regular_modes(
            repo_root=repo_root,
            head=r_head,
            paths=Q_TO_R_CHANGED_PATHS,
            prefix=f"{prefix}.Q_to_R.modes",
        )
    )
    errors.extend(
        _validate_exact_direct_child(
            repo_root=repo_root,
            parent=r_head,
            child=p_head,
            expected_paths=R_TO_P_CHANGED_PATHS,
            prefix=f"{prefix}.R_to_P",
        )
    )
    errors.extend(
        _validate_git_regular_modes(
            repo_root=repo_root,
            head=p_head,
            paths=R_TO_P_CHANGED_PATHS,
            prefix=f"{prefix}.R_to_P.modes",
        )
    )
    actual_q_tree = _git(repo_root, "rev-parse", "--verify", f"{q_head}^{{tree}}")
    if actual_q_tree.returncode != 0 or actual_q_tree.stdout.strip() != q_tree:
        errors.append(f"{prefix}.Q.tree: Git tree mismatch")

    future_paths = (
        LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH,
        LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH,
        *OPERATOR_ACCEPTANCE_RECEIPT_RELATIVE_PATHS,
        PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
        PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH,
    )
    for path in future_paths:
        for label, head in (("Q", q_head), ("R", r_head), ("P", p_head)):
            should_exist = (
                path == LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
                and label in {"R", "P"}
            ) or (
                path == LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
                and label == "P"
            )
            observed = _git(repo_root, "cat-file", "-e", f"{head}:{path}")
            if (observed.returncode == 0) is not should_exist:
                expectation = "present" if should_exist else "absent"
                errors.append(f"{prefix}.{label}.{path}: expected {expectation}")

    authorization_path = PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
    authorization_blobs: dict[str, bytes] = {}
    for label, head in (("Q", q_head), ("R", r_head), ("P", p_head)):
        blob = _git_bytes(repo_root, "show", f"{head}:{authorization_path}")
        if blob.returncode != 0 or len(blob.stdout) > (
            MAX_PROVIDER_FALLBACK_AUTHORIZATION_BYTES
        ):
            errors.append(f"{prefix}.{label}.authorization: unavailable or oversized")
            continue
        authorization_blobs[label] = blob.stdout
        try:
            authorization = _load_json_bytes(
                blob.stdout,
                name=f"{label}-provider-fallback-authorization.json",
            )
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}.{label}.authorization: {exc}")
        else:
            expected_schema = (
                PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA
                if label == "P"
                else PROVIDER_FALLBACK_POLICY_AUTHORIZATION_SCHEMA
            )
            if authorization.get("schema") != expected_schema:
                errors.append(
                    f"{prefix}.{label}.authorization: exact schema required"
                )
    if authorization_blobs.get("Q") != authorization_blobs.get("R"):
        errors.append(f"{prefix}.Q_to_R.authorization: @1 bytes changed")

    manifest_path = f"{_CONVERGENCE_RELATIVE_ROOT}/{MANIFEST_FILENAME}"
    manifests: dict[str, Mapping[str, Any]] = {}
    manifest_raw: dict[str, bytes] = {}
    for label, head in (("Q", q_head), ("R", r_head), ("P", p_head)):
        blob = _git_bytes(repo_root, "show", f"{head}:{manifest_path}")
        if blob.returncode != 0 or len(blob.stdout) > MAX_EVIDENCE_SNAPSHOT_BYTES:
            errors.append(f"{prefix}.{label}.manifest: unavailable or oversized")
            continue
        manifest_raw[label] = blob.stdout
        try:
            manifest = _load_json_bytes(blob.stdout, name=f"{label}-{MANIFEST_FILENAME}")
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}.{label}.manifest: {exc}")
        else:
            manifests[label] = manifest
            if manifest.get("schema") != CONVERGENCE_MANIFEST_SCHEMA:
                errors.append(f"{prefix}.{label}.manifest: exact @1 required")
    if manifest_raw.get("Q") != manifest_raw.get("R"):
        errors.append(f"{prefix}.Q_to_R.manifest: bytes changed")
    q_manifest = manifests.get("Q")
    p_manifest = manifests.get("P")
    p_authorization = authorization_blobs.get("P")
    if q_manifest is not None and p_manifest is not None and p_authorization is not None:
        expected_p = dict(q_manifest)
        components = q_manifest.get("components")
        if not isinstance(components, Mapping):
            errors.append(f"{prefix}.Q.manifest.components: expected object")
        else:
            expected_components = dict(components)
            expected_components[PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME] = (
                "sha256:" + hashlib.sha256(p_authorization).hexdigest()
            )
            expected_p["components"] = expected_components
            _validate_exact_structure(
                errors,
                prefix=f"{prefix}.R_to_P.manifest",
                actual=p_manifest,
                expected=expected_p,
            )

    board_path = PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
    q_board = _git_bytes(repo_root, "show", f"{q_head}:{board_path}")
    p_board = _git_bytes(repo_root, "show", f"{p_head}:{board_path}")
    if q_board.returncode != 0 or p_board.returncode != 0:
        errors.append(f"{prefix}.Q.taskboard: unavailable")
    elif q_board.stdout != p_board.stdout:
        errors.append(f"{prefix}.Q_to_P.taskboard: final-prep bytes changed")
    else:
        try:
            q_tasks = _parse_taskboard_metadata(q_board.stdout.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            errors.append(f"{prefix}.Q.taskboard: {exc}")
        else:
            for task_id in _ACCEPTANCE_TASK_CONTRACTS:
                if q_tasks.get(task_id, {}).get("status") != "todo":
                    errors.append(f"{prefix}.Q.{task_id}.status: expected todo")
            if q_tasks.get("ASE3-022", {}).get("status") != "blocked":
                errors.append(f"{prefix}.Q.ASE3-022.status: expected blocked")
    return errors


def validate_acceptance_child_transition(
    *,
    repo_root: Path | str,
    acceptance_head: str,
    preparation_head: str,
    preparation_tree: str,
    consumed_acceptance_blobs: Mapping[str, bytes] | None = None,
    lifecycle_root_pin_raw: bytes | None = None,
    lifecycle_witness_raw: bytes | None = None,
    fallback_authorization_raw: bytes | None = None,
    expected_root_identity_did: str | None = None,
    expected_final_values: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Reject the retired one-shot P-to-A fan-in entry point."""

    return (
        (
            "operator_acceptance.transition: obsolete atomic fan-in forbidden; "
            "use validate_sequential_acceptance_child_transition"
        ),
    )


def _obsolete_validate_acceptance_child_transition(
    *,
    repo_root: Path | str,
    acceptance_head: str,
    preparation_head: str,
    preparation_tree: str,
    consumed_acceptance_blobs: Mapping[str, bytes] | None = None,
    lifecycle_root_pin_raw: bytes | None = None,
    lifecycle_witness_raw: bytes | None = None,
    fallback_authorization_raw: bytes | None = None,
    expected_root_identity_did: str | None = None,
    expected_final_values: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Compatibility shim that remains fail-closed after fan-in retirement."""

    del (
        consumed_acceptance_blobs,
        lifecycle_root_pin_raw,
        lifecycle_witness_raw,
        fallback_authorization_raw,
        expected_root_identity_did,
        expected_final_values,
    )
    return validate_acceptance_child_transition(
        repo_root=repo_root,
        acceptance_head=acceptance_head,
        preparation_head=preparation_head,
        preparation_tree=preparation_tree,
    )


def validate_reload_child_transition(
    *,
    repo_root: Path | str,
    reload_head: str,
    acceptance_head: str,
    acceptance_tree: str,
    consumed_reload_blobs: Mapping[str, bytes] | None = None,
) -> tuple[str, ...]:
    """Compatibility entry for the exact A023/027-to-L sequential child."""

    return validate_sequential_acceptance_child_transition(
        repo_root=repo_root,
        phase="L",
        child_head=reload_head,
        parent_head=acceptance_head,
        parent_tree=acceptance_tree,
        consumed_child_blobs=consumed_reload_blobs,
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

    def validate(
        self,
        *,
        lifecycle_witness: LocalOperatorLifecycleWitnessSnapshot | None = None,
        root_pin: LocalProfileLifecycleRootPinSnapshot | None = None,
        expected_source_head: str | None = None,
        expected_source_tree: str | None = None,
        expected_final_values: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        prefix = "provider_fallback_policy_authorization"
        schema = self.payload.get("schema")
        if schema == PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA:
            if set(self.payload) != set(
                _PROVIDER_FALLBACK_AUTHORIZATION_V2_REQUIRED_FIELDS
            ):
                errors.append(f"{prefix}: field population mismatch")
            if self.payload.get("board_namespace") != BOARD_NAMESPACE:
                errors.append(f"{prefix}.board_namespace: mismatch")
            errors.extend(
                self._validate_v2(
                    lifecycle_witness=lifecycle_witness,
                    root_pin=root_pin,
                    expected_source_head=expected_source_head,
                    expected_source_tree=expected_source_tree,
                    expected_final_values=expected_final_values,
                )
            )
            return tuple(errors)

        expected_fields = _PROVIDER_FALLBACK_AUTHORIZATION_V1_REQUIRED_FIELDS
        if set(self.payload) != set(expected_fields):
            errors.append(f"{prefix}: field population mismatch")
        if schema != PROVIDER_FALLBACK_POLICY_AUTHORIZATION_SCHEMA:
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
            ("authorization_source", _PROVIDER_FALLBACK_AUTHORIZATION_SOURCE),
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

    def _validate_v2(
        self,
        *,
        lifecycle_witness: LocalOperatorLifecycleWitnessSnapshot | None,
        root_pin: LocalProfileLifecycleRootPinSnapshot | None,
        expected_source_head: str | None,
        expected_source_tree: str | None,
        expected_final_values: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        prefix = "provider_fallback_policy_authorization"
        expected_source = dict(_PROVIDER_FALLBACK_AUTHORIZATION_SOURCE)
        expected_source["source_head"] = (
            self.source_head if expected_source_head is None else expected_source_head
        )
        expected_source["source_tree"] = (
            self.source_tree if expected_source_tree is None else expected_source_tree
        )
        for field, expected in (
            ("authorization_source", expected_source),
            ("route", _PROVIDER_FALLBACK_AUTHORIZATION_ROUTE),
            (
                "ownership_contract",
                _PROVIDER_FALLBACK_AUTHORIZATION_V2_OWNERSHIP_CONTRACT,
            ),
            (
                "bootstrap_route_guarantees",
                _PROVIDER_FALLBACK_AUTHORIZATION_V2_BOOTSTRAP_GUARANTEES,
            ),
        ):
            _validate_exact_policy_object(
                errors,
                prefix=f"{prefix}.{field}",
                actual=self.payload.get(field),
                expected=expected,
            )

        reviewer = _require_exact_keys(
            errors,
            prefix=f"{prefix}.reviewer",
            value=self.payload.get("reviewer"),
            expected=_PROVIDER_FALLBACK_AUTHORIZATION_V2_REVIEWER_FIELDS,
        )
        reviewer_identity: Any = None
        if reviewer is not None:
            reviewer_identity = reviewer.get("identity")
            try:
                _ed25519_public_key_from_did_key(reviewer_identity)
            except ValueError as exc:
                errors.append(f"{prefix}.reviewer.identity: {exc}")
            if reviewer.get("provider") != "local_operator":
                errors.append(
                    f"{prefix}.reviewer.provider: expected 'local_operator'"
                )
            _require_trimmed_string(
                errors,
                prefix=f"{prefix}.reviewer.profile_id",
                value=reviewer.get("profile_id"),
            )
            _require_sha256(
                errors,
                f"{prefix}.reviewer.profile_content_id",
                reviewer.get("profile_content_id"),
            )
            anchor_id = reviewer.get("lifecycle_anchor_id")
            if not isinstance(anchor_id, str) or _HEX64.fullmatch(anchor_id) is None:
                errors.append(
                    f"{prefix}.reviewer.lifecycle_anchor_id: expected lowercase 64-hex"
                )
            _require_exact_integer(
                errors,
                prefix=f"{prefix}.reviewer.generation",
                value=reviewer.get("generation"),
                minimum=1,
            )
            if reviewer.get("witness_path") != (
                LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
            ):
                errors.append(
                    f"{prefix}.reviewer.witness_path: protected path mismatch"
                )
            _require_sha256(
                errors,
                f"{prefix}.reviewer.witness_sha256",
                reviewer.get("witness_sha256"),
            )

        bounds = _require_exact_keys(
            errors,
            prefix=f"{prefix}.authority_bounds",
            value=self.payload.get("authority_bounds"),
            expected=_PROVIDER_FALLBACK_AUTHORIZATION_V2_AUTHORITY_BOUNDS_FIELDS,
        )
        if bounds is not None:
            for field in (
                "repository_cid",
                "budget_cid",
                "resource_cid",
                "authority_cid",
            ):
                _require_trimmed_string(
                    errors,
                    prefix=f"{prefix}.authority_bounds.{field}",
                    value=bounds.get(field),
                )
            _require_hex40(
                errors,
                f"{prefix}.authority_bounds.baseline_commit",
                bounds.get("baseline_commit"),
            )
            effects = _require_sorted_unique_string_array(
                errors,
                prefix=f"{prefix}.authority_bounds.effects",
                value=bounds.get("effects"),
            )
            if effects != _PROVIDER_FALLBACK_AUTHORIZATION_V2_EFFECTS:
                errors.append(
                    f"{prefix}.authority_bounds.effects: exact scoped effects required"
                )
            if bounds.get("baseline_commit") != expected_source["source_head"]:
                errors.append(
                    f"{prefix}.authority_bounds.baseline_commit: "
                    "authorization source mismatch"
                )
        if self.payload.get("fallback_implementer_identity") != "codex":
            errors.append(
                f"{prefix}.fallback_implementer_identity: expected 'codex'"
            )
        if self.payload.get("lifecycle_root_pin_path") != (
            LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
        ):
            errors.append(f"{prefix}.lifecycle_root_pin_path: protected path mismatch")
        _require_sha256(
            errors,
            f"{prefix}.lifecycle_root_pin_sha256",
            self.payload.get("lifecycle_root_pin_sha256"),
        )
        _require_trimmed_string(
            errors,
            prefix=f"{prefix}.lifecycle_witness_nonce",
            value=self.payload.get("lifecycle_witness_nonce"),
            maximum=512,
        )
        try:
            _ed25519_public_key_from_did_key(
                self.payload.get("lifecycle_root_identity_did")
            )
        except ValueError as exc:
            errors.append(f"{prefix}.lifecycle_root_identity_did: {exc}")
        authorized_at = _require_exact_integer(
            errors,
            prefix=f"{prefix}.authorized_at_ms",
            value=self.payload.get("authorized_at_ms"),
            minimum=1,
            maximum=10**16,
        )

        final_values = (
            _ACCEPTANCE_REVIEWER_FINAL_VALUES
            if expected_final_values is None
            else expected_final_values
        )
        final_equalities = {
            "identity": "reviewer_identity",
            "profile_id": "profile_id",
            "profile_content_id": "profile_content_id",
            "lifecycle_anchor_id": "lifecycle_anchor_id",
            "generation": "lifecycle_generation",
        }
        for reviewer_field, final_field in final_equalities.items():
            expected = final_values.get(final_field)
            if _is_unpopulated_final_value(expected):
                errors.append(
                    f"{prefix}.reviewer.{reviewer_field}: final pin is not populated"
                )
            elif reviewer is not None and reviewer.get(reviewer_field) != expected:
                errors.append(
                    f"{prefix}.reviewer.{reviewer_field}: final pin mismatch"
                )

        if lifecycle_witness is not None:
            witness = lifecycle_witness.payload
            profile = witness.get("profile")
            anchor = witness.get("anchor")
            if not isinstance(profile, Mapping) or not isinstance(anchor, Mapping):
                errors.append(f"{prefix}: lifecycle witness projections unavailable")
            else:
                witness_equalities = {
                    "identity": profile.get("identity_did"),
                    "profile_id": profile.get("profile_id"),
                    "profile_content_id": witness.get("profile_content_id"),
                    "lifecycle_anchor_id": anchor.get("anchor_id"),
                    "generation": profile.get("lifecycle_generation"),
                    "witness_sha256": lifecycle_witness.sha256,
                }
                for field, expected in witness_equalities.items():
                    if reviewer is not None and reviewer.get(field) != expected:
                        errors.append(
                            f"{prefix}.reviewer.{field}: witness equality mismatch"
                        )
                top_equalities = {
                    "lifecycle_witness_nonce": witness.get("nonce"),
                    "lifecycle_root_identity_did": witness.get(
                        "root_identity_did"
                    ),
                }
                for field, expected in top_equalities.items():
                    if self.payload.get(field) != expected:
                        errors.append(f"{prefix}.{field}: witness equality mismatch")
                if bounds is not None:
                    profile_bounds = {
                        "repository_cid": profile.get("repository_cid"),
                        "baseline_commit": profile.get("baseline_commit"),
                        "effects": profile.get("effect_bounds"),
                        "budget_cid": profile.get("budget_cid"),
                        "resource_cid": profile.get("resource_cid"),
                        "authority_cid": witness.get("profile_content_id"),
                    }
                    for field, expected in profile_bounds.items():
                        if bounds.get(field) != expected:
                            errors.append(
                                f"{prefix}.authority_bounds.{field}: "
                                "profile equality mismatch"
                            )
                observed = witness.get("observed_at_ms")
                expires = witness.get("expires_at_ms")
                if (
                    authorized_at is not None
                    and type(observed) is int
                    and type(expires) is int
                    and not observed <= authorized_at <= expires
                ):
                    errors.append(
                        f"{prefix}.authorized_at_ms: outside witness validity"
                    )
        if root_pin is not None:
            root_equalities = {
                "lifecycle_root_identity_did": root_pin.root_identity_did,
                "lifecycle_root_pin_sha256": root_pin.sha256,
            }
            for field, expected in root_equalities.items():
                if self.payload.get(field) != expected:
                    errors.append(f"{prefix}.{field}: root-pin equality mismatch")

        source = self.payload.get("authorization_source")
        route = self.payload.get("route")
        if reviewer is not None and isinstance(source, Mapping) and isinstance(
            route, Mapping
        ) and bounds is not None:
            review_payload = {
                "schema": PROVIDER_FALLBACK_POLICY_REVIEW_V2_SCHEMA,
                "board_namespace": BOARD_NAMESPACE,
                "authorization_source": {
                    field: source.get(field)
                    for field in ("kind", "source_head", "source_tree")
                },
                "route": dict(route),
                "authority_bounds": dict(bounds),
                "reviewer": {
                    field: reviewer.get(field)
                    for field in _PROVIDER_FALLBACK_AUTHORIZATION_V2_REVIEWER_FIELDS
                    if field != "signature"
                },
                "lifecycle_root_identity_did": self.payload.get(
                    "lifecycle_root_identity_did"
                ),
                "lifecycle_witness_nonce": self.payload.get(
                    "lifecycle_witness_nonce"
                ),
                "lifecycle_root_pin_path": self.payload.get(
                    "lifecycle_root_pin_path"
                ),
                "lifecycle_root_pin_sha256": self.payload.get(
                    "lifecycle_root_pin_sha256"
                ),
                "authorized_at_ms": self.payload.get("authorized_at_ms"),
                "fallback_implementer_identity": self.payload.get(
                    "fallback_implementer_identity"
                ),
            }
            _verify_standard_ed25519_signature(
                errors,
                prefix=f"{prefix}.reviewer.signature",
                signer_identity_did=reviewer_identity,
                signature_token=reviewer.get("signature"),
                message=_canonical_json_bytes(review_payload),
            )
        return tuple(errors)

    def authorization_id(self, *, raw_sha256: str) -> str:
        source = self.payload.get("authorization_source")
        reviewer = self.payload.get("reviewer")
        bounds = self.payload.get("authority_bounds")
        if not all(isinstance(item, Mapping) for item in (source, reviewer, bounds)):
            return ""
        assert isinstance(source, Mapping)
        assert isinstance(reviewer, Mapping)
        assert isinstance(bounds, Mapping)
        material = {
            "schema": PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA,
            "board_namespace": self.payload.get("board_namespace"),
            "artifact_path": PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
            "artifact_sha256": raw_sha256,
            "authorization_kind": source.get("kind"),
            "source_head": source.get("source_head"),
            "source_tree": source.get("source_tree"),
            "reviewer_identity": reviewer.get("identity"),
            "reviewer_provider": reviewer.get("provider"),
            "reviewer_signature": reviewer.get("signature"),
            "reviewer_profile_id": reviewer.get("profile_id"),
            "reviewer_profile_content_id": reviewer.get("profile_content_id"),
            "reviewer_lifecycle_anchor_id": reviewer.get("lifecycle_anchor_id"),
            "reviewer_lifecycle_generation": reviewer.get("generation"),
            "reviewer_witness_path": reviewer.get("witness_path"),
            "reviewer_witness_sha256": reviewer.get("witness_sha256"),
            "lifecycle_root_identity_did": self.payload.get(
                "lifecycle_root_identity_did"
            ),
            "lifecycle_witness_nonce": self.payload.get(
                "lifecycle_witness_nonce"
            ),
            "lifecycle_root_pin_path": self.payload.get(
                "lifecycle_root_pin_path"
            ),
            "lifecycle_root_pin_sha256": self.payload.get(
                "lifecycle_root_pin_sha256"
            ),
            "authorized_at_ms": self.payload.get("authorized_at_ms"),
            "fallback_implementer_identity": self.payload.get(
                "fallback_implementer_identity"
            ),
            "authority_bounds": dict(bounds),
            "authorization_id": "",
        }
        encoded = json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def acceptance_review_authority(
        self,
        *,
        raw_sha256: str,
        lifecycle_witness: LocalOperatorLifecycleWitnessSnapshot,
        root_pin: LocalProfileLifecycleRootPinSnapshot,
    ) -> Mapping[str, Any]:
        reviewer = self.payload.get("reviewer")
        if not isinstance(reviewer, Mapping):
            return {}
        return {
            "reviewer_identity": reviewer.get("identity"),
            "reviewer_provider": reviewer.get("provider"),
            "profile_id": reviewer.get("profile_id"),
            "profile_content_id": reviewer.get("profile_content_id"),
            "lifecycle_anchor_id": reviewer.get("lifecycle_anchor_id"),
            "lifecycle_anchor_digest": lifecycle_witness.payload.get(
                "anchor_digest"
            ),
            "lifecycle_generation": reviewer.get("generation"),
            "lifecycle_witness_path": (
                LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
            ),
            "lifecycle_witness_sha256": lifecycle_witness.sha256,
            "lifecycle_witness_id": lifecycle_witness.witness_id,
            "lifecycle_witness_nonce": lifecycle_witness.payload.get("nonce"),
            "lifecycle_root_pin_path": (
                LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
            ),
            "lifecycle_root_pin_sha256": root_pin.sha256,
            "lifecycle_root_identity_did": root_pin.root_identity_did,
            "fallback_authorization_id": self.authorization_id(
                raw_sha256=raw_sha256
            ),
            "fallback_authorization_sha256": raw_sha256,
            "lifecycle_witness_observed_at_ms": lifecycle_witness.payload.get(
                "observed_at_ms"
            ),
            "lifecycle_witness_expires_at_ms": lifecycle_witness.payload.get(
                "expires_at_ms"
            ),
            "fallback_authorized_at_ms": self.payload.get("authorized_at_ms"),
        }


@dataclass(frozen=True)
class ConvergenceManifest:
    """Root binding for the bounded ASE3-000 evidence packet."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConvergenceManifest:
        return cls(dict(payload))

    def validate(self, baseline: CurrentMainBaseline) -> tuple[str, ...]:
        errors: list[str] = []
        manifest_schema = self.payload.get("schema")
        if manifest_schema not in {
            CONVERGENCE_MANIFEST_SCHEMA,
            ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            RELOAD_CONVERGENCE_MANIFEST_SCHEMA,
        }:
            errors.append("convergence_manifest.schema: unsupported schema")
        elif manifest_schema == CONVERGENCE_MANIFEST_SCHEMA:
            if set(self.payload) != set(
                _CONVERGENCE_MANIFEST_V1_TOP_LEVEL_FIELDS
            ):
                errors.append(
                    "convergence_manifest: exact @1 top-level population required"
                )
        elif manifest_schema == ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA and set(
            self.payload
        ) != set(_CONVERGENCE_MANIFEST_V2_TOP_LEVEL_FIELDS):
            errors.append(
                "convergence_manifest: exact @2 top-level population required"
            )
        elif manifest_schema == RELOAD_CONVERGENCE_MANIFEST_SCHEMA and set(
            self.payload
        ) != set(_CONVERGENCE_MANIFEST_V3_TOP_LEVEL_FIELDS):
            errors.append(
                "convergence_manifest: exact @3 top-level population required"
            )
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("convergence_manifest.board_namespace: mismatch")
        if self.payload.get("task_id") != "ASE3-000":
            errors.append("convergence_manifest.task_id: expected ASE3-000")
        if self.payload.get("goal_id") != "ASE3-G010":
            errors.append("convergence_manifest.goal_id: expected ASE3-G010")
        created_at = self.payload.get("created_at")
        if not isinstance(created_at, str) or _UTC_TIMESTAMP.fullmatch(created_at) is None:
            errors.append("convergence_manifest.created_at: expected UTC timestamp")
        elif (
            manifest_schema == CONVERGENCE_MANIFEST_SCHEMA
            and created_at != CONVERGENCE_MANIFEST_CREATED_AT
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
        if manifest_schema in {
            ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            RELOAD_CONVERGENCE_MANIFEST_SCHEMA,
        }:
            acceptance = _require_exact_keys(
                errors,
                prefix="convergence_manifest.acceptance",
                value=self.payload.get("acceptance"),
                expected=_ACCEPTANCE_MANIFEST_REQUIRED_FIELDS,
            )
            if acceptance is not None:
                acceptance_phase = acceptance.get("phase")
                allowed_acceptance_phases = {
                    "A019",
                    "A030",
                    "P031",
                    "A031",
                    "A032",
                    "A023/027",
                }
                if acceptance_phase not in allowed_acceptance_phases:
                    errors.append("convergence_manifest.acceptance.phase: exact sequential phase required")
                    acceptance_phase = ""
                expected_parent_phase = SEQUENTIAL_PHASE_PARENT.get(
                    str(acceptance_phase),
                    "",
                )
                if acceptance.get("parent_phase") != expected_parent_phase:
                    errors.append(
                        "convergence_manifest.acceptance.parent_phase: "
                        "direct prior phase required"
                    )
                _require_hex40(
                    errors,
                    "convergence_manifest.acceptance.parent_head",
                    acceptance.get("parent_head"),
                )
                _require_hex40(
                    errors,
                    "convergence_manifest.acceptance.parent_tree",
                    acceptance.get("parent_tree"),
                )
                _require_sha256(
                    errors,
                    "convergence_manifest.acceptance.parent_manifest_sha256",
                    acceptance.get("parent_manifest_sha256"),
                )
                artifact_bindings = acceptance.get("artifacts")
                expected_artifact_paths = {
                    path
                    for path in SEQUENTIAL_PHASE_CHANGED_PATHS.get(
                        str(acceptance_phase),
                        (),
                    )
                    if path not in {
                        _CONVERGENCE_MANIFEST_RELATIVE_PATH,
                        PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
                    }
                }
                if not isinstance(artifact_bindings, Mapping) or set(
                    artifact_bindings
                ) != expected_artifact_paths:
                    errors.append(
                        "convergence_manifest.acceptance.artifacts: exact phase population required"
                    )
                elif isinstance(artifact_bindings, Mapping):
                    for path, digest in artifact_bindings.items():
                        _require_sha256(
                            errors,
                            f"convergence_manifest.acceptance.artifacts.{path}",
                            digest,
                        )
                _validate_exact_structure(
                    errors,
                    prefix="convergence_manifest.acceptance.task_statuses",
                    actual=acceptance.get("task_statuses"),
                    expected=_sequential_task_statuses_after(str(acceptance_phase)),
                )
                if acceptance.get("reload_gate_status") != "blocked":
                    errors.append(
                        "convergence_manifest.acceptance.reload_gate_status: "
                        "must be blocked"
                    )
                expected_authorization_only = acceptance_phase == "P031"
                if acceptance.get("pre_launch_authorization_only") is not (
                    expected_authorization_only
                ):
                    errors.append(
                        "convergence_manifest.acceptance.pre_launch_authorization_only: "
                        "phase mismatch"
                    )
                expected_effect = SEQUENTIAL_PHASE_RUNTIME_EFFECT_CLAIMS.get(
                    str(acceptance_phase)
                )
                if acceptance.get("runtime_effect_claimed") is not expected_effect:
                    errors.append(
                        "convergence_manifest.acceptance.runtime_effect_claimed: "
                        "phase mismatch"
                    )
        elif "acceptance" in self.payload:
            errors.append("convergence_manifest.acceptance: forbidden in preparation @1")
        if manifest_schema == RELOAD_CONVERGENCE_MANIFEST_SCHEMA:
            reload_binding = _require_exact_keys(
                errors,
                prefix="convergence_manifest.reload",
                value=self.payload.get("reload"),
                expected=_RELOAD_MANIFEST_REQUIRED_FIELDS,
            )
            if reload_binding is not None:
                if reload_binding.get("phase") != "provider_attempt_daemon_reload":
                    errors.append(
                        "convergence_manifest.reload.phase: exact phase required"
                    )
                _require_hex40(
                    errors,
                    "convergence_manifest.reload.acceptance_head",
                    reload_binding.get("acceptance_head"),
                )
                _require_hex40(
                    errors,
                    "convergence_manifest.reload.acceptance_tree",
                    reload_binding.get("acceptance_tree"),
                )
                receipt = reload_binding.get("receipt")
                if not isinstance(receipt, Mapping) or set(receipt) != {
                    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
                }:
                    errors.append(
                        "convergence_manifest.reload.receipt: exact population required"
                    )
                elif isinstance(receipt, Mapping):
                    _require_sha256(
                        errors,
                        "convergence_manifest.reload.receipt."
                        + PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
                        receipt.get(PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME),
                    )
                _validate_exact_structure(
                    errors,
                    prefix="convergence_manifest.reload.task",
                    actual=reload_binding.get("task"),
                    expected=_RELOAD_TASK_CONTRACT,
                )
                _validate_exact_structure(
                    errors,
                    prefix="convergence_manifest.reload.accepted_task_statuses",
                    actual=reload_binding.get("accepted_task_statuses"),
                    expected={
                        task_id: "completed"
                        for task_id in SEQUENTIAL_ACCEPTANCE_TASK_IDS
                    },
                )
                if reload_binding.get("reload_gate_completed") is not True:
                    errors.append(
                        "convergence_manifest.reload.reload_gate_completed: must be true"
                    )
                if reload_binding.get("launch_authorization_only") is not True:
                    errors.append(
                        "convergence_manifest.reload.launch_authorization_only: "
                        "must be true"
                    )
                if (
                    reload_binding.get("post_launch_birth_receipt_required")
                    is not True
                ):
                    errors.append(
                        "convergence_manifest.reload.post_launch_birth_receipt_required: "
                        "must be true"
                    )
                if reload_binding.get("post_launch_birth_receipt_schema") != (
                    PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
                ):
                    errors.append(
                        "convergence_manifest.reload.post_launch_birth_receipt_schema: "
                        "exact separate schema required"
                    )
        elif "reload" in self.payload:
            errors.append("convergence_manifest.reload: forbidden before @3")
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


def _load_taskboard_snapshot(
    taskboard_path: Path,
) -> tuple[bytes, dict[str, dict[str, str]]]:
    """Read and parse one stable regular nonsymlink board snapshot."""

    raw = _read_regular_bytes(taskboard_path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{taskboard_path.name}: expected UTF-8 Markdown") from exc
    return raw, _parse_taskboard_metadata(text)


def _load_taskboard_metadata(taskboard_path: Path) -> dict[str, dict[str, str]]:
    """Read one regular nonsymlink board snapshot and reject malformed UTF-8."""

    return _load_taskboard_snapshot(taskboard_path)[1]


def _validate_protected_task_block_bytes(
    text: str,
    *,
    phase: str = "Q",
) -> list[str]:
    """Keep protected blocks Q-sealed except for the exact phase Status field."""

    errors: list[str] = []
    prefix = "protected_task_block_bytes"
    expected_statuses = _sequential_task_statuses_after(phase)
    q_statuses = _sequential_task_statuses_after("Q")
    if not expected_statuses:
        return [f"{prefix}.phase: unsupported sequential phase"]
    for task_id, expected_sha256 in _PROTECTED_TASK_BLOCK_SHA256S.items():
        heading = f"## {task_id} "
        if text.count(heading) != 1:
            errors.append(f"{prefix}.{task_id}: expected exactly one heading")
            continue
        start = text.index(heading)
        end = text.find("\n## ASE3-", start + len(heading))
        if end < 0:
            errors.append(f"{prefix}.{task_id}: missing following task boundary")
            continue
        block = text[start:end] + "\n"
        status_matches = list(
            re.finditer(r"(?m)^- Status: ([^\r\n]+)$", block)
        )
        if len(status_matches) != 1:
            errors.append(f"{prefix}.{task_id}: expected exactly one Status field")
            continue
        observed_status = status_matches[0].group(1).strip().lower()
        expected_status = expected_statuses[task_id]
        if observed_status != expected_status:
            errors.append(
                f"{prefix}.{task_id}.status: exact {phase} phase status required"
            )
        match = status_matches[0]
        normalized = (
            block[: match.start()]
            + f"- Status: {q_statuses[task_id]}"
            + block[match.end() :]
        ).encode("utf-8")
        if hashlib.sha256(normalized).hexdigest() != expected_sha256:
            errors.append(f"{prefix}.{task_id}: protected task block bytes changed")
    return errors


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


def _mapping_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one JSON-compatible policy mapping with the sealed canonical form."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _exact_json_contract_value(actual: Any, expected: Any) -> bool:
    """Compare parsed JSON without Python's bool/int equality aliasing."""

    if type(expected) is bool:
        return type(actual) is bool and actual is expected
    if type(actual) is not type(expected):
        return False
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _exact_json_contract_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _exact_json_contract_value(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    return bool(actual == expected)


def _normalized_markdown_section_contract_sha256(
    text: str,
    *,
    section_heading: str,
    containing_heading: str,
    end_heading: str,
) -> str:
    """Hash one uniquely bounded Markdown section after whitespace normalization."""

    if text.count(section_heading) != 1:
        raise ValueError(f"expected exactly one heading {section_heading!r}")
    containing_start = text.find(containing_heading)
    section_start = text.find(section_heading)
    section_end = text.find(end_heading, section_start + len(section_heading))
    if (
        containing_start < 0
        or section_start <= containing_start
        or section_end <= section_start
    ):
        raise ValueError("section is not inside its required heading boundary")
    bounded = text[section_start:section_end]
    normalized = re.sub(r"\s+", " ", bounded).strip()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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


def _operator_acceptance_phase(
    *,
    tasks: Mapping[str, Mapping[str, str]],
    artifact_root: Path,
    manifest: ConvergenceManifest,
) -> tuple[str, list[str]]:
    """Classify one exact dormant or sequential protected phase."""

    errors: list[str] = []
    prefix = "operator_acceptance.phase"
    relative_to_filename = {
        path: PurePosixPath(path).name
        for path in SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE
    }
    present: set[str] = set()
    for relative_path, filename in relative_to_filename.items():
        try:
            (artifact_root / filename).lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(
                f"{prefix}.{filename}: unable to inspect reserved path: {exc}"
            )
            present.add(relative_path)
        else:
            present.add(relative_path)
    birth_path = artifact_root / PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_FILENAME
    try:
        birth_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        errors.append(f"{prefix}.birth: unable to inspect reserved path: {exc}")
    else:
        errors.append(
            f"{prefix}.{PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_FILENAME}: "
            "post-launch birth evidence is forbidden in every protected "
            "pre-effect phase through L"
        )

    matching_phases = [
        phase
        for phase in SEQUENTIAL_ACCEPTANCE_PHASES
        if present
        == {
            path
            for path, introduction in (
                SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE.items()
            )
            if _sequential_phase_index(introduction)
            <= _sequential_phase_index(phase)
        }
    ]
    if len(matching_phases) != 1:
        errors.append(
            f"{prefix}.artifacts: no exact sequential phase population matches"
        )
        return "invalid", errors
    phase = matching_phases[0]
    observed_statuses = {
        task_id: str(tasks.get(task_id, {}).get("status", "")).strip().lower()
        for task_id in _sequential_task_statuses_after(phase)
    }
    expected_statuses = _sequential_task_statuses_after(phase)
    if observed_statuses != expected_statuses:
        errors.append(f"{prefix}.{phase}.task_statuses: exact phase order required")
    manifest_schema = manifest.payload.get("schema")
    expected_schema = (
        CONVERGENCE_MANIFEST_SCHEMA
        if phase in {"Q", "R", "P019"}
        else (
            RELOAD_CONVERGENCE_MANIFEST_SCHEMA
            if phase == "L"
            else ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
        )
    )
    if manifest_schema != expected_schema:
        errors.append(f"{prefix}.{phase}.manifest_schema: phase mismatch")
    if phase not in {"Q", "R", "P019", "L"}:
        acceptance = manifest.payload.get("acceptance")
        if not isinstance(acceptance, Mapping) or acceptance.get("phase") != phase:
            errors.append(f"{prefix}.{phase}.manifest: exact acceptance phase required")
    if phase == "L":
        reload_binding = manifest.payload.get("reload")
        if (
            not isinstance(reload_binding, Mapping)
            or reload_binding.get("phase") != "provider_attempt_daemon_reload"
        ):
            errors.append(f"{prefix}.L.manifest: exact reload phase required")
    # Artifact population is the phase authority.  Preserve the uniquely
    # discovered phase even when its status/manifest projection is invalid so
    # every phase-coupled validator can report (and fail closed on) the same
    # attempted transition.
    return phase, errors


def _receipt_acceptance_parent(
    task_id: str,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    if task_id == "ASE3-019":
        parent = payload.get("merge")
        if not isinstance(parent, Mapping):
            return "", ""
        return (
            str(parent.get("acceptance_parent_head", "")),
            str(parent.get("acceptance_parent_tree", "")),
        )
    parent = payload.get("acceptance_parent")
    if not isinstance(parent, Mapping):
        return "", ""
    return str(parent.get("head", "")), str(parent.get("tree", ""))


def _validate_sequential_phase_chronology(
    *,
    phase: str,
    repo_root: Path,
    phase_heads: Mapping[str, str],
    phase_trees: Mapping[str, str],
    snapshots: Mapping[str, OperatorAcceptanceReceiptSnapshot],
    reload_snapshot: OperatorAcceptanceReceiptSnapshot | None,
    lifecycle_authority: Mapping[str, Any] | None,
) -> list[str]:
    """Prove signed receipt chronology from each introducing Git commit."""

    errors: list[str] = []
    prefix = "protected_acceptance.chronology"
    phase_index = _sequential_phase_index(phase)
    specs = (
        (
            "A019",
            "A019",
            "P019",
            f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_SALVAGE_RECEIPT_019_FILENAME}",
            snapshots.get("ASE3-019"),
        ),
        (
            "A030",
            "A030",
            "A019",
            HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
            snapshots.get("ASE3-030"),
        ),
        (
            "P031",
            "P031",
            "A030",
            NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
            snapshots.get("P031"),
        ),
        (
            "A031",
            "A031",
            "P031",
            NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
            snapshots.get("ASE3-031"),
        ),
        (
            "A032",
            "A032",
            "A031",
            DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
            snapshots.get("ASE3-032"),
        ),
        (
            "A023",
            "A023/027",
            "A032",
            f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME}",
            snapshots.get("ASE3-023"),
        ),
        (
            "A027",
            "A023/027",
            "A032",
            f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME}",
            snapshots.get("ASE3-027"),
        ),
        (
            "L",
            "L",
            "A023/027",
            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
            reload_snapshot,
        ),
    )
    event_times: dict[str, tuple[int, ...]] = {}
    for label, owner_phase, parent_phase, relative_path, snapshot in specs:
        if phase_index < _sequential_phase_index(owner_phase):
            continue
        if snapshot is None:
            continue
        owner_head = phase_heads.get(owner_phase, "")
        committed = _git_bytes(
            repo_root,
            "--no-replace-objects",
            "show",
            f"{owner_head}:{relative_path}",
        )
        if committed.returncode != 0:
            errors.append(f"{prefix}.{label}.committed_bytes: unavailable")
            continue
        if committed.stdout != snapshot.raw:
            errors.append(
                f"{prefix}.{label}.committed_bytes: "
                "exact introducing-commit bytes required"
            )
        try:
            payload = _load_json_bytes(committed.stdout, name=PurePosixPath(relative_path).name)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}.{label}.committed_bytes: {exc}")
            continue

        observed_head, observed_tree = _receipt_acceptance_parent(
            "ASE3-019" if label == "A019" else "other",
            payload,
        )
        if (
            observed_head != phase_heads.get(parent_phase, "")
            or observed_tree != phase_trees.get(parent_phase, "")
        ):
            errors.append(
                f"{prefix}.{label}.acceptance_parent: "
                "exact discovered phase parent required"
            )

        signature_errors = validate_operator_acceptance_signature(
            payload,
            expected_authority=lifecycle_authority,
        )
        errors.extend(
            f"{prefix}.{label}.signed_receipt.{error}"
            for error in signature_errors
        )
        created_at_ms = _utc_timestamp_to_ms(payload.get("created_at"))
        review = payload.get("review")
        signed_at_ms = (
            _utc_timestamp_to_ms(review.get("signed_at"))
            if isinstance(review, Mapping)
            else None
        )
        if created_at_ms is None:
            errors.append(
                f"{prefix}.{label}.created_at: "
                "valid non-coerced UTC calendar timestamp required"
            )
        if signed_at_ms is None:
            errors.append(
                f"{prefix}.{label}.review.signed_at: "
                "valid non-coerced UTC calendar timestamp required"
            )
        times: list[int] = []
        if created_at_ms is not None:
            times.append(created_at_ms)
        if signed_at_ms is not None:
            times.append(signed_at_ms)
        if label == "A031":
            preload = payload.get("preload_evidence")
            effect_started_at_ms = (
                _utc_timestamp_to_ms(preload.get("runtime_effect_started_at"))
                if isinstance(preload, Mapping)
                else None
            )
            if effect_started_at_ms is None:
                errors.append(
                    f"{prefix}.A031.preload_evidence.runtime_effect_started_at: "
                    "valid non-coerced UTC calendar timestamp required"
                )
            else:
                times.append(effect_started_at_ms)
                if (
                    signed_at_ms is not None
                    and effect_started_at_ms > signed_at_ms
                ):
                    errors.append(
                        f"{prefix}.A031.effect_to_receipt: "
                        "native effect must not follow signed A031 receipt"
                    )
        if times:
            event_times[label] = tuple(times)

    edges = (
        ("A019", "A030"),
        ("A030", "P031"),
        ("P031", "A031"),
        ("A031", "A032"),
        ("A032", "A023"),
        ("A032", "A027"),
        ("A023", "L"),
        ("A027", "L"),
    )
    for earlier, later in edges:
        earlier_times = event_times.get(earlier)
        later_times = event_times.get(later)
        if (
            earlier_times is not None
            and later_times is not None
            and max(earlier_times) > min(later_times)
        ):
            errors.append(
                f"{prefix}.{earlier}_to_{later}: "
                "signed receipt chronology inverted"
            )
    return errors


def _validate_sequential_phase_packet(
    *,
    phase: str,
    artifact_root: Path,
    manifest: ConvergenceManifest,
    repo_root: Path | None,
    fallback_authorization: ProviderFallbackPolicyAuthorization,
    fallback_authorization_raw: bytes,
    manifest_raw: bytes,
    taskboard_raw: bytes,
    expected_root_identity_did: str | None = None,
    phase_head_override: str | None = None,
) -> tuple[list[str], tuple[str, ...]]:
    """Validate the cumulative evidence and exact Git chain for one live phase."""

    errors: list[str] = []
    checked: list[str] = []
    phase_index = _sequential_phase_index(phase)
    if phase_index < 0:
        return ["protected_acceptance.packet: unsupported phase"], ()
    if phase == "Q":
        return errors, ()
    if repo_root is None:
        errors.append("protected_acceptance.packet: repository root required")

    phase_heads: dict[str, str] = {}
    phase_trees: dict[str, str] = {}
    if repo_root is not None:
        expected_phases = SEQUENTIAL_ACCEPTANCE_PHASES[: phase_index + 1]
        validation_head = (
            "HEAD" if phase_head_override is None else phase_head_override
        )
        head = _git(repo_root, "rev-parse", "--verify", validation_head)
        if head.returncode != 0 or _HEX40.fullmatch(head.stdout.strip()) is None:
            errors.append(
                "protected_acceptance.packet.history: exact first-parent prefix unavailable"
            )
        else:
            discovered, discovery_errors = _discover_sequential_phase_heads(
                repo_root=repo_root,
                head=head.stdout.strip(),
                through_phase=phase,
            )
            errors.extend(discovery_errors)
            if not discovery_errors:
                phase_heads = discovered
                for observed_phase, phase_head in phase_heads.items():
                    tree = _git(
                        repo_root,
                        "rev-parse",
                        "--verify",
                        f"{phase_head}^{{tree}}",
                    )
                    if (
                        tree.returncode != 0
                        or _HEX40.fullmatch(tree.stdout.strip()) is None
                    ):
                        errors.append(
                            "protected_acceptance.packet."
                            f"{observed_phase}.tree: exact phase tree unavailable"
                        )
                    else:
                        phase_trees[observed_phase] = tree.stdout.strip()
                errors.extend(
                    validate_protected_acceptance_sequence(
                        repo_root=repo_root,
                        phase_heads=phase_heads,
                        through_phase=phase,
                    )
                )

    raw_by_path: dict[str, bytes] = {
        PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH: (
            fallback_authorization_raw
        ),
    }
    sha_by_path: dict[str, str] = {
        PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH: (
            "sha256:" + hashlib.sha256(fallback_authorization_raw).hexdigest()
        ),
    }
    root_pin: LocalProfileLifecycleRootPinSnapshot | None = None
    lifecycle_witness: LocalOperatorLifecycleWitnessSnapshot | None = None
    lifecycle_authority: Mapping[str, Any] | None = None

    checked.append(LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME)
    try:
        root_pin = load_local_profile_lifecycle_root_pin(
            artifact_root / LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME,
            repository_root=repo_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME}: {exc}")
    else:
        raw_by_path[LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH] = root_pin.raw
        sha_by_path[LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH] = root_pin.sha256
        errors.extend(
            validate_local_profile_lifecycle_root_pin(
                root_pin.payload,
                expected_root_identity_did=expected_root_identity_did,
                expected_base_head=phase_heads.get("Q", ""),
                expected_base_tree=phase_trees.get("Q", ""),
            )
        )

    if phase_index >= _sequential_phase_index("P019"):
        checked.append(LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME)
        witness_final_values: Mapping[str, Any] | None = None
        try:
            lifecycle_witness = load_local_operator_lifecycle_witness(
                artifact_root / LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME,
                repository_root=repo_root,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME}: {exc}")
            lifecycle_witness = None
        else:
            raw_by_path[LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH] = (
                lifecycle_witness.raw
            )
            sha_by_path[LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH] = (
                lifecycle_witness.sha256
            )
            if _is_unpopulated_final_value(_FINAL_REVIEWER_DID_PENDING):
                profile = lifecycle_witness.payload.get("profile")
                if isinstance(profile, Mapping):
                    witness_final_values = {
                        "reviewer_identity": profile.get("identity_did"),
                        "profile_id": profile.get("profile_id"),
                        "profile_content_id": lifecycle_witness.payload.get(
                            "profile_content_id"
                        ),
                        "lifecycle_anchor_id": profile.get("lifecycle_anchor_id"),
                        "lifecycle_anchor_digest": lifecycle_witness.payload.get(
                            "anchor_digest"
                        ),
                        "lifecycle_generation": profile.get("lifecycle_generation"),
                    }
            errors.extend(
                validate_local_operator_lifecycle_witness(
                    lifecycle_witness.payload,
                    root_identity_did=(
                        root_pin.root_identity_did if root_pin is not None else ""
                    ),
                    expected_base_head=phase_heads.get("R", ""),
                    expected_base_tree=phase_trees.get("R", ""),
                    reference_time_ms=(
                        fallback_authorization.payload.get("authorized_at_ms")
                        if type(
                            fallback_authorization.payload.get("authorized_at_ms")
                        )
                        is int
                        else None
                    ),
                    earliest_observed_at_ms=(
                        root_pin.payload.get("pinned_at_ms")
                        if root_pin is not None
                        and type(root_pin.payload.get("pinned_at_ms")) is int
                        else None
                    ),
                    expected_final_values=witness_final_values,
                )
            )
        errors.extend(
            fallback_authorization.validate(
                lifecycle_witness=lifecycle_witness,
                root_pin=root_pin,
                expected_source_head=phase_heads.get("R", ""),
                expected_source_tree=phase_trees.get("R", ""),
                expected_final_values=witness_final_values,
            )
        )
        if root_pin is not None and lifecycle_witness is not None:
            lifecycle_authority = fallback_authorization.acceptance_review_authority(
                raw_sha256=sha_by_path[
                    PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
                ],
                lifecycle_witness=lifecycle_witness,
                root_pin=root_pin,
            )

    snapshots: dict[str, OperatorAcceptanceReceiptSnapshot] = {}

    def load_legacy(task_id: str) -> None:
        expected = _ACCEPTANCE_TASK_CONTRACTS[task_id]
        filename = str(expected["filename"])
        checked.append(filename)
        try:
            snapshot = load_operator_acceptance_receipt(
                artifact_root / filename,
                task_id=task_id,
                repository_root=repo_root,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{filename}: {exc}")
            return
        snapshots[task_id] = snapshot
        relative_path = f"{_CONVERGENCE_RELATIVE_ROOT}/{filename}"
        raw_by_path[relative_path] = snapshot.raw
        sha_by_path[relative_path] = snapshot.sha256
        if task_id == "ASE3-019":
            errors.extend(
                validate_operator_salvage_receipt_019(
                    snapshot.payload,
                    repo_root=repo_root,
                    lifecycle_authority=lifecycle_authority,
                )
            )
        elif task_id == "ASE3-030":
            errors.extend(
                validate_hermetic_identity_acceptance_receipt(
                    snapshot.payload,
                    repo_root=repo_root,
                    lifecycle_authority=lifecycle_authority,
                )
            )
        else:
            errors.extend(
                validate_operator_repair_acceptance_receipt(
                    snapshot.payload,
                    task_id=task_id,
                    repo_root=repo_root,
                    lifecycle_authority=lifecycle_authority,
                )
            )

    if phase_index >= _sequential_phase_index("A019"):
        load_legacy("ASE3-019")
    if phase_index >= _sequential_phase_index("A030"):
        load_legacy("ASE3-030")

    p031: OperatorAcceptanceReceiptSnapshot | None = None
    if phase_index >= _sequential_phase_index("P031"):
        checked.append(NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME)
        try:
            p031 = load_native_dependency_launch_authorization(
                artifact_root / NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME,
                repository_root=repo_root,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME}: {exc}")
        else:
            snapshots["P031"] = p031
            raw_by_path[NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH] = (
                p031.raw
            )
            sha_by_path[NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH] = (
                p031.sha256
            )
            errors.extend(
                validate_native_dependency_launch_authorization(
                    p031.payload,
                    repo_root=repo_root,
                    lifecycle_authority=lifecycle_authority,
                )
            )

    if phase_index >= _sequential_phase_index("A031"):
        checked.append(NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME)
        try:
            a031 = load_native_dependency_acceptance_receipt(
                artifact_root / NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME,
                repository_root=repo_root,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME}: {exc}")
        else:
            snapshots["ASE3-031"] = a031
            raw_by_path[NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH] = (
                a031.raw
            )
            sha_by_path[NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH] = (
                a031.sha256
            )
            errors.extend(
                validate_native_dependency_acceptance_receipt(
                    a031.payload,
                    launch_authorization=(p031.payload if p031 is not None else None),
                    launch_authorization_raw=(p031.raw if p031 is not None else None),
                    repo_root=repo_root,
                    lifecycle_authority=lifecycle_authority,
                )
            )

    if phase_index >= _sequential_phase_index("A032"):
        checked.append(DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME)
        try:
            a032 = load_duckdb_connection_policy_acceptance_receipt(
                artifact_root / DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME,
                repository_root=repo_root,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(
                f"{DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_FILENAME}: {exc}"
            )
        else:
            snapshots["ASE3-032"] = a032
            raw_by_path[DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH] = (
                a032.raw
            )
            sha_by_path[DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH] = (
                a032.sha256
            )
            errors.extend(
                validate_duckdb_connection_policy_acceptance_receipt(
                    a032.payload,
                    repo_root=repo_root,
                    lifecycle_authority=lifecycle_authority,
                )
            )

    if phase_index >= _sequential_phase_index("A023/027"):
        load_legacy("ASE3-023")
        load_legacy("ASE3-027")

    reload_snapshot: OperatorAcceptanceReceiptSnapshot | None = None
    if phase == "L":
        checked.append(PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME)
        try:
            reload_snapshot = load_provider_attempt_reload_receipt(
                artifact_root / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
                repository_root=repo_root,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME}: {exc}")
        else:
            raw_by_path[PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH] = (
                reload_snapshot.raw
            )
            sha_by_path[PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH] = (
                reload_snapshot.sha256
            )
            salvage = snapshots.get("ASE3-019")
            accepted_control_plane = (
                salvage.payload.get("accepted_control_plane")
                if salvage is not None
                and isinstance(
                    salvage.payload.get("accepted_control_plane"), Mapping
                )
                else None
            )
            errors.extend(
                validate_provider_attempt_reload_receipt(
                    reload_snapshot.payload,
                    repo_root=repo_root,
                    lifecycle_authority=lifecycle_authority,
                    accepted_control_plane=accepted_control_plane,
                )
            )

    current_artifacts = {
        path: sha_by_path.get(path)
        for path in SEQUENTIAL_PHASE_CHANGED_PATHS[phase]
        if path
        not in {
            _CONVERGENCE_MANIFEST_RELATIVE_PATH,
            PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(),
        }
    }
    if phase not in {"R", "P019", "L"}:
        acceptance = manifest.payload.get("acceptance")
        bindings = acceptance.get("artifacts") if isinstance(acceptance, Mapping) else None
        if bindings != current_artifacts:
            errors.append(
                f"protected_acceptance.packet.{phase}.manifest_artifacts: "
                "live byte digest mismatch"
            )
    elif phase == "L" and reload_snapshot is not None:
        reload_binding = manifest.payload.get("reload")
        receipt_binding = (
            reload_binding.get("receipt")
            if isinstance(reload_binding, Mapping)
            else None
        )
        if not isinstance(receipt_binding, Mapping) or receipt_binding.get(
            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
        ) != reload_snapshot.sha256:
            errors.append(
                "protected_acceptance.packet.L.manifest_receipt: live digest mismatch"
            )

    parent_specs: tuple[tuple[str, OperatorAcceptanceReceiptSnapshot | None, str], ...] = (
        ("A019", snapshots.get("ASE3-019"), "P019"),
        ("A030", snapshots.get("ASE3-030"), "A019"),
        ("P031", snapshots.get("P031"), "A030"),
        ("A031", snapshots.get("ASE3-031"), "P031"),
        ("A032", snapshots.get("ASE3-032"), "A031"),
        ("A023/027:ASE3-023", snapshots.get("ASE3-023"), "A032"),
        ("A023/027:ASE3-027", snapshots.get("ASE3-027"), "A032"),
        ("L", reload_snapshot, "A023/027"),
    )
    for label, snapshot, parent_phase in parent_specs:
        owner_phase = label.split(":", 1)[0]
        if snapshot is None or phase_index < _sequential_phase_index(owner_phase):
            continue
        observed_head, observed_tree = _receipt_acceptance_parent(
            "ASE3-019" if owner_phase == "A019" else "other",
            snapshot.payload,
        )
        if owner_phase == "L":
            parent = snapshot.payload.get("acceptance_parent")
            observed_head = str(parent.get("head", "")) if isinstance(parent, Mapping) else ""
            observed_tree = str(parent.get("tree", "")) if isinstance(parent, Mapping) else ""
        expected_head = phase_heads.get(parent_phase, "")
        expected_tree_result = (
            _git(repo_root, "rev-parse", "--verify", f"{expected_head}^{{tree}}")
            if repo_root is not None and expected_head
            else None
        )
        expected_tree = (
            expected_tree_result.stdout.strip()
            if expected_tree_result is not None
            and expected_tree_result.returncode == 0
            else ""
        )
        if observed_head != expected_head or observed_tree != expected_tree:
            errors.append(
                f"protected_acceptance.packet.{label}.acceptance_parent: "
                "direct phase parent mismatch"
            )

        sequential_parent = snapshot.payload.get("acceptance_parent")
        if owner_phase in {"P031", "A031", "A032"} and isinstance(
            sequential_parent, Mapping
        ):
            prior = sequential_parent.get("prior_artifacts")
            expected_prior = {
                path: sha_by_path.get(path)
                for path in _sequential_artifacts_after(parent_phase)
            }
            if prior != expected_prior:
                errors.append(
                    f"protected_acceptance.packet.{label}.prior_artifacts: "
                    "live cumulative digest mismatch"
                )

    if reload_snapshot is not None:
        reload_parent = reload_snapshot.payload.get("acceptance_parent")
        receipt_bindings = (
            reload_parent.get("acceptance_receipts")
            if isinstance(reload_parent, Mapping)
            else None
        )
        expected_receipt_bindings = {
            PurePosixPath(path).name: sha_by_path.get(path)
            for path in _sequential_artifacts_after("A023/027")
            if PurePosixPath(path).name in SEQUENTIAL_ACCEPTANCE_ARTIFACT_FILENAMES
        }
        if receipt_bindings != expected_receipt_bindings:
            errors.append(
                "protected_acceptance.packet.L.acceptance_receipts: "
                "live cumulative digest mismatch"
            )

    if repo_root is not None and phase_heads:
        errors.extend(
            _validate_sequential_phase_chronology(
                phase=phase,
                repo_root=repo_root,
                phase_heads=phase_heads,
                phase_trees=phase_trees,
                snapshots=snapshots,
                reload_snapshot=reload_snapshot,
                lifecycle_authority=lifecycle_authority,
            )
        )

    if repo_root is not None and phase_heads:
        changed_paths = SEQUENTIAL_PHASE_CHANGED_PATHS[phase]
        consumed: dict[str, bytes] = {}
        for path in changed_paths:
            if path == _CONVERGENCE_MANIFEST_RELATIVE_PATH:
                consumed[path] = manifest_raw
            elif path == PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix():
                consumed[path] = taskboard_raw
            elif path in raw_by_path:
                consumed[path] = raw_by_path[path]
        parent_phase = SEQUENTIAL_PHASE_PARENT[phase]
        parent_head = phase_heads[parent_phase]
        parent_tree = _git(
            repo_root,
            "rev-parse",
            "--verify",
            f"{parent_head}^{{tree}}",
        ).stdout.strip()
        errors.extend(
            validate_sequential_acceptance_child_transition(
                repo_root=repo_root,
                phase=phase,
                child_head=phase_heads[phase],
                parent_head=parent_head,
                parent_tree=parent_tree,
                consumed_child_blobs=consumed,
            )
        )
    return errors, tuple(checked)


def _obsolete_validate_operator_acceptance_packet(
    *,
    artifact_root: Path,
    manifest: ConvergenceManifest,
    repo_root: Path | None,
    fallback_authorization: ProviderFallbackPolicyAuthorization | None = None,
    fallback_authorization_raw: bytes | None = None,
    manifest_raw: bytes | None = None,
    taskboard_raw: bytes | None = None,
    expected_root_identity_did: str | None = None,
    expected_final_values: Mapping[str, Any] | None = None,
    acceptance_head_override: str | None = None,
) -> tuple[list[str], tuple[str, ...]]:
    """Compatibility shim that cannot validate the retired fan-in packet."""

    del (
        artifact_root,
        manifest,
        repo_root,
        fallback_authorization,
        fallback_authorization_raw,
        manifest_raw,
        taskboard_raw,
        expected_root_identity_did,
        expected_final_values,
        acceptance_head_override,
    )
    return [
        "operator_acceptance.packet: obsolete atomic fan-in forbidden"
    ], ()
def _validate_provider_attempt_reload_packet(
    *,
    artifact_root: Path,
    manifest: ConvergenceManifest,
    repo_root: Path | None,
    fallback_authorization: ProviderFallbackPolicyAuthorization,
    fallback_authorization_raw: bytes,
    manifest_raw: bytes,
    taskboard_raw: bytes,
    expected_root_identity_did: str | None = None,
    expected_lifecycle_final_values: Mapping[str, Any] | None = None,
) -> tuple[list[str], tuple[str, ...]]:
    """Fail closed for the retired direct A-to-L packet entry point."""

    del (
        artifact_root,
        manifest,
        repo_root,
        fallback_authorization,
        fallback_authorization_raw,
        manifest_raw,
        taskboard_raw,
        expected_root_identity_did,
        expected_lifecycle_final_values,
    )
    return [
        (
            "provider_attempt_reload.packet: obsolete direct A-to-L validation "
            "forbidden; use the sequential L phase packet"
        )
    ], ()


def _obsolete_validate_provider_attempt_reload_packet(
    *,
    artifact_root: Path,
    manifest: ConvergenceManifest,
    repo_root: Path | None,
    fallback_authorization: ProviderFallbackPolicyAuthorization,
    fallback_authorization_raw: bytes,
    manifest_raw: bytes,
    taskboard_raw: bytes,
    expected_root_identity_did: str | None = None,
    expected_lifecycle_final_values: Mapping[str, Any] | None = None,
) -> tuple[list[str], tuple[str, ...]]:
    errors: list[str] = []
    checked = (PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,)
    if repo_root is None:
        return ["provider_attempt_reload: repository root required"], checked

    lifecycle_authority: Mapping[str, Any] | None = None
    try:
        root_pin = load_local_profile_lifecycle_root_pin(
            artifact_root / LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME,
            repository_root=repo_root,
        )
        witness = load_local_operator_lifecycle_witness(
            artifact_root / LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME,
            repository_root=repo_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"provider_attempt_reload.authority: {exc}")
    else:
        errors.extend(
            validate_local_profile_lifecycle_root_pin(
                root_pin.payload,
                expected_root_identity_did=expected_root_identity_did,
            )
        )
        errors.extend(
            validate_local_operator_lifecycle_witness(
                witness.payload,
                root_identity_did=root_pin.root_identity_did,
                expected_base_head=fallback_authorization.source_head,
                expected_base_tree=fallback_authorization.source_tree,
                reference_time_ms=(
                    fallback_authorization.payload.get("authorized_at_ms")
                    if type(
                        fallback_authorization.payload.get("authorized_at_ms")
                    )
                    is int
                    else None
                ),
                expected_final_values=expected_lifecycle_final_values,
            )
        )
        errors.extend(
            fallback_authorization.validate(
                lifecycle_witness=witness,
                root_pin=root_pin,
                expected_source_head=fallback_authorization.source_head,
                expected_source_tree=fallback_authorization.source_tree,
                expected_final_values=expected_lifecycle_final_values,
            )
        )
        lifecycle_authority = fallback_authorization.acceptance_review_authority(
            raw_sha256=(
                "sha256:" + hashlib.sha256(fallback_authorization_raw).hexdigest()
            ),
            lifecycle_witness=witness,
            root_pin=root_pin,
        )

    try:
        receipt = load_provider_attempt_reload_receipt(
            artifact_root / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
            repository_root=repo_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME}: {exc}")
        return errors, checked
    reload_binding = manifest.payload.get("reload")
    if not isinstance(reload_binding, Mapping):
        errors.append("convergence_manifest.reload: expected object")
        return errors, checked
    acceptance_head = str(reload_binding.get("acceptance_head", ""))
    acceptance_tree = str(reload_binding.get("acceptance_tree", ""))
    accepted_control_plane: Mapping[str, Any] | None = None
    committed_salvage_path = (
        f"{_CONVERGENCE_RELATIVE_ROOT}/{OPERATOR_SALVAGE_RECEIPT_019_FILENAME}"
    )
    committed_salvage = _git_bytes(
        repo_root,
        "show",
        f"{acceptance_head}:{committed_salvage_path}",
    )
    if (
        committed_salvage.returncode != 0
        or len(committed_salvage.stdout) > MAX_OPERATOR_ACCEPTANCE_RECEIPT_BYTES
    ):
        errors.append(
            "provider_attempt_reload.accepted_control_plane: "
            "committed A receipt unavailable or oversized"
        )
    else:
        try:
            committed_salvage_payload = _load_json_bytes(
                committed_salvage.stdout,
                name="committed-" + OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"provider_attempt_reload.accepted_control_plane: {exc}")
        else:
            if set(committed_salvage_payload) != set(
                _ASE3_019_OPERATOR_SALVAGE_REQUIRED_FIELDS
            ):
                errors.append(
                    "provider_attempt_reload.accepted_control_plane: "
                    "committed A receipt exact schema required"
                )
            if committed_salvage_payload.get("schema") != (
                OPERATOR_SALVAGE_RECEIPT_019_SCHEMA
            ):
                errors.append(
                    "provider_attempt_reload.accepted_control_plane: "
                    "committed A receipt schema mismatch"
                )
            errors.extend(
                validate_operator_acceptance_signature(
                    committed_salvage_payload,
                    expected_authority=lifecycle_authority,
                )
            )
            candidate = committed_salvage_payload.get("accepted_control_plane")
            if isinstance(candidate, Mapping):
                accepted_control_plane = candidate
            else:
                errors.append(
                    "provider_attempt_reload.accepted_control_plane: "
                    "committed A object required"
                )
            acceptance = manifest.payload.get("acceptance")
            receipt_bindings = (
                acceptance.get("receipts")
                if isinstance(acceptance, Mapping)
                else None
            )
            committed_salvage_sha256 = (
                "sha256:" + hashlib.sha256(committed_salvage.stdout).hexdigest()
            )
            if not isinstance(receipt_bindings, Mapping) or receipt_bindings.get(
                OPERATOR_SALVAGE_RECEIPT_019_FILENAME
            ) != committed_salvage_sha256:
                errors.append(
                    "provider_attempt_reload.accepted_control_plane: "
                    "committed A receipt raw digest mismatch"
                )
    errors.extend(
        validate_provider_attempt_reload_receipt(
            receipt.payload,
            repo_root=repo_root,
            lifecycle_authority=lifecycle_authority,
            accepted_control_plane=accepted_control_plane,
        )
    )
    bound_receipt = reload_binding.get("receipt")
    if not isinstance(bound_receipt, Mapping) or bound_receipt.get(
        PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
    ) != receipt.sha256:
        errors.append("convergence_manifest.reload.receipt: digest mismatch")

    parent = receipt.payload.get("acceptance_parent")
    if isinstance(parent, Mapping):
        if parent.get("head") != acceptance_head or parent.get("tree") != acceptance_tree:
            errors.append("provider_attempt_reload.acceptance_parent: A mismatch")
        acceptance = manifest.payload.get("acceptance")
        receipt_bindings = (
            acceptance.get("receipts") if isinstance(acceptance, Mapping) else None
        )
        if parent.get("acceptance_receipts") != receipt_bindings:
            errors.append(
                "provider_attempt_reload.acceptance_parent.acceptance_receipts: "
                "manifest mismatch"
            )

    head = _git(repo_root, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0 or _HEX40.fullmatch(head.stdout.strip()) is None:
        errors.append("provider_attempt_reload.transition: unable to resolve HEAD")
    else:
        consumed = {
            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH: receipt.raw,
            f"{_CONVERGENCE_RELATIVE_ROOT}/{MANIFEST_FILENAME}": manifest_raw,
            PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix(): taskboard_raw,
        }
        errors.extend(
            validate_reload_child_transition(
                repo_root=repo_root,
                reload_head=head.stdout.strip(),
                acceptance_head=acceptance_head,
                acceptance_tree=acceptance_tree,
                consumed_reload_blobs=consumed,
            )
        )
    return errors, checked


def _validate_provider_attempt_reload_gate(
    *,
    tasks: Mapping[str, Mapping[str, str]],
    artifact_root: Path,
    acceptance_phase: bool = False,
    reload_phase: bool = False,
) -> list[str]:
    """Keep ASE3-022 blocked through A and complete it only in exact L."""

    errors: list[str] = []
    prefix = "provider_attempt_reload_gate"
    receipt_path = artifact_root / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
    try:
        receipt_path.lstat()
    except FileNotFoundError:
        if reload_phase:
            errors.append(f"{prefix}.receipt: required in reload phase")
    except OSError as exc:
        errors.append(f"{prefix}.receipt: unable to inspect protected path: {exc}")
    else:
        if not reload_phase:
            errors.append(f"{prefix}.receipt: forbidden before reload phase")

    gate = tasks.get(_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID)
    if gate is None:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}: expected exactly one task"
        )
        return errors
    gate_status = gate.get("status", "todo").strip().lower()
    expected_status = "completed" if reload_phase else "blocked"
    if gate_status != expected_status:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.status: "
            f"expected {expected_status}"
        )
    expected_contract = _RELOAD_TASK_CONTRACT[
        "completed_contract_sha256" if reload_phase else "blocked_contract_sha256"
    ]
    if _task_contract_sha256(gate) != expected_contract:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.contract_sha256: "
            "exact status-only C1 reload-gate contract required"
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
    elif provider_task.get("status") != (
        "completed" if acceptance_phase or reload_phase else "todo"
    ):
        required_status = (
            "completed" if acceptance_phase or reload_phase else "todo"
        )
        errors.append(
            f"{prefix}.ASE3-019.status: expected {required_status} for the "
            "current operator-acceptance phase"
        )
    return errors


def _validate_provider_fallback_task_contract(
    *,
    tasks: Mapping[str, Mapping[str, str]],
    expected_status: str = "todo",
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
    expected_contract = _ACCEPTANCE_TASK_CONTRACTS["ASE3-019"][
        "completed_contract_sha256"
        if expected_status == "completed"
        else "todo_contract_sha256"
    ]
    if _task_contract_sha256(task) != expected_contract:
        errors.append(
            f"{prefix}.ASE3-019.contract_sha256: exact metadata/prose required"
        )
    if task.get("status") != expected_status:
        errors.append(
            f"{prefix}.ASE3-019.status: expected {expected_status!r}"
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
    expected_status: str = "todo",
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
        expected_contract = _ACCEPTANCE_TASK_CONTRACTS[task_id][
            "completed_contract_sha256"
            if expected_status == "completed"
            else "todo_contract_sha256"
        ]
        if _task_contract_sha256(task) != expected_contract:
            errors.append(
                f"{prefix}.{task_id}.contract_sha256: "
                "exact metadata/prose required"
            )
        for field, expected_value in {
            "status": expected_status,
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
        expected_static_validation = expected.get("static validation commands json")
        if (
            expected_static_validation is not None
            and task.get("static validation commands json")
            != expected_static_validation
        ):
            errors.append(
                f"{prefix}.{task_id}.static_validation: exact commands required"
            )
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
    expected_statuses: Mapping[str, str] | None = None,
) -> list[str]:
    """Validate the canonical 30-task expansion and protected activation gate."""

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
        errors.append(f"{prefix}.canonical_tasks: expected exact 30-task population")
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
        expected_status = (
            expected_statuses.get(task_id, "todo")
            if expected_statuses is not None
            else "todo"
        )
        sequential_contract = _SEQUENTIAL_TASK_CONTRACTS.get(task_id)
        expected_contract = (
            sequential_contract["completed_contract_sha256"]
            if sequential_contract is not None and expected_status == "completed"
            else expected["contract_sha256"]
        )
        if _task_contract_sha256(task) != expected_contract:
            errors.append(
                f"{prefix}.{task_id}.contract_sha256: exact metadata/prose required"
            )
        for field, expected_value in {
            "status": expected_status,
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
        searchable = " ".join(task.values())
        for requirement in expected.get("requirements", ()):
            if requirement not in searchable:
                errors.append(
                    f"{prefix}.{task_id}.contract: missing {requirement!r}"
                )
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
        if (
            expected_statuses is not None
            and expected_statuses.get("ASE3-030") == "completed"
        ):
            pass
        else:
            errors.append(
                f"{prefix}.ASE3-030.acceptance_receipt: present before A030"
            )
    identity_task = tasks.get("ASE3-030")
    if (
        identity_task is not None
        and identity_task.get("status") == "completed"
        and (
            expected_statuses is None
            or expected_statuses.get("ASE3-030") != "completed"
        )
    ):
        errors.append(
            f"{prefix}.ASE3-030.status: completion requires A030"
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
            f"{prefix}.ASE3-026.authorization_receipt: present without strict "
            "validation and convergence-manifest binding"
        )
    observation_path = (
        artifact_root
        / PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_FILENAME
    )
    try:
        observation_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        errors.append(f"{prefix}.ASE3-026.observation_receipt: unable to inspect: {exc}")
    else:
        errors.append(
            f"{prefix}.ASE3-026.observation_receipt: present without strict "
            "post-activation validation and convergence-manifest binding"
        )
    activation = tasks.get(_PROTECTED_RUNTIME_ACTIVATION_TASK_ID)
    if activation is None:
        errors.append(f"{prefix}.ASE3-026: expected exactly one activation task")
    else:
        if activation.get(_TASK_TITLE_KEY) != _PROTECTED_RUNTIME_ACTIVATION_TASK_TITLE:
            errors.append(f"{prefix}.ASE3-026.title: exact split activation title required")
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
        if (
            _taskboard_csv(activation, "depends on")
            != _PROTECTED_RUNTIME_ACTIVATION_DEPENDENCIES
        ):
            errors.append(f"{prefix}.ASE3-026.depends_on: exact expansion required")
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
            errors.append(
                f"{prefix}.ASE3-026.outputs: activation authorization receipt required"
            )
        if (
            PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_RELATIVE_PATH
            not in _taskboard_csv(activation, "outputs")
        ):
            errors.append(
                f"{prefix}.ASE3-026.outputs: post-activation observation receipt "
                "required"
            )
        searchable = " ".join(activation.values())
        for requirement in _PROTECTED_RUNTIME_ACTIVATION_REQUIREMENTS:
            if requirement not in searchable:
                errors.append(
                    f"{prefix}.ASE3-026.contract: missing {requirement!r}"
                )

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
        "ASE3-031",
        "ASE3-032",
        "ASE3-033",
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
                f"{prefix}.task_order: exact hermetic/native/transition/layering chain "
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
            "Wave 3b native:    ASE3-031",
            "Wave 3b database:  ASE3-032",
            "Wave 3b adaptive:  ASE3-023",
            "Wave 3b transition: ASE3-033",
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
        for requirement in _MONITOR_STRATEGY_PLAN_REQUIREMENTS:
            if requirement not in plan_text:
                errors.append(
                    f"{prefix}.plan.monitor_strategy: missing {requirement!r}"
                )
        try:
            activation_section_hash = (
                _normalized_markdown_section_contract_sha256(
                    plan_text,
                    section_heading=_ASE3_026_PLAN_SECTION_HEADING,
                    containing_heading=_ASE3_026_PLAN_CONTAINING_HEADING,
                    end_heading=_ASE3_026_PLAN_SECTION_END_HEADING,
                )
            )
        except ValueError as exc:
            errors.append(f"{prefix}.plan.ASE3-026.section: {exc}")
        else:
            if activation_section_hash != _ASE3_026_PLAN_SECTION_CONTRACT_SHA256:
                errors.append(
                    f"{prefix}.plan.ASE3-026.contract_sha256: exact normalized "
                    "protected activation section required"
                )
        try:
            transition_construction_section_hash = (
                _normalized_markdown_section_contract_sha256(
                    plan_text,
                    section_heading=(
                        _TRANSITION_CONSTRUCTION_PLAN_SECTION_HEADING
                    ),
                    containing_heading=(
                        _TRANSITION_CONSTRUCTION_PLAN_CONTAINING_HEADING
                    ),
                    end_heading=(
                        _TRANSITION_CONSTRUCTION_PLAN_SECTION_END_HEADING
                    ),
                )
            )
        except ValueError as exc:
            errors.append(f"{prefix}.plan.ASE3-033.section: {exc}")
        else:
            if transition_construction_section_hash != (
                _TRANSITION_CONSTRUCTION_PLAN_SECTION_CONTRACT_SHA256
            ):
                errors.append(
                    f"{prefix}.plan.ASE3-033.contract_sha256: exact normalized "
                    "protected transition construction section required"
                )
        try:
            native_duckdb_section_hash = (
                _normalized_markdown_section_contract_sha256(
                    plan_text,
                    section_heading=_NATIVE_DUCKDB_PLAN_SECTION_HEADING,
                    containing_heading=_NATIVE_DUCKDB_PLAN_CONTAINING_HEADING,
                    end_heading=_NATIVE_DUCKDB_PLAN_SECTION_END_HEADING,
                )
            )
        except ValueError as exc:
            errors.append(f"{prefix}.plan.ASE3-031-032.section: {exc}")
        else:
            if (
                native_duckdb_section_hash
                != _NATIVE_DUCKDB_PLAN_SECTION_CONTRACT_SHA256
            ):
                errors.append(
                    f"{prefix}.plan.ASE3-031-032.contract_sha256: exact "
                    "normalized native DuckDB gate section required"
                )
        try:
            contract_layering_section_hash = (
                _normalized_markdown_section_contract_sha256(
                    plan_text,
                    section_heading=_CONTRACT_LAYERING_PLAN_SECTION_HEADING,
                    containing_heading=_CONTRACT_LAYERING_PLAN_CONTAINING_HEADING,
                    end_heading=_CONTRACT_LAYERING_PLAN_SECTION_END_HEADING,
                )
            )
        except ValueError as exc:
            errors.append(f"{prefix}.plan.ASE3-029.section: {exc}")
        else:
            if (
                contract_layering_section_hash
                != _CONTRACT_LAYERING_PLAN_SECTION_CONTRACT_SHA256
            ):
                errors.append(
                    f"{prefix}.plan.ASE3-029.contract_sha256: exact normalized "
                    "content-bound contract-layering section required"
                )
        for (
            projection_name,
            section_heading,
            containing_heading,
            end_heading,
            expected_hash,
        ) in _CONTRACT_LAYERING_PLAN_OUTER_SECTION_CONTRACTS:
            projection_prefix = (
                f"{prefix}.plan.ASE3-029.{projection_name}"
            )
            try:
                observed_hash = _normalized_markdown_section_contract_sha256(
                    plan_text,
                    section_heading=section_heading,
                    containing_heading=containing_heading,
                    end_heading=end_heading,
                )
            except ValueError as exc:
                errors.append(f"{projection_prefix}.section: {exc}")
            else:
                if observed_hash != expected_hash:
                    errors.append(
                        f"{projection_prefix}.contract_sha256: exact normalized "
                        "ASE3-029 normative plan projection required"
                    )
        for requirement in _CONTRACT_LAYERING_PLAN_REQUIREMENTS:
            if requirement not in plan_text:
                errors.append(
                    f"{prefix}.plan.ASE3-029.contract: missing {requirement!r}"
                )
        for field, fragment in {
            "hermetic_identity_acceptance_receipt": (
                HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH
            ),
            "hermetic_identity_acceptance_schema": (
                HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA
            ),
            "native_duckdb_acceptance_sequence": (
                "Q→R(root pin)→P019(witness+provider auth@2+manifest)→A019→"
                "A030→P031(native auth+manifest)→A031→A032→A023/027→"
                "L(ASE3-022 reload authorization)"
            ),
            "native_dependency_reviewed_commit": (
                "25fedf091dad928dad1f83c9f81a54c2d401eabe"
            ),
            "native_dependency_launch_authorization": (
                NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH
            ),
            "native_dependency_launch_authorization_schema": (
                NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA
            ),
            "native_dependency_acceptance_receipt": (
                NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH
            ),
            "native_dependency_acceptance_schema": (
                NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA
            ),
            "duckdb_connection_policy_acceptance_receipt": (
                DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH
            ),
            "duckdb_connection_policy_acceptance_schema": (
                DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA
            ),
        }.items():
            if fragment not in plan_text:
                errors.append(f"{prefix}.plan.{field}: exact protected join required")

    initial = config.get("initial_projection")
    if not isinstance(initial, Mapping):
        errors.append(f"{prefix}.initial_projection: expected object")
    else:
        if type(initial.get("task_count")) is not int or initial.get(
            "task_count"
        ) != 30:
            errors.append(
                f"{prefix}.initial_projection.task_count: expected exact integer 30"
            )
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
            errors.append(f"{prefix}.task_groups: exact 30-task population required")
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
        for goal_id, expected_hash in (
            _MONITOR_STRATEGY_OBJECTIVE_CONTRACT_SHA256S.items()
        ):
            goal = goals.get(goal_id)
            if goal is None:
                errors.append(f"{prefix}.objectives.{goal_id}: missing sealed goal")
            elif _task_contract_sha256(goal) != expected_hash:
                errors.append(
                    f"{prefix}.objectives.{goal_id}.contract_sha256: exact "
                    "monitor-strategy goal contract required"
                )
        for goal_id, expected_hash in (
            _NATIVE_DUCKDB_OBJECTIVE_CONTRACT_SHA256S.items()
        ):
            goal = goals.get(goal_id)
            if goal is None:
                errors.append(
                    f"{prefix}.objectives.{goal_id}: missing sealed native "
                    "DuckDB goal"
                )
            elif _task_contract_sha256(goal) != expected_hash:
                errors.append(
                    f"{prefix}.objectives.{goal_id}.contract_sha256: exact "
                    "native DuckDB goal contract required"
                )
        for goal_id, expected_hash in (
            _CONTRACT_LAYERING_OBJECTIVE_CONTRACT_SHA256S.items()
        ):
            goal = goals.get(goal_id)
            if goal is None:
                errors.append(
                    f"{prefix}.objectives.{goal_id}: missing sealed contract-"
                    "layering goal"
                )
            elif _task_contract_sha256(goal) != expected_hash:
                errors.append(
                    f"{prefix}.objectives.{goal_id}.contract_sha256: exact "
                    "contract-layering goal contract required"
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
        "ASE3-023": ["ASE3-030", "ASE3-031", "ASE3-032"],
        "ASE3-022": ["ASE3-030", "ASE3-031", "ASE3-032"],
    }
    if config.get("acceptance_prerequisites") != expected_acceptance_prerequisites:
        errors.append(
            f"{prefix}.acceptance_prerequisites: exact ASE3-030/031/032 "
            "fail-closed acceptance join required"
        )

    transition_policy = config.get("protected_transition_construction")
    transition_prefix = f"{prefix}.protected_transition_construction"
    if type(transition_policy) is not dict:
        errors.append(f"{transition_prefix}: expected exact JSON object")
    else:
        try:
            transition_hash = _mapping_contract_sha256(transition_policy)
        except (TypeError, ValueError):
            errors.append(f"{transition_prefix}.contract_sha256: noncanonical JSON")
        else:
            if transition_hash != _TRANSITION_CONSTRUCTION_POLICY_SHA256:
                errors.append(
                    f"{transition_prefix}.contract_sha256: exact protected "
                    "transition construction policy required"
                )
        exact_transition_fields = {
            "task_id": "ASE3-033",
            "ordinary_dependencies": ["ASE3-000"],
            "pre_q_integration_freeze_prerequisites": list(
                _TRANSITION_CONSTRUCTION_PRE_Q_REVIEWS
            ),
            "prerequisite_review_kind": "independent",
            "prerequisite_acceptance_status_required": False,
            "q_parent_requires_independently_audited_replayed_ase3_033_tooling": (
                True
            ),
            "q_status_transition": {
                "task_id": "ASE3-033",
                "from": "todo",
                "to": "completed",
                "other_status_changes_allowed": False,
                "changed_paths": list(_TRANSITION_Q_CHANGED_PATHS),
            },
            "ase3_033_completed_required_from_phase": "R",
            "required_before_phases": list(
                _TRANSITION_CONSTRUCTION_REQUIRED_PHASES
            ),
            "product_outputs": list(_TRANSITION_CONSTRUCTION_OUTPUTS),
            "public_apis": list(_TRANSITION_CONSTRUCTION_PUBLIC_APIS),
            "required_transition_tests": list(
                _TRANSITION_CONSTRUCTION_REQUIRED_TESTS
            ),
        }
        for field, expected_value in exact_transition_fields.items():
            if not _exact_json_contract_value(
                transition_policy.get(field),
                expected_value,
            ):
                errors.append(
                    f"{transition_prefix}.{field}: exact construction contract "
                    "required"
                )
        transition_task = tasks.get("ASE3-033")
        if transition_task is None:
            errors.append(f"{transition_prefix}.task_id: ASE3-033 task is missing")
        else:
            if _taskboard_csv(transition_task, "depends on") != ("ASE3-000",):
                errors.append(
                    f"{transition_prefix}.ordinary_dependencies: ASE3-000 only"
                )
            if _taskboard_csv(transition_task, "outputs") != (
                _TRANSITION_CONSTRUCTION_OUTPUTS
            ):
                errors.append(
                    f"{transition_prefix}.product_outputs: taskboard mismatch"
                )
            interfaces = _taskboard_csv(transition_task, "interfaces")
            if interfaces != _TRANSITION_CONSTRUCTION_PUBLIC_APIS:
                errors.append(
                    f"{transition_prefix}.public_apis: taskboard mismatch"
                )
            if _taskboard_csv(
                transition_task,
                "required transition tests",
            ) != _TRANSITION_CONSTRUCTION_REQUIRED_TESTS:
                errors.append(
                    f"{transition_prefix}.required_transition_tests: "
                    "taskboard mismatch"
                )
        pre_q_status_edges = set(_TRANSITION_CONSTRUCTION_PRE_Q_REVIEWS) & set(
            dependencies.get("ASE3-033", [])
            if isinstance(dependencies, Mapping)
            else []
        )
        if pre_q_status_edges:
            errors.append(
                f"{transition_prefix}.pre_q_reviews: acceptance-status dependency "
                "forbidden"
            )
        if transition_task is not None and transition_task.get("status") == "todo":
            for relative_path in _TRANSITION_CONSTRUCTION_RESERVED_PATHS:
                try:
                    (repo_root / relative_path).lstat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    errors.append(
                        f"{transition_prefix}.{relative_path}: unable to inspect "
                        f"reserved path: {exc}"
                    )
                else:
                    errors.append(
                        f"{transition_prefix}.{relative_path}: Q inventory "
                        "present before ASE3-033 Q acceptance"
                    )

    layering_policy = config.get("neutral_contract_layering")
    layering_prefix = f"{prefix}.neutral_contract_layering"
    if type(layering_policy) is not dict:
        errors.append(f"{layering_prefix}: expected exact JSON object")
    else:
        if not _exact_json_contract_value(
            layering_policy,
            _CONTRACT_LAYERING_POLICY,
        ):
            errors.append(
                f"{layering_prefix}: exact content-bound lower-effect contract "
                "required"
            )
        try:
            layering_policy_hash = _mapping_contract_sha256(layering_policy)
        except (TypeError, ValueError):
            errors.append(f"{layering_prefix}.contract_sha256: noncanonical JSON")
        else:
            if (
                layering_policy_hash
                != _CONTRACT_LAYERING_POLICY_CONFIG_SHA256
            ):
                errors.append(
                    f"{layering_prefix}.contract_sha256: exact parsed policy "
                    "contract required"
                )
        layering_task = tasks.get("ASE3-029")
        if layering_task is None:
            errors.append(f"{layering_prefix}.task_id: ASE3-029 task is missing")
        else:
            observed_contract = _task_contract_sha256(layering_task)
            if layering_policy.get("task_contract_sha256") != observed_contract:
                errors.append(
                    f"{layering_prefix}.task_contract_sha256: taskboard mismatch"
                )
            try:
                observed_task_cid = _canonical_task_cid_from_metadata(
                    layering_task
                )
            except ValueError as exc:
                errors.append(f"{layering_prefix}.canonical_task_cid: {exc}")
            else:
                if layering_policy.get("canonical_task_cid") != observed_task_cid:
                    errors.append(
                        f"{layering_prefix}.canonical_task_cid: taskboard mismatch"
                    )
        router_task = tasks.get("ASE3-028")
        if router_task is None or _taskboard_csv(
            router_task,
            "depends on",
        ) != ("ASE3-029",):
            errors.append(
                f"{layering_prefix}.downstream_task_id: ASE3-028 must remain "
                "sequential after ASE3-029"
            )

    expected_identity_acceptance = {
        "task_id": "ASE3-030",
        "status": "reserved",
        "receipt_path": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        "receipt_schema": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA,
        "artifact_phase": "A",
        "sequence_phase": "A030",
        "strict_validator_and_manifest_binding_required": True,
        "required_before_task_acceptance": ["ASE3-023", "ASE3-022"],
    }
    if config.get("protected_identity_acceptance") != expected_identity_acceptance:
        errors.append(
            f"{prefix}.protected_identity_acceptance: exact reserved ASE3-030 "
            "receipt contract required"
        )

    expected_native_gates = {
        "protected_native_dependency_launch_authorization": {
            "task_id": "ASE3-031",
            "status": "reserved",
            "authorization_path": (
                NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH
            ),
            "authorization_schema": NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA,
            "artifact_phase": "P",
            "sequence_phase": "P031",
            "signed_by_accepted_local_profile_required": True,
            "accepted_authorization_id_exact_match_required": True,
            "inspection_evidence_is_authority": False,
            "authorization_may_claim_launch_effect": False,
            "strict_validator_and_manifest_binding_required": True,
            "required_before_task_acceptance": ["ASE3-031"],
            "required_before_runtime_effects": ["ASE3-023"],
        },
        "protected_native_dependency_acceptance": {
            "task_id": "ASE3-031",
            "status": "reserved",
            "receipt_path": NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
            "receipt_schema": NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA,
            "artifact_phase": "A",
            "sequence_phase": "A031",
            "authorization_path": (
                NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH
            ),
            "accepted_authorization_id_exact_match_required": True,
            "strict_validator_and_manifest_binding_required": True,
            "required_before_task_acceptance": ["ASE3-023", "ASE3-022"],
        },
        "protected_duckdb_connection_policy_acceptance": {
            "task_id": "ASE3-032",
            "status": "reserved",
            "receipt_path": (
                DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH
            ),
            "receipt_schema": DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
            "artifact_phase": "A",
            "sequence_phase": "A032",
            "requires_prior_acceptance_tasks": ["ASE3-030", "ASE3-031"],
            "strict_validator_and_manifest_binding_required": True,
            "required_before_task_acceptance": ["ASE3-023", "ASE3-022"],
        },
    }
    for section, expected_gate in expected_native_gates.items():
        observed_gate = config.get(section)
        if type(observed_gate) is not dict:
            errors.append(f"{prefix}.{section}: expected exact JSON object")
            continue
        if not _exact_json_contract_value(observed_gate, expected_gate):
            errors.append(f"{prefix}.{section}: exact typed reserved gate required")
        try:
            observed_gate_hash = _mapping_contract_sha256(observed_gate)
        except (TypeError, ValueError):
            errors.append(
                f"{prefix}.{section}.contract_sha256: gate is not canonical JSON"
            )
        else:
            if observed_gate_hash != _NATIVE_DUCKDB_GATE_CONFIG_SHA256S[section]:
                errors.append(
                    f"{prefix}.{section}.contract_sha256: exact parsed gate "
                    "contract required"
                )

    observed_sequence = config.get("protected_native_duckdb_acceptance_sequence")
    sequence_prefix = f"{prefix}.protected_native_duckdb_acceptance_sequence"
    if type(observed_sequence) is not dict:
        errors.append(f"{sequence_prefix}: expected exact JSON object")
    else:
        if not _exact_json_contract_value(
            observed_sequence,
            _NATIVE_DUCKDB_ACCEPTANCE_SEQUENCE,
        ):
            errors.append(f"{sequence_prefix}: exact sequential phase DAG required")
        try:
            sequence_hash = _mapping_contract_sha256(observed_sequence)
        except (TypeError, ValueError):
            errors.append(f"{sequence_prefix}.contract_sha256: noncanonical JSON")
        else:
            if sequence_hash != _NATIVE_DUCKDB_GATE_CONFIG_SHA256S[
                "protected_native_duckdb_acceptance_sequence"
            ]:
                errors.append(
                    f"{sequence_prefix}.contract_sha256: exact parsed phase DAG "
                    "contract required"
                )

        phase_index_by_task: dict[str, int] = {}
        for phase_index, phase_record in enumerate(
            _NATIVE_DUCKDB_ACCEPTANCE_SEQUENCE["phases"]
        ):
            for task_id in phase_record["task_ids"]:
                phase_index_by_task[task_id] = phase_index

        phase_dependency_edges: list[tuple[str, str]] = []
        if isinstance(dependencies, Mapping):
            for task_id, raw_dependencies in dependencies.items():
                if isinstance(raw_dependencies, list):
                    phase_dependency_edges.extend(
                        (str(task_id), str(dependency))
                        for dependency in raw_dependencies
                    )
        reload_task = tasks.get("ASE3-022")
        if reload_task is not None:
            phase_dependency_edges.extend(
                ("ASE3-022", dependency)
                for dependency in _taskboard_csv(reload_task, "depends on")
            )
        observed_prerequisites = config.get("acceptance_prerequisites")
        if isinstance(observed_prerequisites, Mapping):
            for task_id, raw_prerequisites in observed_prerequisites.items():
                if isinstance(raw_prerequisites, list):
                    phase_dependency_edges.extend(
                        (str(task_id), str(prerequisite))
                        for prerequisite in raw_prerequisites
                    )
        for task_id, dependency in phase_dependency_edges:
            task_phase = phase_index_by_task.get(task_id)
            dependency_phase = phase_index_by_task.get(dependency)
            if (
                task_phase is not None
                and dependency_phase is not None
                and dependency_phase >= task_phase
            ):
                errors.append(
                    f"{sequence_prefix}.phase_dependency_dag: {task_id} depends "
                    f"on {dependency} without a strictly earlier committed "
                    "acceptance phase"
                )

    required_protected_acceptance_paths = {
        HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
        NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
    }
    protected_paths = config.get("protected_paths")
    if (
        type(protected_paths) is not list
        or any(type(path) is not str for path in protected_paths)
        or len(protected_paths) != len(set(protected_paths))
        or not required_protected_acceptance_paths.issubset(set(protected_paths))
    ):
        errors.append(
            f"{prefix}.protected_paths: all reserved native DuckDB acceptance "
            "paths must be unique and protected"
        )

    expected_activation = {
        "task_id": "ASE3-026",
        "status": "blocked",
        "receipt_path": PROTECTED_RUNTIME_ACTIVATION_RECEIPT_RELATIVE_PATH,
        "receipt_schema": PROTECTED_RUNTIME_ACTIVATION_AUTHORIZATION_SCHEMA,
        "receipt_phase": "pre_effect_authorization",
        "authorization_may_claim_activation_effect": False,
        "post_activation_observation_receipt_path": (
            PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_RELATIVE_PATH
        ),
        "post_activation_observation_receipt_schema": (
            PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_SCHEMA
        ),
        "post_activation_observation_required_for_completion": True,
        "post_activation_required_observations": [
            "lifecycle_process_birth",
            "lifecycle_lease_fence_heartbeat_and_cursor",
            "monitor_process_birth",
            "monitor_lease_fence_heartbeat_and_cursor",
            "refill_append_recompile_dispatch_or_adoption",
        ],
        "one_generation_cas_lease_required": True,
        "operator_review_required": True,
        "strict_validator_and_manifest_binding_required": True,
    }
    activation_config = config.get("protected_runtime_activation")
    if type(activation_config) is not dict:
        errors.append(
            f"{prefix}.protected_runtime_activation: expected exact JSON object"
        )
    else:
        if set(activation_config) != set(expected_activation):
            errors.append(
                f"{prefix}.protected_runtime_activation.keys: exact population "
                "required"
            )
        for field in sorted(set(activation_config) & set(expected_activation)):
            actual_value = activation_config[field]
            expected_value = expected_activation[field]
            if type(expected_value) is bool:
                if type(actual_value) is not bool or actual_value is not expected_value:
                    errors.append(
                        f"{prefix}.protected_runtime_activation.{field}: expected "
                        f"exact JSON boolean {str(expected_value).lower()}"
                    )
            elif not _exact_json_contract_value(actual_value, expected_value):
                errors.append(
                    f"{prefix}.protected_runtime_activation.{field}: expected "
                    f"exact JSON {type(expected_value).__name__} contract"
                )
        if not _exact_json_contract_value(activation_config, expected_activation):
            errors.append(
                f"{prefix}.protected_runtime_activation: exact typed gate required"
            )
        if (
            _mapping_contract_sha256(activation_config)
            != _PROTECTED_RUNTIME_ACTIVATION_CONFIG_SHA256
        ):
            errors.append(
                f"{prefix}.protected_runtime_activation.contract_sha256: exact "
                "parsed gate contract required"
            )
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
        expected_refill_policy = {
            "enable_after_task": "ASE3-026",
            "activation_task_id": "ASE3-026",
            "prompt_program_refill_enabled": False,
            "saga_schema": "ipfs_accelerate_py.agent_supervisor.durable-refill-saga@1",
            "saga_cursor_states": [
                "EVALUATING",
                "APPEND_RESERVED",
                "APPENDED",
                "PLAN_INVALIDATED",
                "RECOMPILED",
                "DISPATCHED",
                "ADOPTED",
            ],
            "saga_terminal_states": ["DISPATCHED", "ADOPTED"],
            "saga_terminal_states_are_alternatives": True,
            "saga_cursor_durable": True,
            "monitor_phase_deadlines_required": True,
            "max_goals_per_epoch": 8,
            "max_tasks_per_epoch": 24,
            "max_open_tasks": 48,
            "max_depth": 3,
            "max_epochs": 3,
            "max_attempts_per_task": 2,
            "unchanged_residual_cooldown_seconds": 3600,
            "mutate_seed_board": False,
        }
        for field, expected in expected_refill_policy.items():
            if refill.get(field) != expected:
                errors.append(f"{prefix}.refill_policy.{field}: expected {expected!r}")
        if refill != expected_refill_policy:
            errors.append(f"{prefix}.refill_policy: exact dormant saga required")
        if _mapping_contract_sha256(refill) != _REFILL_POLICY_CONFIG_SHA256:
            errors.append(
                f"{prefix}.refill_policy.contract_sha256: exact sealed policy "
                "required"
            )
    monitor = config.get("monitor_policy")
    if not isinstance(monitor, Mapping):
        errors.append(f"{prefix}.monitor_policy: expected object")
    else:
        expected_monitor_policy = {
            "enabled": False,
            "detached": True,
            "activation_task_id": "ASE3-026",
            "durable_guardian": "ReviewedHostNamespaceReconciler",
            "guardian_scope": "host_namespace",
            "guardian_review_required": True,
            "semantic_progress_source": "configured_board_scheduler",
            "heartbeat_seconds": 5,
            "stale_control_seconds": 30,
            "semantic_progress_seconds": 300,
            "max_recoveries_per_window": 3,
            "recovery_window_seconds": 1800,
            "canary_task_id": "ASE3-013",
            "canary_observation_seconds": 900,
            "post_recovery_continuous_health_seconds": 900,
            "continuous_health_required": True,
            "monotonic_elapsed_receipt_required": True,
            "prompt_may_override_observation_window": False,
            "running_requires_joined_lifecycle_monitor_evidence": True,
            "running_join_fields": [
                "lifecycle_process_birth",
                "lifecycle_lease",
                "lifecycle_fence",
                "lifecycle_heartbeat",
                "lifecycle_event_cursor",
                "monitor_process_birth",
                "monitor_lease",
                "monitor_fence",
                "monitor_heartbeat",
                "monitor_event_cursor",
            ],
            "immutable_history_and_cursor_vectors_required": True,
            "unknown_outcome_effect_replay_authorized": False,
            "queue_drain_is_completion": False,
            "branch_local_completion_is_completion": False,
        }
        for field, expected in expected_monitor_policy.items():
            if monitor.get(field) != expected:
                errors.append(f"{prefix}.monitor_policy.{field}: expected {expected!r}")
        if monitor != expected_monitor_policy:
            errors.append(f"{prefix}.monitor_policy: exact dormant guardian policy required")
        if _mapping_contract_sha256(monitor) != _MONITOR_POLICY_CONFIG_SHA256:
            errors.append(
                f"{prefix}.monitor_policy.contract_sha256: exact sealed policy "
                "required"
            )
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

    paths_result = _git_diff_names(
        repo_root,
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

    delta = _git_diff_patch(
        repo_root,
        str(_ASE3_019_ATTEMPT2_PRIOR_SEED["merge_base"]),
        source_commit,
    )
    if delta.returncode != 0:
        errors.append(f"{attempt2_prefix}.prior_delta: unable to compute")
    else:
        delta_digest = "sha256:" + hashlib.sha256(delta.stdout).hexdigest()
        if delta_digest != _ASE3_019_ATTEMPT2_PRIOR_SEED[
            "binary_full_index_delta_sha256"
        ]:
            errors.append(f"{attempt2_prefix}.prior_delta: exact digest mismatch")
    changed_paths = _git_diff_names(
        repo_root,
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
    file_snapshots: dict[str, _RegularFileSnapshot] = {}
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
            file_snapshot = _read_regular_snapshot(
                path,
                maximum_bytes=_EVIDENCE_SNAPSHOT_BYTE_BOUNDS.get(
                    filename,
                    MAX_EVIDENCE_SNAPSHOT_BYTES,
                ),
            )
            raw = file_snapshot.raw
            file_snapshots[filename] = file_snapshot
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

    authority_specs: list[tuple[str, str]] = []
    if payloads[PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME].get(
        "schema"
    ) == PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA:
        authority_specs.append(
            (
                PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME,
                PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
            )
        )
    if payloads[MANIFEST_FILENAME].get("schema") in {
        ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
        RELOAD_CONVERGENCE_MANIFEST_SCHEMA,
    }:
        authority_specs.append(
            (
                MANIFEST_FILENAME,
                f"{_CONVERGENCE_RELATIVE_ROOT}/{MANIFEST_FILENAME}",
            )
        )
    for filename, relative_path in authority_specs:
        if repo_root is None:
            errors.append(
                f"{filename}: repository root is required for authority files"
            )
            continue
        try:
            _require_authority_file_snapshot(
                file_snapshots[filename],
                repository_root=Path(repo_root),
                expected_relative_path=relative_path,
            )
        except ValueError as exc:
            errors.append(str(exc))
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
        board_raw, board_tasks = _load_taskboard_snapshot(board_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"taskboard_snapshot: {exc}")
    else:
        try:
            board_text = board_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"taskboard_snapshot: {exc}")
            board_text = ""
        acceptance_phase, phase_errors = _operator_acceptance_phase(
            tasks=board_tasks,
            artifact_root=root,
            manifest=manifest,
        )
        errors.extend(phase_errors)
        if board_text:
            errors.extend(
                _validate_protected_task_block_bytes(
                    board_text,
                    phase=acceptance_phase,
                )
            )
        phase_statuses = _sequential_task_statuses_after(acceptance_phase)
        is_reload = acceptance_phase == "L"
        has_accepted_019 = (
            phase_statuses.get("ASE3-019") == "completed"
        )
        errors.extend(
            _validate_provider_attempt_reload_gate(
                tasks=board_tasks,
                artifact_root=root,
                acceptance_phase=has_accepted_019 and not is_reload,
                reload_phase=is_reload,
            )
        )
        errors.extend(
            _validate_provider_fallback_task_contract(
                tasks=board_tasks,
                expected_status=phase_statuses.get("ASE3-019", "todo"),
            )
        )
        errors.extend(
            _validate_false_completion_repair_tasks(
                tasks=board_tasks,
                expected_status=phase_statuses.get("ASE3-023", "todo"),
            )
        )
        errors.extend(
            _validate_program_plan_expansion(
                tasks=board_tasks,
                artifact_root=root,
                expected_statuses=phase_statuses,
            )
        )
        if taskboard_path is None and repo_root is not None:
            errors.extend(
                _validate_program_scheduler_projection(
                    repo_root=Path(repo_root),
                    tasks=board_tasks,
                )
            )
        if (
            check_repository
            and repo_root is not None
            and taskboard_path is None
            and acceptance_phase in SEQUENTIAL_ACCEPTANCE_PHASES
        ):
            errors.extend(
                _validate_protected_file_authority(
                    repo_root=Path(repo_root),
                    phase=acceptance_phase,
                )
            )
        if (
            acceptance_phase in SEQUENTIAL_ACCEPTANCE_PHASES
            and acceptance_phase != "Q"
        ):
            phase_packet_errors, phase_packet_checked = (
                _validate_sequential_phase_packet(
                    phase=acceptance_phase,
                    artifact_root=root,
                    manifest=manifest,
                    repo_root=(
                        Path(repo_root)
                        if check_repository and repo_root is not None
                        else None
                    ),
                    fallback_authorization=fallback_authorization,
                    fallback_authorization_raw=raw_artifacts[
                        PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME
                    ],
                    manifest_raw=raw_artifacts[MANIFEST_FILENAME],
                    taskboard_raw=board_raw,
                )
            )
            errors.extend(phase_packet_errors)
            checked.extend(phase_packet_checked)

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
