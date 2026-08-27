#!/usr/bin/env python3
"""Run the additive LGCVF DuckDB + Quack successor controller.

The canonical run-v16 database is forensic input and the sealed run-v17/run-v23
generations remain preserved recovery history.  The active operator has two
explicit stages:

* ``bootstrap`` materializes the exact tracked candidate projection and
  atomically publishes one no-overwrite run-v39 database with provenance;
* ``bootstrap-sealed-continuity`` admits a separately preserved run-v17 only
  into the legacy run-v23 boundary through six explicit raw-byte pins;
* ``launch`` owns run-v39 in-process, starts exactly one foreground
  configured-board scheduler child, and services the closed mutation inbox.

The Quack attach credential exists only in the controller's memory and in the
trusted scheduler process environment.  It is never placed in argv, status,
logs, or a token-vault file.  Implementation-provider environments are still
scrubbed by the existing multi-supervisor boundary.

DuckLake is deliberately a separate, stopped-checkpoint observation.  The
``projection-once`` command writes a physically distinct BoardControlPlane
catalog and marks it non-authoritative; neither ``launch`` nor the configured
scheduler reads that projection for scheduling, leasing, or completion.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import copy
import ctypes
import errno
import fcntl
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
_AMBIENT_PYTHONPATH: Final = frozenset(
    item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
)
_NESTED_DATASETS_ROOT: Final = ROOT / "ipfs_datasets_py"
sys.path[:] = [
    str(ROOT),
    str(_NESTED_DATASETS_ROOT),
    *(
        item
        for item in sys.path
        if item
        and item not in _AMBIENT_PYTHONPATH
        and item not in {str(ROOT), str(_NESTED_DATASETS_ROOT)}
        and not item.startswith("__editable__.")
    ),
]
_RUNTIME_PYCACHE: Final = tempfile.TemporaryDirectory(
    prefix=f"lgcvf-quack-pycache-{os.geteuid()}-"
)
os.chmod(_RUNTIME_PYCACHE.name, 0o700)
sys.pycache_prefix = _RUNTIME_PYCACHE.name
PROGRAM_ROOT_RELATIVE: Final = Path(
    "data/agent_supervisor/logic_governed_compositional_verification_fabric"
)
SOURCE_RUN_RELATIVE: Final = PROGRAM_ROOT_RELATIVE / "run-v17"
LEGACY_SUCCESSOR_RUN_RELATIVE: Final = PROGRAM_ROOT_RELATIVE / "run-v23"
SUCCESSOR_RUN_RELATIVE: Final = PROGRAM_ROOT_RELATIVE / "run-v39"
SOURCE_DATABASE_RELATIVE: Final = SOURCE_RUN_RELATIVE / "control.duckdb"
SUCCESSOR_DATABASE_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "control.duckdb"
OWNER_STATE_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "quack-owner"
PROVENANCE_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence" / "quack-successor-provenance.json"
)
CONTROLLER_STATUS_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "controller.status.json"
CONTROLLER_LOCK_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "controller.lock"
CONTROLLER_LOG_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "logs" / "scheduler.log"
PROJECTION_ROOT_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "ducklake-board-projection"
PROJECTION_RECEIPT_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence" / "ducklake-board-projection.json"
)
STOPPED_STATE_CONTINUITY_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence" / "stopped-state-continuity.json"
)
STOPPED_STATE_RESTART_ADMISSION_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence" / "stopped-state-restart-admission.json"
)
ABANDONED_OWNER_RECOVERY_EVIDENCE_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence"
)
MATERIALIZER_RELATIVE: Final = Path(
    "scripts/materialize_logic_governed_compositional_verification_fabric_control_plane.py"
)
DEFAULT_SUCCESSOR_CONFIG_RELATIVE: Final = Path(
    "config/agent_supervisor_logic_governed_compositional_verification_fabric_quack_candidate_scheduler.json"
)

PROVENANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-quack-successor-provenance@2"
)
NATIVE_RESUME_ADMISSION_MODE: Final = "tracked_candidate_initial_projection_reset"
NATIVE_RESUME_SOURCE_GENERATION: Final = "lgcvf-tracked-candidate-projection"
NATIVE_RESUME_LIVE_CONTINUITY_REQUIRED_ERROR: Final = (
    "native-resume state changed after initial admission; restart requires an "
    "unimplemented live-continuity receipt"
)
NATIVE_RESUME_PROVENANCE_BINDING_ERROR: Final = (
    "native-resume provenance binding differs"
)
SUCCESSOR_STORE_GENERATION: Final = "lgcvf-run-v39"
LEGACY_BOARD_UNSTALL_POLICY_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_LEGACY_BOARD_UNSTALL_POLICY"
)
INTERNAL_CLIENT_GRANT_TTL_SECONDS: Final = 86_400.0
INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS: Final = 43_200.0
STATE_OWNER_BOOTSTRAP_CLIENT_TIMEOUT_SECONDS: Final = 1.0
STATE_OWNER_BOOTSTRAP_PROCESS_STOP_GRACE_SECONDS: Final = 35.0
# Four lanes cold-start serially before the controller can observe the
# required stable all-ready interval.  Keep that finite boundary comfortably
# above the measured four-lane cold-start plus stability budget.
STATE_OWNER_BOOTSTRAP_READY_TIMEOUT_SECONDS: Final = 180.0
STATE_OWNER_BOOTSTRAP_STABILITY_SECONDS: Final = 12.0
LGCVF_SCHEDULER_TREE_STOP_GRACE_SECONDS: Final = 300.0
LGCVF_DATABASE_OWNER_SESSIONS: Final = tuple(
    f"lgcvf-quack-lane-{index}" for index in range(4)
)
LGCVF_TASK_ALIASES: Final = (
    "LGCVF-001",
    "LGCVF-002",
    "LGCVF-010",
    "LGCVF-020",
    "LGCVF-030",
    "LGCVF-040",
    "LGCVF-050",
    "LGCVF-051",
    "LGCVF-060",
    "LGCVF-061",
    "LGCVF-070",
    "LGCVF-071",
    "LGCVF-080",
    "LGCVF-081",
    "LGCVF-090",
    "LGCVF-091",
    "LGCVF-100",
    "LGCVF-101",
    "LGCVF-102",
    "LGCVF-110",
    "LGCVF-111",
    "LGCVF-112",
    "LGCVF-113",
    "LGCVF-120",
    "LGCVF-121",
    "LGCVF-122",
    "LGCVF-123",
    "LGCVF-124",
)
SEALED_CONTINUITY_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-target-only-initial-continuity-verification@1"
)
SEALED_CONTINUITY_MODE: Final = "target_only_initial_continuity"
SEALED_CONTINUITY_AUTHORITY_CEILING: Final = "operational_continuity_only"
FRESH_RECOVERY_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-receipt@1"
)
FRESH_RECOVERY_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-manifest@1"
)
BOOTSTRAP_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-duckdb-materialization@1"
)
NATIVE_RESUME_STAGE_DIRECTORIES: Final = frozenset(
    {
        "evidence",
        "evidence/bootstrap",
    }
)
NATIVE_RESUME_STAGE_LOCK_FILES: Final = frozenset(
    {
        ".control.coordination.duckdb.lock",
        ".control.coordination.duckdb.writer.lock",
        ".control.duckdb.intent.lock",
        ".control.duckdb.lock",
        ".control.duckdb.migration.lock",
        ".control.execution.duckdb.lock",
        ".control.execution.duckdb.writer.lock",
    }
)
NATIVE_RESUME_STAGE_DATA_FILES: Final = frozenset(
    {
        "control.coordination.duckdb",
        "control.duckdb",
        "control.execution.duckdb",
        "evidence/bootstrap/materialization.json",
        "evidence/quack-successor-provenance.json",
    }
)
CONTROLLER_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-quack-successor-status@1"
)
PROJECTION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-ducklake-board-projection@2"
)
STOPPED_STATE_CONTINUITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-stopped-state-continuity@2"
)
STOPPED_STATE_CONTINUITY_ADMISSION_MODE: Final = (
    "typed_stopped_state_continuity"
)
STOPPED_STATE_LIVE_OWNER_EVIDENCE_MODE: Final = "live_owner_clean_stop"
STOPPED_STATE_RECOVERED_EVIDENCE_MODE: Final = (
    "durable_stopped_status_recovery"
)
STOPPED_RECOVERY_DURABLE_ANCHOR_MODE: Final = (
    "durable_stopped_status_anchors"
)
STOPPED_RECOVERY_REVIEWED_LEGACY_MODE: Final = (
    "reviewed_legacy_preflight"
)
QUACK_STATE_SERVER_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-state-server@1"
)
STOPPED_RECOVERY_ANCHORS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-stopped-recovery-anchors@1"
)
STOPPED_RECOVERY_PREFLIGHT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-stopped-recovery-preflight@1"
)
STOPPED_RECOVERY_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-stopped-recovery-result@1"
)
STOPPED_RECOVERY_OPERATION: Final = "reviewed_stopped_continuity_recovery"
FAILED_START_CONTINUITY_ADMISSION_MODE: Final = (
    "typed_failed_start_continuity"
)
FAILED_START_LIVE_OWNER_EVIDENCE_MODE: Final = (
    "live_owner_failed_start_stop"
)
FAILED_START_REVIEWED_EVIDENCE_MODE: Final = (
    "reviewed_failed_start_status_recovery"
)
FAILED_START_RECOVERY_ANCHORS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-failed-start-recovery-anchors@1"
)
FAILED_START_RECOVERY_PREFLIGHT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-failed-start-recovery-preflight@1"
)
FAILED_START_RECOVERY_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-failed-start-recovery-result@1"
)
FAILED_START_RECOVERY_OPERATION: Final = (
    "reviewed_failed_start_continuity_recovery"
)
FAILED_START_SOURCE_MAINTENANCE_PREFLIGHT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-failed-start-source-maintenance-preflight@1"
)
FAILED_START_SOURCE_MAINTENANCE_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-failed-start-source-maintenance-result@1"
)
FAILED_START_SOURCE_MAINTENANCE_OPERATION: Final = (
    "reviewed_failed_start_source_maintenance_reseal"
)
STOPPED_TASK_HISTORY_AUDIT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-stopped-task-history-audit@1"
)
PROTECTED_QUALIFICATION_COMPLETION_PREFLIGHT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-protected-qualification-completion-preflight@1"
)
PROTECTED_QUALIFICATION_COMPLETION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-protected-qualification-completion-result@1"
)
PROTECTED_QUALIFICATION_COMPLETION_INTENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-protected-qualification-completion-intent@1"
)
PROTECTED_QUALIFICATION_COMPLETION_OPERATION: Final = (
    "database_legacy_history_gap_protected_qualification_complete"
)
PROTECTED_QUALIFICATION_COMPLETION_PRIOR_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-database-claim-recovery@1"
)
PROTECTED_QUALIFICATION_COMPLETION_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-protected-qualification-completion-status@1"
)
PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS: Final = "LGCVF-113"
PROTECTED_QUALIFICATION_COMPLETION_TASK_CID: Final = (
    "baguqeerakwvsckoysv5edcru3makxvmcwjjm2alzam5umsyqbkp3efngyqpa"
)
PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION: Final = 8
PROTECTED_QUALIFICATION_COMPLETION_PRIOR_BODY_SHA256: Final = (
    "sha256:f8fe4ec9875cf786f2ca909b2268c784b080e9c86599b1e8d367d982e606829d"
)
PROTECTED_QUALIFICATION_COMPLETION_DEPENDENCIES: Final = {
    "LGCVF-111": (
        "baguqeerau4vdgcyn3sdorik7zawwgwrhy7mxxydb6gashctytbmdzwtmumea"
    ),
    "LGCVF-112": (
        "baguqeeramxjolmqp2rh7r5vfrrs5ne3mqqhbxh3pn74k27h5tbpf4luasfkq"
    ),
}
PROTECTED_QUALIFICATION_RELATIVE: Final = Path(
    "scripts/qualify_logic_governed_compositional_verification_fabric.py"
)
PROTECTED_QUALIFICATION_RESULT_RELATIVE: Final = (
    PROGRAM_ROOT_RELATIVE / "independent_qualification_result.json"
)
PROTECTED_QUALIFICATION_CHECK_TIMEOUT_SECONDS: Final = 1_800.0
PROTECTED_QUALIFICATION_COMMAND_TIMEOUT_SECONDS: Final = 300.0
PROTECTED_QUALIFICATION_COMMAND_GRANT_TTL_SECONDS: Final = 600.0
ABANDONED_OWNER_RECOVERY_PREFLIGHT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-abandoned-owner-recovery-preflight@1"
)
ABANDONED_OWNER_RECOVERY_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-abandoned-owner-recovery-result@1"
)
ABANDONED_OWNER_RECOVERY_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-abandoned-owner-recovery-status@1"
)
ABANDONED_OWNER_RECOVERY_INTENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-abandoned-owner-recovery-intent@1"
)
ABANDONED_OWNER_RECOVERY_OPERATION: Final = (
    "reviewed_abandoned_owner_checkpoint_recovery"
)
FAILED_START_TRUSTED_FINALLY_MODE: Final = "trusted_operator_finally"
FAILED_START_REVIEWED_LEGACY_MODE: Final = "reviewed_legacy_preflight"
FAILED_START_STATUS_ERROR: Final = "unclean_controller_shutdown"
FAILED_START_REASON_SCHEDULER_EXITED: Final = (
    "scheduler_exited_before_lane_attach"
)
FAILED_START_REASON_BOOTSTRAP_FAILED: Final = (
    "state_owner_bootstrap_failed_closed"
)
FAILED_START_REASON_BOOTSTRAP_TIMEOUT: Final = (
    "state_owner_bootstrap_readiness_timeout"
)
FAILED_START_REASON_LEGACY_UNCLASSIFIED: Final = (
    "legacy_unclassified_pre_ready_failure"
)
FAILED_START_REASON_ABANDONED_OWNER_RECOVERED: Final = (
    "abandoned_pre_ready_owner_recovered"
)
FAILED_START_REASON_OPERATOR_STOP: Final = "operator_stop_before_lane_attach"
FAILED_START_TRUSTED_RECOVERY_REASONS: Final = frozenset(
    {
        FAILED_START_REASON_SCHEDULER_EXITED,
        FAILED_START_REASON_BOOTSTRAP_FAILED,
        FAILED_START_REASON_BOOTSTRAP_TIMEOUT,
        FAILED_START_REASON_ABANDONED_OWNER_RECOVERED,
        FAILED_START_REASON_OPERATOR_STOP,
    }
)
FAILED_START_RECOVERY_REASONS: Final = (
    FAILED_START_TRUSTED_RECOVERY_REASONS
    | {FAILED_START_REASON_LEGACY_UNCLASSIFIED}
)
STOPPED_SNAPSHOT_REQUIRED_SEALS: Final = (
    getattr(fcntl, "F_SEAL_SEAL", 0)
    | getattr(fcntl, "F_SEAL_SHRINK", 0)
    | getattr(fcntl, "F_SEAL_GROW", 0)
    | getattr(fcntl, "F_SEAL_WRITE", 0)
)
STOPPED_SNAPSHOT_AGGREGATE_MAX_BYTES: Final = 512 * 1024 * 1024
INITIAL_PROVENANCE_PROJECTION_ADMISSION_MODE: Final = "initial_provenance"
TOKEN_ENV: Final = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
TOKEN_FILE_ENV: Final = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE"
DATABASE_PROGRAM_JSON_ENV: Final = "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON"
STORE_GENERATION_ENV: Final = "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"
BOARD_EXTENSION_INSTALL_POLICY_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY"
)
BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY: Final = "load_only"
LGCVF_LIVE_NATIVE_AUTHORIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-configured-board-native-launch-authorization@1"
)
LGCVF_LIVE_SCHEDULER_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler"
)
LGCVF_LIVE_CAPSULE_MANIFEST_MEMBER: Final = (
    ".lgcvf-configured-board-live-capsule-manifest.json"
)
LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES: Final = (
    "ipfs_accelerate_py.agent_implementation_route",
    "ipfs_accelerate_py.llm_router",
    "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts",
    "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
    "ipfs_accelerate_py.agent_supervisor.merge.database_coordination",
    "ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry",
    "ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle",
    "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
    "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
    "ipfs_accelerate_py.agent_supervisor.runtime.process_security",
    "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
    "ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane",
    "ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client",
    "ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap",
    "ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source",
)
LGCVF_LIVE_REPOSITORY_MODULE_PREFIXES: Final = (
    "ipfs_accelerate_py",
    "ipfs_datasets_py",
    "scripts",
)
LGCVF_LIVE_QUALIFICATION_HOMES_RELATIVE: Final = Path(
    SUCCESSOR_RUN_RELATIVE / "qualification-homes"
)
LGCVF_LIVE_RENDERED_ENV_NAMES: Final = frozenset(
    {
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT",
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        "IPFS_ACCELERATE_AGENT_EVENT_STORE_PATH",
        "IPFS_ACCELERATE_AGENT_EXPORT_PROFILE",
        "IPFS_ACCELERATE_AGENT_GROK_MODEL",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER",
        "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT",
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH",
        "IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE",
        "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE",
        "IPFS_ACCELERATE_AGENT_STATE_FAILOVER_POLICY",
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
        "IPFS_ACCELERATE_AGENT_TASK_SOURCE_KIND",
    }
)
SECRET_HANDLE: Final = f"env://{TOKEN_ENV}"
APPROVED_BOARD_BRANCH: Final = (
    "agent/logic-governed-compositional-verification-fabric-v1"
)
APPROVED_REMOTE_BRANCH_REF: Final = "refs/remotes/github/" + APPROVED_BOARD_BRANCH
MAX_DATABASE_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_JSON_BYTES: Final = 4 * 1024 * 1024
MAX_SECRET_SURFACE_BYTES: Final = 1024 * 1024 * 1024
MAX_STOP_SECONDS: Final = 360.0
UNIX_SOCKET_PATH_CEILING: Final = 100
COMPLETED_TASK_IDS: Final = (
    "LGCVF-001",
    "LGCVF-002",
    "LGCVF-010",
    "LGCVF-020",
    "LGCVF-030",
    "LGCVF-040",
    "LGCVF-050",
    "LGCVF-051",
    "LGCVF-060",
    "LGCVF-061",
    "LGCVF-070",
    "LGCVF-071",
    "LGCVF-080",
)
TODO_TASK_IDS: Final = (
    "LGCVF-081",
    "LGCVF-090",
    "LGCVF-091",
    "LGCVF-100",
    "LGCVF-101",
    "LGCVF-102",
    "LGCVF-110",
    "LGCVF-111",
    "LGCVF-112",
    "LGCVF-113",
    "LGCVF-120",
    "LGCVF-122",
    "LGCVF-124",
)
BLOCKED_TASK_IDS: Final = ("LGCVF-121", "LGCVF-123")
CONSTRUCTION_COMPLETED_TASK_IDS: Final = COMPLETED_TASK_IDS[:7]
RECOVERED_COMPLETED_TASK_IDS: Final = COMPLETED_TASK_IDS[7:]
SEALED_CONTINUITY_EXPECTED_PINS: Final = {
    "control_sha256": (
        "sha256:c931eb71c8ef861c0b4823341989298311a11414b5a7e69ec13f74db62c09238"
    ),
    "coordination_sha256": (
        "sha256:1882695aba63a3d872cbbb6bb737eb173ea81fd9e0b8b6a5131f11f10f7fa2c4"
    ),
    "execution_sha256": (
        "sha256:ca13093d54c55461eea9250b36a06f16764b51e70f0e25965efb207bafd7e9a5"
    ),
    "bootstrap_sha256": (
        "sha256:dd8baaeaf285a23a4e848f03e4a1fd0532c4127e67210d63896e557219b126ab"
    ),
    "manifest_sha256": (
        "sha256:ba418511fec39660765763b012781b8109d437dc02008c01aa1374f843727c71"
    ),
    "recovery_receipt_sha256": (
        "sha256:24fcad13eb74537b1cd0f7531e27282833a77782323aba4a9e2b98c787b013f2"
    ),
}
SEALED_CONTINUITY_EXPECTED_IDENTITIES: Final = {
    "bootstrap_receipt_cid": (
        "baguqeeraujtyr6ywjlmjagd5ijtvkvcxkag5hrtdhyonb66cyhfq55zpfvaa"
    ),
    "manifest_cid": ("baguqeeravix5cxsnflvjmvniwzpqtkrstappy3z5vgjehgk2xlwdn3yhq62a"),
    "receipt_cid": ("baguqeeramzbpvvpb262jwlqa627d4zbqip6tlg6q5gxycdvr4gaoqonpt5ca"),
    "population_root": (
        "baguqeerar2vrvf44pbumffg65zh5etmged3va3ocumu75v3fdgzqbzlk4nja"
    ),
    "source_evidence_cid": (
        "baguqeera4aybmwbobzlojc4u2cdqxxznmd4bgjkwv2kqka5cukhywnmhy4uq"
    ),
    "sealed_operational_verification_root": (
        "baguqeeraqdjtxgx6wjxkb6u3635s633xy7ymqjby4xxo7xq6wrstfnzym4pa"
    ),
    "target_source_head": "092c95725b9642daa479162d631eff3983e67af6",
    "target_source_tree": "83488b19d20f06da44762a2dfecb4a2666c3b192",
}
GIT_EXECUTABLE: Final = Path("/usr/bin/git")
GIT_TIMEOUT_SECONDS: Final = 120.0
MAX_RUNTIME_GITLINK_DEPTH: Final = 16


class SuccessorOperatorError(RuntimeError):
    """The successor cannot be admitted without weakening a boundary."""


def _closed_option_values(argv: Sequence[str], option: str) -> tuple[str, ...]:
    """Read one closed CLI option without accepting a missing value."""

    values: list[str] = []
    tokens = tuple(str(item) for item in argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise SuccessorOperatorError(f"{option} has no value")
            values.append(tokens[index + 1])
            index += 2
            continue
        prefix = option + "="
        if token.startswith(prefix):
            value = token[len(prefix) :]
            if not value:
                raise SuccessorOperatorError(f"{option} has no value")
            values.append(value)
        index += 1
    return tuple(values)


def _seal_lgcvf_execution_route_policy(
    *,
    server: Any,
    program: Any,
    identity: Any,
    controller_birth: Any,
    owner_socket: Path,
) -> Any:
    """Seal the exact 28-task Grok/Codex route through a temporary grant."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
        process_birth_id,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
        QuackStateClient,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        TypedDatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_SOCKET_ENV,
        TYPED_STATE_OWNER_TOKEN_ENV,
    )

    birth_id = process_birth_id(controller_birth)
    if str(getattr(identity, "process_birth_id", "") or "") != birth_id:
        raise SuccessorOperatorError(
            "route sealer process birth differs from the state owner"
        )
    token, grant = server.issue_typed_client_grant_record(
        client_id="lgcvf-route-sealer",
        process_birth_id=birth_id,
        allowed_operations=(
            "whoami_metadata",
            "load_store_generation",
            "executor_control_snapshot",
            "executor_task_projection_page",
        ),
        allowed_command_operations=(),
        peer_pid=os.getpid(),
        ttl_seconds=60.0,
    )
    client: Any | None = None
    projection: Any | None = None
    previous_token = os.environ.get(TYPED_STATE_OWNER_TOKEN_ENV)
    previous_socket = os.environ.get(TYPED_STATE_OWNER_SOCKET_ENV)
    try:
        os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = token
        os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = str(owner_socket)
        client = QuackStateClient(
            owner_id="lgcvf-route-sealer",
            store_id=str(program.store_id),
            process_birth_id=birth_id,
        )
        client.attach(
            str(program.quack_endpoint),
            server_id=str(identity.server_id),
        )
        if previous_token is None:
            os.environ.pop(TYPED_STATE_OWNER_TOKEN_ENV, None)
        else:
            os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = previous_token
        if previous_socket is None:
            os.environ.pop(TYPED_STATE_OWNER_SOCKET_ENV, None)
        else:
            os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = previous_socket
        projection = TypedDatabaseTaskSource(client, owns_client=False)
        execution_modes = {
            alias: GROK_CODEX_EXECUTION_MODE for alias in LGCVF_TASK_ALIASES
        }
        policy = projection.seal_execution_route_policy(execution_modes)
        entries = tuple(policy.entries_by_cid.values())
        if (
            len(entries) != len(LGCVF_TASK_ALIASES)
            or {entry.task_alias for entry in entries} != set(LGCVF_TASK_ALIASES)
            or any(
                entry.execution_mode != GROK_CODEX_EXECUTION_MODE
                for entry in entries
            )
        ):
            raise SuccessorOperatorError(
                "sealed execution route differs from the admitted LGCVF population"
            )
        return policy
    finally:
        if previous_token is None:
            os.environ.pop(TYPED_STATE_OWNER_TOKEN_ENV, None)
        else:
            os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = previous_token
        if previous_socket is None:
            os.environ.pop(TYPED_STATE_OWNER_SOCKET_ENV, None)
        else:
            os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = previous_socket
        if projection is not None:
            projection.close()
        if client is not None:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)


class _LgcvfStateOwnerBootstrapBroker:
    """Mint one exact-birth typed grant per live LGCVF lane daemon."""

    def __init__(
        self,
        *,
        channel: socket.socket,
        descriptor: int,
        server: Any,
        scheduler_birth: Any,
        endpoint: str,
        socket_path: Path,
        store_id: str,
        execution_route_policy: Any,
        process_stop_grace_seconds: float = (
            STATE_OWNER_BOOTSTRAP_PROCESS_STOP_GRACE_SECONDS
        ),
    ) -> None:
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            validate_state_owner_bootstrap_listener,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
            TaskExecutionRoutePolicy,
        )

        if not isinstance(execution_route_policy, TaskExecutionRoutePolicy):
            raise SuccessorOperatorError(
                "bootstrap broker requires an immutable execution route policy"
            )
        validate_state_owner_bootstrap_listener(descriptor)
        self.channel = channel
        self.descriptor = int(descriptor)
        self.server = server
        self.scheduler_birth = scheduler_birth
        self.endpoint = str(endpoint)
        self.socket_path = Path(socket_path)
        self.store_id = str(store_id)
        self.execution_route_policy = execution_route_policy
        self.process_stop_grace_seconds = float(process_stop_grace_seconds)
        if (
            not math.isfinite(self.process_stop_grace_seconds)
            or self.process_stop_grace_seconds < 0.05
            or self.process_stop_grace_seconds > 300.0
        ):
            raise SuccessorOperatorError(
                "bootstrap broker process-stop grace is invalid"
            )
        self.stopping = threading.Event()
        self.failure = ""
        self.last_rejection = ""
        self.rejection_count = 0
        self._lock = threading.RLock()
        self._accepted: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="lgcvf-state-owner-bootstrap",
            daemon=True,
        )
        self._started = False
        self.current_by_session: dict[str, dict[str, Any]] = {}
        self.active_grants: dict[str, str] = {}

    @property
    def ready_sessions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                session
                for session in LGCVF_DATABASE_OWNER_SESSIONS
                if session in self.current_by_session
                and session in self.active_grants
            )

    @property
    def live_ready_signature(self) -> tuple[str, ...]:
        """Return all four exact daemon births only while each remains alive."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            ProcessBirthIdentity,
            owner_liveness,
        )

        with self._lock:
            records = {
                session: dict(self.current_by_session.get(session) or {})
                for session in LGCVF_DATABASE_OWNER_SESSIONS
            }
        signature: list[str] = []
        for session in LGCVF_DATABASE_OWNER_SESSIONS:
            record = records[session]
            raw_birth = record.get("daemon_process_birth")
            birth_id = str(record.get("daemon_process_birth_id") or "")
            if (
                not isinstance(raw_birth, Mapping)
                or not birth_id
                or owner_liveness(ProcessBirthIdentity.from_dict(raw_birth))
                is not OwnerLiveness.ALIVE
            ):
                return ()
            signature.append(birth_id)
        return tuple(signature)

    def start(self) -> None:
        if self._started:
            raise SuccessorOperatorError("bootstrap broker was already started")
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self.stopping.set()
        with self._lock:
            accepted = self._accepted
        if accepted is not None:
            try:
                accepted.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                accepted.close()
            except OSError:
                pass
        try:
            self.channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.channel.close()
        except OSError:
            pass
        if self._started:
            self._thread.join(timeout=5.0)
        if self._started and self._thread.is_alive():
            raise SuccessorOperatorError(
                "state-owner bootstrap broker did not stop"
            )
        self._fence_admitted_births()
        with self._lock:
            grant_ids = tuple(self.active_grants.values())
        revoke_failure = ""
        for grant_id in grant_ids:
            try:
                self.server.revoke_typed_client_grant(grant_id)
            except Exception as exc:  # noqa: BLE001 - revoke every lane.
                revoke_failure = revoke_failure or type(exc).__name__
        if revoke_failure:
            raise SuccessorOperatorError(
                "state-owner bootstrap grant revocation failed: "
                + revoke_failure
            )
        with self._lock:
            for session, grant_id in tuple(self.active_grants.items()):
                if grant_id in grant_ids:
                    self.active_grants.pop(session, None)

    def _admitted_births(self) -> tuple[Any, ...]:
        """Return each exact current credential-holder birth once."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            ProcessBirthIdentity,
        )

        with self._lock:
            records = tuple(
                dict(record) for record in self.current_by_session.values()
            )
        result: list[Any] = []
        seen: set[tuple[int, int, str]] = set()
        for field in (
            "supervisor_process_birth",
            "daemon_process_birth",
        ):
            for record in records:
                raw = record.get(field)
                if not isinstance(raw, Mapping):
                    raise SuccessorOperatorError(
                        "state-owner admitted process birth is unavailable"
                    )
                try:
                    birth = ProcessBirthIdentity.from_dict(raw)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise SuccessorOperatorError(
                        "state-owner admitted process birth is malformed"
                    ) from exc
                if birth.pid <= 1 or birth.start_time_ticks <= 0:
                    raise SuccessorOperatorError(
                        "state-owner admitted process birth is unsafe"
                    )
                key = (birth.pid, birth.start_time_ticks, birth.boot_id)
                if key not in seen:
                    seen.add(key)
                    result.append(birth)
        return tuple(result)

    @staticmethod
    def _signal_admitted_birth(birth: Any, signum: int) -> None:
        """Signal one PID-reuse-resistant admitted birth, or prove it dead."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            owner_liveness,
        )

        if not hasattr(os, "pidfd_open") or not hasattr(
            signal,
            "pidfd_send_signal",
        ):
            raise SuccessorOperatorError(
                "state-owner admitted process fencing requires Linux pidfds"
            )
        pidfd = -1
        try:
            pidfd = os.pidfd_open(birth.pid, 0)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise SuccessorOperatorError(
                "state-owner admitted process pidfd is unavailable"
            ) from exc
        try:
            # Opening the pidfd first makes the subsequent signal immune to a
            # PID disappearing and being reused after this identity check.
            state = owner_liveness(birth)
            if state is OwnerLiveness.DEAD:
                return
            if state is not OwnerLiveness.ALIVE:
                raise SuccessorOperatorError(
                    "state-owner admitted process birth is uninspectable"
                )
            try:
                signal.pidfd_send_signal(pidfd, signum)
            except ProcessLookupError:
                return
            except OSError as exc:
                raise SuccessorOperatorError(
                    "state-owner admitted process could not be signalled"
                ) from exc
        finally:
            os.close(pidfd)

    @staticmethod
    def _live_admitted_births(births: Sequence[Any]) -> tuple[Any, ...]:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            owner_liveness,
        )

        live: list[Any] = []
        for birth in births:
            state = owner_liveness(birth)
            if state is OwnerLiveness.UNKNOWN:
                raise SuccessorOperatorError(
                    "state-owner admitted process became uninspectable"
                )
            if state is OwnerLiveness.ALIVE:
                live.append(birth)
        return tuple(live)

    def _fence_admitted_births(self) -> None:
        """Prove every credential-holding lane birth dead before revocation."""

        births = self._admitted_births()
        live = self._live_admitted_births(births)
        for birth in live:
            self._signal_admitted_birth(birth, signal.SIGTERM)
        deadline = time.monotonic() + self.process_stop_grace_seconds
        while live and time.monotonic() < deadline:
            time.sleep(0.02)
            live = self._live_admitted_births(live)
        for birth in live:
            self._signal_admitted_birth(birth, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while live and time.monotonic() < deadline:
            time.sleep(0.02)
            live = self._live_admitted_births(live)
        if live:
            raise SuccessorOperatorError(
                "state-owner admitted process births survived bounded stop"
            )

    @staticmethod
    def _require_dead(birth_payload: Mapping[str, Any], *, noun: str) -> None:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            ProcessBirthIdentity,
            owner_liveness,
        )

        birth = ProcessBirthIdentity.from_dict(birth_payload)
        if owner_liveness(birth) is not OwnerLiveness.DEAD:
            raise SuccessorOperatorError(f"prior {noun} birth remains live")

    def _supervisor_for_daemon(self, daemon_birth: Any, *, session: str) -> Any:
        from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
            process_birth_id,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            read_process_birth,
        )

        observed_scheduler = read_process_birth(self.scheduler_birth.pid)
        if observed_scheduler != self.scheduler_birth:
            raise SuccessorOperatorError(
                "bootstrap scheduler process birth is no longer exact"
            )
        supervisor = read_process_birth(int(daemon_birth.parent_pid))
        if (
            supervisor is None
            or supervisor.pid <= 1
            or supervisor.parent_pid != self.scheduler_birth.pid
        ):
            raise SuccessorOperatorError(
                "bootstrap daemon is not a child of an admitted lane supervisor"
            )
        before = supervisor
        try:
            raw = Path(f"/proc/{supervisor.pid}/cmdline").read_bytes()
        except OSError as exc:
            raise SuccessorOperatorError(
                "bootstrap lane supervisor argv is unavailable"
            ) from exc
        after = read_process_birth(supervisor.pid)
        if before != after or len(raw) > 1_048_576:
            raise SuccessorOperatorError(
                "bootstrap lane supervisor identity changed during inspection"
            )
        try:
            argv = tuple(
                item.decode("utf-8") for item in raw.split(b"\0") if item
            )
        except UnicodeError as exc:
            raise SuccessorOperatorError(
                "bootstrap lane supervisor argv is malformed"
            ) from exc
        lane_index = LGCVF_DATABASE_OWNER_SESSIONS.index(session)
        exact = {
            "--board-namespace": (
                "logic-governed-compositional-verification-fabric-v1"
            ),
            "--task-shard-count": "4",
            "--task-shard-index": str(lane_index),
            "--state-prefix": f"lgcvf_lane_{lane_index}",
            "--database-owner-session-id": session,
            "--state-owner-bootstrap-fd": str(self.descriptor),
            "--state-owner-bootstrap-store-id": self.store_id,
        }
        if any(
            _closed_option_values(argv, option) != (expected,)
            for option, expected in exact.items()
        ):
            raise SuccessorOperatorError(
                "bootstrap lane supervisor argv differs from its sealed lane"
            )
        supervisor_id = process_birth_id(supervisor)
        for other_session, record in self.current_by_session.items():
            if (
                other_session != session
                and record.get("supervisor_process_birth_id") == supervisor_id
            ):
                raise SuccessorOperatorError(
                    "one lane supervisor requested multiple owner sessions"
                )
        return supervisor

    def _admit(
        self,
        request: Mapping[str, Any],
        *,
        peer_pid: int,
        peer_uid: int,
    ) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
            process_birth_id,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            ProcessBirthIdentity,
            read_process_birth,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA,
            STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
            daemon_required_owner_command_operations,
            daemon_required_owner_operations,
        )

        required = {
            "schema",
            "pid",
            "process_birth",
            "process_birth_id",
            "client_id",
            "store_id",
        }
        if (
            set(request) != required
            or request.get("schema") != STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA
            or self.stopping.is_set()
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap request differs from its closed schema"
            )
        request_birth = request.get("process_birth")
        raw_pid = request.get("pid")
        if (
            isinstance(raw_pid, bool)
            or not isinstance(raw_pid, int)
            or raw_pid <= 1
            or not isinstance(request_birth, Mapping)
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap request identity is malformed"
            )
        pid = raw_pid
        if pid != peer_pid or peer_uid != os.geteuid():
            raise SuccessorOperatorError(
                "state-owner bootstrap SO_PEERCRED identity differs"
            )
        birth_integer_fields = ("pid", "start_time_ticks", "parent_pid")
        if (
            set(request_birth)
            != {"pid", "start_time_ticks", "boot_id", "parent_pid"}
            or any(
                isinstance(request_birth.get(name), bool)
                or not isinstance(request_birth.get(name), int)
                for name in birth_integer_fields
            )
            or request_birth.get("pid") != pid
            or request_birth.get("start_time_ticks", 0) <= 0
            or request_birth.get("parent_pid", -1) < 0
            or not isinstance(request_birth.get("boot_id"), str)
            or len(request_birth.get("boot_id", "")) > 128
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap process birth is malformed"
            )
        try:
            supplied = ProcessBirthIdentity.from_dict(request_birth)
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise SuccessorOperatorError(
                "state-owner bootstrap process birth is malformed"
            ) from exc
        observed = read_process_birth(pid)
        supplied_birth_id = request.get("process_birth_id")
        if (
            not isinstance(supplied_birth_id, str)
            or not supplied_birth_id
            or observed is None
            or observed != supplied
            or process_birth_id(observed) != supplied_birth_id
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap process birth is stale"
            )
        client_id = request.get("client_id")
        requested_store = request.get("store_id")
        if not isinstance(client_id, str) or not isinstance(
            requested_store,
            str,
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap lane scope is malformed"
            )
        matching_sessions = tuple(
            session
            for session in LGCVF_DATABASE_OWNER_SESSIONS
            if client_id == f"database-implementation-daemon:{session}"
        )
        if (
            len(matching_sessions) != 1
            or requested_store != self.store_id
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap lane scope differs from admission"
            )
        session = matching_sessions[0]
        with self._lock:
            supervisor = self._supervisor_for_daemon(observed, session=session)
            prior = self.current_by_session.get(session)
            if prior is not None:
                prior_daemon = prior.get("daemon_process_birth")
                if not isinstance(prior_daemon, Mapping):
                    raise SuccessorOperatorError(
                        "prior daemon bootstrap record is malformed"
                    )
                self._require_dead(prior_daemon, noun="lane daemon")
                prior_supervisor = prior.get("supervisor_process_birth")
                if (
                    isinstance(prior_supervisor, Mapping)
                    and dict(prior_supervisor) != supervisor.to_dict()
                ):
                    self._require_dead(
                        prior_supervisor,
                        noun="lane supervisor",
                    )
            prior_grant = self.active_grants.pop(session, "")
            if prior_grant:
                self.server.revoke_typed_client_grant(prior_grant)
            token, grant = self.server.issue_typed_client_grant_record(
                client_id=client_id,
                process_birth_id=supplied_birth_id,
                allowed_operations=daemon_required_owner_operations(),
                allowed_command_operations=(
                    daemon_required_owner_command_operations()
                ),
                peer_pid=pid,
                ttl_seconds=INTERNAL_CLIENT_GRANT_TTL_SECONDS,
            )
            if self.stopping.is_set():
                self.server.revoke_typed_client_grant(grant.grant_id)
                raise SuccessorOperatorError(
                    "state-owner bootstrap admission closed during grant issue"
                )
            owner_identity = self.server.identity
            if owner_identity is None:
                self.server.revoke_typed_client_grant(grant.grant_id)
                raise SuccessorOperatorError(
                    "state owner lost identity during bootstrap"
                )
            self.current_by_session[session] = {
                "session": session,
                "client_id": client_id,
                "daemon_process_birth": supplied.to_dict(),
                "daemon_process_birth_id": supplied_birth_id,
                "supervisor_process_birth": supervisor.to_dict(),
                "supervisor_process_birth_id": process_birth_id(supervisor),
                "execution_route_policy": (
                    self.execution_route_policy.public_summary()
                ),
                "grant_expires_at_ms": int(grant.expires_at),
                "grant_renew_after": (
                    time.monotonic() + INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS
                ),
            }
            self.active_grants[session] = grant.grant_id
            return {
                "schema": STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
                "ok": True,
                "endpoint": self.endpoint,
                "socket_path": str(self.socket_path),
                "store_id": self.store_id,
                "server_id": str(owner_identity.server_id),
                "client_id": client_id,
                "process_birth_id": supplied_birth_id,
                "token": token,
                "execution_route_policy": self.execution_route_policy.to_dict(),
            }

    def _renew_due_grants(self) -> None:
        """Keep exact live-birth grants bounded and usable indefinitely."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            ProcessBirthIdentity,
            owner_liveness,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
            TypedStateOwnerAuthorizationError,
        )

        now = time.monotonic()
        with self._lock:
            due = tuple(
                (
                    session,
                    grant_id,
                    float(
                        (self.current_by_session.get(session) or {}).get(
                            "grant_renew_after",
                            0.0,
                        )
                    ),
                    dict(self.current_by_session.get(session) or {}),
                )
                for session, grant_id in self.active_grants.items()
            )
        for session, grant_id, renew_after, record in due:
            if now < renew_after:
                continue
            raw_birth = record.get("daemon_process_birth")
            if not isinstance(raw_birth, Mapping):
                raise SuccessorOperatorError(
                    "state-owner renewal daemon birth is unavailable"
                )
            daemon_birth = ProcessBirthIdentity.from_dict(raw_birth)
            liveness = owner_liveness(daemon_birth)
            if liveness is OwnerLiveness.DEAD:
                # The supervisor may already be creating this lane's next
                # exact birth.  Leave the old grant for `_admit` to revoke so
                # this serial broker can accept the replacement immediately.
                continue
            if liveness is not OwnerLiveness.ALIVE:
                raise SuccessorOperatorError(
                    "state-owner renewal daemon birth is uninspectable"
                )
            try:
                renewed = self.server.renew_typed_client_grant(
                    grant_id,
                    ttl_seconds=INTERNAL_CLIENT_GRANT_TTL_SECONDS,
                )
            except TypedStateOwnerAuthorizationError:
                if owner_liveness(daemon_birth) is OwnerLiveness.DEAD:
                    continue
                raise
            with self._lock:
                if self.active_grants.get(session) != grant_id:
                    raise SuccessorOperatorError(
                        "state-owner grant rotated during renewal"
                    )
                record = self.current_by_session.get(session)
                if not isinstance(record, dict):
                    raise SuccessorOperatorError(
                        "state-owner renewal record is unavailable"
                    )
                record["grant_expires_at_ms"] = int(renewed.expires_at)
                record["grant_renew_after"] = (
                    time.monotonic() + INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS
                )

    def _run(self) -> None:
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            StateOwnerBootstrapError,
            _receive_frame,
            _send_frame,
        )

        self.channel.settimeout(1.0)
        while not self.stopping.is_set():
            accepted: socket.socket | None = None
            try:
                self._renew_due_grants()
                accepted, _address = self.channel.accept()
                with self._lock:
                    if self.stopping.is_set():
                        accepted.close()
                        return
                    self._accepted = accepted
                accepted.settimeout(
                    STATE_OWNER_BOOTSTRAP_CLIENT_TIMEOUT_SECONDS
                )
                peer = accepted.getsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_PEERCRED,
                    struct.calcsize("3i"),
                )
                peer_pid, peer_uid, _peer_gid = struct.unpack("3i", peer)
                response = self._admit(
                    _receive_frame(accepted),
                    peer_pid=int(peer_pid),
                    peer_uid=int(peer_uid),
                )
                _send_frame(accepted, response)
            except TimeoutError:
                continue
            except (EOFError, StateOwnerBootstrapError, SuccessorOperatorError) as exc:
                if not self.stopping.is_set():
                    self.last_rejection = type(exc).__name__
                    self.rejection_count += 1
                continue
            except OSError:
                if not self.stopping.is_set() and accepted is None:
                    self.failure = "state_owner_bootstrap_channel_closed"
                    return
                continue
            except BaseException as exc:
                self.failure = type(exc).__name__
                try:
                    self.channel.close()
                except OSError:
                    pass
                return
            finally:
                if accepted is not None:
                    with self._lock:
                        if self._accepted is accepted:
                            self._accepted = None
                    try:
                        accepted.close()
                    except OSError:
                        pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    return content_identity(value)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _contained(root: Path, relative: Path | str) -> Path:
    base = root.resolve()
    candidate = (base / Path(relative)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SuccessorOperatorError(
            f"runtime path escapes repository: {relative}"
        ) from exc
    return candidate


def _paths(root: Path = ROOT) -> dict[str, Path]:
    paths = {
        "source_database": _contained(root, SOURCE_DATABASE_RELATIVE),
        "successor_database": _contained(root, SUCCESSOR_DATABASE_RELATIVE),
        "owner_state": _contained(root, OWNER_STATE_RELATIVE),
        "provenance": _contained(root, PROVENANCE_RELATIVE),
        "controller_status": _contained(root, CONTROLLER_STATUS_RELATIVE),
        "controller_lock": _contained(root, CONTROLLER_LOCK_RELATIVE),
        "controller_log": _contained(root, CONTROLLER_LOG_RELATIVE),
        "projection_root": _contained(root, PROJECTION_ROOT_RELATIVE),
        "projection_receipt": _contained(root, PROJECTION_RECEIPT_RELATIVE),
        "stopped_state_continuity": _contained(
            root, STOPPED_STATE_CONTINUITY_RELATIVE
        ),
        "stopped_state_restart_admission": _contained(
            root, STOPPED_STATE_RESTART_ADMISSION_RELATIVE
        ),
        "abandoned_owner_recovery_evidence": _contained(
            root, ABANDONED_OWNER_RECOVERY_EVIDENCE_RELATIVE
        ),
    }
    socket_identity = hashlib.sha256(
        _canonical_bytes(
            {
                "program": "lgcvf-quack-successor-v1",
                "repository_root": str(root.resolve()),
                "runtime_root": str(_contained(root, SUCCESSOR_RUN_RELATIVE)),
                "database": str(paths["successor_database"]),
            }
        )
    ).hexdigest()[:20]
    owner_socket = (
        Path(tempfile.gettempdir())
        / f"ipfs-accelerate-lgcvf-{os.geteuid()}"
        / f"owner-{socket_identity}.sock"
    )
    if len(os.fsencode(owner_socket)) > UNIX_SOCKET_PATH_CEILING:
        raise SuccessorOperatorError(
            "derived state-owner socket path exceeds its bound"
        )
    paths["owner_socket"] = owner_socket
    return paths


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    noun: str,
    require_private_owner: bool = False,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unreadable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > max_bytes
            or (
                require_private_owner
                and (
                    before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) & 0o077
                )
            )
        ):
            raise SuccessorOperatorError(
                f"{noun} is not a bounded private regular file: {path}"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) > max_bytes or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SuccessorOperatorError(f"{noun} changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _strict_json(
    path: Path,
    *,
    expected_schema: str = "",
    require_private_owner: bool = False,
    verify_content_identity: bool = True,
) -> dict[str, Any]:
    raw = _read_bounded_regular_file(
        path,
        max_bytes=MAX_JSON_BYTES,
        noun="required receipt",
        require_private_owner=require_private_owner,
    )
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(f"receipt is malformed: {path}") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise SuccessorOperatorError(f"receipt is not a canonical object: {path}")
    if expected_schema and value.get("schema") != expected_schema:
        raise SuccessorOperatorError(f"receipt schema differs: {path}")
    claimed = str(value.get("receipt_cid") or value.get("status_cid") or "")
    if claimed and verify_content_identity:
        unsigned = dict(value)
        unsigned.pop("receipt_cid", None)
        unsigned.pop("status_cid", None)
        if claimed != _content_id(unsigned):
            raise SuccessorOperatorError(f"receipt content identity differs: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            if replace:
                os.replace(temporary, path)
            else:
                _rename_noreplace(
                    directory,
                    temporary.name,
                    path.name,
                    noun="immutable JSON receipt",
                )
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _rename_noreplace(
    parent_descriptor: int,
    source_name: str,
    target_name: str,
    *,
    noun: str,
) -> None:
    """Atomically publish one same-parent object without an overwrite fallback."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise SuccessorOperatorError(
            f"atomic no-replace {noun} publication is unavailable"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(target_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    observed_errno = ctypes.get_errno()
    if observed_errno in (errno.EEXIST, errno.ENOTEMPTY):
        raise SuccessorOperatorError(f"refusing to overwrite existing {noun}")
    raise SuccessorOperatorError(
        f"atomic no-replace {noun} publication failed: "
        + os.strerror(observed_errno)
    )


def _rename_directory_noreplace(
    parent_descriptor: int, source_name: str, target_name: str
) -> None:
    """Atomically publish one same-parent directory without overwriting."""

    _rename_noreplace(
        parent_descriptor,
        source_name,
        target_name,
        noun="successor directory",
    )


def _cleanup_successor_stage(
    stage: Path, *, staged_database: Path, staged_provenance: Path
) -> None:
    """Remove only the exact unpublished objects this process created."""

    lock_paths = tuple(
        stage / name
        for name in (
            f".{staged_database.name}.intent.lock",
            f".{staged_database.name}.lock",
            f".{staged_database.name}.migration.lock",
        )
    )
    for path in (staged_provenance, staged_database, *lock_paths):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    cursor = staged_provenance.parent
    while cursor != stage:
        try:
            cursor.rmdir()
        except (FileNotFoundError, OSError):
            break
        cursor = cursor.parent
    try:
        stage.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _remove_staged_database_locks(stage: Path, database_name: str) -> None:
    """Remove only empty, owner-held lock artifacts created by read verification."""

    for name in (
        f".{database_name}.intent.lock",
        f".{database_name}.lock",
        f".{database_name}.migration.lock",
    ):
        path = stage / name
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != 0
        ):
            raise SuccessorOperatorError("staged database lock custody differs")
        path.unlink()


def _open_private_lock(path: Path) -> Any:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise SuccessorOperatorError("controller lock custody is unsafe")
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "a+b")
    except BaseException:
        os.close(descriptor)
        raise


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino))


def _revalidate_generation_bound_controller_lock(
    paths: Mapping[str, Path],
    custody: Mapping[str, Any],
) -> None:
    """Prove that the held lock still belongs to the named run generation."""

    generation = paths["controller_lock"].parent
    handle = custody.get("lock_handle")
    generation_descriptor = custody.get("generation_descriptor")
    if (
        str(custody.get("generation_path") or "") != str(generation)
        or not hasattr(handle, "fileno")
        or getattr(handle, "closed", True)
        or type(generation_descriptor) is not int
        or generation_descriptor < 3
    ):
        raise SuccessorOperatorError(
            "successor generation/controller lock custody is malformed"
        )
    try:
        held_generation = os.fstat(generation_descriptor)
        held_lock = os.fstat(handle.fileno())
        observed_generation_descriptor = os.open(
            generation,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            named_generation = os.fstat(observed_generation_descriptor)
        finally:
            os.close(observed_generation_descriptor)
        named_lock = os.stat(
            paths["controller_lock"].name,
            dir_fd=generation_descriptor,
            follow_symlinks=False,
        )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        raise SuccessorOperatorError(
            "successor generation/controller lock binding changed"
        ) from exc
    if (
        not stat.S_ISDIR(held_generation.st_mode)
        or held_generation.st_uid != os.geteuid()
        or stat.S_IMODE(held_generation.st_mode) & 0o077
        or _inode_identity(held_generation)
        != tuple(custody.get("generation_identity") or ())
        or _inode_identity(named_generation) != _inode_identity(held_generation)
        or not stat.S_ISREG(held_lock.st_mode)
        or held_lock.st_uid != os.geteuid()
        or held_lock.st_nlink != 1
        or stat.S_IMODE(held_lock.st_mode) & 0o077
        or _inode_identity(held_lock) != tuple(custody.get("lock_identity") or ())
        or _inode_identity(named_lock) != _inode_identity(held_lock)
    ):
        raise SuccessorOperatorError(
            "successor generation/controller lock binding changed"
        )


def _open_generation_bound_controller_lock(
    paths: Mapping[str, Path],
    *,
    read_only_existing: bool = False,
) -> dict[str, Any]:
    """Open ``controller.lock`` through a pinned run-generation directory."""

    generation = paths["controller_lock"].parent
    generation_descriptor = -1
    lock_descriptor = -1
    handle: Any | None = None
    try:
        generation_descriptor = os.open(
            generation,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        generation_metadata = os.fstat(generation_descriptor)
        if (
            not stat.S_ISDIR(generation_metadata.st_mode)
            or generation_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(generation_metadata.st_mode) & 0o077
        ):
            raise SuccessorOperatorError(
                "successor generation directory custody is unsafe"
            )
        lock_flags = (
            os.O_RDONLY if read_only_existing else os.O_RDWR | os.O_CREAT
        ) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_descriptor = (
                os.open(
                    paths["controller_lock"].name,
                    lock_flags,
                    dir_fd=generation_descriptor,
                )
                if read_only_existing
                else os.open(
                    paths["controller_lock"].name,
                    lock_flags,
                    0o600,
                    dir_fd=generation_descriptor,
                )
            )
        except FileNotFoundError as exc:
            if read_only_existing:
                raise SuccessorOperatorError(
                    "existing controller lock is unavailable for read-only custody"
                ) from exc
            raise
        lock_metadata = os.fstat(lock_descriptor)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.geteuid()
            or lock_metadata.st_nlink != 1
            or (
                read_only_existing
                and stat.S_IMODE(lock_metadata.st_mode) != 0o600
            )
        ):
            raise SuccessorOperatorError("controller lock custody is unsafe")
        if not read_only_existing:
            os.fchmod(lock_descriptor, 0o600)
        handle = os.fdopen(
            lock_descriptor,
            "rb" if read_only_existing else "a+b",
        )
        lock_descriptor = -1
        custody = {
            "generation_path": str(generation),
            "generation_descriptor": generation_descriptor,
            "generation_identity": _inode_identity(generation_metadata),
            "lock_handle": handle,
            "lock_identity": _inode_identity(lock_metadata),
        }
        _revalidate_generation_bound_controller_lock(paths, custody)
        return custody
    except BaseException:
        if handle is not None:
            handle.close()
        elif lock_descriptor >= 0:
            os.close(lock_descriptor)
        if generation_descriptor >= 0:
            os.close(generation_descriptor)
        raise


def _close_generation_bound_controller_lock(custody: Mapping[str, Any]) -> None:
    handle = custody.get("lock_handle")
    generation_descriptor = custody.get("generation_descriptor")
    if hasattr(handle, "close"):
        handle.close()
    if type(generation_descriptor) is int and generation_descriptor >= 0:
        os.close(generation_descriptor)


def _generation_bound_runtime_path(
    paths: Mapping[str, Path],
    custody: Mapping[str, Any],
    logical_path: Path,
) -> Path:
    """Address one generation member through the already-pinned directory."""

    _revalidate_generation_bound_controller_lock(paths, custody)
    generation = paths["controller_lock"].parent
    try:
        relative = logical_path.relative_to(generation)
    except ValueError as exc:
        raise SuccessorOperatorError(
            "generation-bound runtime path escaped its generation"
        ) from exc
    if not relative.parts:
        raise SuccessorOperatorError(
            "generation-bound runtime path cannot name the generation itself"
        )
    return (
        Path(f"/proc/self/fd/{int(custody['generation_descriptor'])}")
        / relative
    )


def _sha256_regular_file(
    path: Path,
    *,
    max_bytes: int = MAX_DATABASE_BYTES,
    noun: str = "database",
    require_private_owner: bool = False,
) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unreadable: {path}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > max_bytes
            or (
                require_private_owner
                and (
                    before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) & 0o077
                )
            )
        ):
            raise SuccessorOperatorError(
                f"{noun} is not a bounded private regular file: {path}"
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SuccessorOperatorError(f"{noun} changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return "sha256:" + digest.hexdigest()


def _stable_file_metadata(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _sha256_descriptor(
    descriptor: int,
    *,
    max_bytes: int = MAX_DATABASE_BYTES,
    noun: str,
) -> str:
    """Hash a pinned regular-file descriptor without changing its offset."""

    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} descriptor is unreadable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > max_bytes
    ):
        raise SuccessorOperatorError(f"{noun} descriptor is out of bounds")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        try:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
        except OSError as exc:
            raise SuccessorOperatorError(f"{noun} descriptor is unreadable") from exc
        if not block:
            raise SuccessorOperatorError(f"{noun} descriptor was truncated")
        digest.update(block)
        offset += len(block)
    try:
        after = os.fstat(descriptor)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} descriptor is unreadable") from exc
    if _stable_file_metadata(before) != _stable_file_metadata(after):
        raise SuccessorOperatorError(f"{noun} descriptor changed while hashing")
    return "sha256:" + digest.hexdigest()


def _require_stopped_snapshot_capability() -> None:
    required = (
        hasattr(os, "memfd_create")
        and bool(getattr(os, "MFD_ALLOW_SEALING", 0))
        and bool(getattr(fcntl, "F_ADD_SEALS", 0))
        and bool(getattr(fcntl, "F_GET_SEALS", 0))
        and STOPPED_SNAPSHOT_REQUIRED_SEALS != 0
    )
    if not required:
        raise SuccessorOperatorError(
            "sealed stopped-state database snapshots are unavailable"
        )


def _copy_stopped_database_to_sealed_memfd(
    *,
    name: str,
    logical_path: Path,
    generation_descriptor: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Copy one pinned stopped database into one immutable anonymous file."""

    _require_stopped_snapshot_capability()
    source_descriptor = -1
    snapshot_descriptor = -1
    try:
        source_descriptor = os.open(
            logical_path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=generation_descriptor,
        )
        source_before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_size <= 0
            or source_before.st_size > min(MAX_DATABASE_BYTES, max_bytes)
            or source_before.st_uid != os.geteuid()
            or source_before.st_nlink != 1
            or stat.S_IMODE(source_before.st_mode) & 0o077
        ):
            raise SuccessorOperatorError(
                f"stopped-state {name} database custody is unsafe"
            )
        try:
            os.stat(
                logical_path.name + ".wal",
                dir_fd=generation_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise SuccessorOperatorError(
                f"stopped-state {name} database has a live WAL"
            )
        snapshot_descriptor = os.memfd_create(
            f"lgcvf-stopped-{name}.duckdb",
            getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
        )
        remaining = int(source_before.st_size)
        while remaining:
            block = os.read(source_descriptor, min(1024 * 1024, remaining))
            if not block:
                raise SuccessorOperatorError(
                    f"stopped-state {name} database was truncated during snapshot"
                )
            view = memoryview(block)
            while view:
                written = os.write(snapshot_descriptor, view)
                if written <= 0:
                    raise SuccessorOperatorError(
                        f"stopped-state {name} snapshot write failed"
                    )
                view = view[written:]
            remaining -= len(block)
        source_after = os.fstat(source_descriptor)
        named_source = os.stat(
            logical_path.name,
            dir_fd=generation_descriptor,
            follow_symlinks=False,
        )
        if (
            _stable_file_metadata(source_before)
            != _stable_file_metadata(source_after)
            or _stable_file_metadata(source_before)
            != _stable_file_metadata(named_source)
        ):
            raise SuccessorOperatorError(
                f"stopped-state {name} database changed during snapshot"
            )
        os.fchmod(snapshot_descriptor, 0o400)
        os.fsync(snapshot_descriptor)
        fcntl.fcntl(
            snapshot_descriptor,
            fcntl.F_ADD_SEALS,
            STOPPED_SNAPSHOT_REQUIRED_SEALS,
        )
        observed_seals = int(
            fcntl.fcntl(snapshot_descriptor, fcntl.F_GET_SEALS)
        )
        if (
            observed_seals & STOPPED_SNAPSHOT_REQUIRED_SEALS
            != STOPPED_SNAPSHOT_REQUIRED_SEALS
        ):
            raise SuccessorOperatorError(
                f"stopped-state {name} snapshot is not immutable"
            )
        digest = _sha256_descriptor(
            snapshot_descriptor,
            noun=f"sealed stopped-state {name} snapshot",
        )
        return {
            "name": name,
            "logical_path": str(logical_path),
            "source_descriptor": source_descriptor,
            "source_metadata": _stable_file_metadata(source_before),
            "snapshot_descriptor": snapshot_descriptor,
            "snapshot_path": f"/proc/self/fd/{snapshot_descriptor}",
            "sha256": digest,
            "seals": observed_seals,
        }
    except BaseException:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
        raise


def _validate_stopped_database_snapshots(
    paths: Mapping[str, Path],
    custody: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Revalidate source inodes and sealed bytes, returning logical evidence."""

    _revalidate_generation_bound_controller_lock(paths, custody)
    databases = _successor_state_databases(paths)
    if set(snapshots) != set(databases):
        raise SuccessorOperatorError("stopped-state snapshot inventory differs")
    generation_descriptor = int(custody["generation_descriptor"])
    observed: dict[str, dict[str, str]] = {}
    for name, logical_path in databases.items():
        snapshot = snapshots[name]
        if (
            set(snapshot)
            != {
                "name",
                "logical_path",
                "source_descriptor",
                "source_metadata",
                "snapshot_descriptor",
                "snapshot_path",
                "sha256",
                "seals",
            }
            or snapshot.get("name") != name
            or snapshot.get("logical_path") != str(logical_path)
        ):
            raise SuccessorOperatorError(
                f"stopped-state {name} snapshot binding differs"
            )
        source_descriptor = snapshot.get("source_descriptor")
        snapshot_descriptor = snapshot.get("snapshot_descriptor")
        if (
            type(source_descriptor) is not int
            or source_descriptor < 3
            or type(snapshot_descriptor) is not int
            or snapshot_descriptor < 3
            or snapshot.get("snapshot_path")
            != f"/proc/self/fd/{snapshot_descriptor}"
        ):
            raise SuccessorOperatorError(
                f"stopped-state {name} snapshot descriptor differs"
            )
        try:
            source_metadata = os.fstat(source_descriptor)
            named_source = os.stat(
                logical_path.name,
                dir_fd=generation_descriptor,
                follow_symlinks=False,
            )
            snapshot_metadata = os.fstat(snapshot_descriptor)
            seals = int(fcntl.fcntl(snapshot_descriptor, fcntl.F_GET_SEALS))
        except (OSError, TypeError, ValueError) as exc:
            raise SuccessorOperatorError(
                f"stopped-state {name} snapshot cannot be revalidated"
            ) from exc
        try:
            os.stat(
                logical_path.name + ".wal",
                dir_fd=generation_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            wal_absent = True
        except OSError as exc:
            raise SuccessorOperatorError(
                f"stopped-state {name} snapshot cannot be revalidated"
            ) from exc
        else:
            wal_absent = False
        if (
            not wal_absent
            or _stable_file_metadata(source_metadata)
            != tuple(snapshot.get("source_metadata") or ())
            or _stable_file_metadata(named_source)
            != _stable_file_metadata(source_metadata)
            or not stat.S_ISREG(snapshot_metadata.st_mode)
            or snapshot_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(snapshot_metadata.st_mode) != 0o400
            or seals & STOPPED_SNAPSHOT_REQUIRED_SEALS
            != STOPPED_SNAPSHOT_REQUIRED_SEALS
            or seals != snapshot.get("seals")
            or _sha256_descriptor(
                snapshot_descriptor,
                noun=f"sealed stopped-state {name} snapshot",
            )
            != snapshot.get("sha256")
        ):
            raise SuccessorOperatorError(
                f"stopped-state {name} snapshot changed"
            )
        observed[name] = {
            "path": str(logical_path),
            "sha256": str(snapshot["sha256"]),
        }
    return observed


@contextlib.contextmanager
def _sealed_stopped_database_snapshots(
    paths: Mapping[str, Path],
    custody: Mapping[str, Any],
) -> Any:
    """Hold immutable copies and their exact source descriptors for one projection."""

    generation = paths["controller_lock"].parent
    databases = _successor_state_databases(paths)
    if any(path.parent != generation for path in databases.values()):
        raise SuccessorOperatorError("successor database escaped its generation")
    _revalidate_generation_bound_controller_lock(paths, custody)
    snapshots: dict[str, dict[str, Any]] = {}
    remaining_snapshot_bytes = STOPPED_SNAPSHOT_AGGREGATE_MAX_BYTES
    try:
        for name, logical_path in databases.items():
            snapshots[name] = _copy_stopped_database_to_sealed_memfd(
                name=name,
                logical_path=logical_path,
                generation_descriptor=int(custody["generation_descriptor"]),
                max_bytes=remaining_snapshot_bytes,
            )
            remaining_snapshot_bytes -= int(
                snapshots[name]["source_metadata"][2]
            )
        _validate_stopped_database_snapshots(paths, custody, snapshots)
        yield snapshots
    finally:
        for snapshot in snapshots.values():
            for field in ("snapshot_descriptor", "source_descriptor"):
                descriptor = snapshot.get(field)
                if type(descriptor) is int and descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass


def _stopped_snapshot_capacity_preflight(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "aggregate_max_bytes": STOPPED_SNAPSHOT_AGGREGATE_MAX_BYTES,
        "source_bytes": 0,
        "reason": "",
    }
    try:
        _require_stopped_snapshot_capability()
        total = 0
        for name, database in _successor_state_databases(paths).items():
            metadata = os.lstat(database)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) & 0o077
                or metadata.st_size <= 0
            ):
                raise SuccessorOperatorError(
                    f"stopped-state {name} database custody is unsafe"
                )
            total += int(metadata.st_size)
            if total > STOPPED_SNAPSHOT_AGGREGATE_MAX_BYTES:
                raise SuccessorOperatorError(
                    "stopped-state database snapshot aggregate exceeds its bound"
                )
    except (OSError, SuccessorOperatorError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    result["available"] = True
    result["source_bytes"] = total
    return result


def _regular_file_contains(path: Path, needle: bytes) -> bool:
    if not needle:
        return False
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SuccessorOperatorError(
            f"could not inspect credential surface: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 0
            or metadata.st_size > MAX_SECRET_SURFACE_BYTES
        ):
            raise SuccessorOperatorError(
                f"credential surface is not a bounded regular file: {path}"
            )
        carry = b""
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return False
            observed = carry + block
            if needle in observed:
                return True
            overlap = max(0, len(needle) - 1)
            carry = observed[-overlap:] if overlap else b""
    finally:
        os.close(descriptor)


def _database_identity(path: Path) -> dict[str, str]:
    import duckdb

    try:
        connection = duckdb.connect(str(path), read_only=True)
        try:
            rows = connection.execute(
                "SELECT key, value FROM control_plane_metadata "
                "WHERE key IN ('database_uuid','schema_version',"
                "'schema_fingerprint','migration_catalog_fingerprint')"
            ).fetchall()
        finally:
            connection.close()
    except Exception as exc:
        raise SuccessorOperatorError(
            f"could not read control-plane identity from {path}: {type(exc).__name__}"
        ) from exc
    return {str(key): str(value or "") for key, value in rows}


def datasets_profile_migration(path: Path) -> Any:
    """Idempotently admit the datasets-authoritative migration catalog."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
        load_datasets_authoritative_operational_catalog,
        verify_datasets_authoritative_operational_schema,
    )

    report = install_datasets_authoritative_operational_schema(
        path,
        application_version="lgcvf-quack-successor-v1",
        tool_version="lgcvf-quack-controller-v1",
        owner_id=f"lgcvf-quack-controller:{os.getpid()}",
    )
    verification = verify_datasets_authoritative_operational_schema(path)
    expected_catalog = load_datasets_authoritative_operational_catalog().fingerprint()
    if (
        verification.get("valid") is not True
        or report.schema_fingerprint != verification.get("schema_fingerprint")
        or report.catalog_fingerprint != expected_catalog
        or verification.get("catalog_fingerprint") != expected_catalog
    ):
        raise SuccessorOperatorError(
            "datasets-authoritative migration report and verification differ"
        )
    return report


def _verify_profile(
    path: Path,
    *,
    sealed_descriptor: int | None = None,
    read_only: bool = False,
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        control_plane_schema as schema_module,
    )

    verifier = schema_module.verify_datasets_authoritative_operational_schema
    if sealed_descriptor is not None or read_only:
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            DuckDBConnection,
            connect_duckdb_with_policy,
        )

        import duckdb

        exact_path = str(path)
        if (
            (sealed_descriptor is not None and type(sealed_descriptor) is not int)
            or (sealed_descriptor is not None and sealed_descriptor < 3)
            or (
                sealed_descriptor is not None
                and exact_path != f"/proc/self/fd/{sealed_descriptor}"
            )
            or (
                sealed_descriptor is None
                and re.fullmatch(r"/proc/self/fd/[1-9][0-9]*/.+", exact_path)
                is None
            )
            or verifier.__closure__ is not None
        ):
            raise SuccessorOperatorError(
                "read-only profile verifier binding is unavailable"
            )

        def exact_read_only_opener(
            requested: Path | str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if str(requested) != exact_path or args or kwargs:
                raise SuccessorOperatorError(
                    "read-only profile verifier requested a foreign database"
                )
            raw = connect_duckdb_with_policy(
                duckdb,
                exact_path,
                read_only=True,
            )
            try:
                wrapped = DuckDBConnection.wrap(raw)
            except BaseException:
                raw.close()
                raise
            return contextlib.closing(wrapped)

        isolated_globals = dict(verifier.__globals__)
        isolated_globals["open_duckdb_connection"] = exact_read_only_opener
        verifier = types.FunctionType(
            verifier.__code__,
            isolated_globals,
            name=verifier.__name__,
            argdefs=verifier.__defaults__,
            closure=None,
        )
        verifier.__kwdefaults__ = (
            dict(schema_module.verify_datasets_authoritative_operational_schema.__kwdefaults__)
            if schema_module.verify_datasets_authoritative_operational_schema.__kwdefaults__
            else None
        )
    verification = verifier(path)
    expected = (
        schema_module.load_datasets_authoritative_operational_catalog().fingerprint()
    )
    if (
        verification.get("valid") is not True
        or verification.get("catalog_fingerprint") != expected
    ):
        raise SuccessorOperatorError(
            f"datasets-authoritative schema verification failed: {path}"
        )
    return verification


def _strict_addressed_mapping(
    value: Mapping[str, Any],
    *,
    identity_field: str,
    noun: str,
) -> dict[str, Any]:
    normalized = dict(value)
    claimed = str(normalized.get(identity_field) or "")
    unsigned = dict(normalized)
    unsigned.pop(identity_field, None)
    if not claimed or claimed != _content_id(unsigned):
        raise SuccessorOperatorError(f"{noun} content identity differs")
    return normalized


def _strict_addressed_json(
    path: Path,
    *,
    expected_schema: str,
    identity_field: str,
    noun: str,
) -> dict[str, Any]:
    value = _strict_json(
        path,
        expected_schema=expected_schema,
        require_private_owner=True,
    )
    return _strict_addressed_mapping(
        value,
        identity_field=identity_field,
        noun=noun,
    )


def _plain_json_object(path: Path, *, noun: str) -> dict[str, Any]:
    raw = _read_bounded_regular_file(path, max_bytes=MAX_JSON_BYTES, noun=noun)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(f"{noun} is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise SuccessorOperatorError(f"{noun} is not an object: {path}")
    return value


def _require_sha256_pin(value: str, *, noun: str) -> str:
    normalized = str(value or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized) is None:
        raise SuccessorOperatorError(f"{noun} SHA-256 pin is malformed")
    return normalized


def _require_private_directory(path: Path, *, noun: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise SuccessorOperatorError(f"{noun} custody is not private: {path}")


def _privatize_owned_directory(path: Path, *, noun: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise SuccessorOperatorError(f"{noun} is not an owned directory: {path}")
    os.chmod(path, 0o700, follow_symlinks=False)
    _require_private_directory(path, noun=noun)


def _sealed_source_paths(source_root: Path) -> dict[str, Path]:
    lexical = Path(os.path.abspath(os.fspath(source_root)))
    if lexical.name != "run-v17":
        raise SuccessorOperatorError("sealed continuity source must be named run-v17")
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor = cursor / component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise SuccessorOperatorError(
                "sealed continuity source path cannot be inspected"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise SuccessorOperatorError(
                "sealed continuity source path contains a link or non-directory"
            )
    _require_private_directory(lexical, noun="sealed continuity source")
    evidence = lexical / "evidence"
    bootstrap_root = evidence / "bootstrap"
    recovery_root = evidence / "fresh-generation-recovery"
    for directory, noun in (
        (evidence, "sealed evidence directory"),
        (bootstrap_root, "sealed bootstrap directory"),
        (recovery_root, "sealed recovery directory"),
    ):
        _require_private_directory(directory, noun=noun)
    paths = {
        "root": lexical,
        "control": lexical / "control.duckdb",
        "coordination": lexical / "control.coordination.duckdb",
        "execution": lexical / "control.execution.duckdb",
        "bootstrap": bootstrap_root / "materialization.json",
        "recovery_root": recovery_root,
        "recovery_receipt": recovery_root / "recovery-receipt.json",
    }
    for key in ("control", "coordination", "execution"):
        if paths[key].with_name(paths[key].name + ".wal").exists():
            raise SuccessorOperatorError(f"sealed {key} database has a live WAL")
    return paths


def _git_text(root: Path, arguments: Sequence[str], *, noun: str) -> str:
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            [
                str(GIT_EXECUTABLE),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                *arguments,
            ],
            cwd=root,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise SuccessorOperatorError(f"{noun} could not be observed") from exc
    if completed.returncode != 0:
        raise SuccessorOperatorError(
            f"{noun} failed: {(completed.stderr or completed.stdout)[-1000:].strip()}"
        )
    return completed.stdout.strip()


def _git_quiet(root: Path, arguments: Sequence[str], *, noun: str) -> None:
    _git_text(root, arguments, noun=noun)


def _regular_git_blob_oid(path: Path, *, noun: str) -> str:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unreadable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > MAX_DATABASE_BYTES
        ):
            raise SuccessorOperatorError(f"{noun} is not a bounded regular file")
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {before.st_size}\0".encode("ascii"))
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SuccessorOperatorError(f"{noun} changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _owned_directory_chain_identity(
    repository: Path,
    relative_path: Path,
    *,
    noun: str,
) -> tuple[Path, Path, tuple[tuple[int, int, int, int, int, int], ...]]:
    """Resolve one same-owner, symlink-free directory chain beneath a repository."""

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SuccessorOperatorError(f"{noun} contains an unsafe gitlink path")
    current = repository
    identities: list[tuple[int, int, int, int, int, int]] = []
    try:
        for component in (None, *relative_path.parts):
            if component is not None:
                current = current / component
            metadata = os.lstat(current)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
            ):
                raise SuccessorOperatorError(f"{noun} gitlink custody differs")
            identities.append(
                (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_uid,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            )
        exact_repository = repository.resolve(strict=True)
        exact_gitlink = current.resolve(strict=True)
        exact_gitlink.relative_to(exact_repository)
    except SuccessorOperatorError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SuccessorOperatorError(
            f"{noun} gitlink custody cannot be resolved"
        ) from exc
    return exact_repository, exact_gitlink, tuple(identities)


def _validate_ignored_runtime_inventory(
    repository: Path,
    *,
    pathspecs: Sequence[str],
    noun: str,
) -> None:
    ignored = _git_text(
        repository,
        (
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            *pathspecs,
        ),
        noun=f"{noun} ignored inventory",
    )
    for raw in ignored.split("\0"):
        if not raw:
            continue
        relative_path = Path(raw)
        path = repository / relative_path
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise SuccessorOperatorError(
                f"{noun} ignored object cannot be inspected"
            ) from exc
        if (
            relative_path.suffix != ".pyc"
            or "__pycache__" not in relative_path.parts
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise SuccessorOperatorError(
                f"{noun} contains an ignored executable or data object"
            )


def _tracked_runtime_inventory(
    repository: Path,
    *,
    head: str,
    pathspecs: Sequence[str],
    noun: str,
    _gitlink_depth: int = 0,
    _gitlink_chain: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, Any]:
    if _gitlink_depth > MAX_RUNTIME_GITLINK_DEPTH:
        raise SuccessorOperatorError(f"{noun} gitlink nesting is too deep")
    if (
        _git_text(
            repository,
            ("rev-parse", "--show-object-format"),
            noun=f"{noun} object format",
        )
        != "sha1"
    ):
        raise SuccessorOperatorError(f"{noun} object format is unsupported")
    special_index = _git_text(
        repository,
        ("ls-files", "-v", "-z", "--", *pathspecs),
        noun=f"{noun} index flags",
    )
    if any(
        record and not record.startswith("H ") for record in special_index.split("\0")
    ):
        raise SuccessorOperatorError(f"{noun} has special index flags")
    raw_records = _git_text(
        repository,
        ("ls-tree", "-r", "-z", head, "--", *pathspecs),
        noun=f"{noun} tracked inventory",
    )
    observed: list[tuple[str, str, str]] = []
    for raw in raw_records.split("\0"):
        if not raw:
            continue
        try:
            metadata, relative = raw.split("\t", 1)
            mode, object_type, expected_oid = metadata.split(" ", 2)
        except ValueError as exc:
            raise SuccessorOperatorError(f"{noun} inventory is malformed") from exc
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SuccessorOperatorError(f"{noun} contains an unsafe tracked object")
        if object_type == "commit" and mode == "160000":
            try:
                (
                    exact_repository_root,
                    gitlink_path,
                    initial_directory_chain,
                ) = _owned_directory_chain_identity(
                    repository,
                    relative_path,
                    noun=noun,
                )
                with os.scandir(gitlink_path) as entries:
                    gitlink_is_empty = next(entries, None) is None
            except OSError as exc:
                raise SuccessorOperatorError(
                    f"{noun} gitlink custody cannot be inspected"
                ) from exc
            if gitlink_is_empty:
                try:
                    (
                        final_repository_root,
                        final_gitlink_path,
                        final_directory_chain,
                    ) = _owned_directory_chain_identity(
                        repository,
                        relative_path,
                        noun=noun,
                    )
                    with os.scandir(gitlink_path) as entries:
                        gitlink_remains_empty = next(entries, None) is None
                    (
                        terminal_repository_root,
                        terminal_gitlink_path,
                        terminal_directory_chain,
                    ) = _owned_directory_chain_identity(
                        repository,
                        relative_path,
                        noun=noun,
                    )
                except OSError as exc:
                    raise SuccessorOperatorError(
                        f"{noun} gitlink custody cannot be revalidated"
                    ) from exc
                if (
                    final_repository_root != exact_repository_root
                    or final_gitlink_path != gitlink_path
                    or final_directory_chain != initial_directory_chain
                    or terminal_repository_root != exact_repository_root
                    or terminal_gitlink_path != gitlink_path
                    or terminal_directory_chain != initial_directory_chain
                    or not gitlink_remains_empty
                ):
                    raise SuccessorOperatorError(
                        f"{noun} uninitialized gitlink custody changed"
                    )
            else:
                def observe_initialized_gitlink() -> tuple[Path, str, str]:
                    nested_root = _git_text(
                        gitlink_path,
                        ("rev-parse", "--show-toplevel"),
                        noun=f"{noun} initialized gitlink root",
                    )
                    nested_head = _git_text(
                        gitlink_path,
                        ("rev-parse", "--verify", "HEAD^{commit}"),
                        noun=f"{noun} initialized gitlink HEAD",
                    )
                    nested_dirty = _git_text(
                        gitlink_path,
                        (
                            "status",
                            "--porcelain=v1",
                            "--untracked-files=all",
                            "--ignore-submodules=none",
                        ),
                        noun=f"{noun} initialized gitlink inventory",
                    )
                    try:
                        observed_nested_root = Path(nested_root).resolve(strict=True)
                    except (OSError, RuntimeError) as exc:
                        raise SuccessorOperatorError(
                            f"{noun} initialized gitlink root cannot be resolved"
                        ) from exc
                    return observed_nested_root, nested_head, nested_dirty

                exact_nested_root = gitlink_path
                gitlink_identity = (str(exact_nested_root), expected_oid)
                if gitlink_identity in _gitlink_chain:
                    raise SuccessorOperatorError(
                        f"{noun} initialized gitlink cycle differs"
                    )
                nested_root, nested_head, nested_dirty = observe_initialized_gitlink()
                if (
                    nested_root != exact_nested_root
                    or nested_head != expected_oid
                    or nested_dirty
                ):
                    raise SuccessorOperatorError(
                        f"{noun} initialized gitlink custody differs"
                    )
                _tracked_runtime_inventory(
                    gitlink_path,
                    head=expected_oid,
                    pathspecs=(".",),
                    noun=f"{noun} initialized gitlink {relative_path.as_posix()}",
                    _gitlink_depth=_gitlink_depth + 1,
                    _gitlink_chain=_gitlink_chain | {gitlink_identity},
                )
                (
                    final_repository_root,
                    final_gitlink_path,
                    final_directory_chain,
                ) = _owned_directory_chain_identity(
                    repository,
                    relative_path,
                    noun=noun,
                )
                nested_root, nested_head, nested_dirty = observe_initialized_gitlink()
                if (
                    final_repository_root != exact_repository_root
                    or final_gitlink_path != gitlink_path
                    or final_directory_chain != initial_directory_chain
                    or nested_root != exact_nested_root
                    or nested_head != expected_oid
                    or nested_dirty
                ):
                    raise SuccessorOperatorError(
                        f"{noun} initialized gitlink custody changed"
                    )
                _validate_ignored_runtime_inventory(
                    gitlink_path,
                    pathspecs=(".",),
                    noun=f"{noun} initialized gitlink {relative_path.as_posix()}",
                )
                (
                    terminal_repository_root,
                    terminal_gitlink_path,
                    terminal_directory_chain,
                ) = _owned_directory_chain_identity(
                    repository,
                    relative_path,
                    noun=noun,
                )
                if (
                    terminal_repository_root != exact_repository_root
                    or terminal_gitlink_path != gitlink_path
                    or terminal_directory_chain != initial_directory_chain
                ):
                    raise SuccessorOperatorError(
                        f"{noun} initialized gitlink custody changed"
                    )
            observed.append((relative_path.as_posix(), mode, expected_oid))
            continue
        if object_type == "blob" and mode == "120000":
            link_path = repository / relative_path
            metadata_status = os.lstat(link_path)
            target_text = os.readlink(link_path)
            target_bytes = os.fsencode(target_text)
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {len(target_bytes)}\0".encode("ascii"))
            digest.update(target_bytes)
            try:
                (link_path.parent / target_text).resolve(strict=True).relative_to(
                    repository.resolve(strict=True)
                )
            except (OSError, ValueError) as exc:
                raise SuccessorOperatorError(
                    f"{noun} tracked link escapes its repository"
                ) from exc
            if (
                not stat.S_ISLNK(metadata_status.st_mode)
                or metadata_status.st_uid != os.geteuid()
                or digest.hexdigest() != expected_oid
            ):
                raise SuccessorOperatorError(f"{noun} tracked link differs from HEAD")
            observed.append((relative_path.as_posix(), mode, expected_oid))
            continue
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise SuccessorOperatorError(f"{noun} contains an unsafe tracked object")
        observed_oid = _regular_git_blob_oid(
            repository / relative_path, noun=f"{noun} tracked object"
        )
        if observed_oid != expected_oid:
            raise SuccessorOperatorError(f"{noun} tracked bytes differ from HEAD")
        observed.append((relative_path.as_posix(), mode, observed_oid))
    _validate_ignored_runtime_inventory(
        repository,
        pathspecs=pathspecs,
        noun=noun,
    )
    inventory_root = "sha256:" + hashlib.sha256(_canonical_bytes(observed)).hexdigest()
    return {
        "tracked_object_count": len(observed),
        "tracked_inventory_root": inventory_root,
    }


def _observe_candidate_runtime_continuity(
    root: Path,
    *,
    require_resolved_remote: bool,
) -> dict[str, Any]:
    """Observe one clean candidate, optionally requiring exact remote equality."""

    if (
        _AMBIENT_PYTHONPATH
        or sys.path[:2] != [str(root), str(root / "ipfs_datasets_py")]
        or sys.pycache_prefix != _RUNTIME_PYCACHE.name
    ):
        raise SuccessorOperatorError("candidate Python import boundary differs")
    quarantine_path = Path(sys.pycache_prefix)
    _require_private_directory(
        quarantine_path, noun="candidate Python bytecode quarantine"
    )
    try:
        quarantine = quarantine_path.resolve(strict=True)
        candidate_root = root.resolve(strict=True)
    except OSError as exc:
        raise SuccessorOperatorError(
            "candidate Python bytecode quarantine cannot be resolved"
        ) from exc
    try:
        quarantine.relative_to(candidate_root)
    except ValueError:
        pass
    else:
        raise SuccessorOperatorError(
            "candidate Python bytecode quarantine is inside the worktree"
        )
    branch = _git_text(root, ("symbolic-ref", "--short", "HEAD"), noun="board branch")
    if branch != APPROVED_BOARD_BRANCH:
        raise SuccessorOperatorError(
            "continuity verification is not on the approved board branch"
        )
    current_head = _git_text(root, ("rev-parse", "HEAD"), noun="current HEAD")
    current_tree = _git_text(root, ("rev-parse", "HEAD^{tree}"), noun="current tree")
    dirty = _git_text(
        root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        noun="candidate source inventory",
    )
    if dirty:
        raise SuccessorOperatorError(
            "continuity verification requires a completely clean candidate worktree"
        )
    datasets_relative = "ipfs_datasets_py"
    datasets = _contained(root, datasets_relative)
    datasets_metadata = os.lstat(datasets)
    if (
        not stat.S_ISDIR(datasets_metadata.st_mode)
        or stat.S_ISLNK(datasets_metadata.st_mode)
        or datasets_metadata.st_uid != os.geteuid()
    ):
        raise SuccessorOperatorError("nested runtime source custody differs")
    datasets_head = _git_text(
        datasets, ("rev-parse", "HEAD"), noun="nested runtime HEAD"
    )
    datasets_tree = _git_text(
        datasets, ("rev-parse", "HEAD^{tree}"), noun="nested runtime tree"
    )
    datasets_dirty = _git_text(
        datasets,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        noun="nested runtime source inventory",
    )
    gitlink = _git_text(
        root,
        ("ls-tree", current_head, "--", datasets_relative),
        noun="nested runtime gitlink",
    ).split()
    if (
        datasets_dirty
        or len(gitlink) < 3
        or gitlink[0] != "160000"
        or gitlink[1] != "commit"
        or gitlink[2] != datasets_head
    ):
        raise SuccessorOperatorError(
            "continuity verification requires the exact clean nested runtime gitlink"
        )
    remote_head = _git_text(
        root,
        ("rev-parse", APPROVED_REMOTE_BRANCH_REF),
        noun="resolved remote board branch",
    )
    if require_resolved_remote:
        if current_head != remote_head:
            raise SuccessorOperatorError(
                "current board candidate is not the resolved remote branch"
            )
    else:
        _git_quiet(
            root,
            ("merge-base", "--is-ancestor", remote_head, current_head),
            noun="resolved remote/final stopped candidate ancestry",
        )
    superproject_inventory = _tracked_runtime_inventory(
        root,
        head=current_head,
        pathspecs=(
            "ipfs_accelerate_py",
            "scripts/ops",
            "scripts/run_logic_governed_compositional_verification_fabric_quack.py",
            "scripts/validate_logic_governed_compositional_verification_fabric_plan.py",
            (
                "config/agent_supervisor_logic_governed_compositional_verification_"
                "fabric_scheduler.json"
            ),
            str(DEFAULT_SUCCESSOR_CONFIG_RELATIVE),
            (
                "docs/architecture/logic_governed_compositional_verification_"
                "fabric.todo.md"
            ),
            (
                "docs/architecture/logic_governed_compositional_verification_"
                "fabric.objectives.md"
            ),
            (
                "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_"
                "FABRIC_PLAN.md"
            ),
            (
                "data/agent_supervisor/logic_governed_compositional_verification_"
                "fabric/formal_work_plan.json"
            ),
        ),
        noun="candidate runtime",
    )
    datasets_inventory = _tracked_runtime_inventory(
        datasets,
        head=datasets_head,
        pathspecs=("__init__.py", "ipfs_datasets_py"),
        noun="nested runtime",
    )
    return {
        "approved_branch": branch,
        "resolved_remote_head": remote_head,
        "current_head": current_head,
        "current_tree": current_tree,
        "candidate_worktree_clean": True,
        "datasets_head": datasets_head,
        "datasets_tree": datasets_tree,
        "datasets_worktree_clean": True,
        "python_bytecode_quarantine": {
            "enabled": True,
            "ephemeral": True,
            "ignored_worktree_pycache": "quarantined_not_imported",
            "outside_candidate_root": True,
            "private": True,
        },
        "superproject_runtime_inventory": superproject_inventory,
        "datasets_runtime_inventory": datasets_inventory,
    }


def _candidate_runtime_continuity(root: Path) -> dict[str, Any]:
    """Observe the exact resolved remote candidate for live/restart admission."""

    return _observe_candidate_runtime_continuity(
        root,
        require_resolved_remote=True,
    )


def _target_source_continuity(
    root: Path,
    *,
    source_head: str,
    source_tree: str,
    config: Mapping[str, Any],
    observed_continuity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
    ):
        raise SuccessorOperatorError("sealed source Git identity is malformed")
    candidate = (
        dict(observed_continuity)
        if observed_continuity is not None
        else _candidate_runtime_continuity(root)
    )
    branch = str(candidate["approved_branch"])
    if config.get("merge_target_branch") != branch:
        raise SuccessorOperatorError(
            "continuity verification is not on the approved board branch"
        )
    current_head = str(candidate["current_head"])
    observed_source_tree = _git_text(
        root,
        ("show", "-s", "--format=%T", source_head),
        noun="sealed source commit",
    )
    if observed_source_tree != source_tree:
        raise SuccessorOperatorError("sealed source commit/tree binding differs")
    _git_quiet(
        root,
        ("merge-base", "--is-ancestor", source_head, current_head),
        noun="sealed source ancestry",
    )
    authority_paths = []
    for field in (
        "taskboard_path",
        "objectives_path",
        "plan_path",
        "formal_plan_path",
        "validator_path",
    ):
        value = str(config.get(field) or "")
        if not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise SuccessorOperatorError(f"scheduler {field} is unsafe")
        authority_paths.append(value)
    config_relative = (
        "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
        "scheduler.json"
    )
    _git_quiet(
        root,
        (
            "diff",
            "--no-ext-diff",
            "--quiet",
            "HEAD",
            "--",
            config_relative,
            *authority_paths,
        ),
        noun="current authority source worktree",
    )
    _git_quiet(
        root,
        (
            "diff",
            "--no-ext-diff",
            "--quiet",
            source_head,
            current_head,
            "--",
            *authority_paths,
        ),
        noun="sealed/current authority source",
    )
    return {
        **candidate,
        "target_source_head": source_head,
        "target_source_tree": source_tree,
    }


def _require_false_authority(value: Mapping[str, Any], *, noun: str) -> None:
    false_fields = (
        "validation_self_authority",
        "validation_completion_authoritative",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
    )
    if any(value.get(field) is not False for field in false_fields):
        raise SuccessorOperatorError(f"{noun} exceeds the continuity authority ceiling")
    if (
        value.get("candidate_authored_validation") is not True
        or value.get("network_isolation_enforced") is not True
        or value.get("model_provider_route") != "none"
        or value.get("source_database_statuses_read") is not False
        or value.get("source_database_completion_records_imported") is not False
        or value.get("synthetic_source_disposition") != "quarantined_not_imported"
    ):
        raise SuccessorOperatorError(f"{noun} recovery limitations differ")


def _validate_recovery_policy_projection(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    policy = config.get("fresh_generation_recovery")
    plan_binding = config.get("plan_binding")
    if not isinstance(policy, Mapping) or not isinstance(plan_binding, Mapping):
        raise SuccessorOperatorError("tracked fresh-recovery policy is unavailable")
    expected_partition = {
        "construction_completed_task_ids": list(CONSTRUCTION_COMPLETED_TASK_IDS),
        "recovered_completed_task_ids": list(RECOVERED_COMPLETED_TASK_IDS),
        "rejected_synthetic_task_ids": list(TODO_TASK_IDS),
        "preserved_blocked_task_ids": list(BLOCKED_TASK_IDS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
    }
    if manifest.get("completion_partition") != expected_partition:
        raise SuccessorOperatorError("sealed completion partition differs")
    retained = manifest.get("retained_completion_binding")
    expected_retained = {
        "binding_cid": policy.get("retained_completion_binding_cid"),
        "construction_completion_count": 7,
        "delta_cid": policy.get("retained_delta_cid"),
        "dynamic_completion_receipt_count": 5,
        "logical_completion_count": 12,
        "path": policy.get("retained_revision_receipt_path"),
        "protected_blocker_binding_cid": policy.get(
            "retained_protected_blocker_binding_cid"
        ),
        "receipt_cid": policy.get("retained_revision_receipt_cid"),
        "sha256": policy.get("retained_revision_receipt_sha256"),
        "successor_revision_cid": policy.get("retained_successor_revision_cid"),
    }
    if retained != expected_retained:
        raise SuccessorOperatorError("sealed retained-completion projection differs")
    quarantine = manifest.get("wrong_default_quarantine")
    if not isinstance(quarantine, Mapping):
        raise SuccessorOperatorError("sealed wrong-default quarantine is unavailable")
    quarantine_projection = {
        "incident_manifest_path": policy.get("wrong_default_incident_manifest_path"),
        "incident_manifest_sha256": policy.get(
            "wrong_default_incident_manifest_sha256"
        ),
        "incident_manifest_cid": policy.get("wrong_default_incident_manifest_cid"),
        "contaminated_coordination_manifest_path": policy.get(
            "contaminated_coordination_projection_path"
        ),
        "contaminated_coordination_manifest_sha256": policy.get(
            "contaminated_coordination_projection_sha256"
        ),
        "contaminated_coordination_manifest_cid": policy.get(
            "contaminated_coordination_projection_manifest_cid"
        ),
        "rejected_record_set_cid": policy.get(
            "contaminated_coordination_rejected_record_set_cid"
        ),
        "rejected_contaminated_coordination_projection_root": policy.get(
            "rejected_contaminated_coordination_projection_root"
        ),
        "rejected_synthetic_task_ids": list(TODO_TASK_IDS),
        "disposition": "preserved_forensic_quarantine_not_imported",
        "source_database_opened": False,
    }
    if any(
        quarantine.get(key) != value for key, value in quarantine_projection.items()
    ):
        raise SuccessorOperatorError(
            "sealed wrong-default quarantine projection differs"
        )
    policy_merges = policy.get("merge_completions")
    manifest_merges = manifest.get("merge_completion_evidence")
    if (
        not isinstance(policy_merges, list)
        or not isinstance(manifest_merges, list)
        or len(policy_merges) != len(RECOVERED_COMPLETED_TASK_IDS)
        or len(manifest_merges) != len(policy_merges)
    ):
        raise SuccessorOperatorError("sealed merge-completion population differs")
    for expected, observed in zip(policy_merges, manifest_merges, strict=True):
        if (
            not isinstance(expected, Mapping)
            or not isinstance(observed, Mapping)
            or any(observed.get(key) != value for key, value in expected.items())
        ):
            raise SuccessorOperatorError("sealed merge-completion projection differs")
    common_fields = (
        "source_generation",
        "target_generation",
        "source_head",
        "source_tree",
        "source_evidence_cid",
        "plan_root_cid",
        "population_root",
        "validation_qualification_cid",
        "candidate_authored_validation",
        "validation_self_authority",
        "validation_completion_authoritative",
        "source_database_statuses_read",
        "source_database_completion_records_imported",
        "synthetic_source_disposition",
        "network_isolation_enforced",
        "model_provider_route",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
    )
    if any(manifest.get(field) != receipt.get(field) for field in common_fields):
        raise SuccessorOperatorError(
            "sealed recovery receipt/manifest projection differs"
        )
    if (
        manifest.get("source_generation") != policy.get("source_generation")
        or manifest.get("target_generation") != policy.get("target_generation")
        or manifest.get("source_runtime_root") != policy.get("source_runtime_root")
        or manifest.get("target_runtime_root") != policy.get("target_runtime_root")
        or manifest.get("plan_root_cid") != plan_binding.get("formal_plan_content_id")
        or receipt.get("completed_task_ids") != list(COMPLETED_TASK_IDS)
        or receipt.get("todo_task_ids") != list(TODO_TASK_IDS)
        or receipt.get("blocked_task_ids") != list(BLOCKED_TASK_IDS)
        or receipt.get("completed_count") != 13
        or receipt.get("todo_count") != 13
        or receipt.get("blocked_count") != 2
        or receipt.get("atomic_publish") is not True
    ):
        raise SuccessorOperatorError("sealed recovery policy binding differs")
    _require_false_authority(manifest, noun="sealed recovery manifest")
    _require_false_authority(receipt, noun="sealed recovery receipt")


def _validate_historical_qualification(manifest: Mapping[str, Any]) -> None:
    qualification = manifest.get("validation_qualification")
    if not isinstance(qualification, Mapping):
        raise SuccessorOperatorError("sealed historical qualification is unavailable")
    normalized = _strict_addressed_mapping(
        qualification,
        identity_field="receipt_cid",
        noun="sealed historical qualification",
    )
    if (
        normalized.get("receipt_cid") != manifest.get("validation_qualification_cid")
        or normalized.get("passed") is not True
        or normalized.get("disposition") != "passed"
        or normalized.get("candidate_authored_replay") is not True
        or normalized.get("completion_authoritative") is not False
        or normalized.get("production_authoritative") is not False
        or normalized.get("production_authorized") is not False
        or normalized.get("objective_complete") is not False
        or normalized.get("provider_route") != "none"
        or normalized.get("network_permitted") is not False
        or normalized.get("cache_reused") is not False
    ):
        raise SuccessorOperatorError(
            "sealed historical qualification limitations differ"
        )
    recovery_manifest = normalized.get("recovery_manifest")
    if not isinstance(recovery_manifest, Mapping):
        raise SuccessorOperatorError("historical qualification manifest is unavailable")
    _strict_addressed_mapping(
        recovery_manifest,
        identity_field="manifest_cid",
        noun="historical qualification manifest",
    )


def _verify_sealed_control_state(
    database: Path,
    *,
    expected_sha256: str,
    manifest: Mapping[str, Any],
    formal_plan: Mapping[str, Any],
) -> dict[str, Any]:
    import duckdb

    before = _sha256_regular_file(
        database,
        noun="sealed control database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed control database SHA-256 differs")
    profile = _verify_profile(database)
    formal_tasks = formal_plan.get("tasks")
    if not isinstance(formal_tasks, list):
        raise SuccessorOperatorError("tracked formal task population is unavailable")
    formal_by_alias = {
        str(item.get("task_id") or ""): dict(item)
        for item in formal_tasks
        if isinstance(item, Mapping)
    }
    all_aliases = set(COMPLETED_TASK_IDS + TODO_TASK_IDS + BLOCKED_TASK_IDS)
    if set(formal_by_alias) != all_aliases:
        raise SuccessorOperatorError("tracked formal task population differs")
    try:
        connection = duckdb.connect(
            str(database),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
            },
        )
        try:
            task_rows = connection.execute(
                "SELECT task_cid, task_alias, status, revision, plan_cid, "
                "identity_json, body_json FROM tasks ORDER BY task_alias"
            ).fetchall()
            plan_rows = connection.execute(
                "SELECT plan_cid, plan_alias, status, revision, body_json "
                "FROM plans ORDER BY plan_cid"
            ).fetchall()
            dependency_rows = connection.execute(
                "SELECT task_cid, dependency_task_cid, kind "
                "FROM task_dependencies ORDER BY task_cid, dependency_task_cid, kind"
            ).fetchall()
            completion_rows = connection.execute(
                "SELECT task_cid FROM completion_receipts ORDER BY task_cid"
            ).fetchall()
            zero_counts = {
                table: int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                for table in (
                    "task_claims",
                    "task_attempts",
                    "task_assignments",
                    "task_blocks",
                    "resource_claims",
                    "maintenance_leases",
                    "leases",
                    "lease_events",
                    "token_history",
                    "client_sessions",
                )
            }
        finally:
            connection.close()
    except Exception as exc:
        if isinstance(exc, SuccessorOperatorError):
            raise
        raise SuccessorOperatorError(
            f"sealed control state cannot be reconstructed: {type(exc).__name__}"
        ) from exc
    if len(task_rows) != 28 or any(zero_counts.values()):
        raise SuccessorOperatorError(
            "sealed control database has unexpected live state"
        )
    expected_status_revision = {
        **{alias: ("completed", 1) for alias in CONSTRUCTION_COMPLETED_TASK_IDS},
        **{alias: ("completed", 2) for alias in RECOVERED_COMPLETED_TASK_IDS},
        **{alias: ("todo", 1) for alias in TODO_TASK_IDS},
        **{alias: ("blocked", 1) for alias in BLOCKED_TASK_IDS},
    }
    tasks_by_cid: dict[str, str] = {}
    rows_by_alias: dict[str, dict[str, Any]] = {}
    for (
        task_cid,
        alias,
        status,
        revision,
        plan_cid,
        identity_raw,
        body_raw,
    ) in task_rows:
        task_cid = str(task_cid)
        alias = str(alias)
        try:
            identity = json.loads(str(identity_raw))
            body = json.loads(str(body_raw))
        except json.JSONDecodeError as exc:
            raise SuccessorOperatorError("sealed task JSON is malformed") from exc
        if (
            alias not in expected_status_revision
            or (str(status), int(revision)) != expected_status_revision[alias]
            or str(plan_cid) != manifest.get("plan_root_cid")
            or not isinstance(identity, Mapping)
            or identity.get("task_alias") != alias
            or identity.get("task_cid") != task_cid
            or identity.get("repository_tree_id")
            != "git-tree:" + str(manifest.get("source_tree") or "")
            or not isinstance(body, Mapping)
            or body.get("formal_record") != formal_by_alias[alias]
            or body.get("formal_task_content_id") != task_cid
            or body.get("board_namespace")
            != "logic-governed-compositional-verification-fabric-v1"
        ):
            raise SuccessorOperatorError(f"{alias}: sealed task authority differs")
        if task_cid in tasks_by_cid or alias in rows_by_alias:
            raise SuccessorOperatorError("sealed task identity is duplicated")
        tasks_by_cid[task_cid] = alias
        rows_by_alias[alias] = {
            "task_cid": task_cid,
            "status": str(status),
            "body": dict(body),
        }
    if set(rows_by_alias) != all_aliases:
        raise SuccessorOperatorError("sealed task alias population differs")
    if len(plan_rows) != 1:
        raise SuccessorOperatorError("sealed plan population differs")
    plan_cid, plan_alias, plan_status, plan_revision, plan_body_raw = plan_rows[0]
    try:
        plan_body = json.loads(str(plan_body_raw))
    except json.JSONDecodeError as exc:
        raise SuccessorOperatorError("sealed plan JSON is malformed") from exc
    if (
        str(plan_cid) != manifest.get("plan_root_cid")
        or str(plan_alias) != "logic-governed-compositional-verification-fabric-v1"
        or str(plan_status) != "active"
        or int(plan_revision) != 1
        or not isinstance(plan_body, Mapping)
        or plan_body.get("source_head") != manifest.get("source_head")
        or plan_body.get("repository_tree_id")
        != "git-tree:" + str(manifest.get("source_tree") or "")
    ):
        raise SuccessorOperatorError("sealed active plan differs")
    observed_dependencies: set[tuple[str, str]] = set()
    for task_cid, dependency_cid, kind in dependency_rows:
        task_alias = tasks_by_cid.get(str(task_cid), "")
        dependency_alias = tasks_by_cid.get(str(dependency_cid), "")
        if not task_alias or not dependency_alias or str(kind) != "depends_on":
            raise SuccessorOperatorError("sealed dependency identity differs")
        observed_dependencies.add((task_alias, dependency_alias))
    expected_dependencies = {
        (alias, str(dependency))
        for alias, task in formal_by_alias.items()
        for dependency in task.get("depends_on") or ()
    }
    if (
        len(dependency_rows) != 46
        or len(observed_dependencies) != 46
        or observed_dependencies != expected_dependencies
    ):
        raise SuccessorOperatorError("sealed dependency graph differs")
    completed_cids = {rows_by_alias[alias]["task_cid"] for alias in COMPLETED_TASK_IDS}
    ready = []
    dependencies_by_alias: dict[str, set[str]] = {alias: set() for alias in all_aliases}
    for alias, dependency in observed_dependencies:
        dependencies_by_alias[alias].add(rows_by_alias[dependency]["task_cid"])
    for alias in TODO_TASK_IDS:
        row = rows_by_alias[alias]
        if (
            row["body"].get("is_schedulable") is True
            and dependencies_by_alias[alias] <= completed_cids
        ):
            ready.append(alias)
    if ready != ["LGCVF-081"]:
        raise SuccessorOperatorError("sealed ready frontier differs")
    completion_aliases = [tasks_by_cid.get(str(row[0]), "") for row in completion_rows]
    if sorted(completion_aliases) != sorted(RECOVERED_COMPLETED_TASK_IDS):
        raise SuccessorOperatorError("sealed reconstructed completion receipts differ")
    after = _sha256_regular_file(
        database,
        noun="sealed control database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed control database changed during verification"
        )
    identity = _database_identity(database)
    return {
        "sha256": before,
        "database_uuid": identity.get("database_uuid", ""),
        "schema_fingerprint": profile.get("schema_fingerprint", ""),
        "catalog_fingerprint": profile.get("catalog_fingerprint", ""),
        "task_count": 28,
        "dependency_count": 46,
        "completion_receipt_count": 6,
        "ready_task_ids": ready,
        "zero_state_counts": zero_counts,
        "task_cids_by_alias": {
            alias: rows_by_alias[alias]["task_cid"] for alias in sorted(rows_by_alias)
        },
    }


def _verify_sealed_coordination_state(
    database: Path,
    *,
    expected_sha256: str,
    control_tasks: Mapping[str, str],
    formal_plan: Mapping[str, Any],
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )

    before = _sha256_regular_file(
        database,
        noun="sealed coordination database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed coordination database SHA-256 differs")
    try:
        projection = read_coordination_registry_projection(database)
    except Exception as exc:
        raise SuccessorOperatorError(
            f"sealed coordination projection is unreadable: {type(exc).__name__}"
        ) from exc
    expected_counts = {
        "registered_tasks": 28,
        "dependency_edges": 46,
        "logical_completions": 13,
        "task_claims": 0,
        "active_task_claims": 0,
        "resource_claims": 0,
        "active_resource_claims": 0,
        "task_attempts": 0,
        "active_task_attempts": 0,
        "fenced_leases": 0,
        "active_fenced_leases": 0,
        "maintenance_leases": 0,
        "active_maintenance_leases": 0,
    }
    if projection.get("counts") != expected_counts or any(
        projection.get(field) != []
        for field in (
            "task_claims",
            "task_attempts",
            "fenced_leases",
            "resource_claims",
            "maintenance_leases",
        )
    ):
        raise SuccessorOperatorError("sealed coordination database has live state")
    registered = {
        str(item.get("task_id") or ""): str(item.get("task_cid") or "")
        for item in projection.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    if registered != dict(control_tasks):
        raise SuccessorOperatorError("sealed coordination task registry differs")
    cid_to_alias = {cid: alias for alias, cid in control_tasks.items()}
    observed_dependencies = {
        (
            cid_to_alias.get(str(item.get("task_cid") or ""), ""),
            cid_to_alias.get(str(item.get("dependency_task_cid") or ""), ""),
        )
        for item in projection.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    formal_tasks = formal_plan.get("tasks") or ()
    expected_dependencies = {
        (str(task.get("task_id") or ""), str(dependency))
        for task in formal_tasks
        if isinstance(task, Mapping)
        for dependency in task.get("depends_on") or ()
    }
    completion_aliases = {
        cid_to_alias.get(str(item.get("task_cid") or ""), "")
        for item in projection.get("logical_completions") or ()
        if isinstance(item, Mapping) and item.get("status") == "succeeded"
    }
    if observed_dependencies != expected_dependencies or completion_aliases != set(
        COMPLETED_TASK_IDS
    ):
        raise SuccessorOperatorError("sealed coordination authority differs")
    after = _sha256_regular_file(
        database,
        noun="sealed coordination database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed coordination database changed during verification"
        )
    return {"sha256": before, "counts": expected_counts}


def _verify_sealed_execution_state(
    database: Path,
    *,
    expected_sha256: str,
    control_schema_fingerprint: str,
) -> dict[str, Any]:
    import duckdb

    before = _sha256_regular_file(
        database,
        noun="sealed execution database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed execution database SHA-256 differs")
    try:
        connection = duckdb.connect(
            str(database),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
            },
        )
        try:
            counts = {
                table: int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                for table in (
                    "attempt_phases",
                    "daemon_execution_events",
                    "database_task_attempts",
                    "effect_claims",
                    "provider_invocations",
                )
            }
            metadata = {
                str(key): str(value)
                for key, value in connection.execute(
                    "SELECT key, value FROM daemon_execution_metadata ORDER BY key"
                ).fetchall()
            }
        finally:
            connection.close()
    except Exception as exc:
        raise SuccessorOperatorError(
            f"sealed execution state is unreadable: {type(exc).__name__}"
        ) from exc
    if any(counts.values()) or (
        metadata.get("authority_mode") != "embedded"
        or metadata.get("control_schema_fingerprint") != control_schema_fingerprint
        or metadata.get("control_schema_profile_id")
        != "datasets-authoritative-operational-control-plane@1"
        or metadata.get("interface") != "DatabaseImplementationDaemon@1"
        or metadata.get("process_instance_id") != "fresh-recovery-bootstrap"
        or metadata.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
        or metadata.get("state_schema_revision")
        != "datasets-authoritative-operational-v1"
        or not str(metadata.get("logical_owner_session_id") or "").startswith(
            "embedded-store:"
        )
    ):
        raise SuccessorOperatorError("sealed execution database has unexpected state")
    after = _sha256_regular_file(
        database,
        noun="sealed execution database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed execution database changed during verification"
        )
    return {"sha256": before, "row_counts": counts, "metadata": metadata}


def _verify_sealed_layout(paths: Mapping[str, Path], *, manifest_name: str) -> None:
    expected_root = {
        ".control.coordination.duckdb.lock",
        ".control.duckdb.intent.lock",
        ".control.duckdb.lock",
        ".control.duckdb.migration.lock",
        ".control.execution.duckdb.lock",
        "control.coordination.duckdb",
        "control.duckdb",
        "control.execution.duckdb",
        "evidence",
    }
    expected_evidence = {"bootstrap", "fresh-generation-recovery"}
    expected_bootstrap = {"materialization.json"}
    expected_recovery = {"recovery-receipt.json", manifest_name}
    observed = {
        "root": {item.name for item in os.scandir(paths["root"])},
        "evidence": {item.name for item in os.scandir(paths["root"] / "evidence")},
        "bootstrap": {
            item.name for item in os.scandir(paths["root"] / "evidence" / "bootstrap")
        },
        "recovery": {item.name for item in os.scandir(paths["recovery_root"])},
    }
    if observed != {
        "root": expected_root,
        "evidence": expected_evidence,
        "bootstrap": expected_bootstrap,
        "recovery": expected_recovery,
    }:
        raise SuccessorOperatorError("sealed run-v17 layout differs")
    for name in sorted(expected_root):
        if not name.startswith("."):
            continue
        lock_path = paths["root"] / name
        metadata = os.lstat(lock_path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != 0
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise SuccessorOperatorError("sealed empty lock-file custody differs")


def _assert_sealed_report_snapshot(
    paths: Mapping[str, Path],
    report: Mapping[str, Any],
) -> None:
    pins = report.get("pins")
    manifest_cid = str(report.get("manifest_cid") or "")
    if (
        not isinstance(pins, Mapping)
        or re.fullmatch(r"bagu[a-z2-7]{20,}", manifest_cid) is None
    ):
        raise SuccessorOperatorError("sealed continuity report pins are unavailable")
    manifest = paths["recovery_root"] / f"{manifest_cid}.manifest.json"
    observed = {
        "control_sha256": _sha256_regular_file(
            paths["control"], noun="sealed control database", require_private_owner=True
        ),
        "coordination_sha256": _sha256_regular_file(
            paths["coordination"],
            noun="sealed coordination database",
            require_private_owner=True,
        ),
        "execution_sha256": _sha256_regular_file(
            paths["execution"],
            noun="sealed execution database",
            require_private_owner=True,
        ),
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if any(pins.get(key) != value for key, value in observed.items()):
        raise SuccessorOperatorError(
            "sealed continuity snapshot changed after verification"
        )
    _verify_sealed_layout(paths, manifest_name=manifest.name)


def _validate_bootstrap_receipt(
    bootstrap: Mapping[str, Any],
    *,
    recovery_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    verification = bootstrap.get("verification")
    if not isinstance(verification, Mapping):
        raise SuccessorOperatorError("sealed bootstrap verification is unavailable")
    _strict_addressed_mapping(
        verification,
        identity_field="verification_root",
        noun="sealed bootstrap verification",
    )
    expected_paths = {
        "control": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.duckdb"
        ),
        "coordination": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.coordination.duckdb"
        ),
        "execution": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.execution.duckdb"
        ),
    }
    if (
        bootstrap.get("receipt_cid") != recovery_receipt.get("bootstrap_receipt_cid")
        or bootstrap.get("population_root") != manifest.get("population_root")
        or bootstrap.get("plan_root_cid") != manifest.get("plan_root_cid")
        or bootstrap.get("source_head") != manifest.get("source_head")
        or bootstrap.get("repository_tree_id")
        != "git-tree:" + str(manifest.get("source_tree") or "")
        or bootstrap.get("authority_mode") != "embedded"
        or bootstrap.get("task_source_kind") != "duckdb"
        or bootstrap.get("maximum_writer_processes") != 1
        or bootstrap.get("quack_qualified") is not False
        or bootstrap.get("schema_revision") != "datasets-authoritative-operational-v1"
        or bootstrap.get("schema_profile") != "datasets-authoritative-operational"
        or bootstrap.get("database_paths") != expected_paths
        or verification.get("valid") is not True
        or verification.get("stores_unchanged") is not True
    ):
        raise SuccessorOperatorError("sealed bootstrap receipt binding differs")


def verify_sealed_target_continuity(
    *,
    root: Path,
    source_root: Path,
    control_sha256: str,
    coordination_sha256: str,
    execution_sha256: str,
    bootstrap_sha256: str,
    manifest_sha256: str,
    recovery_receipt_sha256: str,
) -> dict[str, Any]:
    """Admit one reviewed hash-pinned snapshot with bounded semantic checks."""

    root = root.resolve(strict=True)
    _candidate_runtime_continuity(root)
    paths = _sealed_source_paths(source_root)
    pins = {
        "control_sha256": _require_sha256_pin(
            control_sha256, noun="sealed control database"
        ),
        "coordination_sha256": _require_sha256_pin(
            coordination_sha256, noun="sealed coordination database"
        ),
        "execution_sha256": _require_sha256_pin(
            execution_sha256, noun="sealed execution database"
        ),
        "bootstrap_sha256": _require_sha256_pin(
            bootstrap_sha256, noun="sealed bootstrap receipt"
        ),
        "manifest_sha256": _require_sha256_pin(
            manifest_sha256, noun="sealed recovery manifest"
        ),
        "recovery_receipt_sha256": _require_sha256_pin(
            recovery_receipt_sha256, noun="sealed recovery receipt"
        ),
    }
    if pins != SEALED_CONTINUITY_EXPECTED_PINS:
        raise SuccessorOperatorError(
            "sealed continuity pins differ from the reviewed board candidate"
        )
    recovery_receipt = _strict_addressed_json(
        paths["recovery_receipt"],
        expected_schema=FRESH_RECOVERY_RECEIPT_SCHEMA,
        identity_field="receipt_cid",
        noun="sealed recovery receipt",
    )
    manifest_cid = str(recovery_receipt.get("manifest_cid") or "")
    if re.fullmatch(r"bagu[a-z2-7]{20,}", manifest_cid) is None:
        raise SuccessorOperatorError("sealed recovery manifest CID is unsafe")
    manifest_path = paths["recovery_root"] / f"{manifest_cid}.manifest.json"
    manifest = _strict_addressed_json(
        manifest_path,
        expected_schema=FRESH_RECOVERY_MANIFEST_SCHEMA,
        identity_field="manifest_cid",
        noun="sealed recovery manifest",
    )
    bootstrap = _strict_addressed_json(
        paths["bootstrap"],
        expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
        identity_field="receipt_cid",
        noun="sealed bootstrap receipt",
    )
    observed_identities = {
        "bootstrap_receipt_cid": bootstrap.get("receipt_cid"),
        "manifest_cid": manifest.get("manifest_cid"),
        "receipt_cid": recovery_receipt.get("receipt_cid"),
        "population_root": recovery_receipt.get("population_root"),
        "source_evidence_cid": recovery_receipt.get("source_evidence_cid"),
        "sealed_operational_verification_root": recovery_receipt.get(
            "operational_verification_root"
        ),
        "target_source_head": manifest.get("source_head"),
        "target_source_tree": manifest.get("source_tree"),
    }
    if observed_identities != SEALED_CONTINUITY_EXPECTED_IDENTITIES:
        raise SuccessorOperatorError(
            "sealed continuity identities differ from the reviewed board candidate"
        )
    observed_artifact_hashes = {
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest_path,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if any(
        observed_artifact_hashes[key] != pins[key] for key in observed_artifact_hashes
    ):
        raise SuccessorOperatorError("sealed recovery artifact SHA-256 differs")
    if (
        manifest.get("manifest_cid") != manifest_cid
        or recovery_receipt.get("bootstrap_receipt_sha256") != pins["bootstrap_sha256"]
    ):
        raise SuccessorOperatorError("sealed recovery artifact cross-binding differs")
    _verify_sealed_layout(paths, manifest_name=manifest_path.name)
    _validate_bootstrap_receipt(
        bootstrap,
        recovery_receipt=recovery_receipt,
        manifest=manifest,
    )
    config = _plain_json_object(
        _contained(
            root,
            "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
            "scheduler.json",
        ),
        noun="tracked scheduler config",
    )
    formal_plan = _plain_json_object(
        _contained(root, str(config.get("formal_plan_path") or "")),
        noun="tracked formal plan",
    )
    _validate_recovery_policy_projection(
        config=config,
        manifest=manifest,
        receipt=recovery_receipt,
    )
    _validate_historical_qualification(manifest)
    source_binding = _target_source_continuity(
        root,
        source_head=str(manifest.get("source_head") or ""),
        source_tree=str(manifest.get("source_tree") or ""),
        config=config,
    )
    control = _verify_sealed_control_state(
        paths["control"],
        expected_sha256=pins["control_sha256"],
        manifest=manifest,
        formal_plan=formal_plan,
    )
    coordination = _verify_sealed_coordination_state(
        paths["coordination"],
        expected_sha256=pins["coordination_sha256"],
        control_tasks=control["task_cids_by_alias"],
        formal_plan=formal_plan,
    )
    execution = _verify_sealed_execution_state(
        paths["execution"],
        expected_sha256=pins["execution_sha256"],
        control_schema_fingerprint=str(control["schema_fingerprint"]),
    )
    after_artifact_hashes = {
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest_path,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if after_artifact_hashes != observed_artifact_hashes:
        raise SuccessorOperatorError(
            "sealed recovery artifacts changed during verification"
        )
    report: dict[str, Any] = {
        "schema": SEALED_CONTINUITY_VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only_hash_pinned_target_snapshot",
        "admission_mode": SEALED_CONTINUITY_MODE,
        "authority_ceiling": SEALED_CONTINUITY_AUTHORITY_CEILING,
        "source_root": str(paths["root"]),
        "candidate_root": str(root),
        "source_generation": "lgcvf-run-v17",
        "target_generation": "lgcvf-run-v17",
        "manifest_cid": manifest_cid,
        "receipt_cid": recovery_receipt["receipt_cid"],
        "bootstrap_receipt_cid": bootstrap["receipt_cid"],
        "source_evidence_cid": recovery_receipt["source_evidence_cid"],
        "population_root": recovery_receipt["population_root"],
        "plan_root_cid": recovery_receipt["plan_root_cid"],
        "sealed_operational_verification_root": recovery_receipt[
            "operational_verification_root"
        ],
        "pins": pins,
        "source_binding": source_binding,
        "control": {
            key: value for key, value in control.items() if key != "task_cids_by_alias"
        },
        "coordination": coordination,
        "execution": execution,
        "completed_task_ids": list(COMPLETED_TASK_IDS),
        "todo_task_ids": list(TODO_TASK_IDS),
        "blocked_task_ids": list(BLOCKED_TASK_IDS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
        "ready_task_ids": ["LGCVF-081"],
        "stores_unchanged": True,
        "target_database_statuses_read": True,
        "source_database_statuses_read": False,
        "fresh_source_evidence_revalidated": False,
        "historical_source_bytes_revalidated": False,
        "source_provenance_authoritative": False,
        "target_snapshot_hash_pinned": True,
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "source_database_completion_records_imported": False,
        "synthetic_source_disposition": "quarantined_not_imported",
        "network_isolation_enforced": True,
        "model_provider_route": "none",
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "authoritative_for_release": False,
        "production_authorized": False,
    }
    report["verification_root"] = _content_id(report)
    return report


def _canonical_recovery_verification(root: Path = ROOT) -> dict[str, Any]:
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(_contained(root, MATERIALIZER_RELATIVE)),
        "recovery-verify",
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300.0,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SuccessorOperatorError(
            "canonical run-v17 recovery verifier returned malformed output"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(report, dict)
        or report.get("valid") is not True
        or report.get("target_generation") != "lgcvf-run-v17"
        or report.get("stores_unchanged") is not True
        or report.get("source_database_statuses_read") is not False
        or report.get("completed_count") != 13
        or report.get("todo_count") != 13
        or report.get("blocked_count") != 2
        or report.get("ready_task_ids") != ["LGCVF-081"]
    ):
        raise SuccessorOperatorError(
            "canonical run-v17 recovery is not a verified 13/13/2 recovery: "
            + str(report.get("error") or completed.stderr[-1000:])
        )
    return report


def clone_verified_successor(
    source_database: Path,
    target_database: Path,
    provenance_path: Path,
    *,
    recovery_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one complete, verified, no-overwrite successor run."""

    source = Path(source_database).resolve(strict=True)
    target = Path(os.path.abspath(os.fspath(target_database)))
    provenance = Path(os.path.abspath(os.fspath(provenance_path)))
    final_run = target.parent
    try:
        provenance_relative = provenance.relative_to(final_run)
    except ValueError as exc:
        raise SuccessorOperatorError(
            "successor provenance must be inside the target generation"
        ) from exc
    if (
        source.parent.name != "run-v17"
        or final_run.name != "run-v23"
        or target.name != "control.duckdb"
        or len(provenance_relative.parts) != 2
        or provenance_relative.parts[0] != "evidence"
    ):
        raise SuccessorOperatorError("successor clone must be run-v17 -> run-v23")
    if source == target:
        raise SuccessorOperatorError("successor source and target are identical")
    try:
        os.lstat(final_run)
    except FileNotFoundError:
        pass
    else:
        raise SuccessorOperatorError("refusing to overwrite an existing successor")
    if os.path.lexists(source.with_name(source.name + ".wal")):
        raise SuccessorOperatorError("run-v17 control database has a live WAL")
    admission_mode = str(
        recovery_verification.get("admission_mode")
        or "canonical_fresh_generation_recovery"
    )
    sealed_source_paths: dict[str, Path] | None = None
    if admission_mode == SEALED_CONTINUITY_MODE:
        sealed_source_paths = _sealed_source_paths(
            Path(str(recovery_verification.get("source_root") or ""))
        )
        if (
            source != sealed_source_paths["control"]
            or recovery_verification.get("authority_ceiling")
            != SEALED_CONTINUITY_AUTHORITY_CEILING
            or recovery_verification.get("target_snapshot_hash_pinned") is not True
            or recovery_verification.get("historical_source_bytes_revalidated")
            is not False
            or recovery_verification.get("source_provenance_authoritative") is not False
            or recovery_verification.get("authoritative_for_release") is not False
            or recovery_verification.get("production_authorized") is not False
        ):
            raise SuccessorOperatorError(
                "sealed target continuity report is not admissible"
            )
        _require_false_authority(
            recovery_verification, noun="sealed target continuity report"
        )
        _assert_sealed_report_snapshot(sealed_source_paths, recovery_verification)
    elif admission_mode != "canonical_fresh_generation_recovery":
        raise SuccessorOperatorError("successor admission mode is unsupported")
    if (
        recovery_verification.get("valid") is not True
        or recovery_verification.get("target_generation") != "lgcvf-run-v17"
        or recovery_verification.get("stores_unchanged") is not True
        or recovery_verification.get("source_database_statuses_read") is not False
    ):
        raise SuccessorOperatorError("run-v17 recovery verification is not admissible")

    source_verification = _verify_profile(source)
    source_identity = _database_identity(source)
    source_digest = _sha256_regular_file(source)
    if sealed_source_paths is not None and source_digest != (
        recovery_verification.get("pins") or {}
    ).get("control_sha256"):
        raise SuccessorOperatorError(
            "sealed control source differs from its admitted pin"
        )

    publish_parent = final_run.parent
    publish_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _privatize_owned_directory(publish_parent, noun="successor publication parent")
    # Keep the unpublished generation under the same reviewed run-v* ignore
    # boundary as the final generation.  Sealed admission is repeated after
    # cloning; a hidden .run-v23.* stage would otherwise appear as untracked
    # worktree dirt and make every real bootstrap fail closed on itself.
    stage = publish_parent / f"{final_run.name}.stage-{uuid.uuid4().hex}"
    os.mkdir(stage, mode=0o700)
    staged_database = stage / target.name
    staged_provenance = stage / provenance_relative
    parent_descriptor = os.open(
        publish_parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_before = os.fstat(parent_descriptor)
    stage_before = os.lstat(stage)
    source_descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    target_descriptor: int | None = None
    published = False
    try:
        source_before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_size <= 0
            or source_before.st_size > MAX_DATABASE_BYTES
        ):
            raise SuccessorOperatorError(
                "run-v17 source is not a bounded regular database"
            )
        target_descriptor = os.open(
            staged_database,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:
                    raise SuccessorOperatorError("run-v23 clone write made no progress")
                view = view[written:]
        os.fsync(target_descriptor)
        os.close(target_descriptor)
        target_descriptor = None

        target_verification = _verify_profile(staged_database)
        target_identity = _database_identity(staged_database)
        target_digest = _sha256_regular_file(
            staged_database,
            noun="staged successor database",
            require_private_owner=True,
        )
        if (
            _sha256_regular_file(source) != source_digest
            or target_digest != source_digest
            or target_identity != source_identity
            or target_verification.get("schema_fingerprint")
            != source_verification.get("schema_fingerprint")
        ):
            raise SuccessorOperatorError("run-v23 clone differs from verified run-v17")
        if sealed_source_paths is not None:
            pins = recovery_verification.get("pins") or {}
            refreshed = verify_sealed_target_continuity(
                root=Path(str(recovery_verification.get("candidate_root") or "")),
                source_root=sealed_source_paths["root"],
                control_sha256=str(pins.get("control_sha256") or ""),
                coordination_sha256=str(pins.get("coordination_sha256") or ""),
                execution_sha256=str(pins.get("execution_sha256") or ""),
                bootstrap_sha256=str(pins.get("bootstrap_sha256") or ""),
                manifest_sha256=str(pins.get("manifest_sha256") or ""),
                recovery_receipt_sha256=str(pins.get("recovery_receipt_sha256") or ""),
            )
            if refreshed != dict(recovery_verification):
                raise SuccessorOperatorError(
                    "sealed continuity report changed before successor publication"
                )

        receipt = {
            "schema": PROVENANCE_SCHEMA,
            "issued_at": _utc_now(),
            "source_generation": "lgcvf-run-v17",
            "target_generation": "lgcvf-run-v23",
            "source_database": str(source),
            "target_database": str(target),
            "source_sha256": source_digest,
            "target_initial_sha256": target_digest,
            "database_uuid": source_identity.get("database_uuid", ""),
            "schema_fingerprint": source_verification["schema_fingerprint"],
            "catalog_fingerprint": source_verification["catalog_fingerprint"],
            "recovery_verification_root": str(
                recovery_verification.get("verification_root") or ""
            ),
            "recovery_receipt_cid": str(recovery_verification.get("receipt_cid") or ""),
            "recovery_manifest_cid": str(
                recovery_verification.get("manifest_cid") or ""
            ),
            "bootstrap_receipt_cid": str(
                recovery_verification.get("bootstrap_receipt_cid") or ""
            ),
            "source_evidence_cid": str(
                recovery_verification.get("source_evidence_cid") or ""
            ),
            "population_root": str(recovery_verification.get("population_root") or ""),
            "plan_root_cid": str(recovery_verification.get("plan_root_cid") or ""),
            "admission_mode": admission_mode,
            "authority_ceiling": str(
                recovery_verification.get("authority_ceiling")
                or "operational_recovery_only"
            ),
            "source_root": str(
                sealed_source_paths["root"]
                if sealed_source_paths is not None
                else source.parent
            ),
            "source_coordination_database": str(
                sealed_source_paths["coordination"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_execution_database": str(
                sealed_source_paths["execution"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_bootstrap_receipt": str(
                sealed_source_paths["bootstrap"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_manifest": str(
                (
                    sealed_source_paths["recovery_root"]
                    / f"{recovery_verification.get('manifest_cid')}.manifest.json"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_receipt": str(
                sealed_source_paths["recovery_receipt"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_coordination_sha256": str(
                (recovery_verification.get("pins") or {}).get("coordination_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_execution_sha256": str(
                (recovery_verification.get("pins") or {}).get("execution_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_bootstrap_sha256": str(
                (recovery_verification.get("pins") or {}).get("bootstrap_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_manifest_sha256": str(
                (recovery_verification.get("pins") or {}).get("manifest_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_receipt_sha256": str(
                (recovery_verification.get("pins") or {}).get("recovery_receipt_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "target_source_head": str(
                (recovery_verification.get("source_binding") or {}).get(
                    "target_source_head"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "target_source_tree": str(
                (recovery_verification.get("source_binding") or {}).get(
                    "target_source_tree"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "sealed_operational_verification_root": str(
                recovery_verification.get("sealed_operational_verification_root") or ""
            ),
            "fresh_source_evidence_revalidated": admission_mode
            != SEALED_CONTINUITY_MODE,
            "historical_source_bytes_revalidated": admission_mode
            != SEALED_CONTINUITY_MODE,
            "source_provenance_authoritative": admission_mode != SEALED_CONTINUITY_MODE,
            "target_snapshot_hash_pinned": admission_mode == SEALED_CONTINUITY_MODE,
            "target_database_statuses_read": admission_mode == SEALED_CONTINUITY_MODE,
            "source_database_statuses_read_scope": (
                "lost_fresh_recovery_source_generation_lgcvf-run-v16"
            ),
            "restart_requires_live_continuity_receipt": admission_mode
            == SEALED_CONTINUITY_MODE,
            "live_continuity_receipt_implemented": False,
            "clone_preserves_database_uuid": True,
            "owner_generation_rotates_on_start": True,
            "source_database_statuses_read": False,
            "source_database_completion_records_imported": False,
            "candidate_authored_validation": True,
            "validation_self_authority": False,
            "validation_completion_authoritative": False,
            "synthetic_source_disposition": "quarantined_not_imported",
            "network_isolation_enforced": True,
            "model_provider_route": "none",
            "task_implementation_complete": False,
            "test_qualification_complete": False,
            "objective_complete": False,
            "release_qualified": False,
            "authoritative_for_release": False,
            "production_authorized": False,
        }
        receipt["receipt_cid"] = _content_id(receipt)
        _atomic_json(staged_provenance, receipt, replace=False)
        if (
            _strict_json(
                staged_provenance,
                expected_schema=PROVENANCE_SCHEMA,
                require_private_owner=True,
            )
            != receipt
        ):
            raise SuccessorOperatorError("staged successor provenance differs")
        _remove_staged_database_locks(stage, staged_database.name)
        if (
            {item.name for item in os.scandir(stage)}
            != {target.name, provenance_relative.parts[0]}
            or {item.name for item in os.scandir(staged_provenance.parent)}
            != {staged_provenance.name}
            or os.path.lexists(staged_database.with_name(staged_database.name + ".wal"))
        ):
            raise SuccessorOperatorError("staged successor inventory differs")
        _require_private_directory(stage, noun="staged successor generation")
        _require_private_directory(
            staged_provenance.parent, noun="staged successor evidence"
        )
        source_after = os.fstat(source_descriptor)
        stage_after = os.lstat(stage)
        parent_after = os.fstat(parent_descriptor)
        if (
            (
                source_before.st_dev,
                source_before.st_ino,
                source_before.st_size,
                source_before.st_mtime_ns,
                source_before.st_ctime_ns,
            )
            != (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
                source_after.st_ctime_ns,
            )
            or (stage_before.st_dev, stage_before.st_ino)
            != (stage_after.st_dev, stage_after.st_ino)
            or (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
            or _sha256_regular_file(
                staged_database,
                noun="staged successor database",
                require_private_owner=True,
            )
            != target_digest
            or _sha256_regular_file(source) != source_digest
        ):
            raise SuccessorOperatorError(
                "source or staged successor changed before publication"
            )
        stage_descriptor = os.open(
            stage,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(stage_descriptor)
        finally:
            os.close(stage_descriptor)
        _rename_directory_noreplace(parent_descriptor, stage.name, final_run.name)
        published = True
        try:
            os.fsync(parent_descriptor)
        except OSError as exc:
            raise SuccessorOperatorError(
                "successor published completely but parent durability is uncertain"
            ) from exc
        return receipt
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(source_descriptor)
        os.close(parent_descriptor)
        if not published:
            _cleanup_successor_stage(
                stage,
                staged_database=staged_database,
                staged_provenance=staged_provenance,
            )


def _require_ignored_successor(
    root: Path,
    *,
    run_relative: Path = SUCCESSOR_RUN_RELATIVE,
) -> None:
    stage_lock = (
        run_relative.with_name(run_relative.name + ".stage-probe")
        / ".control.duckdb.lock"
    )
    for relative, noun in (
        (run_relative / "control.duckdb", "successor Git-ignore policy"),
        (stage_lock, "successor staging Git-ignore policy"),
    ):
        _git_quiet(
            root,
            ("check-ignore", "-q", "--no-index", str(relative)),
            noun=noun,
        )


def _load_native_resume_config(root: Path) -> tuple[dict[str, Any], bytes]:
    """Load the exact tracked run-v39 profile with duplicate-key rejection."""

    path = _contained(root, DEFAULT_SUCCESSOR_CONFIG_RELATIVE)
    raw = _read_bounded_regular_file(
        path,
        max_bytes=MAX_JSON_BYTES,
        noun="LGCVF native-resume candidate config",
    )

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate candidate config key: {key}")
            value[key] = item
        return value

    try:
        config = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(
            "LGCVF native-resume candidate config is invalid"
        ) from exc
    if not isinstance(config, dict):
        raise SuccessorOperatorError(
            "LGCVF native-resume candidate config must be an object"
        )
    program = config.get("database_program")
    runtime = config.get("runtime_paths")
    projection = config.get("initial_projection")
    bootstrap = config.get("bootstrap_writer_policy")
    expected_projection = {
        "task_count": 28,
        "completed_task_ids": list(CONSTRUCTION_COMPLETED_TASK_IDS),
        "ready_task_ids": ["LGCVF-051", "LGCVF-060", "LGCVF-070", "LGCVF-080"],
        "blocked_task_ids": list(BLOCKED_TASK_IDS),
        "terminal_task_id": "LGCVF-124",
        "goal_count": 14,
        "root_goal_id": "LGCVF-G000",
    }
    if (
        config.get("schema")
        != (
            "ipfs_accelerate_py.agent_supervisor."
            "logic_governed_compositional_verification_fabric.scheduler_config@1"
        )
        or config.get("board_namespace")
        != "logic-governed-compositional-verification-fabric-v1"
        or not isinstance(program, dict)
        or not isinstance(runtime, dict)
        or program.get("store_id") != SUCCESSOR_DATABASE_RELATIVE.as_posix()
        or program.get("store_generation") != SUCCESSOR_STORE_GENERATION
        or program.get("authority_mode") != "quack"
        or runtime.get("root") != SUCCESSOR_RUN_RELATIVE.as_posix()
        or projection != expected_projection
        or bootstrap
        != {
            "maximum_processes": 1,
            "quack_required": False,
            "offline_single_writer_materialization_permitted": True,
            "quack_required_after_publish": True,
            "direct_multi_process_duckdb_permitted": False,
            "automatic_installation_permitted": False,
        }
    ):
        raise SuccessorOperatorError(
            "LGCVF native-resume candidate projection or generation differs"
        )
    return config, raw


def _native_resume_stage_config(
    config: Mapping[str, Any],
    *,
    root: Path,
    stage: Path,
) -> dict[str, Any]:
    """Retarget only unpublished materializer paths into the private stage."""

    staged = copy.deepcopy(dict(config))
    try:
        relative = stage.relative_to(root.resolve(strict=True)).as_posix()
        program = staged["database_program"]
        runtime = staged["runtime_paths"]
        program["store_id"] = f"{relative}/control.duckdb"
        runtime["evidence"] = f"{relative}/evidence"
    except (KeyError, TypeError, ValueError) as exc:
        raise SuccessorOperatorError(
            "LGCVF native-resume staging paths are unavailable"
        ) from exc
    return staged


def _native_resume_materialized_projection(
    config: Mapping[str, Any],
    *,
    task_ids: Sequence[str],
    completed_task_ids: Sequence[str],
    todo_task_ids: Sequence[str],
    blocked_task_ids: Sequence[str],
    ready_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Reconstruct the one immutable initial task frontier and its CID."""

    projection = config.get("initial_projection")
    if not isinstance(projection, Mapping):
        raise SuccessorOperatorError("native-resume initial projection is unavailable")
    tasks = list(task_ids)
    completed = list(completed_task_ids)
    todo = list(todo_task_ids)
    blocked = list(blocked_task_ids)
    ready = list(ready_task_ids)
    expected_completed = list(projection.get("completed_task_ids") or ())
    expected_ready = list(projection.get("ready_task_ids") or ())
    expected_blocked = list(projection.get("blocked_task_ids") or ())
    expected_tasks = list(LGCVF_TASK_ALIASES)
    terminal = set(expected_completed) | set(expected_blocked)
    expected_todo = [alias for alias in expected_tasks if alias not in terminal]
    if (
        projection.get("task_count") != len(expected_tasks)
        or tasks != expected_tasks
        or completed != expected_completed
        or todo != expected_todo
        or ready != expected_ready
        or blocked != expected_blocked
    ):
        raise SuccessorOperatorError(
            "materialized native-resume authority differs from initial_projection"
        )
    result = {
        "task_count": len(tasks),
        "completed_count": len(completed),
        "todo_count": len(todo),
        "blocked_count": len(blocked),
        "completed_task_ids": completed,
        "ready_task_ids": ready,
        "blocked_task_ids": blocked,
    }
    result["projection_root"] = _content_id(result)
    return result


def _verify_native_resume_projection(
    database: Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact initial task frontier from the unpublished DB."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with DatabaseTaskSource(database, install_schema=False) as source:
        records = list(source.list_tasks(limit=100).tasks)
        ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    return _native_resume_materialized_projection(
        config,
        task_ids=[item.task_alias for item in records],
        completed_task_ids=[
            item.task_alias for item in records if item.status == "completed"
        ],
        todo_task_ids=[item.task_alias for item in records if item.status == "todo"],
        blocked_task_ids=[
            item.task_alias for item in records if item.status == "blocked"
        ],
        ready_task_ids=ready,
    )


def _validate_native_bootstrap_receipt(
    receipt: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    database_paths: Mapping[str, str],
    source_head: str,
    repository_tree_id: str,
    population_root: str,
    plan_root_cid: str,
    schema_fingerprint: str,
    catalog_fingerprint: str,
) -> None:
    """Replay the exact initial materializer receipt semantics."""

    projection = config.get("initial_projection")
    materialization = receipt.get("materialization")
    verification = receipt.get("verification")
    schema_install = receipt.get("schema_install")
    if not all(
        isinstance(item, Mapping)
        for item in (projection, materialization, verification, schema_install)
    ):
        raise SuccessorOperatorError(
            "native-resume bootstrap receipt structure differs"
        )
    assert isinstance(projection, Mapping)
    assert isinstance(materialization, Mapping)
    assert isinstance(verification, Mapping)
    assert isinstance(schema_install, Mapping)
    receipt_body = dict(receipt)
    claimed_receipt_cid = str(receipt_body.pop("receipt_cid", ""))
    verification_body = dict(verification)
    claimed_verification_root = str(
        verification_body.pop("verification_root", "")
    )
    task_source = materialization.get("task_source")
    control = verification.get("control")
    coordination = verification.get("coordination")
    execution = verification.get("execution")
    if not all(
        isinstance(item, Mapping)
        for item in (task_source, control, coordination, execution)
    ):
        raise SuccessorOperatorError(
            "native-resume bootstrap receipt projection differs"
        )
    assert isinstance(task_source, Mapping)
    assert isinstance(control, Mapping)
    assert isinstance(coordination, Mapping)
    assert isinstance(execution, Mapping)
    registered = materialization.get("registered_task_cids")
    completed = materialization.get("bootstrap_completed_task_cids")
    task_cids = task_source.get("task_cids")
    statuses = control.get("statuses")
    ready = list(projection.get("ready_task_ids") or ())
    completed_aliases = list(projection.get("completed_task_ids") or ())
    blocked_aliases = list(projection.get("blocked_task_ids") or ())
    task_count = int(projection.get("task_count") or 0)
    goal_count = int(projection.get("goal_count") or 0)
    expected_top_level_fields = {
        "schema",
        "authority_mode",
        "task_source_kind",
        "maximum_writer_processes",
        "quack_qualified",
        "schema_revision",
        "schema_profile",
        "semantic_truth_authority",
        "operational_coordination_authority",
        "population_root",
        "plan_root_cid",
        "repository_tree_id",
        "source_head",
        "database_paths",
        "schema_install",
        "materialization",
        "verification",
        "receipt_cid",
    }
    expected_verification_fields = {
        "schema",
        "valid",
        "verification_mode",
        "expected_stage",
        "population_root",
        "plan_root_cid",
        "repository_tree_id",
        "control",
        "coordination",
        "execution",
        "stores_unchanged",
        "maximum_writer_processes",
        "quack_qualified",
        "verification_root",
    }
    expected_coordination_counts = {
        "active_fenced_leases": 0,
        "active_maintenance_leases": 0,
        "active_resource_claims": 0,
        "active_task_attempts": 0,
        "active_task_claims": 0,
        "dependency_edges": 46,
        "fenced_leases": 0,
        "logical_completions": len(completed_aliases),
        "maintenance_leases": 0,
        "registered_tasks": task_count,
        "resource_claims": 0,
        "task_attempts": 0,
        "task_claims": 0,
    }
    expected_execution_counts = {
        "attempt_phases": 0,
        "daemon_execution_events": 0,
        "database_task_attempts": 0,
        "effect_claims": 0,
        "provider_invocations": 0,
    }
    expected_schema_install_fields = {
        "catalog_fingerprint",
        "changed",
        "from_version",
        "receipts",
        "schema",
        "schema_fingerprint",
        "to_version",
    }
    expected_migration_receipt_fields = {
        "application_version",
        "checksum",
        "error_text",
        "finished_at",
        "migration_id",
        "outcome",
        "receipt_cid",
        "schema",
        "schema_fingerprint",
        "started_at",
        "tool_version",
        "version",
    }
    expected_task_source_fields = {
        "event_watermark",
        "goal_count",
        "goal_edge_count",
        "plan_count",
        "plan_root_cid",
        "projection_cid",
        "repository_tree_id",
        "schema",
        "task_cids",
        "task_count",
    }
    expected_control_fields = {
        "catalog_projection",
        "completion_receipts",
        "dependency_count",
        "event_stream_root",
        "evidence",
        "goal_count",
        "objective_revision_history",
        "plan_projection",
        "plan_revision_history",
        "ready_task_aliases",
        "relation_count",
        "relation_inventory",
        "residual_content_projection",
        "runtime_progress_observed",
        "schema_verification",
        "semantic_event_stream_root",
        "semantic_events",
        "statuses",
        "table_counts",
        "task_count",
        "task_revision_histories",
        "tasks",
    }
    expected_coordination_fields = {
        "catalog_projection",
        "counts",
        "projection_root",
    }
    expected_execution_fields = {
        "catalog_projection",
        "metadata",
        "row_counts",
        "runtime_progress_observed",
        "schema_inventory",
    }

    def exact_integer(value: Any, expected: int) -> bool:
        return type(value) is int and value == expected

    def exact_integer_mapping(value: Any, expected: Mapping[str, int]) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(
                exact_integer(value.get(key), item)
                for key, item in expected.items()
            )
        )

    def canonical_cid(value: Any) -> bool:
        return (
            isinstance(value, str)
            and re.fullmatch(r"b[a-z2-7]{60}", value) is not None
        )

    migration_receipts = schema_install.get("receipts")
    migration_receipt: Mapping[str, Any] = (
        migration_receipts[0]
        if isinstance(migration_receipts, list)
        and len(migration_receipts) == 1
        and isinstance(migration_receipts[0], Mapping)
        else {}
    )
    semantic_difference = (
        set(receipt) != expected_top_level_fields
        or claimed_receipt_cid != _content_id(receipt_body)
        or not exact_integer(projection.get("task_count"), 28)
        or not exact_integer(projection.get("goal_count"), 14)
        or receipt.get("schema") != BOOTSTRAP_RECEIPT_SCHEMA
        or receipt.get("authority_mode") != "embedded"
        or receipt.get("task_source_kind") != "duckdb"
        or not exact_integer(receipt.get("maximum_writer_processes"), 1)
        or receipt.get("quack_qualified") is not False
        or receipt.get("schema_revision")
        != "datasets-authoritative-operational-v1"
        or receipt.get("schema_profile")
        != "datasets-authoritative-operational"
        or receipt.get("semantic_truth_authority") != "ipfs_datasets_py"
        or receipt.get("operational_coordination_authority")
        != "ipfs_accelerate_py"
        or receipt.get("population_root") != population_root
        or receipt.get("plan_root_cid") != plan_root_cid
        or receipt.get("repository_tree_id") != repository_tree_id
        or receipt.get("source_head") != source_head
        or receipt.get("database_paths") != dict(database_paths)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in database_paths.items()
        )
        or set(schema_install) != expected_schema_install_fields
        or schema_install.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/control-plane-migration-run@1"
        or schema_install.get("changed") is not True
        or not exact_integer(schema_install.get("from_version"), 0)
        or not exact_integer(schema_install.get("to_version"), 1)
        or schema_install.get("schema_fingerprint") != schema_fingerprint
        or schema_install.get("catalog_fingerprint") != catalog_fingerprint
        or set(migration_receipt) != expected_migration_receipt_fields
        or migration_receipt.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "control-plane-migration-receipt@1"
        )
        or migration_receipt.get("schema_fingerprint") != schema_fingerprint
        or migration_receipt.get("outcome") != "applied"
        or migration_receipt.get("application_version") != "lgcvf-v1"
        or not exact_integer(migration_receipt.get("version"), 1)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(migration_receipt.get("checksum") or ""),
        )
        is None
        or not canonical_cid(migration_receipt.get("receipt_cid"))
        or set(materialization)
        != {
            "bootstrap_completed_task_cids",
            "registered_task_cids",
            "task_source",
        }
        or not isinstance(registered, list)
        or len(registered) != task_count
        or not all(canonical_cid(item) for item in registered)
        or len(set(registered)) != task_count
        or task_cids != registered
        or not isinstance(completed, list)
        or completed != registered[: len(completed_aliases)]
        or set(task_source) != expected_task_source_fields
        or task_source.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/database-task-source@1"
        or not exact_integer(task_source.get("task_count"), task_count)
        or not exact_integer(task_source.get("goal_count"), goal_count)
        or not exact_integer(task_source.get("goal_edge_count"), 38)
        or not exact_integer(task_source.get("plan_count"), 1)
        or not exact_integer(task_source.get("event_watermark"), 82)
        or not canonical_cid(task_source.get("projection_cid"))
        or task_source.get("plan_root_cid") != plan_root_cid
        or task_source.get("repository_tree_id") != repository_tree_id
        or set(verification) != expected_verification_fields
        or claimed_verification_root != _content_id(verification_body)
        or verification.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "lgcvf-duckdb-read-only-verification@1"
        )
        or verification.get("valid") is not True
        or verification.get("verification_mode") != "read_only"
        or verification.get("expected_stage") != "initial"
        or verification.get("population_root") != population_root
        or verification.get("plan_root_cid") != plan_root_cid
        or verification.get("repository_tree_id") != repository_tree_id
        or verification.get("stores_unchanged") is not True
        or not exact_integer(verification.get("maximum_writer_processes"), 1)
        or verification.get("quack_qualified") is not False
        or set(control) != expected_control_fields
        or not exact_integer(control.get("task_count"), task_count)
        or not exact_integer(control.get("goal_count"), goal_count)
        or not exact_integer(control.get("dependency_count"), 46)
        or control.get("ready_task_aliases") != ready
        or control.get("runtime_progress_observed") is not False
        or not isinstance(statuses, Mapping)
        or not all(
            isinstance(alias, str) and isinstance(status, str)
            for alias, status in statuses.items()
        )
        or len(statuses) != task_count
        or [alias for alias, status in statuses.items() if status == "completed"]
        != completed_aliases
        or [alias for alias, status in statuses.items() if status == "blocked"]
        != blocked_aliases
        or sum(status == "todo" for status in statuses.values())
        != task_count - len(completed_aliases) - len(blocked_aliases)
        or set(coordination) != expected_coordination_fields
        or not exact_integer_mapping(
            coordination.get("counts"), expected_coordination_counts
        )
        or set(execution) != expected_execution_fields
        or not exact_integer_mapping(
            execution.get("row_counts"), expected_execution_counts
        )
        or execution.get("runtime_progress_observed") is not False
    )
    if semantic_difference:
        raise SuccessorOperatorError(
            "native-resume bootstrap receipt semantics differ"
        )


def _verify_native_resume_stage_allowlist(
    stage: Path,
    *,
    include_provenance: bool,
) -> None:
    """Require the materializer to leave only the declared initial objects."""

    expected_files = set(NATIVE_RESUME_STAGE_DATA_FILES)
    if not include_provenance:
        expected_files.remove("evidence/quack-successor-provenance.json")
    observed_directories: set[str] = set()
    observed_files: set[str] = set()
    for path in stage.rglob("*"):
        relative = path.relative_to(stage).as_posix()
        metadata = os.lstat(path)
        if stat.S_ISDIR(metadata.st_mode):
            if (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise SuccessorOperatorError(
                    "native-resume stage directory custody differs"
                )
            observed_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            if (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                raise SuccessorOperatorError(
                    "native-resume stage file custody differs"
                )
            observed_files.add(relative)
        else:
            raise SuccessorOperatorError(
                "native-resume stage contains an undeclared object"
            )
    if (
        observed_directories != set(NATIVE_RESUME_STAGE_DIRECTORIES)
        or observed_files
        != expected_files | set(NATIVE_RESUME_STAGE_LOCK_FILES)
        or any(
            os.lstat(stage / relative).st_size != 0
            for relative in NATIVE_RESUME_STAGE_LOCK_FILES
        )
    ):
        raise SuccessorOperatorError(
            "native-resume stage inventory differs from the exact allowlist"
        )


def _privatize_and_sync_native_resume_stage(stage: Path) -> None:
    """Reject special/aliased stage members, privatize them, and fsync all."""

    _privatize_owned_directory(stage, noun="native-resume stage root")
    entries = sorted(stage.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    if len(entries) > 128:
        raise SuccessorOperatorError("native-resume stage inventory exceeds its bound")
    for path in entries:
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise SuccessorOperatorError("native-resume stage custody differs")
        if stat.S_ISDIR(metadata.st_mode):
            os.chmod(path, 0o700)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            os.chmod(path, 0o600)
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise SuccessorOperatorError(
                "native-resume stage contains a special or aliased object"
            )
    for directory in [
        *[item for item in entries if item.is_dir()],
        stage,
    ]:
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _native_resume_stage_inventory(stage: Path) -> tuple[tuple[Any, ...], ...]:
    """Return an inode-bound inventory for final publication race checks."""

    inventory: list[tuple[Any, ...]] = []
    for path in sorted(stage.rglob("*")):
        metadata = os.lstat(path)
        kind = (
            "directory"
            if stat.S_ISDIR(metadata.st_mode)
            else "file" if stat.S_ISREG(metadata.st_mode) else "special"
        )
        inventory.append(
            (
                path.relative_to(stage).as_posix(),
                kind,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        )
    return tuple(inventory)


def _cleanup_native_resume_stage(stage: Path, *, publish_parent: Path) -> None:
    """Remove only an unpublished, owner-private stage created by this call."""

    try:
        metadata = os.lstat(stage)
    except FileNotFoundError:
        return
    try:
        valid_parent = stage.parent.resolve(strict=True) == publish_parent.resolve(
            strict=True
        )
    except OSError:
        valid_parent = False
    if (
        valid_parent
        and stage.name.startswith(SUCCESSOR_RUN_RELATIVE.name + ".stage-")
        and stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
    ):
        shutil.rmtree(stage)


def bootstrap_native_resume(root: Path = ROOT) -> dict[str, Any]:
    """Atomically publish run-v39 from the tracked candidate projection."""

    root = root.resolve(strict=True)
    paths = _paths(root)
    _require_ignored_successor(root)
    config, config_raw = _load_native_resume_config(root)
    continuity_before = _candidate_runtime_continuity(root)
    final_run = paths["successor_database"].parent
    try:
        os.lstat(final_run)
    except FileNotFoundError:
        pass
    else:
        raise SuccessorOperatorError("refusing to overwrite an existing successor")
    publish_parent = final_run.parent
    publish_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _privatize_owned_directory(
        publish_parent,
        noun="native-resume publication parent",
    )
    parent_descriptor = os.open(
        publish_parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_before = os.fstat(parent_descriptor)
    stage = publish_parent / (
        f"{SUCCESSOR_RUN_RELATIVE.name}.stage-{uuid.uuid4().hex}"
    )
    os.mkdir(stage, mode=0o700)
    published = False
    try:
        materializer = importlib.import_module(
            "scripts."
            "materialize_logic_governed_compositional_verification_fabric_control_plane"
        )
        population = materializer.build_population(config, root=root)
        staged_config = _native_resume_stage_config(
            config,
            root=root,
            stage=stage,
        )
        bootstrap = materializer._materialize_canonical(
            staged_config,
            population,
            root=root,
            recheck_source=True,
        )
        staged_database = stage / "control.duckdb"
        staged_coordination = stage / "control.coordination.duckdb"
        staged_execution = stage / "control.execution.duckdb"
        staged_bootstrap = stage / "evidence" / "bootstrap" / "materialization.json"
        staged_provenance = stage / "evidence" / paths["provenance"].name
        for database in (staged_database, staged_coordination, staged_execution):
            if os.path.lexists(database.with_name(database.name + ".wal")):
                raise SuccessorOperatorError(
                    "native-resume materialization retained a live WAL"
                )
        projection = _verify_native_resume_projection(
            staged_database,
            config=config,
        )
        final_database_paths = {
            "control": SUCCESSOR_DATABASE_RELATIVE.as_posix(),
            "coordination": SUCCESSOR_DATABASE_RELATIVE.with_name(
                "control.coordination.duckdb"
            ).as_posix(),
            "execution": SUCCESSOR_DATABASE_RELATIVE.with_name(
                "control.execution.duckdb"
            ).as_posix(),
        }
        bootstrap = dict(bootstrap)
        bootstrap["database_paths"] = final_database_paths
        bootstrap.pop("receipt_cid", None)
        bootstrap["receipt_cid"] = _content_id(bootstrap)
        profile = _verify_profile(staged_database)
        _validate_native_bootstrap_receipt(
            bootstrap,
            config=config,
            database_paths=final_database_paths,
            source_head=str(population["source_head"]),
            repository_tree_id=str(population["repository_tree_id"]),
            population_root=str(population["population_root"]),
            plan_root_cid=str(population["plan_root_cid"]),
            schema_fingerprint=str(profile.get("schema_fingerprint") or ""),
            catalog_fingerprint=str(profile.get("catalog_fingerprint") or ""),
        )
        _atomic_json(staged_bootstrap, bootstrap, replace=True)

        identity = _database_identity(staged_database)
        config_after, config_raw_after = _load_native_resume_config(root)
        continuity_after = _candidate_runtime_continuity(root)
        if (
            config_after != config
            or config_raw_after != config_raw
            or continuity_after != continuity_before
            or population.get("source_head") != continuity_before.get("current_head")
            or population.get("repository_tree_id")
            != "git-tree:" + str(continuity_before.get("current_tree") or "")
        ):
            raise SuccessorOperatorError(
                "candidate source changed during native-resume materialization"
            )
        target_digest = _sha256_regular_file(staged_database)
        coordination_digest = _sha256_regular_file(staged_coordination)
        execution_digest = _sha256_regular_file(staged_execution)
        receipt = {
            "schema": PROVENANCE_SCHEMA,
            "issued_at": _utc_now(),
            "admission_mode": NATIVE_RESUME_ADMISSION_MODE,
            "source_generation": NATIVE_RESUME_SOURCE_GENERATION,
            "target_generation": SUCCESSOR_STORE_GENERATION,
            "source_database": "",
            "target_database": str(paths["successor_database"]),
            "source_head": str(population["source_head"]),
            "source_tree": str(continuity_before["current_tree"]),
            "source_forest_root": str(population["source_forest_root"]),
            "datasets_head": str(continuity_before["datasets_head"]),
            "datasets_tree": str(continuity_before["datasets_tree"]),
            "candidate_config_path": DEFAULT_SUCCESSOR_CONFIG_RELATIVE.as_posix(),
            "candidate_config_sha256": (
                "sha256:" + hashlib.sha256(config_raw).hexdigest()
            ),
            "population_root": str(population["population_root"]),
            "plan_root_cid": str(population["plan_root_cid"]),
            "initial_projection": copy.deepcopy(config["initial_projection"]),
            "materialized_projection": projection,
            "bootstrap_receipt_cid": str(bootstrap["receipt_cid"]),
            "bootstrap_verification_root": str(
                (bootstrap.get("verification") or {}).get("verification_root") or ""
            ),
            "target_initial_sha256": target_digest,
            "target_coordination_initial_sha256": coordination_digest,
            "target_execution_initial_sha256": execution_digest,
            "database_uuid": str(identity.get("database_uuid") or ""),
            "schema_fingerprint": str(profile.get("schema_fingerprint") or ""),
            "catalog_fingerprint": str(profile.get("catalog_fingerprint") or ""),
            "initial_projection_reset": True,
            "continuity_completion_records_imported": False,
            "source_database_statuses_read": False,
            "source_database_completion_records_imported": False,
            "quack_required_after_publish": True,
            "direct_multi_process_duckdb_permitted": False,
            "ducklake_projection_authoritative": False,
            "restart_requires_live_continuity_receipt": True,
            "live_continuity_receipt_implemented": False,
            "candidate_authored_validation": True,
            "validation_self_authority": False,
            "validation_completion_authoritative": False,
            "network_isolation_enforced": True,
            "model_provider_route": "none",
            "task_implementation_complete": False,
            "test_qualification_complete": False,
            "objective_complete": False,
            "release_qualified": False,
            "authoritative_for_release": False,
            "production_authorized": False,
        }
        receipt["receipt_cid"] = _content_id(receipt)
        _atomic_json(staged_provenance, receipt, replace=False)
        _privatize_and_sync_native_resume_stage(stage)
        _verify_native_resume_stage_allowlist(stage, include_provenance=True)
        if _strict_json(
            staged_bootstrap,
            expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
            require_private_owner=True,
        ) != bootstrap:
            raise SuccessorOperatorError(
                "native-resume bootstrap receipt replay differs"
            )
        stage_sealed = os.lstat(stage)
        sealed_inventory = _native_resume_stage_inventory(stage)
        if _strict_json(
            staged_provenance,
            expected_schema=PROVENANCE_SCHEMA,
            require_private_owner=True,
        ) != receipt:
            raise SuccessorOperatorError("native-resume provenance replay differs")
        _require_private_directory(stage, noun="native-resume stage root")
        _verify_native_resume_stage_allowlist(stage, include_provenance=True)
        parent_after = os.fstat(parent_descriptor)
        stage_after = os.lstat(stage)
        if (
            (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
            or (
                stage_sealed.st_dev,
                stage_sealed.st_ino,
                stage_sealed.st_uid,
                stage_sealed.st_mode,
                stage_sealed.st_nlink,
            )
            != (
                stage_after.st_dev,
                stage_after.st_ino,
                stage_after.st_uid,
                stage_after.st_mode,
                stage_after.st_nlink,
            )
            or os.path.lexists(final_run)
            or _candidate_runtime_continuity(root) != continuity_before
            or _native_resume_stage_inventory(stage) != sealed_inventory
            or _sha256_regular_file(staged_database) != target_digest
            or _sha256_regular_file(staged_coordination) != coordination_digest
            or _sha256_regular_file(staged_execution) != execution_digest
            or _strict_json(
                staged_bootstrap,
                expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
                require_private_owner=True,
            )
            != bootstrap
            or _strict_json(
                staged_provenance,
                expected_schema=PROVENANCE_SCHEMA,
                require_private_owner=True,
            )
            != receipt
        ):
            raise SuccessorOperatorError(
                "native-resume publication boundary changed before rename"
            )
        _rename_directory_noreplace(
            parent_descriptor,
            stage.name,
            final_run.name,
        )
        published = True
        try:
            os.fsync(parent_descriptor)
        except OSError as exc:
            raise SuccessorOperatorError(
                "native resume published completely but parent durability is uncertain"
            ) from exc
        return receipt
    finally:
        os.close(parent_descriptor)
        if not published:
            _cleanup_native_resume_stage(stage, publish_parent=publish_parent)


def bootstrap_successor(root: Path = ROOT) -> dict[str, Any]:
    return bootstrap_native_resume(root)


def bootstrap_sealed_successor(
    *,
    root: Path,
    source_root: Path,
    control_sha256: str,
    coordination_sha256: str,
    execution_sha256: str,
    bootstrap_sha256: str,
    manifest_sha256: str,
    recovery_receipt_sha256: str,
) -> dict[str, Any]:
    _require_ignored_successor(root, run_relative=LEGACY_SUCCESSOR_RUN_RELATIVE)
    verification = verify_sealed_target_continuity(
        root=root,
        source_root=source_root,
        control_sha256=control_sha256,
        coordination_sha256=coordination_sha256,
        execution_sha256=execution_sha256,
        bootstrap_sha256=bootstrap_sha256,
        manifest_sha256=manifest_sha256,
        recovery_receipt_sha256=recovery_receipt_sha256,
    )
    source_paths = _sealed_source_paths(source_root)
    return clone_verified_successor(
        source_paths["control"],
        _contained(root, LEGACY_SUCCESSOR_RUN_RELATIVE / "control.duckdb"),
        _contained(
            root,
            LEGACY_SUCCESSOR_RUN_RELATIVE
            / "evidence"
            / "quack-successor-provenance.json",
        ),
        recovery_verification=verification,
    )


def _load_provenance(
    paths: Mapping[str, Path],
    *,
    root: Path = ROOT,
    expected_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    database = paths["successor_database"]
    coordination = database.with_name("control.coordination.duckdb")
    execution = database.with_name("control.execution.duckdb")
    _require_private_directory(database.parent, noun="successor generation")
    _require_private_directory(
        paths["provenance"].parent, noun="successor evidence directory"
    )
    for noun, store in (
        ("control", database),
        ("coordination", coordination),
        ("execution", execution),
    ):
        if os.path.lexists(store.with_name(store.name + ".wal")):
            raise SuccessorOperatorError(
                f"successor {noun} database has a live WAL"
            )
    initial_receipt = _strict_json(
        paths["provenance"],
        expected_schema=PROVENANCE_SCHEMA,
        require_private_owner=True,
        verify_content_identity=False,
    )
    if expected_receipt is not None and initial_receipt != dict(expected_receipt):
        raise SuccessorOperatorError(
            "verified successor provenance differs from native authorization"
        )
    if initial_receipt.get("admission_mode") == SEALED_CONTINUITY_MODE:
        _candidate_runtime_continuity(root)
    receipt = _strict_json(
        paths["provenance"],
        expected_schema=PROVENANCE_SCHEMA,
        require_private_owner=True,
    )
    if receipt != initial_receipt:
        raise SuccessorOperatorError("successor provenance changed during admission")
    target_digest = _sha256_regular_file(
        database,
        noun="successor control database",
        require_private_owner=True,
    )
    admission_mode = str(receipt.get("admission_mode") or "")
    if receipt.get("target_database") != str(database):
        raise SuccessorOperatorError("successor provenance target differs")
    source_database: Path | None = None
    if admission_mode == NATIVE_RESUME_ADMISSION_MODE:
        config, config_raw = _load_native_resume_config(root)
        stopped_restart_present = os.path.lexists(
            paths["stopped_state_continuity"]
        )
        continuity = (
            _observe_candidate_runtime_continuity(
                root,
                require_resolved_remote=False,
            )
            if stopped_restart_present
            else _candidate_runtime_continuity(root)
        )
        source_head_value = receipt.get("source_head")
        source_tree_value = receipt.get("source_tree")
        source_head = source_head_value if type(source_head_value) is str else ""
        source_tree = source_tree_value if type(source_tree_value) is str else ""
        native_fields = {
            "schema",
            "issued_at",
            "admission_mode",
            "source_generation",
            "target_generation",
            "source_database",
            "target_database",
            "source_head",
            "source_tree",
            "source_forest_root",
            "datasets_head",
            "datasets_tree",
            "candidate_config_path",
            "candidate_config_sha256",
            "population_root",
            "plan_root_cid",
            "initial_projection",
            "materialized_projection",
            "bootstrap_receipt_cid",
            "bootstrap_verification_root",
            "target_initial_sha256",
            "target_coordination_initial_sha256",
            "target_execution_initial_sha256",
            "database_uuid",
            "schema_fingerprint",
            "catalog_fingerprint",
            "initial_projection_reset",
            "continuity_completion_records_imported",
            "source_database_statuses_read",
            "source_database_completion_records_imported",
            "quack_required_after_publish",
            "direct_multi_process_duckdb_permitted",
            "ducklake_projection_authoritative",
            "restart_requires_live_continuity_receipt",
            "live_continuity_receipt_implemented",
            "candidate_authored_validation",
            "validation_self_authority",
            "validation_completion_authoritative",
            "network_isolation_enforced",
            "model_provider_route",
            "task_implementation_complete",
            "test_qualification_complete",
            "objective_complete",
            "release_qualified",
            "authoritative_for_release",
            "production_authorized",
            "receipt_cid",
        }

        def native_content_cid(field: str) -> bool:
            value = receipt.get(field)
            return (
                type(value) is str
                and re.fullmatch(r"b[a-z2-7]{60}", value) is not None
            )

        def native_sha256(field: str) -> bool:
            value = receipt.get(field)
            return (
                type(value) is str
                and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
            )

        if (
            set(receipt) != native_fields
            or receipt.get("source_generation") != NATIVE_RESUME_SOURCE_GENERATION
            or receipt.get("target_generation") != SUCCESSOR_STORE_GENERATION
            or receipt.get("source_database") != ""
            or type(receipt.get("issued_at")) is not str
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
                receipt["issued_at"],
            )
            is None
            or source_head != continuity.get("current_head")
            or source_tree != continuity.get("current_tree")
            or receipt.get("datasets_head") != continuity.get("datasets_head")
            or receipt.get("datasets_tree") != continuity.get("datasets_tree")
            or receipt.get("candidate_config_path")
            != DEFAULT_SUCCESSOR_CONFIG_RELATIVE.as_posix()
            or receipt.get("candidate_config_sha256")
            != "sha256:" + hashlib.sha256(config_raw).hexdigest()
            or receipt.get("initial_projection") != config.get("initial_projection")
            or not native_content_cid("source_forest_root")
            or not native_content_cid("population_root")
            or not native_content_cid("plan_root_cid")
            or not native_content_cid("bootstrap_receipt_cid")
            or not native_content_cid("bootstrap_verification_root")
            or not native_content_cid("schema_fingerprint")
            or not native_content_cid("catalog_fingerprint")
            or not native_content_cid("receipt_cid")
            or not native_sha256("target_initial_sha256")
            or not native_sha256("target_coordination_initial_sha256")
            or not native_sha256("target_execution_initial_sha256")
            or type(receipt.get("database_uuid")) is not str
            or not str(receipt.get("database_uuid") or "")
            or receipt.get("initial_projection_reset") is not True
            or receipt.get("continuity_completion_records_imported") is not False
            or receipt.get("source_database_statuses_read") is not False
            or receipt.get("source_database_completion_records_imported") is not False
            or receipt.get("quack_required_after_publish") is not True
            or receipt.get("direct_multi_process_duckdb_permitted") is not False
            or receipt.get("ducklake_projection_authoritative") is not False
            or receipt.get("restart_requires_live_continuity_receipt") is not True
            or receipt.get("live_continuity_receipt_implemented") is not False
            or receipt.get("candidate_authored_validation") is not True
            or receipt.get("validation_self_authority") is not False
            or receipt.get("validation_completion_authoritative") is not False
            or receipt.get("network_isolation_enforced") is not True
            or receipt.get("model_provider_route") != "none"
            or receipt.get("task_implementation_complete") is not False
            or receipt.get("test_qualification_complete") is not False
            or receipt.get("objective_complete") is not False
            or receipt.get("release_qualified") is not False
            or receipt.get("authoritative_for_release") is not False
            or receipt.get("production_authorized") is not False
            or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
            or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
            or _git_text(
                root,
                ("show", "-s", "--format=%T", source_head),
                noun="native-resume source commit",
            )
            != source_tree
        ):
            if stopped_restart_present:
                return _load_stopped_restart_provenance(
                    paths,
                    root=root,
                    provenance=receipt,
                )
            raise SuccessorOperatorError(NATIVE_RESUME_PROVENANCE_BINDING_ERROR)
        _git_quiet(
            root,
            (
                "merge-base",
                "--is-ancestor",
                source_head,
                str(continuity.get("current_head") or ""),
            ),
            noun="native-resume source ancestry",
        )
        bootstrap_path = (
            database.parent / "evidence" / "bootstrap" / "materialization.json"
        )
        bootstrap = _strict_json(
            bootstrap_path,
            expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
            require_private_owner=True,
        )
        _validate_native_bootstrap_receipt(
            bootstrap,
            config=config,
            database_paths={
                "control": SUCCESSOR_DATABASE_RELATIVE.as_posix(),
                "coordination": SUCCESSOR_DATABASE_RELATIVE.with_name(
                    "control.coordination.duckdb"
                ).as_posix(),
                "execution": SUCCESSOR_DATABASE_RELATIVE.with_name(
                    "control.execution.duckdb"
                ).as_posix(),
            },
            source_head=source_head,
            repository_tree_id="git-tree:" + source_tree,
            population_root=receipt["population_root"],
            plan_root_cid=receipt["plan_root_cid"],
            schema_fingerprint=receipt["schema_fingerprint"],
            catalog_fingerprint=receipt["catalog_fingerprint"],
        )
        bootstrap_verification = bootstrap["verification"]
        assert isinstance(bootstrap_verification, Mapping)
        if (
            bootstrap.get("receipt_cid") != receipt.get("bootstrap_receipt_cid")
            or bootstrap_verification.get("verification_root")
            != receipt.get("bootstrap_verification_root")
            or bootstrap.get("population_root") != receipt.get("population_root")
            or bootstrap.get("plan_root_cid") != receipt.get("plan_root_cid")
        ):
            raise SuccessorOperatorError(
                "native-resume bootstrap/provenance cross-binding differs"
            )
        if (
            target_digest != receipt.get("target_initial_sha256")
            or _sha256_regular_file(
                coordination,
                noun="native-resume coordination database",
                require_private_owner=True,
            )
            != receipt.get("target_coordination_initial_sha256")
            or _sha256_regular_file(
                execution,
                noun="native-resume execution database",
                require_private_owner=True,
            )
            != receipt.get("target_execution_initial_sha256")
        ):
            if stopped_restart_present:
                return _load_stopped_restart_provenance(
                    paths,
                    root=root,
                    provenance=receipt,
                )
            raise SuccessorOperatorError(
                NATIVE_RESUME_LIVE_CONTINUITY_REQUIRED_ERROR
            )
        projection = _verify_native_resume_projection(database, config=config)
        if projection != receipt.get("materialized_projection"):
            raise SuccessorOperatorError(
                "native-resume initial projection replay differs"
            )
    elif admission_mode == "canonical_fresh_generation_recovery":
        source_database = paths["source_database"]
        if receipt.get("source_database") != str(source_database):
            raise SuccessorOperatorError("successor provenance no longer binds run-v17")
    elif admission_mode == SEALED_CONTINUITY_MODE:
        if (
            receipt.get("authority_ceiling") != SEALED_CONTINUITY_AUTHORITY_CEILING
            or receipt.get("fresh_source_evidence_revalidated") is not False
            or receipt.get("historical_source_bytes_revalidated") is not False
            or receipt.get("source_provenance_authoritative") is not False
            or receipt.get("target_snapshot_hash_pinned") is not True
            or receipt.get("target_database_statuses_read") is not True
            or receipt.get("source_database_statuses_read_scope")
            != "lost_fresh_recovery_source_generation_lgcvf-run-v16"
            or receipt.get("restart_requires_live_continuity_receipt") is not True
            or receipt.get("live_continuity_receipt_implemented") is not False
            or receipt.get("authoritative_for_release") is not False
            or receipt.get("production_authorized") is not False
            or receipt.get("source_generation") != "lgcvf-run-v17"
            or receipt.get("target_generation") != "lgcvf-run-v23"
            or receipt.get("clone_preserves_database_uuid") is not True
            or receipt.get("owner_generation_rotates_on_start") is not True
        ):
            raise SuccessorOperatorError("sealed successor authority ceiling differs")
        _require_false_authority(receipt, noun="sealed successor provenance")
        sealed = _sealed_source_paths(Path(str(receipt.get("source_root") or "")))
        source_database = sealed["control"]
        expected_manifest = (
            sealed["recovery_root"]
            / f"{receipt.get('recovery_manifest_cid')}.manifest.json"
        )
        expected_paths = {
            "source_database": sealed["control"],
            "source_coordination_database": sealed["coordination"],
            "source_execution_database": sealed["execution"],
            "source_bootstrap_receipt": sealed["bootstrap"],
            "source_recovery_receipt": sealed["recovery_receipt"],
            "source_recovery_manifest": expected_manifest,
        }
        if any(
            receipt.get(field) != str(path) for field, path in expected_paths.items()
        ):
            raise SuccessorOperatorError("sealed successor source path binding differs")
        sealed_hashes = {
            "source_sha256": _sha256_regular_file(
                sealed["control"],
                noun="sealed control database",
                require_private_owner=True,
            ),
            "source_coordination_sha256": _sha256_regular_file(
                sealed["coordination"],
                noun="sealed coordination database",
                require_private_owner=True,
            ),
            "source_execution_sha256": _sha256_regular_file(
                sealed["execution"],
                noun="sealed execution database",
                require_private_owner=True,
            ),
            "source_bootstrap_sha256": _sha256_regular_file(
                sealed["bootstrap"],
                max_bytes=MAX_JSON_BYTES,
                noun="sealed bootstrap receipt",
                require_private_owner=True,
            ),
            "source_recovery_manifest_sha256": _sha256_regular_file(
                expected_manifest,
                max_bytes=MAX_JSON_BYTES,
                noun="sealed recovery manifest",
                require_private_owner=True,
            ),
            "source_recovery_receipt_sha256": _sha256_regular_file(
                sealed["recovery_receipt"],
                max_bytes=MAX_JSON_BYTES,
                noun="sealed recovery receipt",
                require_private_owner=True,
            ),
        }
        if any(receipt.get(field) != digest for field, digest in sealed_hashes.items()):
            raise SuccessorOperatorError("sealed successor source hash binding differs")
        if sealed_hashes != {
            "source_sha256": SEALED_CONTINUITY_EXPECTED_PINS["control_sha256"],
            "source_coordination_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "coordination_sha256"
            ],
            "source_execution_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "execution_sha256"
            ],
            "source_bootstrap_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "bootstrap_sha256"
            ],
            "source_recovery_manifest_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "manifest_sha256"
            ],
            "source_recovery_receipt_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "recovery_receipt_sha256"
            ],
        }:
            raise SuccessorOperatorError("sealed successor reviewed pins differ")
        refreshed = verify_sealed_target_continuity(
            root=root,
            source_root=sealed["root"],
            **SEALED_CONTINUITY_EXPECTED_PINS,
        )
        source_binding = refreshed.get("source_binding") or {}
        semantic_bindings = {
            "recovery_verification_root": refreshed.get("verification_root"),
            "recovery_receipt_cid": refreshed.get("receipt_cid"),
            "recovery_manifest_cid": refreshed.get("manifest_cid"),
            "bootstrap_receipt_cid": refreshed.get("bootstrap_receipt_cid"),
            "source_evidence_cid": refreshed.get("source_evidence_cid"),
            "population_root": refreshed.get("population_root"),
            "plan_root_cid": refreshed.get("plan_root_cid"),
            "target_source_head": source_binding.get("target_source_head"),
            "target_source_tree": source_binding.get("target_source_tree"),
            "sealed_operational_verification_root": refreshed.get(
                "sealed_operational_verification_root"
            ),
        }
        if any(
            receipt.get(field) != expected
            for field, expected in semantic_bindings.items()
        ):
            raise SuccessorOperatorError(
                "sealed successor provenance cross-binding differs"
            )
        if (
            target_digest != receipt.get("target_initial_sha256")
            or target_digest != SEALED_CONTINUITY_EXPECTED_PINS["control_sha256"]
        ):
            raise SuccessorOperatorError(
                "sealed successor changed after its initial admission; restart "
                "requires an unimplemented live-continuity receipt"
            )
    else:
        raise SuccessorOperatorError("successor provenance admission mode differs")
    if source_database is not None and _sha256_regular_file(
        source_database,
        noun="successor provenance source database",
        require_private_owner=admission_mode == SEALED_CONTINUITY_MODE,
    ) != receipt.get("source_sha256"):
        raise SuccessorOperatorError("successor provenance no longer binds run-v17")
    verification = _verify_profile(database)
    identity = _database_identity(database)
    if (
        verification.get("schema_fingerprint") != receipt.get("schema_fingerprint")
        or verification.get("catalog_fingerprint") != receipt.get("catalog_fingerprint")
        or identity.get("database_uuid") != receipt.get("database_uuid")
    ):
        raise SuccessorOperatorError(
            "successor database identity differs from provenance"
        )
    return receipt


def _load_lgcvf_live_raw_provenance_receipt(
    paths: Mapping[str, Path],
    *,
    _receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Read the content-addressed receipt without importing the database stack."""

    receipt_path = paths["provenance"] if _receipt_path is None else _receipt_path
    _require_private_directory(
        receipt_path.parent,
        noun="successor evidence directory",
    )
    receipt = _strict_json(
        receipt_path,
        expected_schema=PROVENANCE_SCHEMA,
        require_private_owner=True,
    )
    receipt_cid = receipt.get("receipt_cid")
    if (
        type(receipt_cid) is not str
        or re.fullmatch(r"[a-z2-7]{32,256}", receipt_cid) is None
        or receipt.get("target_database") != str(paths["successor_database"])
        or (
            receipt.get("source_generation"),
            receipt.get("target_generation"),
            receipt.get("admission_mode"),
        )
        not in {
            (
                NATIVE_RESUME_SOURCE_GENERATION,
                SUCCESSOR_STORE_GENERATION,
                NATIVE_RESUME_ADMISSION_MODE,
            ),
        }
    ):
        raise SuccessorOperatorError(
            "raw successor provenance is not the exact live generation receipt"
        )
    return receipt


def _parse_quack_endpoint(endpoint: str) -> tuple[str, int]:
    match = re.fullmatch(r"quack:(?://)?(127\.0\.0\.1|localhost):(\d{1,5})", endpoint)
    if match is None or not 1 <= int(match.group(2)) <= 65535:
        raise SuccessorOperatorError("successor Quack endpoint must be fixed loopback")
    return match.group(1), int(match.group(2))


def _validate_successor_board(
    config_path: Path,
    root: Path = ROOT,
    *,
    config_bytes: bytes | None = None,
    admitted_live_validator_sha256: str = "",
) -> tuple[Any, Any, str, int]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
        preflight_configured_board,
    )

    board = load_configured_board(
        config_path,
        repo_root=root,
        config_bytes=config_bytes,
    )
    program = board.resolved_database_program()
    raw_program = board.payload.get("database_program")
    expected_store = SUCCESSOR_DATABASE_RELATIVE.as_posix()
    expected_registry = OWNER_STATE_RELATIVE.as_posix()
    provider = board.payload.get("provider")
    bootstrap = board.payload.get("bootstrap_writer_policy")
    projection = board.payload.get("ducklake_projection_program")
    if (
        board.max_lanes != 4
        or board.board_namespace
        != "logic-governed-compositional-verification-fabric-v1"
        or program.authority_mode != "quack"
        or program.task_source_kind != "duckdb"
        or program.failover_policy != "fail_closed"
        or program.endpoint_secret_handle != SECRET_HANDLE
        or program.store_id != expected_store
        or program.runtime_registry_path != expected_registry
        or program.store_generation != SUCCESSOR_STORE_GENERATION
        or program.schema_revision != "datasets-authoritative-operational-v1"
        or not isinstance(raw_program, Mapping)
        or raw_program.get("schema_profile") != "datasets-authoritative-operational"
        or board.runtime_paths.get("root") != SUCCESSOR_RUN_RELATIVE.as_posix()
        or not isinstance(provider, Mapping)
        or provider.get("primary_provider_id") != "grok_cli"
        or provider.get("primary_model_id") != "grok-4.6"
        or provider.get("fallback_provider_id") != "codex"
        or provider.get("fallback_model_id") != "gpt-5.6-terra"
        or provider.get("fallback_trigger") != "primary_quota_exhausted"
        or provider.get("fallback_reasoning_effort") != "high"
        or provider.get("max_concurrency") != 4
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("maximum_processes") != 1
        or bootstrap.get("quack_required") is not False
        or bootstrap.get("offline_single_writer_materialization_permitted") is not True
        or bootstrap.get("quack_required_after_publish") is not True
        or bootstrap.get("direct_multi_process_duckdb_permitted") is not False
        or not isinstance(projection, Mapping)
        or projection.get("root") != PROJECTION_ROOT_RELATIVE.as_posix()
        or projection.get("catalog_path")
        != (PROJECTION_ROOT_RELATIVE / "lake.ducklake").as_posix()
        or projection.get("data_path")
        != (PROJECTION_ROOT_RELATIVE / "lake-data").as_posix()
        or projection.get("authority") is not False
        or projection.get("scheduling_prerequisite") is not False
        or projection.get("completion_prerequisite") is not False
        or "fresh_generation_recovery" in board.payload
    ):
        raise SuccessorOperatorError(
            "scheduler config is not the exact four-lane successor"
        )
    host, port = _parse_quack_endpoint(program.quack_endpoint)
    preflight = preflight_configured_board(
        board,
        admitted_live_validator_sha256=admitted_live_validator_sha256,
    )
    if preflight.get("valid") is not True:
        raise SuccessorOperatorError(
            "configured-board preflight failed: "
            + ", ".join(preflight.get("errors") or ())
        )
    return board, program, host, port


def _status_payload(
    *,
    lifecycle: str,
    controller_birth: Mapping[str, Any],
    provenance_cid: str,
    owner_identity: Mapping[str, Any] | None = None,
    scheduler_birth: Mapping[str, Any] | None = None,
    scheduler_returncode: int | None = None,
    error: str = "",
    projection_root: Path | None = None,
) -> dict[str, Any]:
    observed_projection_root = (
        _paths()["projection_root"]
        if projection_root is None
        else Path(projection_root).resolve()
    )
    payload: dict[str, Any] = {
        "schema": CONTROLLER_STATUS_SCHEMA,
        "lifecycle": lifecycle,
        "updated_at": _utc_now(),
        "controller_birth": dict(controller_birth),
        "provenance_cid": provenance_cid,
        "owner_identity": dict(owner_identity or {}),
        "scheduler_birth": dict(scheduler_birth or {}),
        "scheduler_returncode": scheduler_returncode,
        "error": error,
        "ducklake_projection": {
            "path": str(observed_projection_root),
            "control_catalog_path": str(observed_projection_root / "control.duckdb"),
            "ducklake_catalog_path": str(observed_projection_root / "lake.ducklake"),
            "ducklake_data_path": str(observed_projection_root / "lake-data"),
            "authoritative": False,
            "read_by_scheduler": False,
            "scheduling_authority": False,
            "completion_authority": False,
            "live_quack_endpoint": False,
            "mode": "separate_stopped_checkpoint",
        },
    }
    payload["status_cid"] = _content_id(payload)
    return payload


def _write_status(path: Path, payload: Mapping[str, Any], *, token: str = "") -> None:
    encoded = _canonical_bytes(payload)
    if token and token.encode("ascii") in encoded:
        raise SuccessorOperatorError("Quack token would enter controller status")
    _atomic_json(path, payload, replace=True)


def _successor_state_databases(paths: Mapping[str, Path]) -> dict[str, Path]:
    control = paths["successor_database"]
    return {
        "control": control,
        "coordination": control.with_name("control.coordination.duckdb"),
        "execution": control.with_name("control.execution.duckdb"),
    }


def _stopped_state_database_digests(
    paths: Mapping[str, Path],
    *,
    _database_paths: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, str]]:
    databases = _successor_state_databases(paths)
    actual_databases = (
        databases if _database_paths is None else dict(_database_paths)
    )
    if set(actual_databases) != set(databases):
        raise SuccessorOperatorError(
            "stopped-state database path custody is incomplete"
        )
    observed: dict[str, dict[str, str]] = {}
    for name, database in databases.items():
        actual = actual_databases[name]
        if os.path.lexists(actual.with_name(actual.name + ".wal")):
            raise SuccessorOperatorError(
                f"stopped-state {name} database has a live WAL"
            )
        observed[name] = {
            "path": str(database),
            "sha256": _sha256_regular_file(
                actual,
                noun=f"stopped-state {name} database",
                require_private_owner=True,
            ),
        }
    return observed


def _stopped_recovery_io_paths(
    paths: Mapping[str, Path],
    lock_custody: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve every recovery surface through the held generation descriptor."""

    databases = _successor_state_databases(paths)
    if lock_custody is None:
        return {
            "provenance": paths["provenance"],
            "controller_status": paths["controller_status"],
            "stopped_state_continuity": paths["stopped_state_continuity"],
            "stopped_state_restart_admission": paths[
                "stopped_state_restart_admission"
            ],
            "owner_status": (
                paths["owner_state"] / "quack-state-server.status.json"
            ),
            "owner_marker": paths["successor_database"].with_name(
                ".control.duckdb.state-owner.json"
            ),
            "bootstrap": (
                paths["successor_database"].parent
                / "evidence"
                / "bootstrap"
                / "materialization.json"
            ),
            "databases": databases,
        }
    return {
        "provenance": _generation_bound_runtime_path(
            paths,
            lock_custody,
            paths["provenance"],
        ),
        "controller_status": _generation_bound_runtime_path(
            paths,
            lock_custody,
            paths["controller_status"],
        ),
        "stopped_state_continuity": _generation_bound_runtime_path(
            paths,
            lock_custody,
            paths["stopped_state_continuity"],
        ),
        "stopped_state_restart_admission": _generation_bound_runtime_path(
            paths,
            lock_custody,
            paths["stopped_state_restart_admission"],
        ),
        "owner_status": _generation_bound_runtime_path(
            paths,
            lock_custody,
            paths["owner_state"] / "quack-state-server.status.json",
        ),
        "owner_marker": _generation_bound_runtime_path(
            paths,
            lock_custody,
            paths["successor_database"].with_name(
                ".control.duckdb.state-owner.json"
            ),
        ),
        "bootstrap": _generation_bound_runtime_path(
            paths,
            lock_custody,
            paths["successor_database"].parent
            / "evidence"
            / "bootstrap"
            / "materialization.json",
        ),
        "databases": {
            name: _generation_bound_runtime_path(
                paths,
                lock_custody,
                database,
            )
            for name, database in databases.items()
        },
    }


def _stopped_receipt_io_view(
    paths: Mapping[str, Path],
    io_paths: Mapping[str, Any],
) -> dict[str, Path]:
    """Retain logical receipt values while pinning receipt/status file I/O."""

    view = dict(paths)
    view["controller_status"] = Path(io_paths["controller_status"])
    view["stopped_state_continuity"] = Path(
        io_paths["stopped_state_continuity"]
    )
    view["stopped_state_restart_admission"] = Path(
        io_paths["stopped_state_restart_admission"]
    )
    return view


def _stopped_recovery_generation_inventory(
    paths: Mapping[str, Path],
    lock_custody: Mapping[str, Any],
) -> tuple[tuple[Any, ...], ...]:
    """Snapshot top-level entries so read-only admission cannot leave sidecars."""

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    descriptor = int(lock_custody["generation_descriptor"])
    try:
        names = tuple(sorted(os.listdir(descriptor)))
        inventory = []
        for name in names:
            if not name or name in {".", ".."} or "/" in name:
                raise SuccessorOperatorError(
                    "stopped recovery generation inventory is malformed"
                )
            metadata = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            inventory.append(
                (
                    name,
                    stat.S_IFMT(metadata.st_mode),
                    stat.S_IMODE(metadata.st_mode),
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            )
    except OSError as exc:
        raise SuccessorOperatorError(
            "stopped recovery generation inventory is unavailable"
        ) from exc
    return tuple(inventory)


def _require_private_stopped_receipt(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        raise SuccessorOperatorError("stopped-state receipt is unavailable")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
    ):
        raise SuccessorOperatorError(
            "stopped-state continuity receipt custody is unsafe"
        )


def _sync_stopped_receipt_directory(path: Path) -> None:
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _rename_stopped_receipt_noreplace(
    source: Path,
    target: Path,
    *,
    noun: str,
) -> None:
    """Move authority custody atomically without a check/overwrite window."""

    if source.parent != target.parent:
        raise SuccessorOperatorError(
            "stopped-state receipt custody move escaped its directory"
        )
    directory = os.open(
        source.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        _rename_noreplace(
            directory,
            source.name,
            target.name,
            noun=noun,
        )
        os.fsync(directory)
    finally:
        os.close(directory)


def _claim_stopped_state_restart_admission(
    paths: Mapping[str, Path],
    *,
    expected_restart: bool,
    expected_receipt_cid: str = "",
    expected_controller_status_cid: str = "",
) -> bool:
    """Authenticate and consume exactly the previously admitted receipt."""

    source = paths["stopped_state_continuity"]
    target = paths["stopped_state_restart_admission"]
    source_present = os.path.lexists(source)
    if source_present is not expected_restart:
        raise SuccessorOperatorError(
            "stopped-state restart receipt presence differs from admission"
        )
    if not source_present:
        if expected_receipt_cid or expected_controller_status_cid:
            raise SuccessorOperatorError(
                "fresh launch unexpectedly carries stopped-state receipt pins"
            )
        if os.path.lexists(target):
            raise SuccessorOperatorError(
                "prior stopped-state restart admission remains unretired"
            )
        return False
    _require_private_stopped_receipt(source)
    if os.path.lexists(target):
        raise SuccessorOperatorError(
            "prior stopped-state restart admission remains unretired"
        )
    receipt = _strict_json(
        source,
        expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
        require_private_owner=True,
    )
    status = _strict_json(
        paths["controller_status"],
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    if (
        not expected_receipt_cid
        or not expected_controller_status_cid
        or receipt.get("receipt_cid") != expected_receipt_cid
        or status.get("status_cid") != expected_controller_status_cid
        or status.get("stopped_state_continuity_receipt_cid")
        != expected_receipt_cid
        or status.get("stopped_state_continuity_status_cid")
        != receipt.get("controller_status_cid")
    ):
        raise SuccessorOperatorError(
            "stopped-state restart receipt/status claim binding differs"
        )
    _rename_stopped_receipt_noreplace(
        source,
        target,
        noun="stopped-state restart admission",
    )
    claimed_receipt = _strict_json(
        target,
        expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
        require_private_owner=True,
    )
    claimed_status = _strict_json(
        paths["controller_status"],
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    if claimed_receipt != receipt or claimed_status != status:
        raise SuccessorOperatorError(
            "stopped-state restart claim changed during custody transfer"
        )
    return True


def _restore_or_retire_stopped_restart_admission(
    paths: Mapping[str, Path],
    *,
    retire_unbound_status_cid: str = "",
) -> str:
    """Recover an interrupted receipt claim or retire it after a clean stop."""

    admission = paths["stopped_state_restart_admission"]
    stopped = paths["stopped_state_continuity"]
    if not os.path.lexists(admission):
        return "absent"
    _require_private_stopped_receipt(admission)
    if os.path.lexists(stopped):
        raise SuccessorOperatorError(
            "stopped and consumed restart receipts coexist"
        )
    receipt = _strict_json(
        admission,
        expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
        require_private_owner=True,
    )
    status = _strict_json(
        paths["controller_status"],
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    linked = status.get("stopped_state_continuity_receipt_cid")
    linked_status = status.get("stopped_state_continuity_status_cid")
    if (
        status.get("lifecycle") == "stopped"
        and status.get("error") == ""
        and type(status.get("scheduler_returncode")) is int
        and status.get("scheduler_returncode") == 0
        and linked is None
        and linked_status is None
    ):
        if not retire_unbound_status_cid:
            return "pending_validated_retirement"
        if status.get("status_cid") != retire_unbound_status_cid:
            raise SuccessorOperatorError(
                "prevalidated clean-stop status changed before receipt retirement"
            )
        admission.unlink()
        _sync_stopped_receipt_directory(admission)
        return "retired_after_clean_stop"
    reconstructed_unbound = dict(status)
    reconstructed_unbound.pop("status_cid", None)
    reconstructed_unbound.pop("stopped_state_continuity_receipt_cid", None)
    reconstructed_unbound.pop("stopped_state_continuity_status_cid", None)
    reconstructed_unbound["status_cid"] = _content_id(
        reconstructed_unbound
    )
    if (
        linked != receipt.get("receipt_cid")
        or linked_status != receipt.get("controller_status_cid")
        or reconstructed_unbound["status_cid"] != linked_status
    ):
        raise SuccessorOperatorError(
            "consumed stopped-state restart admission/status binding differs"
        )
    _rename_stopped_receipt_noreplace(
        admission,
        stopped,
        noun="restored stopped-state continuity receipt",
    )
    return "restored_interrupted_claim"


def _stopped_owner_status_sha256(
    paths: Mapping[str, Path],
    *,
    controller_status: Mapping[str, Any],
    _status_path: Path | None = None,
    _marker_path: Path | None = None,
) -> str:
    """Authenticate the durable Quack-owner stopped projection."""

    status_path = (
        paths["owner_state"] / "quack-state-server.status.json"
        if _status_path is None
        else _status_path
    )
    owner_status = _strict_json(
        status_path,
        expected_schema=QUACK_STATE_SERVER_STATUS_SCHEMA,
        require_private_owner=True,
        verify_content_identity=False,
    )
    controller_identity = controller_status.get("owner_identity")
    stopped_identity = owner_status.get("identity")
    if not isinstance(controller_identity, Mapping) or not isinstance(
        stopped_identity, Mapping
    ):
        raise SuccessorOperatorError(
            "stopped-state owner identity projection is unavailable"
        )
    comparable_stopped = dict(stopped_identity)
    if comparable_stopped.pop("status", None) != "stopped":
        raise SuccessorOperatorError("stopped-state owner identity is not stopped")
    comparable_controller = dict(controller_identity)
    comparable_controller.pop("status", None)
    expected_marker = paths["successor_database"].with_name(
        ".control.duckdb.state-owner.json"
    )
    observed_marker = expected_marker if _marker_path is None else _marker_path
    if (
        owner_status.get("lifecycle") != "stopped"
        or owner_status.get("database_path") != str(paths["successor_database"])
        or owner_status.get("state_dir") != str(paths["owner_state"])
        or owner_status.get("store_id")
        != SUCCESSOR_DATABASE_RELATIVE.as_posix()
        or owner_status.get("secret_handle") != SECRET_HANDLE
        or owner_status.get("owner_marker_path") != str(expected_marker)
        or comparable_stopped != comparable_controller
        or os.path.lexists(observed_marker)
    ):
        raise SuccessorOperatorError(
            "stopped-state durable owner status binding differs"
        )
    return _sha256_regular_file(
        status_path,
        max_bytes=MAX_JSON_BYTES,
        noun="stopped-state durable owner status",
        require_private_owner=True,
    )


def _validate_stopped_controller_tree_births(
    stopped_status: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    """Parse one exact controller/owner/scheduler process tree."""

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )

    owner_identity = stopped_status.get("owner_identity")
    raw_controller = stopped_status.get("controller_birth")
    raw_scheduler = stopped_status.get("scheduler_birth")
    raw_owner = (
        owner_identity.get("process_birth")
        if isinstance(owner_identity, Mapping)
        else None
    )

    def exact_birth(
        raw: Any,
        *,
        noun: str,
    ) -> ProcessBirthIdentity:
        exact_fields = {"pid", "start_time_ticks", "boot_id", "parent_pid"}
        if (
            not isinstance(raw, Mapping)
            or set(raw) != exact_fields
            or type(raw.get("pid")) is not int
            or int(raw["pid"]) <= 1
            or type(raw.get("start_time_ticks")) is not int
            or int(raw["start_time_ticks"]) <= 0
            or type(raw.get("parent_pid")) is not int
            or int(raw["parent_pid"]) < 0
            or type(raw.get("boot_id")) is not str
            or not str(raw["boot_id"])
            or len(str(raw["boot_id"])) > 128
            or any(ord(character) < 0x21 for character in str(raw["boot_id"]))
        ):
            raise SuccessorOperatorError(
                f"stopped-state {noun} process birth binding is malformed"
            )
        try:
            parsed_birth = ProcessBirthIdentity.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise SuccessorOperatorError(
                f"stopped-state {noun} process birth binding is malformed"
            ) from exc
        if parsed_birth.to_dict() != dict(raw):
            raise SuccessorOperatorError(
                f"stopped-state {noun} process birth binding is malformed"
            )
        return parsed_birth

    controller = exact_birth(raw_controller, noun="controller")
    owner = exact_birth(raw_owner, noun="owner")
    recovery = stopped_status.get("abandoned_owner_recovery")
    if recovery is not None:
        if (
            not isinstance(recovery, Mapping)
            or set(recovery)
            != {
                "schema",
                "preflight_cid",
                "abandoned_owner_server_id",
                "scheduling_attempted",
            }
            or recovery.get("schema")
            != ABANDONED_OWNER_RECOVERY_STATUS_SCHEMA
            or type(recovery.get("preflight_cid")) is not str
            or re.fullmatch(
                r"b[a-z2-7]{20,200}",
                str(recovery.get("preflight_cid") or ""),
            )
            is None
            or type(recovery.get("abandoned_owner_server_id")) is not str
            or not str(recovery.get("abandoned_owner_server_id") or "")
            or recovery.get("scheduling_attempted") is not False
            or raw_owner != raw_controller
            or raw_scheduler != {}
        ):
            raise SuccessorOperatorError(
                "stopped-state abandoned owner recovery binding differs"
            )
        return controller, owner, owner

    protected_completion = stopped_status.get(
        "protected_qualification_completion"
    )
    if protected_completion is not None:
        if (
            not isinstance(protected_completion, Mapping)
            or set(protected_completion)
            != {
                "schema",
                "preflight_cid",
                "completion_receipt_cid",
                "completed",
                "scheduling_attempted",
            }
            or protected_completion.get("schema")
            != PROTECTED_QUALIFICATION_COMPLETION_STATUS_SCHEMA
            or re.fullmatch(
                r"b[a-z2-7]{20,200}",
                str(protected_completion.get("preflight_cid") or ""),
            )
            is None
            or type(protected_completion.get("completed")) is not bool
            or (
                protected_completion.get("completed") is True
                and re.fullmatch(
                    r"b[a-z2-7]{20,200}",
                    str(
                        protected_completion.get("completion_receipt_cid") or ""
                    ),
                )
                is None
            )
            or (
                protected_completion.get("completed") is False
                and protected_completion.get("completion_receipt_cid") != ""
            )
            or protected_completion.get("scheduling_attempted") is not False
            or raw_owner != raw_controller
            or raw_scheduler != {}
        ):
            raise SuccessorOperatorError(
                "stopped-state protected qualification binding differs"
            )
        return controller, owner, owner

    scheduler = exact_birth(raw_scheduler, noun="scheduler")
    if (
        raw_owner != raw_controller
        or raw_scheduler == raw_controller
        or scheduler.parent_pid != controller.pid
        or scheduler.boot_id != controller.boot_id
    ):
        raise SuccessorOperatorError(
            "stopped-state controller/owner/scheduler birth relation differs"
        )
    return controller, scheduler, owner


def _require_stopped_controller_tree_dead(
    stopped_status: Mapping[str, Any],
) -> None:
    """Require the exact controller, owner, and scheduler births to be dead."""

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        owner_liveness,
    )

    births = _validate_stopped_controller_tree_births(stopped_status)
    if any(owner_liveness(birth) is not OwnerLiveness.DEAD for birth in births):
        raise SuccessorOperatorError(
            "stopped-state controller tree is not exactly dead"
        )


def _owner_schema_fingerprint_matches_canonical_cid(
    owner_fingerprint: Any,
    canonical_fingerprint: Any,
) -> bool:
    """Bridge only the Quack owner's typed SHA-256/DAG-JSON CID forms."""

    if (
        type(owner_fingerprint) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", owner_fingerprint) is None
        or type(canonical_fingerprint) is not str
        or canonical_fingerprint != canonical_fingerprint.lower()
        or not canonical_fingerprint.startswith("b")
    ):
        return False
    payload = canonical_fingerprint[1:]
    if (
        not payload
        or re.fullmatch(r"[a-z2-7]+", payload) is None
        or len(payload) % 8 in {1, 3, 6}
    ):
        return False
    try:
        raw = base64.b32decode(
            payload.upper() + ("=" * ((-len(payload)) % 8)),
            casefold=False,
        )
    except (binascii.Error, ValueError):
        return False
    if (
        "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")
        != canonical_fingerprint
    ):
        return False
    # CIDv1 + dag-json(0x0129) + sha2-256 + 32-byte digest.
    prefix = b"\x01\xa9\x02\x12\x20"
    return (
        len(raw) == len(prefix) + 32
        and raw.startswith(prefix)
        and owner_fingerprint == f"sha256:{raw[len(prefix):].hex()}"
    )


def _validate_unbound_stopped_controller_status(
    stopped_status: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> None:
    """Validate the exact durable status shape allowed to mint continuity."""

    owner_identity = stopped_status.get("owner_identity")
    controller_birth = stopped_status.get("controller_birth")
    scheduler_birth = stopped_status.get("scheduler_birth")
    base_fields = {
        "schema",
        "lifecycle",
        "updated_at",
        "controller_birth",
        "provenance_cid",
        "owner_identity",
        "scheduler_birth",
        "scheduler_returncode",
        "error",
        "ducklake_projection",
        "status_cid",
    }
    expected_fields = set(base_fields)
    for optional_field in (
        "stopped_recovery_anchors",
        "protected_qualification_completion",
    ):
        if optional_field in stopped_status:
            expected_fields.add(optional_field)
    updated_at = stopped_status.get("updated_at")
    projection = stopped_status.get("ducklake_projection")
    projection_fields = {
        "path",
        "control_catalog_path",
        "ducklake_catalog_path",
        "ducklake_data_path",
        "authoritative",
        "read_by_scheduler",
        "scheduling_authority",
        "completion_authority",
        "live_quack_endpoint",
        "mode",
    }
    if (
        set(stopped_status) != expected_fields
        or stopped_status.get("schema") != CONTROLLER_STATUS_SCHEMA
        or stopped_status.get("lifecycle") != "stopped"
        or stopped_status.get("error") != ""
        or type(stopped_status.get("scheduler_returncode")) is not int
        or stopped_status.get("scheduler_returncode") != 0
        or type(updated_at) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            updated_at,
        )
        is None
        or stopped_status.get("provenance_cid") != provenance.get("receipt_cid")
        or "stopped_state_continuity_receipt_cid" in stopped_status
        or "stopped_state_continuity_status_cid" in stopped_status
        or not isinstance(owner_identity, Mapping)
        or not isinstance(controller_birth, Mapping)
        or not isinstance(scheduler_birth, Mapping)
        or owner_identity.get("process_birth") != controller_birth
        or owner_identity.get("database_uuid") != provenance.get("database_uuid")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            provenance.get("schema_fingerprint"),
        )
        or owner_identity.get("store_id")
        != SUCCESSOR_DATABASE_RELATIVE.as_posix()
        or owner_identity.get("secret_handle") != SECRET_HANDLE
        or not isinstance(projection, Mapping)
        or set(projection) != projection_fields
        or projection.get("mode") != "separate_stopped_checkpoint"
        or any(
            projection.get(field) is not False
            for field in (
                "authoritative",
                "read_by_scheduler",
                "scheduling_authority",
                "completion_authority",
                "live_quack_endpoint",
            )
        )
    ):
        raise SuccessorOperatorError(
            "unbound stopped-state controller status differs"
        )
    _require_stopped_controller_tree_dead(stopped_status)


def _failed_start_reason_from_exception(exc: BaseException | None) -> str:
    """Classify only the three pre-ready failures approved for continuity."""

    if not isinstance(exc, SuccessorOperatorError):
        return ""
    message = str(exc)
    if message == "scheduler exited before all lane daemons attached":
        return FAILED_START_REASON_SCHEDULER_EXITED
    if message == "lane state-owner bootstrap readiness timed out":
        return FAILED_START_REASON_BOOTSTRAP_TIMEOUT
    if message == "controller stop requested before all lane daemons attached":
        return FAILED_START_REASON_OPERATOR_STOP
    prefix = "lane state-owner bootstrap failed closed: "
    detail = message[len(prefix) :] if message.startswith(prefix) else ""
    if (
        detail
        and len(detail) <= 512
        and all(character.isprintable() for character in detail)
        and "\x00" not in detail
    ):
        return FAILED_START_REASON_BOOTSTRAP_FAILED
    return ""


def _validate_unbound_failed_start_controller_status(
    failed_status: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    require_dead: bool,
) -> None:
    """Validate the sole unbound pre-ready failure status shape."""

    owner_identity = failed_status.get("owner_identity")
    controller_birth = failed_status.get("controller_birth")
    scheduler_birth = failed_status.get("scheduler_birth")
    base_fields = {
        "schema",
        "lifecycle",
        "updated_at",
        "controller_birth",
        "provenance_cid",
        "owner_identity",
        "scheduler_birth",
        "scheduler_returncode",
        "error",
        "ducklake_projection",
        "status_cid",
    }
    expected_fields = set(base_fields)
    for optional_field in (
        "failed_start_recovery_anchors",
        "abandoned_owner_recovery",
    ):
        if optional_field in failed_status:
            expected_fields.add(optional_field)
    updated_at = failed_status.get("updated_at")
    returncode = failed_status.get("scheduler_returncode")
    projection = failed_status.get("ducklake_projection")
    projection_fields = {
        "path",
        "control_catalog_path",
        "ducklake_catalog_path",
        "ducklake_data_path",
        "authoritative",
        "read_by_scheduler",
        "scheduling_authority",
        "completion_authority",
        "live_quack_endpoint",
        "mode",
    }
    if (
        set(failed_status) != expected_fields
        or failed_status.get("schema") != CONTROLLER_STATUS_SCHEMA
        or failed_status.get("lifecycle") != "stopped"
        or failed_status.get("error") != FAILED_START_STATUS_ERROR
        or type(returncode) is not int
        or not (-255 <= returncode <= 255)
        or type(updated_at) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            updated_at,
        )
        is None
        or failed_status.get("provenance_cid")
        != provenance.get("receipt_cid")
        or "stopped_state_continuity_receipt_cid" in failed_status
        or "stopped_state_continuity_status_cid" in failed_status
        or not isinstance(owner_identity, Mapping)
        or not isinstance(controller_birth, Mapping)
        or not isinstance(scheduler_birth, Mapping)
        or owner_identity.get("process_birth") != controller_birth
        or owner_identity.get("database_uuid") != provenance.get("database_uuid")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            provenance.get("schema_fingerprint"),
        )
        or owner_identity.get("store_id")
        != SUCCESSOR_DATABASE_RELATIVE.as_posix()
        or owner_identity.get("secret_handle") != SECRET_HANDLE
        or not isinstance(projection, Mapping)
        or set(projection) != projection_fields
        or projection.get("mode") != "separate_stopped_checkpoint"
        or any(
            projection.get(field) is not False
            for field in (
                "authoritative",
                "read_by_scheduler",
                "scheduling_authority",
                "completion_authority",
                "live_quack_endpoint",
            )
        )
    ):
        raise SuccessorOperatorError(
            "unbound failed-start controller status differs"
        )
    _validate_stopped_controller_tree_births(failed_status)
    if require_dead:
        _require_stopped_controller_tree_dead(failed_status)


def _failed_start_superseded_archive_path(
    paths: Mapping[str, Path],
    receipt: Mapping[str, Any],
) -> Path:
    receipt_cid = receipt.get("receipt_cid")
    if (
        type(receipt_cid) is not str
        or re.fullmatch(r"b[a-z2-7]{20,200}", receipt_cid) is None
    ):
        raise SuccessorOperatorError(
            "superseded restart receipt identity is malformed"
        )
    return paths["stopped_state_restart_admission"].with_name(
        "superseded-stopped-state-restart-admission."
        f"{receipt_cid}.json"
    )


def _validate_failed_start_superseded_admission_snapshot(
    paths: Mapping[str, Path],
    snapshot: Any,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise SuccessorOperatorError(
            "failed-start superseded admission snapshot is malformed"
        )
    receipt = snapshot.get("receipt")
    if not isinstance(receipt, Mapping):
        raise SuccessorOperatorError(
            "failed-start superseded admission receipt is malformed"
        )
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=provenance,
    )
    archive_path = _failed_start_superseded_archive_path(paths, receipt)
    expected_sha256 = "sha256:" + hashlib.sha256(
        _canonical_bytes(dict(receipt)) + b"\n"
    ).hexdigest()
    if (
        set(snapshot) != {"receipt", "file_sha256", "archive_path"}
        or snapshot.get("file_sha256") != expected_sha256
        or snapshot.get("archive_path") != str(archive_path)
    ):
        raise SuccessorOperatorError(
            "failed-start superseded admission snapshot differs"
        )
    return {
        "receipt": dict(receipt),
        "file_sha256": expected_sha256,
        "archive_path": str(archive_path),
    }


def _capture_failed_start_superseded_admission(
    paths: Mapping[str, Path],
    *,
    io_paths: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Capture consumed prior authority only as typed superseded history."""

    canonical = Path(io_paths["stopped_state_continuity"])
    admission = Path(io_paths["stopped_state_restart_admission"])
    if os.path.lexists(canonical):
        raise SuccessorOperatorError(
            "failed-start publication found unconsumed restart authority"
        )
    if not os.path.lexists(admission):
        return None
    receipt = _strict_json(
        admission,
        expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
        require_private_owner=True,
    )
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=provenance,
    )
    archive = _failed_start_superseded_archive_path(paths, receipt)
    bound_archive = admission.with_name(archive.name)
    if os.path.lexists(bound_archive):
        raise SuccessorOperatorError(
            "failed-start superseded receipt archive already exists"
        )
    snapshot = {
        "receipt": receipt,
        "file_sha256": _sha256_regular_file(
            admission,
            max_bytes=MAX_JSON_BYTES,
            noun="superseded stopped-state restart admission",
            require_private_owner=True,
        ),
        "archive_path": str(archive),
    }
    return _validate_failed_start_superseded_admission_snapshot(
        paths,
        snapshot,
        provenance=provenance,
    )


def _archive_failed_start_superseded_admission(
    paths: Mapping[str, Path],
    *,
    io_paths: Mapping[str, Any],
    provenance: Mapping[str, Any],
    expected_snapshot: Mapping[str, Any] | None,
) -> str:
    """Move superseded custody only after its replacement status is durable."""

    snapshot = _validate_failed_start_superseded_admission_snapshot(
        paths,
        expected_snapshot,
        provenance=provenance,
    )
    state = _observe_failed_start_superseded_admission(
        paths,
        io_paths=io_paths,
        provenance=provenance,
        expected_snapshot=snapshot,
    )
    if state == "absent":
        return state
    if state == "archived":
        return "already_archived"
    assert snapshot is not None
    admission = Path(io_paths["stopped_state_restart_admission"])
    receipt = snapshot["receipt"]
    assert isinstance(receipt, Mapping)
    logical_archive = _failed_start_superseded_archive_path(paths, receipt)
    archive = admission.with_name(logical_archive.name)
    _rename_stopped_receipt_noreplace(
        admission,
        archive,
        noun="superseded stopped-state restart admission archive",
    )
    if (
        _strict_json(
            archive,
            expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
            require_private_owner=True,
        )
        != receipt
        or _sha256_regular_file(
            archive,
            max_bytes=MAX_JSON_BYTES,
            noun="archived superseded restart admission",
            require_private_owner=True,
        )
        != snapshot["file_sha256"]
    ):
        raise SuccessorOperatorError(
            "superseded restart admission archive changed during publication"
        )
    return "archived"


def _observe_failed_start_superseded_admission(
    paths: Mapping[str, Path],
    *,
    io_paths: Mapping[str, Any],
    provenance: Mapping[str, Any],
    expected_snapshot: Mapping[str, Any] | None,
) -> str:
    """Read one prior receipt from consumed or archived custody without mutation."""

    snapshot = _validate_failed_start_superseded_admission_snapshot(
        paths,
        expected_snapshot,
        provenance=provenance,
    )
    admission = Path(io_paths["stopped_state_restart_admission"])
    if snapshot is None:
        if os.path.lexists(admission):
            raise SuccessorOperatorError(
                "failed-start found an unanchored prior restart admission"
            )
        return "absent"
    receipt = snapshot["receipt"]
    assert isinstance(receipt, Mapping)
    logical_archive = _failed_start_superseded_archive_path(paths, receipt)
    archive = admission.with_name(logical_archive.name)
    source_present = os.path.lexists(admission)
    archive_present = os.path.lexists(archive)
    if source_present and archive_present:
        raise SuccessorOperatorError(
            "superseded restart admission and archive coexist"
        )
    if not source_present and not archive_present:
        raise SuccessorOperatorError(
            "anchored superseded restart admission custody is unavailable"
        )
    observed_path = admission if source_present else archive
    if (
        _strict_json(
            observed_path,
            expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
            require_private_owner=True,
        )
        != receipt
        or _sha256_regular_file(
            observed_path,
            max_bytes=MAX_JSON_BYTES,
            noun="anchored superseded restart admission",
            require_private_owner=True,
        )
        != snapshot["file_sha256"]
    ):
        raise SuccessorOperatorError(
            "anchored superseded restart admission changed"
        )
    return "consumed" if source_present else "archived"


def _capture_stopped_recovery_anchors(
    paths: Mapping[str, Path],
    *,
    root: Path,
    stopped_status: Mapping[str, Any],
    provenance: Mapping[str, Any],
    io_paths: Mapping[str, Any],
    lock_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture exact stopped bytes before the unbound status is published."""

    generation_inventory = (
        _stopped_recovery_generation_inventory(paths, lock_custody)
        if lock_custody is not None
        else None
    )
    durable_provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    if durable_provenance != dict(provenance):
        raise SuccessorOperatorError(
            "clean-stop recovery anchor provenance changed"
        )
    owner_identity = stopped_status.get("owner_identity")
    if (
        stopped_status.get("lifecycle") != "stopped"
        or stopped_status.get("error") != ""
        or type(stopped_status.get("scheduler_returncode")) is not int
        or stopped_status.get("scheduler_returncode") != 0
        or stopped_status.get("provenance_cid") != provenance.get("receipt_cid")
        or not isinstance(owner_identity, Mapping)
        or owner_identity.get("process_birth")
        != stopped_status.get("controller_birth")
        or owner_identity.get("database_uuid") != provenance.get("database_uuid")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            provenance.get("schema_fingerprint"),
        )
        or owner_identity.get("store_id")
        != SUCCESSOR_DATABASE_RELATIVE.as_posix()
        or owner_identity.get("secret_handle") != SECRET_HANDLE
    ):
        raise SuccessorOperatorError(
            "clean-stop recovery anchor status binding differs"
        )
    final_continuity = _observe_candidate_runtime_continuity(
        root,
        require_resolved_remote=False,
    )
    _validate_stopped_projection_native_provenance(
        paths,
        root=root,
        receipt=provenance,
        final_continuity=final_continuity,
        _bootstrap_path=Path(io_paths["bootstrap"]),
    )
    databases = _stopped_state_database_digests(
        paths,
        _database_paths=io_paths["databases"],
    )
    owner_status_sha256 = _stopped_owner_status_sha256(
        paths,
        controller_status=stopped_status,
        _status_path=Path(io_paths["owner_status"]),
        _marker_path=Path(io_paths["owner_marker"]),
    )
    verification = _verify_profile(
        Path(io_paths["databases"]["control"]),
        read_only=True,
    )
    identity = _database_identity(Path(io_paths["databases"]["control"]))
    if (
        verification.get("schema_fingerprint")
        != provenance.get("schema_fingerprint")
        or verification.get("catalog_fingerprint")
        != provenance.get("catalog_fingerprint")
        or identity.get("database_uuid") != provenance.get("database_uuid")
        or identity.get("schema_fingerprint")
        != provenance.get("schema_fingerprint")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            identity.get("schema_fingerprint"),
        )
    ):
        raise SuccessorOperatorError(
            "clean-stop recovery anchor database identity differs"
        )
    if (
        lock_custody is not None
        and _stopped_recovery_generation_inventory(paths, lock_custody)
        != generation_inventory
    ):
        raise SuccessorOperatorError(
            "clean-stop recovery anchor admission changed generation inventory"
        )
    anchors: dict[str, Any] = {
        "schema": STOPPED_RECOVERY_ANCHORS_SCHEMA,
        "captured_at": _utc_now(),
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "source_provenance_cid": provenance["receipt_cid"],
        "final_source_continuity": final_continuity,
        "databases": databases,
        "owner_status_sha256": owner_status_sha256,
    }
    anchors["anchors_cid"] = _content_id(anchors)
    return anchors


def _bind_stopped_recovery_anchors_status(
    stopped_status: Mapping[str, Any],
    anchors: Mapping[str, Any],
) -> dict[str, Any]:
    bound = dict(stopped_status)
    bound.pop("status_cid", None)
    bound["stopped_recovery_anchors"] = dict(anchors)
    bound["status_cid"] = _content_id(bound)
    return bound


def _failed_start_recovery_reviewed_pins(
    *,
    failed_status: Mapping[str, Any],
    provenance: Mapping[str, Any],
    failed_start_reason: str,
    final_continuity: Mapping[str, Any],
    databases: Mapping[str, Any],
    owner_status_sha256: str,
    bootstrap_sha256: str,
    superseded_restart_admission: Mapping[str, Any] | None,
    recovery_authorization_mode: str,
    owner_stop_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "controller_status_cid": failed_status["status_cid"],
        "controller_status": dict(failed_status),
        "source_provenance_cid": provenance["receipt_cid"],
        "failed_start_reason": failed_start_reason,
        "source_continuity": dict(final_continuity),
        "databases": dict(databases),
        "owner_status_sha256": owner_status_sha256,
        "bootstrap_sha256": bootstrap_sha256,
        "superseded_restart_admission": (
            dict(superseded_restart_admission)
            if superseded_restart_admission is not None
            else None
        ),
        "recovery_authorization_mode": recovery_authorization_mode,
        "owner_stop_receipt": (
            dict(owner_stop_receipt)
            if owner_stop_receipt is not None
            else None
        ),
    }


def _failed_start_preflight_cid(reviewed_pins: Mapping[str, Any]) -> str:
    return _content_id(
        {
            "schema": FAILED_START_RECOVERY_PREFLIGHT_SCHEMA,
            "operation": FAILED_START_RECOVERY_OPERATION,
            "reviewed_pins": dict(reviewed_pins),
        }
    )


def _validate_failed_start_owner_stop_receipt(
    failed_status: Mapping[str, Any],
    receipt: Any,
) -> dict[str, Any]:
    owner_identity = failed_status.get("owner_identity")
    owner_server_id = (
        str(owner_identity.get("server_id") or "")
        if isinstance(owner_identity, Mapping)
        else ""
    )
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != {"stopped", "server_id", "at"}
        or receipt.get("stopped") is not True
        or receipt.get("server_id") != owner_server_id
        or type(receipt.get("at")) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            receipt["at"],
        )
        is None
    ):
        raise SuccessorOperatorError(
            "failed-start owner stop receipt differs"
        )
    return dict(receipt)


def _capture_failed_start_recovery_anchors(
    paths: Mapping[str, Path],
    *,
    root: Path,
    failed_status: Mapping[str, Any],
    provenance: Mapping[str, Any],
    failed_start_reason: str,
    owner_stop: Mapping[str, Any],
    io_paths: Mapping[str, Any],
    lock_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal current post-stop bytes in the trusted controller finally path."""

    if failed_start_reason not in FAILED_START_TRUSTED_RECOVERY_REASONS:
        raise SuccessorOperatorError(
            "failed-start recovery reason is not allowlisted"
        )
    _validate_unbound_failed_start_controller_status(
        failed_status,
        provenance=provenance,
        require_dead=False,
    )
    owner_stop_receipt = _validate_failed_start_owner_stop_receipt(
        failed_status,
        owner_stop,
    )
    generation_inventory = (
        _stopped_recovery_generation_inventory(paths, lock_custody)
        if lock_custody is not None
        else None
    )
    durable_provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    if durable_provenance != dict(provenance):
        raise SuccessorOperatorError(
            "failed-start recovery anchor provenance changed"
        )
    final_continuity = _observe_candidate_runtime_continuity(
        root,
        require_resolved_remote=False,
    )
    bootstrap_path = Path(io_paths["bootstrap"])
    _validate_stopped_projection_native_provenance(
        paths,
        root=root,
        receipt=provenance,
        final_continuity=final_continuity,
        _bootstrap_path=bootstrap_path,
    )
    databases = _stopped_state_database_digests(
        paths,
        _database_paths=io_paths["databases"],
    )
    owner_status_sha256 = _stopped_owner_status_sha256(
        paths,
        controller_status=failed_status,
        _status_path=Path(io_paths["owner_status"]),
        _marker_path=Path(io_paths["owner_marker"]),
    )
    bootstrap_sha256 = _sha256_regular_file(
        bootstrap_path,
        max_bytes=MAX_JSON_BYTES,
        noun="failed-start bootstrap receipt",
        require_private_owner=True,
    )
    verification = _verify_profile(
        Path(io_paths["databases"]["control"]),
        read_only=True,
    )
    identity = _database_identity(Path(io_paths["databases"]["control"]))
    owner_identity = failed_status["owner_identity"]
    assert isinstance(owner_identity, Mapping)
    if (
        verification.get("schema_fingerprint")
        != provenance.get("schema_fingerprint")
        or verification.get("catalog_fingerprint")
        != provenance.get("catalog_fingerprint")
        or identity.get("database_uuid") != provenance.get("database_uuid")
        or identity.get("schema_fingerprint")
        != provenance.get("schema_fingerprint")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            identity.get("schema_fingerprint"),
        )
    ):
        raise SuccessorOperatorError(
            "failed-start recovery anchor database identity differs"
        )
    superseded = _capture_failed_start_superseded_admission(
        paths,
        io_paths=io_paths,
        provenance=provenance,
    )
    reviewed_pins = _failed_start_recovery_reviewed_pins(
        failed_status=failed_status,
        provenance=provenance,
        failed_start_reason=failed_start_reason,
        final_continuity=final_continuity,
        databases=databases,
        owner_status_sha256=owner_status_sha256,
        bootstrap_sha256=bootstrap_sha256,
        superseded_restart_admission=superseded,
        recovery_authorization_mode=FAILED_START_TRUSTED_FINALLY_MODE,
        owner_stop_receipt=owner_stop_receipt,
    )
    anchors: dict[str, Any] = {
        "schema": FAILED_START_RECOVERY_ANCHORS_SCHEMA,
        "captured_at": _utc_now(),
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "source_provenance_cid": provenance["receipt_cid"],
        "source_controller_status_cid": failed_status["status_cid"],
        "failed_start_reason": failed_start_reason,
        "final_source_continuity": final_continuity,
        "databases": databases,
        "owner_status_sha256": owner_status_sha256,
        "bootstrap_sha256": bootstrap_sha256,
        "superseded_restart_admission": superseded,
        "recovery_authorization_mode": FAILED_START_TRUSTED_FINALLY_MODE,
        "owner_stop_receipt": owner_stop_receipt,
        "recovery_preflight_cid": _failed_start_preflight_cid(reviewed_pins),
    }
    anchors["anchors_cid"] = _content_id(anchors)
    if (
        lock_custody is not None
        and _stopped_recovery_generation_inventory(paths, lock_custody)
        != generation_inventory
    ):
        raise SuccessorOperatorError(
            "failed-start recovery anchor changed generation inventory"
        )
    if (
        _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        != durable_provenance
        or _stopped_state_database_digests(
            paths,
            _database_paths=io_paths["databases"],
        )
        != databases
        or _stopped_owner_status_sha256(
            paths,
            controller_status=failed_status,
            _status_path=Path(io_paths["owner_status"]),
            _marker_path=Path(io_paths["owner_marker"]),
        )
        != owner_status_sha256
        or _sha256_regular_file(
            bootstrap_path,
            max_bytes=MAX_JSON_BYTES,
            noun="failed-start bootstrap receipt",
            require_private_owner=True,
        )
        != bootstrap_sha256
        or _capture_failed_start_superseded_admission(
            paths,
            io_paths=io_paths,
            provenance=provenance,
        )
        != superseded
        or _observe_candidate_runtime_continuity(
            root,
            require_resolved_remote=False,
        )
        != final_continuity
    ):
        raise SuccessorOperatorError(
            "failed-start recovery evidence changed during anchor capture"
        )
    return anchors


def _bind_failed_start_recovery_anchors_status(
    failed_status: Mapping[str, Any],
    anchors: Mapping[str, Any],
) -> dict[str, Any]:
    bound = dict(failed_status)
    bound.pop("status_cid", None)
    bound["failed_start_recovery_anchors"] = dict(anchors)
    bound["status_cid"] = _content_id(bound)
    return bound


def _validate_stopped_continuity_receipt_shape(
    paths: Mapping[str, Path],
    receipt: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    controller_status_cid: str | None = None,
) -> None:
    """Validate every immutable receipt field before any status mutation."""

    expected_fields = {
        "schema",
        "issued_at",
        "admission_mode",
        "target_generation",
        "source_provenance_cid",
        "controller_status_cid",
        "stop_evidence",
        "owner_status_sha256",
        "final_source_continuity",
        "databases",
        "controller_lock_held_at_issue",
        "live_wal_absent",
        "requires_stopped_checkpoint",
        "projection_only",
        "same_generation_restart_only",
        "restart_authority",
        "authoritative",
        "scheduling_authority",
        "completion_authority",
        "read_by_scheduler",
        "quack_endpoint_served",
        "production_authorized",
        "receipt_cid",
    }
    issued_at = receipt.get("issued_at")
    admission_mode = receipt.get("admission_mode")
    expected_checkpoint = (
        admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
    )
    databases = receipt.get("databases")
    logical_databases = _successor_state_databases(paths)
    database_shape_valid = isinstance(databases, Mapping) and set(
        databases
    ) == set(logical_databases)
    if database_shape_valid:
        for name, logical_path in logical_databases.items():
            item = databases[name]
            if (
                not isinstance(item, Mapping)
                or set(item) != {"path", "sha256"}
                or item.get("path") != str(logical_path)
                or type(item.get("sha256")) is not str
                or re.fullmatch(r"sha256:[0-9a-f]{64}", item["sha256"])
                is None
            ):
                database_shape_valid = False
                break
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != STOPPED_STATE_CONTINUITY_SCHEMA
        or admission_mode
        not in {
            STOPPED_STATE_CONTINUITY_ADMISSION_MODE,
            FAILED_START_CONTINUITY_ADMISSION_MODE,
        }
        or receipt.get("target_generation") != SUCCESSOR_STORE_GENERATION
        or receipt.get("source_provenance_cid")
        != provenance.get("receipt_cid")
        or (
            controller_status_cid is not None
            and receipt.get("controller_status_cid") != controller_status_cid
        )
        or type(receipt.get("controller_status_cid")) is not str
        or not str(receipt.get("controller_status_cid") or "")
        or type(issued_at) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            issued_at,
        )
        is None
        or not isinstance(receipt.get("stop_evidence"), Mapping)
        or type(receipt.get("owner_status_sha256")) is not str
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            receipt["owner_status_sha256"],
        )
        is None
        or not isinstance(receipt.get("final_source_continuity"), Mapping)
        or not database_shape_valid
        or receipt.get("controller_lock_held_at_issue") is not True
        or receipt.get("live_wal_absent") is not True
        or receipt.get("requires_stopped_checkpoint")
        is not expected_checkpoint
        or receipt.get("projection_only") is not False
        or receipt.get("same_generation_restart_only") is not True
        or receipt.get("restart_authority") is not True
        or any(
            receipt.get(field) is not False
            for field in (
                "authoritative",
                "scheduling_authority",
                "completion_authority",
                "read_by_scheduler",
                "quack_endpoint_served",
                "production_authorized",
            )
        )
        or receipt.get("receipt_cid")
        != _content_id(
            {
                name: value
                for name, value in receipt.items()
                if name != "receipt_cid"
            }
        )
    ):
        raise SuccessorOperatorError(
            "stopped-state continuity receipt shape differs"
        )


def _validate_stopped_recovery_anchors(
    stopped_status: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    final_continuity: Mapping[str, Any],
    databases: Mapping[str, Any],
    owner_status_sha256: str,
) -> dict[str, Any] | None:
    """Replay future stopped-time anchors exactly; identify legacy statuses."""

    anchors = stopped_status.get("stopped_recovery_anchors")
    if anchors is None:
        return None
    if not isinstance(anchors, Mapping):
        raise SuccessorOperatorError("stopped recovery anchors are malformed")
    expected_fields = {
        "schema",
        "captured_at",
        "target_generation",
        "source_provenance_cid",
        "final_source_continuity",
        "databases",
        "owner_status_sha256",
        "anchors_cid",
    }
    captured_at = anchors.get("captured_at")
    if (
        set(anchors) != expected_fields
        or anchors.get("schema") != STOPPED_RECOVERY_ANCHORS_SCHEMA
        or type(captured_at) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            captured_at,
        )
        is None
        or anchors.get("target_generation") != SUCCESSOR_STORE_GENERATION
        or anchors.get("source_provenance_cid")
        != provenance.get("receipt_cid")
        or anchors.get("final_source_continuity") != dict(final_continuity)
        or anchors.get("databases") != dict(databases)
        or anchors.get("owner_status_sha256") != owner_status_sha256
        or anchors.get("anchors_cid")
        != _content_id(
            {
                name: value
                for name, value in anchors.items()
                if name != "anchors_cid"
            }
        )
    ):
        raise SuccessorOperatorError(
            "durable stopped recovery anchors differ from current bytes"
        )
    return dict(anchors)


def _validate_failed_start_recovery_anchors(
    paths: Mapping[str, Path],
    failed_status: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    final_continuity: Mapping[str, Any],
    databases: Mapping[str, Any],
    owner_status_sha256: str,
    bootstrap_sha256: str,
    require_dead: bool,
) -> dict[str, Any] | None:
    """Replay durable failed-start anchors against current stopped bytes."""

    raw_anchors = failed_status.get("failed_start_recovery_anchors")
    if raw_anchors is None:
        return None
    if not isinstance(raw_anchors, Mapping):
        raise SuccessorOperatorError(
            "failed-start recovery anchors are malformed"
        )
    source_status = dict(failed_status)
    source_status.pop("status_cid", None)
    source_status.pop("failed_start_recovery_anchors", None)
    source_status["status_cid"] = _content_id(source_status)
    _validate_unbound_failed_start_controller_status(
        source_status,
        provenance=provenance,
        require_dead=require_dead,
    )
    expected_fields = {
        "schema",
        "captured_at",
        "target_generation",
        "source_provenance_cid",
        "source_controller_status_cid",
        "failed_start_reason",
        "final_source_continuity",
        "databases",
        "owner_status_sha256",
        "bootstrap_sha256",
        "superseded_restart_admission",
        "recovery_authorization_mode",
        "owner_stop_receipt",
        "recovery_preflight_cid",
        "anchors_cid",
    }
    captured_at = raw_anchors.get("captured_at")
    reason = raw_anchors.get("failed_start_reason")
    authorization_mode = raw_anchors.get("recovery_authorization_mode")
    superseded = _validate_failed_start_superseded_admission_snapshot(
        paths,
        snapshot=raw_anchors.get("superseded_restart_admission"),
        provenance=provenance,
    )
    owner_stop_raw = raw_anchors.get("owner_stop_receipt")
    if authorization_mode == FAILED_START_TRUSTED_FINALLY_MODE:
        owner_stop = _validate_failed_start_owner_stop_receipt(
            source_status,
            owner_stop_raw,
        )
    elif authorization_mode == FAILED_START_REVIEWED_LEGACY_MODE:
        if owner_stop_raw is not None:
            raise SuccessorOperatorError(
                "reviewed failed-start recovery fabricated an owner receipt"
            )
        owner_stop = None
    else:
        raise SuccessorOperatorError(
            "failed-start recovery authorization mode differs"
        )
    reviewed_pins = _failed_start_recovery_reviewed_pins(
        failed_status=source_status,
        provenance=provenance,
        failed_start_reason=str(reason or ""),
        final_continuity=final_continuity,
        databases=databases,
        owner_status_sha256=owner_status_sha256,
        bootstrap_sha256=bootstrap_sha256,
        superseded_restart_admission=superseded,
        recovery_authorization_mode=authorization_mode,
        owner_stop_receipt=owner_stop,
    )
    if (
        set(raw_anchors) != expected_fields
        or raw_anchors.get("schema") != FAILED_START_RECOVERY_ANCHORS_SCHEMA
        or type(captured_at) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            captured_at,
        )
        is None
        or raw_anchors.get("target_generation")
        != SUCCESSOR_STORE_GENERATION
        or raw_anchors.get("source_provenance_cid")
        != provenance.get("receipt_cid")
        or raw_anchors.get("source_controller_status_cid")
        != source_status.get("status_cid")
        or reason not in FAILED_START_RECOVERY_REASONS
        or raw_anchors.get("final_source_continuity")
        != dict(final_continuity)
        or raw_anchors.get("databases") != dict(databases)
        or raw_anchors.get("owner_status_sha256") != owner_status_sha256
        or raw_anchors.get("bootstrap_sha256") != bootstrap_sha256
        or raw_anchors.get("superseded_restart_admission") != superseded
        or raw_anchors.get("owner_stop_receipt") != owner_stop
        or raw_anchors.get("recovery_preflight_cid")
        != _failed_start_preflight_cid(reviewed_pins)
        or raw_anchors.get("anchors_cid")
        != _content_id(
            {
                name: value
                for name, value in raw_anchors.items()
                if name != "anchors_cid"
            }
        )
    ):
        raise SuccessorOperatorError(
            "durable failed-start recovery anchors differ from current bytes"
        )
    return dict(raw_anchors)


def _stopped_recovery_preflight_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    sealed_source_continuity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute deterministic exact pins without modifying stopped evidence."""

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    generation_inventory = _stopped_recovery_generation_inventory(
        paths,
        lock_custody,
    )
    observed_provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    if provenance is not None and observed_provenance != dict(provenance):
        raise SuccessorOperatorError(
            "stopped recovery provenance changed before preflight"
        )
    durable_status = _strict_json(
        Path(io_paths["controller_status"]),
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    _validate_unbound_stopped_controller_status(
        durable_status,
        provenance=observed_provenance,
    )
    raw_anchors = durable_status.get("stopped_recovery_anchors")
    anchored_source_continuity = (
        raw_anchors.get("final_source_continuity")
        if isinstance(raw_anchors, Mapping)
        else None
    )
    if sealed_source_continuity is not None:
        final_continuity = dict(sealed_source_continuity)
        source_continuity_is_sealed = True
    elif isinstance(anchored_source_continuity, Mapping):
        final_continuity = dict(anchored_source_continuity)
        source_continuity_is_sealed = True
    else:
        final_continuity = _observe_candidate_runtime_continuity(
            root,
            require_resolved_remote=False,
        )
        source_continuity_is_sealed = False
    observed_source_continuity = (
        _observe_stopped_projection_source_continuity(
            root,
            final_continuity,
        )
        if source_continuity_is_sealed
        else final_continuity
    )
    databases = _stopped_state_database_digests(
        paths,
        _database_paths=io_paths["databases"],
    )
    owner_status_sha256 = _stopped_owner_status_sha256(
        paths,
        controller_status=durable_status,
        _status_path=Path(io_paths["owner_status"]),
        _marker_path=Path(io_paths["owner_marker"]),
    )
    bootstrap_sha256 = _sha256_regular_file(
        Path(io_paths["bootstrap"]),
        max_bytes=MAX_JSON_BYTES,
        noun="stopped recovery bootstrap receipt",
        require_private_owner=True,
    )
    _validate_stopped_projection_native_provenance(
        paths,
        root=root,
        receipt=observed_provenance,
        final_continuity=final_continuity,
        _bootstrap_path=Path(io_paths["bootstrap"]),
    )
    verification = _verify_profile(
        Path(io_paths["databases"]["control"]),
        read_only=True,
    )
    identity = _database_identity(Path(io_paths["databases"]["control"]))
    owner_identity = durable_status["owner_identity"]
    assert isinstance(owner_identity, Mapping)
    if (
        verification.get("schema_fingerprint")
        != observed_provenance.get("schema_fingerprint")
        or verification.get("catalog_fingerprint")
        != observed_provenance.get("catalog_fingerprint")
        or identity.get("database_uuid")
        != observed_provenance.get("database_uuid")
        or identity.get("schema_fingerprint")
        != observed_provenance.get("schema_fingerprint")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            identity.get("schema_fingerprint"),
        )
    ):
        raise SuccessorOperatorError(
            "stopped recovery database identity differs from provenance"
        )
    anchors = _validate_stopped_recovery_anchors(
        durable_status,
        provenance=observed_provenance,
        final_continuity=final_continuity,
        databases=databases,
        owner_status_sha256=owner_status_sha256,
    )
    reviewed_pins: dict[str, Any] = {
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "controller_status_cid": durable_status["status_cid"],
        "controller_status": durable_status,
        "source_provenance_cid": observed_provenance["receipt_cid"],
        "source_continuity": final_continuity,
        "databases": databases,
        "owner_status_sha256": owner_status_sha256,
        "durable_stopped_anchors_cid": (
            str(anchors["anchors_cid"]) if anchors is not None else ""
        ),
    }
    preflight_binding = {
        "schema": STOPPED_RECOVERY_PREFLIGHT_SCHEMA,
        "operation": STOPPED_RECOVERY_OPERATION,
        "reviewed_pins": reviewed_pins,
    }
    report: dict[str, Any] = {
        "schema": STOPPED_RECOVERY_PREFLIGHT_SCHEMA,
        "operation": STOPPED_RECOVERY_OPERATION,
        "observed_at": _utc_now(),
        "reviewed_pins": reviewed_pins,
        "preflight_cid": _content_id(preflight_binding),
        "durable_stopped_anchors_present": anchors is not None,
        "generic_recovery_authorized": anchors is not None,
        "legacy_explicit_review_required": anchors is None,
        "controller_lock_held": True,
        "live_wal_absent": True,
        "restart_authority": False,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "production_authorized": False,
    }
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    if source_continuity_is_sealed:
        _observe_stopped_projection_source_continuity(
            root,
            final_continuity,
            minimum_remote_head=str(
                observed_source_continuity["resolved_remote_head"]
            ),
        )
        source_changed_during_preflight = False
    else:
        source_changed_during_preflight = (
            _observe_candidate_runtime_continuity(
                root,
                require_resolved_remote=False,
            )
            != final_continuity
        )
    if (
        _stopped_recovery_generation_inventory(paths, lock_custody)
        != generation_inventory
        or _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        != observed_provenance
        or _strict_json(
            Path(io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        )
        != durable_status
        or _sha256_regular_file(
            Path(io_paths["bootstrap"]),
            max_bytes=MAX_JSON_BYTES,
            noun="stopped recovery bootstrap receipt",
            require_private_owner=True,
        )
        != bootstrap_sha256
        or _stopped_state_database_digests(
            paths,
            _database_paths=io_paths["databases"],
        )
        != databases
        or _stopped_owner_status_sha256(
            paths,
            controller_status=durable_status,
            _status_path=Path(io_paths["owner_status"]),
            _marker_path=Path(io_paths["owner_marker"]),
        )
        != owner_status_sha256
        or source_changed_during_preflight
    ):
        raise SuccessorOperatorError(
            "stopped recovery evidence changed during preflight"
        )
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    return report


def _failed_start_recovery_preflight_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    failed_start_reason: str = "",
    _allow_continuity_receipt: bool = False,
    _require_dead_controller_tree: bool = True,
) -> dict[str, Any]:
    """Pin one stopped pre-ready failure without mutating its generation."""

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    if (
        os.path.lexists(io_paths["stopped_state_continuity"])
        and not _allow_continuity_receipt
    ):
        raise SuccessorOperatorError(
            "failed-start continuity is already published"
        )
    generation_inventory = _stopped_recovery_generation_inventory(
        paths,
        lock_custody,
    )
    observed_provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    if provenance is not None and observed_provenance != dict(provenance):
        raise SuccessorOperatorError(
            "failed-start recovery provenance changed before preflight"
        )
    durable_status = _strict_json(
        Path(io_paths["controller_status"]),
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    raw_anchors = durable_status.get("failed_start_recovery_anchors")
    if isinstance(raw_anchors, Mapping):
        source_status = dict(durable_status)
        source_status.pop("status_cid", None)
        source_status.pop("failed_start_recovery_anchors", None)
        source_status["status_cid"] = _content_id(source_status)
        anchored_reason = str(raw_anchors.get("failed_start_reason") or "")
        if failed_start_reason and failed_start_reason != anchored_reason:
            raise SuccessorOperatorError(
                "reviewed failed-start reason differs from durable anchors"
            )
        selected_reason = anchored_reason
        authorization_mode = str(
            raw_anchors.get("recovery_authorization_mode") or ""
        )
        owner_stop_raw = raw_anchors.get("owner_stop_receipt")
        sealed_continuity = raw_anchors.get("final_source_continuity")
        if not isinstance(sealed_continuity, Mapping):
            raise SuccessorOperatorError(
                "failed-start anchored source continuity is malformed"
            )
        final_continuity = dict(sealed_continuity)
        observed_source = _observe_stopped_projection_source_continuity(
            root,
            final_continuity,
        )
    elif raw_anchors is None:
        source_status = durable_status
        if failed_start_reason and (
            failed_start_reason != FAILED_START_REASON_LEGACY_UNCLASSIFIED
        ):
            raise SuccessorOperatorError(
                "unanchored legacy failed-start reason cannot be inferred"
            )
        selected_reason = FAILED_START_REASON_LEGACY_UNCLASSIFIED
        authorization_mode = FAILED_START_REVIEWED_LEGACY_MODE
        owner_stop_raw = None
        final_continuity = _observe_candidate_runtime_continuity(
            root,
            require_resolved_remote=False,
        )
        observed_source = final_continuity
    else:
        raise SuccessorOperatorError(
            "failed-start recovery anchors are malformed"
        )
    if selected_reason not in FAILED_START_RECOVERY_REASONS:
        raise SuccessorOperatorError(
            "an exact allowlisted failed-start reason is required"
        )
    _validate_unbound_failed_start_controller_status(
        source_status,
        provenance=observed_provenance,
        require_dead=_require_dead_controller_tree,
    )
    databases = _stopped_state_database_digests(
        paths,
        _database_paths=io_paths["databases"],
    )
    owner_status_sha256 = _stopped_owner_status_sha256(
        paths,
        controller_status=source_status,
        _status_path=Path(io_paths["owner_status"]),
        _marker_path=Path(io_paths["owner_marker"]),
    )
    bootstrap_path = Path(io_paths["bootstrap"])
    bootstrap_sha256 = _sha256_regular_file(
        bootstrap_path,
        max_bytes=MAX_JSON_BYTES,
        noun="failed-start recovery bootstrap receipt",
        require_private_owner=True,
    )
    _validate_stopped_projection_native_provenance(
        paths,
        root=root,
        receipt=observed_provenance,
        final_continuity=final_continuity,
        _bootstrap_path=bootstrap_path,
    )
    verification = _verify_profile(
        Path(io_paths["databases"]["control"]),
        read_only=True,
    )
    identity = _database_identity(Path(io_paths["databases"]["control"]))
    owner_identity = source_status["owner_identity"]
    assert isinstance(owner_identity, Mapping)
    if (
        verification.get("schema_fingerprint")
        != observed_provenance.get("schema_fingerprint")
        or verification.get("catalog_fingerprint")
        != observed_provenance.get("catalog_fingerprint")
        or identity.get("database_uuid")
        != observed_provenance.get("database_uuid")
        or identity.get("schema_fingerprint")
        != observed_provenance.get("schema_fingerprint")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            identity.get("schema_fingerprint"),
        )
    ):
        raise SuccessorOperatorError(
            "failed-start recovery database identity differs from provenance"
        )
    if raw_anchors is None:
        superseded = _capture_failed_start_superseded_admission(
            paths,
            io_paths=io_paths,
            provenance=observed_provenance,
        )
        owner_stop = None
        anchors = None
    else:
        assert isinstance(raw_anchors, Mapping)
        superseded = _validate_failed_start_superseded_admission_snapshot(
            paths,
            raw_anchors.get("superseded_restart_admission"),
            provenance=observed_provenance,
        )
        _observe_failed_start_superseded_admission(
            paths,
            io_paths=io_paths,
            provenance=observed_provenance,
            expected_snapshot=superseded,
        )
        owner_stop = (
            _validate_failed_start_owner_stop_receipt(
                source_status,
                owner_stop_raw,
            )
            if authorization_mode == FAILED_START_TRUSTED_FINALLY_MODE
            else None
        )
        anchors = _validate_failed_start_recovery_anchors(
            paths,
            durable_status,
            provenance=observed_provenance,
            final_continuity=final_continuity,
            databases=databases,
            owner_status_sha256=owner_status_sha256,
            bootstrap_sha256=bootstrap_sha256,
            require_dead=_require_dead_controller_tree,
        )
    reviewed_pins = _failed_start_recovery_reviewed_pins(
        failed_status=source_status,
        provenance=observed_provenance,
        failed_start_reason=selected_reason,
        final_continuity=final_continuity,
        databases=databases,
        owner_status_sha256=owner_status_sha256,
        bootstrap_sha256=bootstrap_sha256,
        superseded_restart_admission=superseded,
        recovery_authorization_mode=authorization_mode,
        owner_stop_receipt=owner_stop,
    )
    preflight_cid = _failed_start_preflight_cid(reviewed_pins)
    if anchors is not None and anchors.get("recovery_preflight_cid") != preflight_cid:
        raise SuccessorOperatorError(
            "durable failed-start preflight binding differs"
        )
    report: dict[str, Any] = {
        "schema": FAILED_START_RECOVERY_PREFLIGHT_SCHEMA,
        "operation": FAILED_START_RECOVERY_OPERATION,
        "observed_at": _utc_now(),
        "reviewed_pins": reviewed_pins,
        "preflight_cid": preflight_cid,
        "durable_failed_start_anchors_present": anchors is not None,
        "generic_recovery_authorized": anchors is not None,
        "legacy_explicit_review_required": anchors is None,
        "controller_lock_held": True,
        "live_wal_absent": True,
        "restart_authority": False,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "production_authorized": False,
    }
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    source_changed = (
        _observe_stopped_projection_source_continuity(
            root,
            final_continuity,
            minimum_remote_head=str(observed_source["resolved_remote_head"]),
        )
        != observed_source
        if anchors is not None
        else _observe_candidate_runtime_continuity(
            root,
            require_resolved_remote=False,
        )
        != final_continuity
    )
    if (
        _stopped_recovery_generation_inventory(paths, lock_custody)
        != generation_inventory
        or _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        != observed_provenance
        or _strict_json(
            Path(io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        )
        != durable_status
        or _stopped_state_database_digests(
            paths,
            _database_paths=io_paths["databases"],
        )
        != databases
        or _stopped_owner_status_sha256(
            paths,
            controller_status=source_status,
            _status_path=Path(io_paths["owner_status"]),
            _marker_path=Path(io_paths["owner_marker"]),
        )
        != owner_status_sha256
        or _sha256_regular_file(
            bootstrap_path,
            max_bytes=MAX_JSON_BYTES,
            noun="failed-start recovery bootstrap receipt",
            require_private_owner=True,
        )
        != bootstrap_sha256
        or (
            _capture_failed_start_superseded_admission(
                paths,
                io_paths=io_paths,
                provenance=observed_provenance,
            )
            != superseded
            if anchors is None
            else _observe_failed_start_superseded_admission(
                paths,
                io_paths=io_paths,
                provenance=observed_provenance,
                expected_snapshot=superseded,
            )
            not in {"absent", "consumed", "archived"}
        )
        or source_changed
    ):
        raise SuccessorOperatorError(
            "failed-start recovery evidence changed during preflight"
        )
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    return report


def _write_stopped_state_continuity(
    paths: Mapping[str, Path],
    *,
    root: Path,
    stopped_status: Mapping[str, Any],
    provenance: Mapping[str, Any],
    owner_checkpoint: Mapping[str, Any],
    owner_stop: Mapping[str, Any],
    _io_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one immutable same-generation receipt after a clean owner stop."""

    io_paths = (
        _stopped_recovery_io_paths(paths, None)
        if _io_paths is None
        else dict(_io_paths)
    )
    receipt_paths = _stopped_receipt_io_view(paths, io_paths)
    durable_provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    if durable_provenance != dict(provenance):
        raise SuccessorOperatorError(
            "clean stopped-state provenance changed before publication"
        )
    durable_status = _strict_json(
        Path(io_paths["controller_status"]),
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    checkpoint = dict(owner_checkpoint)
    stopped = dict(owner_stop)
    owner_identity = durable_status.get("owner_identity")
    owner_server_id = (
        str(owner_identity.get("server_id") or "")
        if isinstance(owner_identity, Mapping)
        else ""
    )
    if (
        durable_status != dict(stopped_status)
        or durable_status.get("lifecycle") != "stopped"
        or durable_status.get("error") != ""
        or durable_status.get("provenance_cid") != provenance.get("receipt_cid")
        or type(durable_status.get("scheduler_returncode")) is not int
        or durable_status.get("scheduler_returncode") != 0
        or set(checkpoint) != {"checkpointed", "server_id", "database_path", "at"}
        or checkpoint.get("checkpointed") is not True
        or checkpoint.get("server_id") != owner_server_id
        or checkpoint.get("database_path") != str(paths["successor_database"])
        or set(stopped) != {"stopped", "server_id", "at"}
        or stopped.get("stopped") is not True
        or stopped.get("server_id") != owner_server_id
    ):
        raise SuccessorOperatorError(
            "clean stopped-state continuity evidence differs"
        )
    final_continuity = _observe_candidate_runtime_continuity(
        root,
        require_resolved_remote=False,
    )
    databases = _stopped_state_database_digests(
        paths,
        _database_paths=io_paths["databases"],
    )
    owner_status_sha256 = _stopped_owner_status_sha256(
        paths,
        controller_status=durable_status,
        _status_path=Path(io_paths["owner_status"]),
        _marker_path=Path(io_paths["owner_marker"]),
    )
    anchors = _validate_stopped_recovery_anchors(
        durable_status,
        provenance=provenance,
        final_continuity=final_continuity,
        databases=databases,
        owner_status_sha256=owner_status_sha256,
    )
    if anchors is None:
        raise SuccessorOperatorError(
            "clean stopped-state status lacks pre-publication recovery anchors"
        )
    retired_claim = _restore_or_retire_stopped_restart_admission(
        receipt_paths,
        retire_unbound_status_cid=str(durable_status["status_cid"]),
    )
    if retired_claim not in {"absent", "retired_after_clean_stop"}:
        raise SuccessorOperatorError(
            "prior stopped-state restart admission was not safely retired"
        )
    receipt: dict[str, Any] = {
        "schema": STOPPED_STATE_CONTINUITY_SCHEMA,
        "issued_at": _utc_now(),
        "admission_mode": STOPPED_STATE_CONTINUITY_ADMISSION_MODE,
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "source_provenance_cid": provenance["receipt_cid"],
        "controller_status_cid": durable_status["status_cid"],
        "stop_evidence": {
            "mode": STOPPED_STATE_LIVE_OWNER_EVIDENCE_MODE,
            "owner_checkpoint": checkpoint,
            "owner_stop": stopped,
            "historical_owner_receipts_reconstructed": False,
        },
        "owner_status_sha256": owner_status_sha256,
        "final_source_continuity": final_continuity,
        "databases": databases,
        "controller_lock_held_at_issue": True,
        "live_wal_absent": True,
        "requires_stopped_checkpoint": True,
        "projection_only": False,
        "same_generation_restart_only": True,
        "restart_authority": True,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "read_by_scheduler": False,
        "quack_endpoint_served": False,
        "production_authorized": False,
    }
    receipt["receipt_cid"] = _content_id(receipt)
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=provenance,
        controller_status_cid=str(durable_status["status_cid"]),
    )
    _atomic_json(
        Path(io_paths["stopped_state_continuity"]),
        receipt,
        replace=False,
    )
    return receipt


def _reviewed_failed_start_anchors_from_preflight(
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    pins = preflight.get("reviewed_pins")
    if not isinstance(pins, Mapping):
        raise SuccessorOperatorError(
            "reviewed failed-start preflight pins are unavailable"
        )
    if (
        preflight.get("schema") != FAILED_START_RECOVERY_PREFLIGHT_SCHEMA
        or preflight.get("operation") != FAILED_START_RECOVERY_OPERATION
        or preflight.get("legacy_explicit_review_required") is not True
        or pins.get("recovery_authorization_mode")
        != FAILED_START_REVIEWED_LEGACY_MODE
        or pins.get("owner_stop_receipt") is not None
        or preflight.get("preflight_cid")
        != _failed_start_preflight_cid(pins)
    ):
        raise SuccessorOperatorError(
            "reviewed failed-start preflight semantics differ"
        )
    anchors: dict[str, Any] = {
        "schema": FAILED_START_RECOVERY_ANCHORS_SCHEMA,
        "captured_at": _utc_now(),
        "target_generation": pins["target_generation"],
        "source_provenance_cid": pins["source_provenance_cid"],
        "source_controller_status_cid": pins["controller_status_cid"],
        "failed_start_reason": pins["failed_start_reason"],
        "final_source_continuity": pins["source_continuity"],
        "databases": pins["databases"],
        "owner_status_sha256": pins["owner_status_sha256"],
        "bootstrap_sha256": pins["bootstrap_sha256"],
        "superseded_restart_admission": pins[
            "superseded_restart_admission"
        ],
        "recovery_authorization_mode": FAILED_START_REVIEWED_LEGACY_MODE,
        "owner_stop_receipt": None,
        "recovery_preflight_cid": preflight["preflight_cid"],
    }
    anchors["anchors_cid"] = _content_id(anchors)
    return anchors


def _validate_failed_start_stop_evidence(
    receipt: Mapping[str, Any],
    *,
    failed_status: Mapping[str, Any],
    anchors: Mapping[str, Any],
) -> None:
    evidence = receipt.get("stop_evidence")
    if not isinstance(evidence, Mapping):
        raise SuccessorOperatorError("failed-start stop evidence is malformed")
    superseded = anchors.get("superseded_restart_admission")
    superseded_receipt = (
        superseded.get("receipt") if isinstance(superseded, Mapping) else None
    )
    superseded_cid = (
        str(superseded_receipt.get("receipt_cid") or "")
        if isinstance(superseded_receipt, Mapping)
        else ""
    )
    common_valid = (
        evidence.get("failed_start_reason")
        == anchors.get("failed_start_reason")
        and evidence.get("durable_failed_start_anchors_cid")
        == anchors.get("anchors_cid")
        and evidence.get("superseded_restart_receipt_cid")
        == superseded_cid
        and evidence.get("historical_owner_receipts_reconstructed") is False
    )
    mode = evidence.get("mode")
    if mode == FAILED_START_LIVE_OWNER_EVIDENCE_MODE:
        if (
            set(evidence)
            != {
                "mode",
                "failed_start_reason",
                "owner_stop",
                "durable_failed_start_anchors_cid",
                "superseded_restart_receipt_cid",
                "historical_owner_receipts_reconstructed",
            }
            or not common_valid
            or anchors.get("recovery_authorization_mode")
            != FAILED_START_TRUSTED_FINALLY_MODE
            or evidence.get("owner_stop")
            != _validate_failed_start_owner_stop_receipt(
                failed_status,
                anchors.get("owner_stop_receipt"),
            )
        ):
            raise SuccessorOperatorError(
                "trusted failed-start stop evidence differs"
            )
        return
    if mode == FAILED_START_REVIEWED_EVIDENCE_MODE:
        recovered_at = evidence.get("recovered_at")
        if (
            set(evidence)
            != {
                "mode",
                "recovered_at",
                "failed_start_reason",
                "source_controller_status_cid",
                "recovery_preflight_cid",
                "durable_failed_start_anchors_cid",
                "superseded_restart_receipt_cid",
                "historical_owner_receipts_reconstructed",
            }
            or not common_valid
            or anchors.get("recovery_authorization_mode")
            != FAILED_START_REVIEWED_LEGACY_MODE
            or anchors.get("owner_stop_receipt") is not None
            or type(recovered_at) is not str
            or recovered_at != receipt.get("issued_at")
            or evidence.get("source_controller_status_cid")
            != anchors.get("source_controller_status_cid")
            or evidence.get("recovery_preflight_cid")
            != anchors.get("recovery_preflight_cid")
        ):
            raise SuccessorOperatorError(
                "reviewed failed-start stop evidence differs"
            )
        return
    raise SuccessorOperatorError("failed-start stop evidence mode differs")


def _build_failed_start_continuity_receipt(
    paths: Mapping[str, Path],
    *,
    failed_status: Mapping[str, Any],
    provenance: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    pins = preflight.get("reviewed_pins")
    anchors = failed_status.get("failed_start_recovery_anchors")
    if not isinstance(pins, Mapping) or not isinstance(anchors, Mapping):
        raise SuccessorOperatorError(
            "failed-start receipt lacks durable reviewed pins"
        )
    issued_at = _utc_now()
    superseded = anchors.get("superseded_restart_admission")
    superseded_receipt = (
        superseded.get("receipt") if isinstance(superseded, Mapping) else None
    )
    superseded_cid = (
        str(superseded_receipt.get("receipt_cid") or "")
        if isinstance(superseded_receipt, Mapping)
        else ""
    )
    authorization_mode = anchors.get("recovery_authorization_mode")
    if authorization_mode == FAILED_START_TRUSTED_FINALLY_MODE:
        stop_evidence: dict[str, Any] = {
            "mode": FAILED_START_LIVE_OWNER_EVIDENCE_MODE,
            "failed_start_reason": anchors["failed_start_reason"],
            "owner_stop": anchors["owner_stop_receipt"],
            "durable_failed_start_anchors_cid": anchors["anchors_cid"],
            "superseded_restart_receipt_cid": superseded_cid,
            "historical_owner_receipts_reconstructed": False,
        }
    elif authorization_mode == FAILED_START_REVIEWED_LEGACY_MODE:
        stop_evidence = {
            "mode": FAILED_START_REVIEWED_EVIDENCE_MODE,
            "recovered_at": issued_at,
            "failed_start_reason": anchors["failed_start_reason"],
            "source_controller_status_cid": anchors[
                "source_controller_status_cid"
            ],
            "recovery_preflight_cid": anchors["recovery_preflight_cid"],
            "durable_failed_start_anchors_cid": anchors["anchors_cid"],
            "superseded_restart_receipt_cid": superseded_cid,
            "historical_owner_receipts_reconstructed": False,
        }
    else:
        raise SuccessorOperatorError(
            "failed-start receipt authorization mode differs"
        )
    receipt: dict[str, Any] = {
        "schema": STOPPED_STATE_CONTINUITY_SCHEMA,
        "issued_at": issued_at,
        "admission_mode": FAILED_START_CONTINUITY_ADMISSION_MODE,
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "source_provenance_cid": provenance["receipt_cid"],
        "controller_status_cid": failed_status["status_cid"],
        "stop_evidence": stop_evidence,
        "owner_status_sha256": pins["owner_status_sha256"],
        "final_source_continuity": pins["source_continuity"],
        "databases": pins["databases"],
        "controller_lock_held_at_issue": True,
        "live_wal_absent": True,
        "requires_stopped_checkpoint": False,
        "projection_only": False,
        "same_generation_restart_only": True,
        "restart_authority": True,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "read_by_scheduler": False,
        "quack_endpoint_served": False,
        "production_authorized": False,
    }
    receipt["receipt_cid"] = _content_id(receipt)
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=provenance,
        controller_status_cid=str(failed_status["status_cid"]),
    )
    _validate_failed_start_stop_evidence(
        receipt,
        failed_status=failed_status,
        anchors=anchors,
    )
    return receipt


def _complete_interrupted_failed_start_publication(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any],
    _require_dead_controller_tree: bool = True,
) -> dict[str, Any] | None:
    """Bind a fully replayed failed-start receipt after an interrupted write."""

    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    receipt = _strict_json(
        Path(io_paths["stopped_state_continuity"]),
        expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
        require_private_owner=True,
    )
    if receipt.get("admission_mode") != FAILED_START_CONTINUITY_ADMISSION_MODE:
        return None
    status = _strict_json(
        Path(io_paths["controller_status"]),
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    linked_receipt = status.get("stopped_state_continuity_receipt_cid")
    linked_status = status.get("stopped_state_continuity_status_cid")
    if linked_receipt is not None or linked_status is not None:
        if (
            linked_receipt == receipt.get("receipt_cid")
            and linked_status == receipt.get("controller_status_cid")
        ):
            return None
        raise SuccessorOperatorError(
            "existing failed-start publication binding differs"
        )
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=provenance,
        controller_status_cid=str(status.get("status_cid") or ""),
    )
    preflight = _failed_start_recovery_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
        _allow_continuity_receipt=True,
        _require_dead_controller_tree=_require_dead_controller_tree,
    )
    expected = _build_failed_start_continuity_receipt(
        paths,
        failed_status=status,
        provenance=provenance,
        preflight=preflight,
    )
    # issued_at is the sole fresh field; replay the stored value before compare.
    expected["issued_at"] = receipt.get("issued_at")
    evidence = dict(expected["stop_evidence"])
    if evidence.get("mode") == FAILED_START_REVIEWED_EVIDENCE_MODE:
        evidence["recovered_at"] = receipt.get("issued_at")
    expected["stop_evidence"] = evidence
    expected["receipt_cid"] = _content_id(
        {name: value for name, value in expected.items() if name != "receipt_cid"}
    )
    if expected != receipt:
        raise SuccessorOperatorError(
            "interrupted failed-start continuity receipt differs"
        )
    anchors = status.get("failed_start_recovery_anchors")
    assert isinstance(anchors, Mapping)
    _validate_failed_start_stop_evidence(
        receipt,
        failed_status=status,
        anchors=anchors,
    )
    bound = _bind_stopped_state_continuity_status(status, receipt)
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    _write_status(Path(io_paths["controller_status"]), bound)
    return receipt


def _recover_interrupted_failed_start_continuity(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any],
    reviewed_preflight_cid: str = "",
    failed_start_reason: str = "",
    _require_dead_controller_tree: bool = True,
) -> dict[str, Any] | None:
    """Publish only anchored current failed-start bytes as new authority."""

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    if os.path.lexists(io_paths["stopped_state_continuity"]):
        return _complete_interrupted_failed_start_publication(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
            _require_dead_controller_tree=_require_dead_controller_tree,
        )
    status = _strict_json(
        Path(io_paths["controller_status"]),
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    if status.get("error") != FAILED_START_STATUS_ERROR:
        return None
    if any(
        name in status
        for name in (
            "stopped_state_continuity_receipt_cid",
            "stopped_state_continuity_status_cid",
        )
    ):
        raise SuccessorOperatorError(
            "failed-start status is already continuity-bound without a receipt"
        )
    if "failed_start_recovery_anchors" not in status:
        preflight = _failed_start_recovery_preflight_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
            failed_start_reason=failed_start_reason,
            _require_dead_controller_tree=_require_dead_controller_tree,
        )
        expected_cid = str(preflight["preflight_cid"])
        if reviewed_preflight_cid != expected_cid:
            raise SuccessorOperatorError(
                "unanchored failed-start status requires an explicitly reviewed "
                "failed-start recovery preflight CID"
            )
        pins = preflight["reviewed_pins"]
        assert isinstance(pins, Mapping)
        if pins.get("controller_status") != status:
            raise SuccessorOperatorError(
                "reviewed failed-start controller status changed"
            )
        anchors = _reviewed_failed_start_anchors_from_preflight(preflight)
        status = _bind_failed_start_recovery_anchors_status(status, anchors)
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        _write_status(Path(io_paths["controller_status"]), status)
        if _strict_json(
            Path(io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        ) != status:
            raise SuccessorOperatorError(
                "failed-start anchored status changed during publication"
            )
    preflight = _failed_start_recovery_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
        failed_start_reason=failed_start_reason,
        _require_dead_controller_tree=_require_dead_controller_tree,
    )
    pins = preflight["reviewed_pins"]
    assert isinstance(pins, Mapping)
    anchors = status.get("failed_start_recovery_anchors")
    if (
        preflight.get("generic_recovery_authorized") is not True
        or not isinstance(anchors, Mapping)
        or pins.get("controller_status_cid")
        != anchors.get("source_controller_status_cid")
    ):
        raise SuccessorOperatorError(
            "failed-start durable recovery admission differs"
        )
    _archive_failed_start_superseded_admission(
        paths,
        io_paths=io_paths,
        provenance=provenance,
        expected_snapshot=anchors.get("superseded_restart_admission"),
    )
    repeated = _failed_start_recovery_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
        failed_start_reason=failed_start_reason,
        _require_dead_controller_tree=_require_dead_controller_tree,
    )
    if repeated.get("preflight_cid") != preflight.get("preflight_cid"):
        raise SuccessorOperatorError(
            "failed-start evidence changed during receipt admission"
        )
    receipt = _build_failed_start_continuity_receipt(
        paths,
        failed_status=status,
        provenance=provenance,
        preflight=repeated,
    )
    _atomic_json(
        Path(io_paths["stopped_state_continuity"]),
        receipt,
        replace=False,
    )
    completed = _complete_interrupted_failed_start_publication(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
        _require_dead_controller_tree=_require_dead_controller_tree,
    )
    if completed != receipt:
        raise SuccessorOperatorError(
            "failed-start recovery publication completion differs"
        )
    return receipt


def _complete_interrupted_stopped_recovery_publication(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Finish only a fully validated receipt-written/status-unbound recovery."""

    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    receipt = _strict_json(
        Path(io_paths["stopped_state_continuity"]),
        expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
        require_private_owner=True,
    )
    status = _strict_json(
        Path(io_paths["controller_status"]),
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    linked_receipt = status.get("stopped_state_continuity_receipt_cid")
    linked_status = status.get("stopped_state_continuity_status_cid")
    if linked_receipt is not None or linked_status is not None:
        if (
            linked_receipt == receipt.get("receipt_cid")
            and linked_status == receipt.get("controller_status_cid")
        ):
            return None
        raise SuccessorOperatorError(
            "existing stopped-state recovery publication binding differs"
        )
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=provenance,
        controller_status_cid=str(status["status_cid"]),
    )
    receipt_source_continuity = receipt.get("final_source_continuity")
    if not isinstance(receipt_source_continuity, Mapping):
        raise SuccessorOperatorError(
            "interrupted stopped-state source continuity differs"
        )
    preflight = _stopped_recovery_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
        sealed_source_continuity=receipt_source_continuity,
    )
    reviewed_pins = preflight["reviewed_pins"]
    assert isinstance(reviewed_pins, Mapping)
    stop_evidence = receipt.get("stop_evidence")
    final_continuity = receipt.get("final_source_continuity")
    databases = receipt.get("databases")
    if (
        status.get("lifecycle") != "stopped"
        or status.get("error") != ""
        or type(status.get("scheduler_returncode")) is not int
        or status.get("scheduler_returncode") != 0
        or status.get("provenance_cid") != provenance.get("receipt_cid")
        or receipt.get("source_provenance_cid") != provenance.get("receipt_cid")
        or receipt.get("controller_status_cid") != status.get("status_cid")
        or receipt.get("target_generation") != SUCCESSOR_STORE_GENERATION
        or receipt.get("admission_mode")
        != STOPPED_STATE_CONTINUITY_ADMISSION_MODE
        or receipt.get("controller_lock_held_at_issue") is not True
        or receipt.get("live_wal_absent") is not True
        or receipt.get("requires_stopped_checkpoint") is not True
        or receipt.get("projection_only") is not False
        or receipt.get("same_generation_restart_only") is not True
        or receipt.get("restart_authority") is not True
        or any(
            receipt.get(field) is not False
            for field in (
                "authoritative",
                "scheduling_authority",
                "completion_authority",
                "read_by_scheduler",
                "quack_endpoint_served",
                "production_authorized",
            )
        )
        or not isinstance(stop_evidence, Mapping)
        or not isinstance(final_continuity, Mapping)
        or not isinstance(databases, Mapping)
        or dict(databases) != reviewed_pins.get("databases")
        or dict(final_continuity) != reviewed_pins.get("source_continuity")
        or receipt.get("owner_status_sha256")
        != reviewed_pins.get("owner_status_sha256")
    ):
        raise SuccessorOperatorError(
            "interrupted stopped-state recovery receipt differs"
        )
    evidence_mode = stop_evidence.get("mode")
    owner_identity = status.get("owner_identity")
    owner_server_id = (
        str(owner_identity.get("server_id") or "")
        if isinstance(owner_identity, Mapping)
        else ""
    )
    if evidence_mode == STOPPED_STATE_RECOVERED_EVIDENCE_MODE:
        anchor_cid = str(
            reviewed_pins.get("durable_stopped_anchors_cid") or ""
        )
        expected_authorization_mode = (
            STOPPED_RECOVERY_DURABLE_ANCHOR_MODE
            if anchor_cid
            else STOPPED_RECOVERY_REVIEWED_LEGACY_MODE
        )
        if (
            set(stop_evidence)
            != {
                "mode",
                "recovered_at",
                "source_controller_status_cid",
                "recovery_preflight_cid",
                "recovery_authorization_mode",
                "durable_stopped_anchors_cid",
                "historical_owner_receipts_reconstructed",
            }
            or stop_evidence.get("recovered_at") != receipt.get("issued_at")
            or stop_evidence.get("source_controller_status_cid")
            != status.get("status_cid")
            or stop_evidence.get("recovery_preflight_cid")
            != preflight.get("preflight_cid")
            or stop_evidence.get("recovery_authorization_mode")
            != expected_authorization_mode
            or stop_evidence.get("durable_stopped_anchors_cid")
            != anchor_cid
            or stop_evidence.get("historical_owner_receipts_reconstructed")
            is not False
        ):
            raise SuccessorOperatorError(
                "interrupted recovered stop evidence differs"
            )
    elif evidence_mode == STOPPED_STATE_LIVE_OWNER_EVIDENCE_MODE:
        checkpoint = stop_evidence.get("owner_checkpoint")
        stopped = stop_evidence.get("owner_stop")
        if (
            preflight.get("durable_stopped_anchors_present") is not True
            or
            set(stop_evidence)
            != {
                "mode",
                "owner_checkpoint",
                "owner_stop",
                "historical_owner_receipts_reconstructed",
            }
            or stop_evidence.get("historical_owner_receipts_reconstructed")
            is not False
            or not isinstance(checkpoint, Mapping)
            or set(checkpoint)
            != {"checkpointed", "server_id", "database_path", "at"}
            or checkpoint.get("checkpointed") is not True
            or checkpoint.get("server_id") != owner_server_id
            or checkpoint.get("database_path")
            != str(paths["successor_database"])
            or not isinstance(stopped, Mapping)
            or set(stopped) != {"stopped", "server_id", "at"}
            or stopped.get("stopped") is not True
            or stopped.get("server_id") != owner_server_id
        ):
            raise SuccessorOperatorError(
                "interrupted live-owner stop evidence differs"
            )
    else:
        raise SuccessorOperatorError(
            "interrupted stopped-state evidence mode differs"
        )
    bound = _bind_stopped_state_continuity_status(status, receipt)
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    _write_status(Path(io_paths["controller_status"]), bound)
    return receipt


def _recover_interrupted_stopped_state_continuity(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any],
    reviewed_preflight_cid: str = "",
) -> dict[str, Any] | None:
    """Recover only the publication interrupted after a proven clean stop.

    The recovery does not synthesize the vanished in-memory checkpoint or stop
    receipts.  It issues fresh evidence from the durable controller and owner
    stopped projections while holding the exact run-generation lock.
    """

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    if os.path.lexists(io_paths["stopped_state_continuity"]):
        completed = _complete_interrupted_stopped_recovery_publication(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
        )
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        return completed
    if not os.path.lexists(io_paths["controller_status"]):
        return None
    durable_status = _strict_json(
        Path(io_paths["controller_status"]),
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    if any(
        name in durable_status
        for name in (
            "stopped_state_continuity_receipt_cid",
            "stopped_state_continuity_status_cid",
        )
    ):
        raise SuccessorOperatorError(
            "interrupted stopped-state status is already continuity-bound"
        )
    if (
        durable_status.get("lifecycle") != "stopped"
        or durable_status.get("error") != ""
        or type(durable_status.get("scheduler_returncode")) is not int
        or durable_status.get("scheduler_returncode") != 0
        or durable_status.get("provenance_cid") != provenance.get("receipt_cid")
    ):
        return None
    preflight = _stopped_recovery_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
    )
    expected_preflight_cid = str(preflight["preflight_cid"])
    if reviewed_preflight_cid and reviewed_preflight_cid != expected_preflight_cid:
        raise SuccessorOperatorError(
            "reviewed stopped recovery preflight CID differs"
        )
    if (
        preflight.get("legacy_explicit_review_required") is True
        and reviewed_preflight_cid != expected_preflight_cid
    ):
        raise SuccessorOperatorError(
            "legacy stopped status is not self-anchored; run "
            "stopped-recovery-preflight, review every exact pin, then run "
            "recover-stopped-continuity --reviewed-preflight-cid <cid>"
        )
    if os.path.lexists(io_paths["stopped_state_restart_admission"]):
        retired_claim = _restore_or_retire_stopped_restart_admission(
            _stopped_receipt_io_view(paths, io_paths),
            retire_unbound_status_cid=str(durable_status["status_cid"]),
        )
        if retired_claim != "retired_after_clean_stop":
            raise SuccessorOperatorError(
                "interrupted clean-stop receipt claim was not safely retired"
            )
    reviewed_pins = preflight["reviewed_pins"]
    assert isinstance(reviewed_pins, Mapping)
    anchor_cid = str(
        reviewed_pins.get("durable_stopped_anchors_cid") or ""
    )
    authorization_mode = (
        STOPPED_RECOVERY_DURABLE_ANCHOR_MODE
        if anchor_cid
        else STOPPED_RECOVERY_REVIEWED_LEGACY_MODE
    )
    recovered_at = _utc_now()
    receipt: dict[str, Any] = {
        "schema": STOPPED_STATE_CONTINUITY_SCHEMA,
        "issued_at": recovered_at,
        "admission_mode": STOPPED_STATE_CONTINUITY_ADMISSION_MODE,
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "source_provenance_cid": provenance["receipt_cid"],
        "controller_status_cid": durable_status["status_cid"],
        "stop_evidence": {
            "mode": STOPPED_STATE_RECOVERED_EVIDENCE_MODE,
            "recovered_at": recovered_at,
            "source_controller_status_cid": durable_status["status_cid"],
            "recovery_preflight_cid": expected_preflight_cid,
            "recovery_authorization_mode": authorization_mode,
            "durable_stopped_anchors_cid": anchor_cid,
            "historical_owner_receipts_reconstructed": False,
        },
        "owner_status_sha256": reviewed_pins["owner_status_sha256"],
        "final_source_continuity": reviewed_pins["source_continuity"],
        "databases": reviewed_pins["databases"],
        "controller_lock_held_at_issue": True,
        "live_wal_absent": True,
        "requires_stopped_checkpoint": True,
        "projection_only": False,
        "same_generation_restart_only": True,
        "restart_authority": True,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "read_by_scheduler": False,
        "quack_endpoint_served": False,
        "production_authorized": False,
    }
    receipt["receipt_cid"] = _content_id(receipt)
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=provenance,
        controller_status_cid=str(durable_status["status_cid"]),
    )
    repeated_preflight = _stopped_recovery_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
    )
    if repeated_preflight.get("preflight_cid") != expected_preflight_cid:
        raise SuccessorOperatorError(
            "stopped recovery evidence changed during admission"
        )
    _atomic_json(
        Path(io_paths["stopped_state_continuity"]),
        receipt,
        replace=False,
    )
    completed = _complete_interrupted_stopped_recovery_publication(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
    )
    if completed != receipt:
        raise SuccessorOperatorError(
            "stopped recovery publication completion differs"
        )
    return completed


def _bind_stopped_state_continuity_status(
    stopped_status: Mapping[str, Any],
    continuity: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-bind an immutable continuity receipt into the final status."""

    admission_mode = continuity.get("admission_mode")
    expected_error = (
        ""
        if admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
        else FAILED_START_STATUS_ERROR
        if admission_mode == FAILED_START_CONTINUITY_ADMISSION_MODE
        else None
    )
    if (
        stopped_status.get("status_cid")
        != continuity.get("controller_status_cid")
        or stopped_status.get("lifecycle") != "stopped"
        or expected_error is None
        or stopped_status.get("error") != expected_error
        or (
            admission_mode == FAILED_START_CONTINUITY_ADMISSION_MODE
            and "failed_start_recovery_anchors" not in stopped_status
        )
        or continuity.get("receipt_cid") != _content_id(
            {
                name: value
                for name, value in continuity.items()
                if name != "receipt_cid"
            }
        )
    ):
        raise SuccessorOperatorError(
            "stopped-state continuity/status cross-binding differs"
        )
    bound = dict(stopped_status)
    bound.pop("status_cid", None)
    bound["stopped_state_continuity_receipt_cid"] = continuity["receipt_cid"]
    bound["stopped_state_continuity_status_cid"] = stopped_status["status_cid"]
    bound["status_cid"] = _content_id(bound)
    return bound


def _token_sink(owner_state: Path) -> Path:
    """Return an impossible child path so legacy helpers cannot persist the token."""

    marker = owner_state / ".ephemeral-token-persistence-disabled"
    payload = b"trusted controller keeps the Quack attach credential in memory\n"
    if marker.exists():
        observed = os.lstat(marker)
        if not stat.S_ISREG(observed.st_mode) or marker.read_bytes() != payload:
            raise SuccessorOperatorError("ephemeral token sink marker is unsafe")
    else:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(marker, 0o400)
    # The parent component is a regular file, so mkdir/open in the legacy
    # persistence helper fails without ever creating credential material.
    return marker / "unavailable"


def _prepare_private_owner_socket(socket_path: Path) -> None:
    """Admit one short same-UID directory without following a symlink."""

    path = Path(socket_path)
    temporary_root = Path(tempfile.gettempdir()).resolve()
    parent = path.parent
    if (
        not path.is_absolute()
        or parent.parent.resolve() != temporary_root
        or parent.name != f"ipfs-accelerate-lgcvf-{os.geteuid()}"
        or not path.name.startswith("owner-")
        or not path.name.endswith(".sock")
        or len(os.fsencode(path)) > UNIX_SOCKET_PATH_CEILING
    ):
        raise SuccessorOperatorError("state-owner socket identity is unsafe")
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        metadata = os.lstat(parent)
    except OSError as exc:
        raise SuccessorOperatorError(
            "state-owner socket directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SuccessorOperatorError("state-owner socket directory custody is unsafe")
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SuccessorOperatorError("state-owner socket cannot be inspected") from exc
    if (
        not stat.S_ISSOCK(existing.st_mode)
        or stat.S_ISLNK(existing.st_mode)
        or existing.st_uid != os.geteuid()
        or existing.st_nlink != 1
        or stat.S_IMODE(existing.st_mode) & 0o077
    ):
        raise SuccessorOperatorError("existing state-owner socket custody is unsafe")


def _installed_extension_version(info_path: Path, *, name: str) -> str:
    """Read the sole short hexadecimal build identity from DuckDB metadata."""

    raw = _read_bounded_regular_file(
        info_path,
        max_bytes=64 * 1024,
        noun=f"installed {name} extension metadata",
    )
    versions = tuple(
        match.decode("ascii")
        for match in re.findall(
            rb"(?<![0-9a-f])([0-9a-f]{7,8})(?![0-9a-f])",
            raw,
        )
    )
    if len(versions) != 1:
        raise SuccessorOperatorError(
            f"installed {name} extension build identity is ambiguous"
        )
    return versions[0]


def _resolve_installed_duckdb_live_runtime() -> dict[str, Any]:
    """Resolve the exact installed DuckDB facade, native ELF, and extensions.

    Resolution deliberately uses import metadata and module specs without
    importing DuckDB.  Native module creation is permitted only after the
    capsule/native/admission join has been verified.
    """

    try:
        distribution = importlib.metadata.distribution("duckdb")
        version = str(distribution.version)
        site_root = Path(distribution.locate_file("")).resolve(strict=True)
        metadata_value = getattr(distribution, "_path", None)
        if metadata_value is None:
            raise SuccessorOperatorError(
                "installed DuckDB distribution metadata root is unavailable"
            )
        metadata_root = Path(metadata_value).resolve(strict=True)
        package_root = (site_root / "duckdb").resolve(strict=True)
        native_spec = importlib.util.find_spec("_duckdb")
        native_origin = getattr(native_spec, "origin", None)
        if not isinstance(native_origin, str) or not native_origin:
            raise SuccessorOperatorError("installed DuckDB native module is absent")
        native_path = Path(native_origin).resolve(strict=True)
    except (ImportError, importlib.metadata.PackageNotFoundError, OSError) as exc:
        raise SuccessorOperatorError(
            "installed DuckDB runtime cannot be resolved"
        ) from exc
    if (
        re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,63}", version) is None
        or metadata_root.parent != site_root
        or metadata_root.name != f"duckdb-{version}.dist-info"
        or package_root.parent != site_root
        or package_root.name != "duckdb"
        or native_path.parent != site_root
        or not native_path.name.startswith("_duckdb.")
        or not native_path.name.endswith(".so")
    ):
        raise SuccessorOperatorError("installed DuckDB runtime layout differs")
    machine = os.uname().machine if hasattr(os, "uname") else ""
    platform_name = {
        "aarch64": "linux_arm64",
        "x86_64": "linux_amd64",
    }.get(machine, "")
    ambient_home = str(os.environ.get("HOME", "") or "")
    if (
        not platform_name
        or not ambient_home
        or "\x00" in ambient_home
        or not Path(ambient_home).is_absolute()
    ):
        raise SuccessorOperatorError(
            "installed DuckDB extension platform is unavailable"
        )
    extension_root = (
        Path(ambient_home)
        / ".duckdb"
        / "extensions"
        / f"v{version}"
        / platform_name
    )
    try:
        extension_root = extension_root.resolve(strict=True)
    except OSError as exc:
        raise SuccessorOperatorError(
            "installed DuckDB extension directory is unavailable"
        ) from exc
    extensions: dict[str, dict[str, str | Path]] = {}
    for name in ("quack", "ducklake", "httpfs"):
        path = extension_root / f"{name}.duckdb_extension"
        info_path = Path(str(path) + ".info")
        extensions[name] = {
            "path": path,
            "version": _installed_extension_version(info_path, name=name),
        }
    return {
        "version": version,
        "engine_version": f"v{version}",
        "package_root": package_root,
        "metadata_root": metadata_root,
        "native_path": native_path,
        "extension_platform": platform_name,
        "quack_path": extensions["quack"]["path"],
        "quack_version": extensions["quack"]["version"],
        "ducklake_path": extensions["ducklake"]["path"],
        "ducklake_version": extensions["ducklake"]["version"],
        "httpfs_path": extensions["httpfs"]["path"],
        "httpfs_version": extensions["httpfs"]["version"],
    }


def _lgcvf_live_native_authorization_id(
    *,
    native_pin: Any,
    provenance: Mapping[str, Any],
    source_head: str,
    source_tree: str,
    candidate_config_sha256: str,
) -> str:
    """Bind native evidence to this exact source/config/provenance admission."""

    pin_payload = getattr(native_pin, "as_dict", lambda: None)()
    receipt_cid = provenance.get("receipt_cid")
    if (
        not isinstance(pin_payload, Mapping)
        or not isinstance(receipt_cid, str)
        or not receipt_cid
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", candidate_config_sha256)
        is None
    ):
        raise SuccessorOperatorError(
            "LGCVF native launch authorization inputs are incomplete"
        )
    body = {
        "schema": LGCVF_LIVE_NATIVE_AUTHORIZATION_SCHEMA,
        "board_namespace": (
            "logic-governed-compositional-verification-fabric-v1"
        ),
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "source_head": source_head,
        "source_tree": source_tree,
        "candidate_config_path": DEFAULT_SUCCESSOR_CONFIG_RELATIVE.as_posix(),
        "candidate_config_sha256": candidate_config_sha256,
        "successor_provenance_cid": receipt_cid,
        "native_pin": dict(pin_payload),
        "claims": {
            "capsule_exact_match_required": True,
            "parent_loader_environment_sanitized_before_exec": True,
            "quack_extension_install_policy": "load_only",
            "ducklake_authority": False,
        },
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _remove_private_live_capsule_parent(path: Path | None) -> None:
    """Remove only this launch's owner-private temporary capsule directory."""

    if path is None:
        return
    parent = Path(path)
    try:
        temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
        observed_parent = os.lstat(parent)
        if (
            parent.parent.resolve(strict=True) != temporary_root
            or not parent.name.startswith(
                f"lgcvf-live-capsule-{os.geteuid()}-"
            )
            or not stat.S_ISDIR(observed_parent.st_mode)
            or stat.S_ISLNK(observed_parent.st_mode)
            or observed_parent.st_uid != os.geteuid()
        ):
            return
        entries = tuple(parent.rglob("*"))
        if any(
            stat.S_ISLNK(os.lstat(entry).st_mode)
            or os.lstat(entry).st_uid != os.geteuid()
            for entry in entries
        ):
            return
        for directory in sorted(
            (entry for entry in entries if entry.is_dir()),
            key=lambda entry: len(entry.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o700)
        os.chmod(parent, 0o700)
        shutil.rmtree(parent)
    except OSError:
        return


def _child_environment(
    *,
    token: str,
    identity: Any,
    owner_state: Path,
    root: Path,
    rendered_environment: Mapping[str, Any] | None = None,
    launch_home: Path | None = None,
) -> dict[str, str]:
    rendered = dict(rendered_environment or {})
    if (
        TOKEN_ENV in rendered
        or TOKEN_FILE_ENV in rendered
        or not set(rendered).issubset(LGCVF_LIVE_RENDERED_ENV_NAMES)
        or any(
            not isinstance(name, str)
            or not isinstance(value, (str, int, float))
            or "\x00" in str(value)
            for name, value in rendered.items()
        )
    ):
        raise SuccessorOperatorError(
            "configured scheduler rendered a foreign environment field"
        )
    environment = {
        name: str(os.environ[name])
        for name in ("LANG", "LC_ALL", "LC_CTYPE", "TZ")
        if name in os.environ and "\x00" not in str(os.environ[name])
    }
    environment["PATH"] = "/usr/bin:/bin"
    environment.update({str(name): str(value) for name, value in rendered.items()})
    environment[TOKEN_ENV] = token
    environment[TOKEN_FILE_ENV] = str(_token_sink(owner_state))
    environment["IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION"] = str(
        identity.generation
    )
    environment["IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION"] = str(
        identity.schema_revision
    )
    environment["IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT"] = str(root)
    environment[LEGACY_BOARD_UNSTALL_POLICY_ENV] = "disabled"
    environment[BOARD_EXTENSION_INSTALL_POLICY_ENV] = (
        BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
    )
    home = Path(launch_home) if launch_home is not None else owner_state
    environment["HOME"] = str(home)
    if launch_home is not None:
        environment["IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME"] = str(home)
        environment["XDG_CACHE_HOME"] = str(home / ".cache" / "xdg")
        environment["CUDA_CACHE_PATH"] = str(home / ".cache" / "cuda")
        environment["CUDA_CACHE_DISABLE"] = "1"
    if any(
        name.startswith(("LD_", "PYTHON")) or name == "GLIBC_TUNABLES"
        for name in environment
    ):
        raise SuccessorOperatorError(
            "scheduler environment retained ambient loader or Python authority"
        )
    return environment


def _exact_birth(pid: int) -> Any:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        read_process_birth,
    )

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        birth = read_process_birth(pid)
        if birth is not None:
            return birth
        time.sleep(0.01)
    raise SuccessorOperatorError("could not capture scheduler process birth")


def _terminate_exact(
    birth: Any,
    *,
    grace_seconds: float = 10.0,
    child_process: subprocess.Popen[Any] | None = None,
) -> str:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        owner_liveness,
    )

    def send(signum: int) -> None:
        if child_process is not None and child_process.poll() is not None:
            return
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return
        if state is not OwnerLiveness.ALIVE:
            raise SuccessorOperatorError("scheduler birth is uninspectable")
        if birth.pid <= 1:
            raise SuccessorOperatorError("refusing to signal an unsafe PID")
        try:
            group = os.getpgid(birth.pid)
            if group == birth.pid:
                os.killpg(group, signum)
            else:
                os.kill(birth.pid, signum)
        except ProcessLookupError:
            return

    if child_process is not None:
        if child_process.pid != birth.pid:
            raise SuccessorOperatorError("scheduler child differs from its birth")
        if child_process.poll() is not None:
            return "already_dead"
    if owner_liveness(birth) is OwnerLiveness.DEAD:
        return "already_dead"
    send(signal.SIGTERM)
    deadline = time.monotonic() + max(0.1, grace_seconds)
    while time.monotonic() < deadline:
        if child_process is not None and child_process.poll() is not None:
            return "terminated"
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return "terminated"
        if state is OwnerLiveness.UNKNOWN:
            raise SuccessorOperatorError("scheduler became uninspectable during stop")
        time.sleep(0.05)
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if child_process is not None and child_process.poll() is not None:
            return "killed"
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return "killed"
        if state is OwnerLiveness.UNKNOWN:
            raise SuccessorOperatorError("scheduler became uninspectable after kill")
        time.sleep(0.05)
    raise SuccessorOperatorError("exact scheduler birth survived bounded stop")


def run_successor(
    config_path: Path,
    *,
    root: Path = ROOT,
    implement: bool,
    duration_seconds: float,
) -> int:
    paths = _paths(root)
    lock_custody = _open_generation_bound_controller_lock(paths)
    lock_handle = lock_custody["lock_handle"]
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuccessorOperatorError(
                "another successor controller owns the lock"
            ) from exc
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        recovered = _automatically_recover_abandoned_owner_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
        )
        if recovered is not None:
            raise SuccessorOperatorError(
                "abandoned state owner recovered; restart the successor "
                "controller against the new continuity receipt"
            )
        protected_publication = (
            _recover_interrupted_protected_qualification_publication_locked(
                paths,
                root=root,
                lock_custody=lock_custody,
            )
        )
        if protected_publication is not None:
            raise SuccessorOperatorError(
                "protected qualification stopped-state publication recovered; "
                "restart the successor controller against the new continuity "
                "receipt"
            )
        result = _run_locked_successor(
            config_path,
            root=root,
            implement=implement,
            duration_seconds=duration_seconds,
            _locked_paths=paths,
            _lock_custody=lock_custody,
        )
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        return result
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            _close_generation_bound_controller_lock(lock_custody)


def _preload_lgcvf_live_controller_dependency_closure() -> tuple[str, ...]:
    """Import every repository module the live controller can call later."""

    loaded: list[str] = []
    for module_name in LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise SuccessorOperatorError(
                "LGCVF live controller dependency closure is unavailable: "
                f"{module_name}: {type(exc).__name__}"
            ) from exc
        if getattr(module, "__name__", None) != module_name:
            raise SuccessorOperatorError(
                "LGCVF live controller dependency identity differs: "
                f"{module_name}"
            )
        loaded.append(module_name)
    return tuple(loaded)


def _lgcvf_live_module_expected_members(module_name: str) -> frozenset[str]:
    """Return the only capsule members allowed to implement one module name."""

    if module_name == "ipfs_datasets_py":
        return frozenset(
            {
                "ipfs_datasets_py/__init__.py",
                "ipfs_datasets_py/ipfs_datasets_py/__init__.py",
            }
        )
    if module_name.startswith("ipfs_datasets_py."):
        stem = "ipfs_datasets_py/" + module_name.replace(".", "/")
    elif any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in ("ipfs_accelerate_py", "scripts")
    ):
        stem = module_name.replace(".", "/")
    else:
        return frozenset()
    return frozenset({stem + ".py", stem + "/__init__.py"})


def _lgcvf_live_manifest_member(
    relative: str,
    *,
    manifest_files: Mapping[str, str],
    read_member: Any,
    noun: str,
) -> bytes:
    """Read one member and bind it to the authenticated manifest digest."""

    digest = manifest_files.get(relative)
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
    ):
        raise SuccessorOperatorError(f"{noun} is absent from the sealed capsule")
    try:
        raw = read_member(relative)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SuccessorOperatorError(
            f"{noun} cannot be read from the sealed capsule"
        ) from exc
    if (
        type(raw) is not bytes
        or len(raw) > 64 * 1024 * 1024
        or "sha256:" + hashlib.sha256(raw).hexdigest() != digest
    ):
        raise SuccessorOperatorError(f"{noun} sealed capsule bytes differ")
    return raw


def _lgcvf_live_sealed_manifest_inventory(
    capsule_pin: Any,
    capsule_descriptor: int,
) -> tuple[str, dict[str, str]]:
    """Read the already-verified manifest from the immutable archive itself."""

    from ipfs_accelerate_py.agent_implementation_route import (
        verify_lgcvf_configured_board_live_sealed_capsule,
    )

    try:
        archive_path = verify_lgcvf_configured_board_live_sealed_capsule(
            capsule_pin,
            capsule_descriptor,
        )
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest_raw = archive.read(LGCVF_LIVE_CAPSULE_MANIFEST_MEMBER)
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise SuccessorOperatorError(
            "LGCVF live sealed capsule manifest is unavailable"
        ) from exc
    if not 0 < len(manifest_raw) <= 8 * 1024 * 1024:
        raise SuccessorOperatorError(
            "LGCVF live sealed capsule manifest is out of bounds"
        )

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = value
        return result

    try:
        manifest = json.loads(
            manifest_raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
        canonical = (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(
            "LGCVF live sealed capsule manifest is invalid"
        ) from exc
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if (
        manifest_raw != canonical
        or not isinstance(files, dict)
        or manifest.get("capsule_id") != getattr(capsule_pin, "capsule_id", None)
        or manifest.get("operator_path")
        != getattr(capsule_pin, "operator_path", None)
        or manifest.get("operator_sha256")
        != getattr(capsule_pin, "operator_sha256", None)
        or manifest.get("candidate_config_path")
        != getattr(capsule_pin, "candidate_config_path", None)
        or manifest.get("candidate_config_sha256")
        != getattr(capsule_pin, "candidate_config_sha256", None)
    ):
        raise SuccessorOperatorError(
            "LGCVF live sealed capsule manifest identity differs"
        )
    normalized: dict[str, str] = {}
    for relative, digest in files.items():
        path = Path(str(relative))
        if (
            not isinstance(relative, str)
            or not relative
            or path.is_absolute()
            or ".." in path.parts
            or relative != path.as_posix()
            or not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise SuccessorOperatorError(
                "LGCVF live sealed capsule manifest inventory differs"
            )
        normalized[relative] = digest
    return archive_path, normalized


def _audit_lgcvf_live_loaded_repository_modules(
    *,
    root: Path,
    operator_path: Path,
    manifest_files: Mapping[str, str],
    read_member: Any,
    modules: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Bind loaded repository source origins and current bytes to the capsule."""

    try:
        exact_root = root.resolve(strict=True)
        exact_datasets_root = (exact_root / "ipfs_datasets_py").resolve(
            strict=True
        )
        exact_operator = operator_path.resolve(strict=True)
    except OSError as exc:
        raise SuccessorOperatorError(
            "LGCVF live loaded-source roots are unavailable"
        ) from exc
    expected_operator = exact_root / (
        "scripts/run_logic_governed_compositional_verification_fabric_quack.py"
    )
    if exact_operator != expected_operator:
        raise SuccessorOperatorError(
            "LGCVF live outer operator origin differs"
        )
    operator_member = expected_operator.relative_to(exact_root).as_posix()
    sealed_operator = _lgcvf_live_manifest_member(
        operator_member,
        manifest_files=manifest_files,
        read_member=read_member,
        noun="LGCVF live outer operator",
    )
    current_operator = _read_bounded_regular_file(
        exact_operator,
        max_bytes=64 * 1024 * 1024,
        noun="LGCVF live outer operator",
    )
    if current_operator != sealed_operator:
        raise SuccessorOperatorError(
            "LGCVF live outer operator bytes differ from the sealed capsule"
        )

    module_table = sys.modules if modules is None else modules
    audited: list[str] = []
    member_owners: dict[str, tuple[str, Any]] = {}
    for module_name, module in sorted(module_table.items()):
        namespace_member = any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in LGCVF_LIVE_REPOSITORY_MODULE_PREFIXES
        )
        module_file = getattr(module, "__file__", None) if module is not None else None
        if isinstance(module_file, str):
            lexical_file = Path(module_file)
            under_repository = False
            if lexical_file.is_absolute():
                try:
                    lexical_file.relative_to(exact_root)
                    under_repository = True
                except ValueError:
                    pass
        else:
            lexical_file = None
            under_repository = False
        if not namespace_member and not under_repository:
            continue
        if module is None or lexical_file is None:
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin is invalid: {module_name}"
            )
        if module_name == "__main__":
            try:
                main_origin = lexical_file.resolve(strict=True)
            except OSError as exc:
                raise SuccessorOperatorError(
                    "LGCVF live outer operator module origin is unavailable"
                ) from exc
            if main_origin == exact_operator:
                continue
        if not lexical_file.is_absolute():
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin is invalid: {module_name}"
            )
        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        if not isinstance(spec_origin, str) or Path(spec_origin) != lexical_file:
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin differs: {module_name}"
            )
        try:
            exact_file = lexical_file.resolve(strict=True)
        except OSError as exc:
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin is unavailable: {module_name}"
            ) from exc
        if exact_file != lexical_file:
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin contains a link: {module_name}"
            )
        if exact_file == exact_operator:
            continue
        try:
            nested_relative = exact_file.relative_to(exact_datasets_root)
        except ValueError:
            try:
                relative = exact_file.relative_to(exact_root).as_posix()
            except ValueError as exc:
                raise SuccessorOperatorError(
                    f"LGCVF live loaded repository module escaped the source root: {module_name}"
                ) from exc
        else:
            relative = (
                Path("ipfs_datasets_py") / nested_relative
            ).as_posix()
        previous_owner = member_owners.setdefault(relative, (module_name, module))
        if previous_owner[1] is not module:
            raise SuccessorOperatorError(
                "LGCVF live loaded repository module origin is aliased"
            )
        sealed_source = _lgcvf_live_manifest_member(
            relative,
            manifest_files=manifest_files,
            read_member=read_member,
            noun=f"LGCVF live loaded repository module {module_name}",
        )
        current_source = _read_bounded_regular_file(
            exact_file,
            max_bytes=64 * 1024 * 1024,
            noun=f"LGCVF live loaded repository module {module_name}",
        )
        if current_source != sealed_source:
            raise SuccessorOperatorError(
                "LGCVF live loaded repository module bytes differ from the "
                f"sealed capsule: {module_name}"
            )
        audited.append(module_name)
    if not set(LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES).issubset(audited):
        missing = sorted(set(LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES) - set(audited))
        raise SuccessorOperatorError(
            "LGCVF live controller dependency closure is incomplete: "
            + ", ".join(missing[:3])
        )
    return tuple(audited)


def _retarget_lgcvf_live_repository_imports(
    *,
    root: Path,
    archive_path: str,
    modules: Mapping[str, Any] | None = None,
    path_entries: list[str] | None = None,
    meta_path: list[Any] | None = None,
) -> tuple[str, ...]:
    """Remove mutable repository import roots and project package paths to ZIP."""

    exact_root = root.resolve(strict=True)
    if (
        not archive_path.startswith("/proc/self/fd/")
        or not Path(archive_path).is_absolute()
    ):
        raise SuccessorOperatorError("LGCVF live sealed import root is invalid")
    capsule_roots = (
        archive_path + "/ipfs_datasets_py",
        archive_path,
    )
    target_path = sys.path if path_entries is None else path_entries
    retained: list[str] = []
    for entry in target_path:
        if entry in capsule_roots or not isinstance(entry, str) or not entry:
            continue
        if entry.startswith("__editable__.") or not Path(entry).is_absolute():
            continue
        try:
            resolved = Path(entry).resolve(strict=False)
            resolved.relative_to(exact_root)
        except ValueError:
            retained.append(entry)
        else:
            continue
    target_path[:] = list(dict.fromkeys((*capsule_roots, *retained)))

    allowed_meta = (
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    )
    target_meta = sys.meta_path if meta_path is None else meta_path
    if any(finder not in target_meta for finder in allowed_meta):
        raise SuccessorOperatorError(
            "LGCVF live standard import machinery is unavailable"
        )
    target_meta[:] = list(allowed_meta)

    module_table = sys.modules if modules is None else modules
    retargeted: list[str] = []
    for module_name, module in sorted(module_table.items()):
        if module is None:
            continue
        package_path = getattr(module, "__path__", None)
        if package_path is None:
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not Path(module_file).is_absolute():
            continue
        try:
            exact_file = Path(module_file).resolve(strict=True)
            nested_relative = exact_file.relative_to(
                exact_root / "ipfs_datasets_py"
            )
        except ValueError:
            try:
                relative = exact_file.relative_to(exact_root)
            except (OSError, ValueError):
                continue
        except OSError:
            continue
        else:
            relative = Path("ipfs_datasets_py") / nested_relative
        if module_name == "ipfs_datasets_py":
            sealed_package = archive_path + "/ipfs_datasets_py/ipfs_datasets_py"
        else:
            sealed_package = archive_path + "/" + relative.parent.as_posix()
        projected = [sealed_package]
        module.__path__ = projected
        spec = getattr(module, "__spec__", None)
        if spec is None or getattr(spec, "submodule_search_locations", None) is None:
            raise SuccessorOperatorError(
                f"LGCVF live loaded package spec differs: {module_name}"
            )
        spec.submodule_search_locations = projected
        retargeted.append(module_name)

    if path_entries is None and modules is None:
        sys.path_importer_cache.clear()
        importlib.invalidate_caches()
    if tuple(target_path[:2]) != capsule_roots:
        raise SuccessorOperatorError("LGCVF live sealed import roots drifted")
    return tuple(retargeted)


def _restore_lgcvf_stopped_candidate_import_boundary(
    *,
    root: Path,
    sealed_import_roots: Sequence[str],
) -> None:
    """Restore the admitted candidate roots after the sealed live child stops.

    Live admission intentionally removes the mutable worktree from ``sys.path``.
    A clean-stop continuity observation, however, must inspect the final board
    tree after admitted merges.  Restore only after proving that the current
    roots are the exact two sealed capsule roots installed by this launch.
    Repository package objects remain projected to the now-closed capsule, so
    this boundary transition cannot import fresh candidate code while the
    controller is unwinding.
    """

    exact_root = root.resolve(strict=True)
    expected = tuple(str(value) for value in sealed_import_roots)
    if (
        len(expected) != 2
        or tuple(sys.path[:2]) != expected
        or any(
            not value.startswith("/proc/self/fd/")
            or not Path(value).is_absolute()
            for value in expected
        )
        or sys.pycache_prefix != _RUNTIME_PYCACHE.name
    ):
        raise SuccessorOperatorError(
            "LGCVF stopped sealed import boundary differs"
        )
    candidate_roots = (str(exact_root), str(exact_root / "ipfs_datasets_py"))
    retained = [
        entry
        for entry in sys.path[2:]
        if isinstance(entry, str)
        and entry
        and entry not in expected
        and entry not in candidate_roots
    ]
    sys.path[:] = [*candidate_roots, *retained]
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()
    if tuple(sys.path[:2]) != candidate_roots:
        raise SuccessorOperatorError(
            "LGCVF stopped candidate import boundary was not restored"
        )


def _prepare_lgcvf_configured_board_live_launch(
    *,
    root: Path,
    config_path: Path,
    provenance: Mapping[str, Any],
    stopped_restart: bool = False,
) -> dict[str, Any]:
    """Materialize and authenticate every byte needed by the live child.

    This function has no database or Quack-owner effect.  It may create only a
    private content-addressed capsule, sealed anonymous descriptors, and the
    verified load-only extension HOME.  It deliberately leaves the candidate
    import boundary intact for provenance verification.  Its caller must close
    both descriptors on every exit path.
    """

    expected_config = _contained(root, DEFAULT_SUCCESSOR_CONFIG_RELATIVE)
    try:
        exact_config = config_path.resolve(strict=True)
    except OSError as exc:
        raise SuccessorOperatorError(
            "LGCVF live candidate config is unavailable"
        ) from exc
    if exact_config != expected_config:
        raise SuccessorOperatorError(
            "LGCVF live capsule requires the exact candidate config"
        )
    def observe_continuity() -> dict[str, Any]:
        if stopped_restart:
            return _observe_candidate_runtime_continuity(
                root,
                require_resolved_remote=False,
            )
        return _candidate_runtime_continuity(root)

    continuity = observe_continuity()
    source_head = str(continuity.get("current_head") or "")
    source_tree = str(continuity.get("current_tree") or "")
    config_raw = _read_bounded_regular_file(
        exact_config,
        max_bytes=4 * 1024 * 1024,
        noun="LGCVF live candidate config",
    )
    candidate_config_sha256 = (
        "sha256:" + hashlib.sha256(config_raw).hexdigest()
    )
    runtime = _resolve_installed_duckdb_live_runtime()

    from ipfs_accelerate_py.agent_implementation_route import (
        materialize_lgcvf_configured_board_live_capsule,
        project_lgcvf_configured_board_live_extensions,
        seal_lgcvf_configured_board_live_capsule,
        verify_lgcvf_configured_board_live_sealed_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        build_lgcvf_configured_board_live_admission,
        verify_lgcvf_configured_board_live_context,
    )
    from ipfs_accelerate_py.llm_router import (
        inspect_agent_supervisor_native_dependency_source,
        seal_agent_supervisor_native_dependency,
        verify_agent_supervisor_native_dependency_sealed_fd,
    )

    native_pin = inspect_agent_supervisor_native_dependency_source(
        runtime["native_path"],
        distribution_version=str(runtime["version"]),
        engine_version=str(runtime["engine_version"]),
    )
    native_authorization_id = _lgcvf_live_native_authorization_id(
        native_pin=native_pin,
        provenance=provenance,
        source_head=source_head,
        source_tree=source_tree,
        candidate_config_sha256=candidate_config_sha256,
    )
    capsule_parent = Path(
        tempfile.mkdtemp(
            prefix=f"lgcvf-live-capsule-{os.geteuid()}-",
            dir=tempfile.gettempdir(),
        )
    )
    os.chmod(capsule_parent, 0o700)
    capsule = None
    native_launch = None
    try:
        capsule_pin = materialize_lgcvf_configured_board_live_capsule(
            source_root=root,
            capsule_parent=capsule_parent,
            source_head=source_head,
            source_tree=source_tree,
            python_executable=sys.executable,
            duckdb_package_root=runtime["package_root"],
            duckdb_distribution_metadata_root=runtime["metadata_root"],
            duckdb_distribution_version=str(runtime["version"]),
            quack_extension_path=runtime["quack_path"],
            quack_extension_version=str(runtime["quack_version"]),
            ducklake_extension_path=runtime["ducklake_path"],
            ducklake_extension_version=str(runtime["ducklake_version"]),
            httpfs_extension_path=runtime["httpfs_path"],
            httpfs_extension_version=str(runtime["httpfs_version"]),
            native_authorization_id=native_authorization_id,
            native_dependency_id=native_pin.dependency_id,
        )
        if (
            capsule_pin.source_head != source_head
            or capsule_pin.source_tree != source_tree
            or capsule_pin.candidate_config_sha256
            != candidate_config_sha256
            or capsule_pin.native_authorization_id
            != native_authorization_id
            or capsule_pin.native_dependency_id != native_pin.dependency_id
            or capsule_pin.duckdb_distribution_version
            != runtime["version"]
            or capsule_pin.quack_extension.version
            != runtime["quack_version"]
            or capsule_pin.ducklake_extension.version
            != runtime["ducklake_version"]
            or capsule_pin.httpfs_extension.version
            != runtime["httpfs_version"]
        ):
            raise SuccessorOperatorError(
                "LGCVF live capsule differs from its controller admission"
            )
        capsule = seal_lgcvf_configured_board_live_capsule(capsule_pin)
        if (
            verify_lgcvf_configured_board_live_sealed_capsule(
                capsule_pin,
                capsule.descriptor,
            )
            != capsule.executable_path
        ):
            raise SuccessorOperatorError(
                "LGCVF live capsule sealed descriptor drifted"
            )
        native_launch = seal_agent_supervisor_native_dependency(
            runtime["native_path"],
            expected_pin=native_pin,
            accepted_authorization_id=native_authorization_id,
        )
        if (
            native_launch.pin != native_pin
            or native_launch.accepted_authorization_id
            != native_authorization_id
            or verify_agent_supervisor_native_dependency_sealed_fd(
                native_launch
            )
            != f"/proc/self/fd/{native_launch.descriptor.descriptor}"
        ):
            raise SuccessorOperatorError(
                "LGCVF native dependency sealed descriptor drifted"
            )
        admission = build_lgcvf_configured_board_live_admission(
            capsule_pin,
            native_launch,
        )
        capsule_pin_json = capsule_pin.to_json()
        admission_json = admission.to_json()
        native_launch_json = native_launch.to_json()
        context = verify_lgcvf_configured_board_live_context(
            capsule_pin_json=capsule_pin_json,
            capsule_descriptor=capsule.descriptor,
            admission_json=admission_json,
            native_launch_json=native_launch_json,
            native_descriptor=native_launch.descriptor.descriptor,
        )
        if context.admission != admission:
            raise SuccessorOperatorError(
                "LGCVF live capsule/native admission join drifted"
            )
        qualification_parent = _contained(
            root,
            LGCVF_LIVE_QUALIFICATION_HOMES_RELATIVE,
        )
        launch_home = project_lgcvf_configured_board_live_extensions(
            capsule_pin,
            capsule.descriptor,
            qualification_parent,
        )
        expected_home = qualification_parent / capsule_pin.capsule_id.removeprefix(
            "sha256:"
        )
        if launch_home != expected_home:
            raise SuccessorOperatorError(
                "LGCVF live extension HOME identity drifted"
            )
        preloaded_modules = _preload_lgcvf_live_controller_dependency_closure()
        archive_path, manifest_files = _lgcvf_live_sealed_manifest_inventory(
            capsule_pin,
            capsule.descriptor,
        )
        try:
            with zipfile.ZipFile(archive_path, mode="r") as archive:
                sealed_config_raw = _lgcvf_live_manifest_member(
                    capsule_pin.candidate_config_path,
                    manifest_files=manifest_files,
                    read_member=archive.read,
                    noun="LGCVF live candidate config",
                )
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise SuccessorOperatorError(
                "LGCVF live sealed controller closure is unreadable"
            ) from exc
        if sealed_config_raw != config_raw:
            raise SuccessorOperatorError(
                "LGCVF live candidate config differs from the sealed capsule"
            )
        board, program, host, port = _validate_successor_board(
            exact_config,
            root,
            config_bytes=sealed_config_raw,
            admitted_live_validator_sha256=capsule_pin.validator_sha256,
        )
        try:
            with zipfile.ZipFile(archive_path, mode="r") as archive:
                audited_modules = _audit_lgcvf_live_loaded_repository_modules(
                    root=root,
                    operator_path=Path(__file__),
                    manifest_files=manifest_files,
                    read_member=archive.read,
                )
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise SuccessorOperatorError(
                "LGCVF live sealed controller closure is unreadable"
            ) from exc
        final_continuity = observe_continuity()
        if final_continuity != continuity:
            raise SuccessorOperatorError(
                "LGCVF live source changed after controller closure admission"
            )
        return {
            "capsule_parent": capsule_parent,
            "capsule_pin": capsule_pin,
            "capsule": capsule,
            "capsule_pin_json": capsule_pin_json,
            "admission": admission,
            "admission_json": admission_json,
            "native_launch": native_launch,
            "native_launch_json": native_launch_json,
            "launch_home": launch_home,
            "pass_fds": context.pass_fds,
            "board": board,
            "program": program,
            "host": host,
            "port": port,
            "sealed_config_raw": sealed_config_raw,
            "archive_path": archive_path,
            "continuity": dict(continuity),
            "stopped_restart": stopped_restart,
            "preloaded_modules": preloaded_modules,
            "audited_modules": audited_modules,
        }
    except BaseException:
        if native_launch is not None:
            try:
                os.close(native_launch.descriptor.descriptor)
            except OSError:
                pass
        if capsule is not None:
            try:
                os.close(capsule.descriptor)
            except OSError:
                pass
        _remove_private_live_capsule_parent(capsule_parent)
        raise


def _verify_lgcvf_live_provenance_before_import_retarget(
    *,
    paths: Mapping[str, Path],
    root: Path,
    raw_provenance: Mapping[str, Any],
    live_launch: Mapping[str, Any],
    lock_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify state, re-audit newly loaded source, then seal import routing."""

    stopped_admission: Mapping[str, Any] | None = None
    if live_launch.get("stopped_restart") is True:
        stopped_admission = _load_stopped_restart_admission(
            paths,
            root=root,
            provenance=raw_provenance,
            lock_custody=lock_custody,
        )
        provenance = dict(stopped_admission["provenance"])
    else:
        provenance = _load_provenance(
            paths,
            root=root,
            expected_receipt=raw_provenance,
        )
    if provenance != raw_provenance:
        raise SuccessorOperatorError(
            "verified successor provenance differs from native authorization"
        )
    archive_path = live_launch.get("archive_path")
    admitted_continuity = live_launch.get("continuity")
    capsule_pin = live_launch.get("capsule_pin")
    capsule_descriptor = getattr(live_launch.get("capsule"), "descriptor", None)
    if (
        not isinstance(archive_path, str)
        or not isinstance(admitted_continuity, Mapping)
        or type(capsule_descriptor) is not int
        or capsule_descriptor < 3
    ):
        raise SuccessorOperatorError(
            "LGCVF live controller admission is incomplete before import sealing"
        )
    post_provenance_preloaded_modules = (
        _preload_lgcvf_live_controller_dependency_closure()
    )
    refreshed_archive_path, manifest_files = (
        _lgcvf_live_sealed_manifest_inventory(
            capsule_pin,
            capsule_descriptor,
        )
    )
    if refreshed_archive_path != archive_path:
        raise SuccessorOperatorError(
            "LGCVF live sealed import root changed during provenance verification"
        )
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            audited_modules = _audit_lgcvf_live_loaded_repository_modules(
                root=root,
                operator_path=Path(__file__),
                manifest_files=manifest_files,
                read_member=archive.read,
            )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise SuccessorOperatorError(
            "LGCVF live sealed controller closure is unreadable"
        ) from exc
    final_continuity = (
        _observe_candidate_runtime_continuity(
            root,
            require_resolved_remote=False,
        )
        if live_launch.get("stopped_restart") is True
        else _candidate_runtime_continuity(root)
    )
    if final_continuity != admitted_continuity:
        raise SuccessorOperatorError(
            "LGCVF live source changed during provenance verification"
        )
    retargeted_packages = _retarget_lgcvf_live_repository_imports(
        root=root,
        archive_path=archive_path,
    )
    result = {
        "provenance": provenance,
        "preloaded_modules": post_provenance_preloaded_modules,
        "audited_modules": audited_modules,
        "retargeted_packages": retargeted_packages,
    }
    if stopped_admission is not None:
        receipt = stopped_admission["receipt"]
        status = stopped_admission["controller_status"]
        assert isinstance(receipt, Mapping)
        assert isinstance(status, Mapping)
        result.update(
            {
                "stopped_restart_receipt_cid": str(receipt["receipt_cid"]),
                "stopped_restart_controller_status_cid": str(
                    status["status_cid"]
                ),
            }
        )
    return result


def _close_lgcvf_configured_board_live_launch(
    launch: Mapping[str, Any] | None,
) -> None:
    """Close only the two controller-owned sealed descriptors and capsule."""

    if not launch:
        return
    descriptors = {
        int(getattr(launch.get("capsule"), "descriptor", -1)),
        int(
            getattr(
                getattr(launch.get("native_launch"), "descriptor", None),
                "descriptor",
                -1,
            )
        ),
    }
    for descriptor in descriptors:
        if descriptor >= 3:
            try:
                os.close(descriptor)
            except OSError:
                pass
    _remove_private_live_capsule_parent(launch.get("capsule_parent"))


def _run_locked_successor(
    config_path: Path,
    *,
    root: Path,
    implement: bool,
    duration_seconds: float,
    _locked_paths: Mapping[str, Path] | None = None,
    _lock_custody: Mapping[str, Any] | None = None,
) -> int:
    if (_locked_paths is None) != (_lock_custody is None):
        raise SuccessorOperatorError(
            "locked successor generation custody is incomplete"
        )
    paths = dict(_locked_paths) if _locked_paths is not None else _paths(root)
    recovery_io_paths = _stopped_recovery_io_paths(paths, _lock_custody)
    receipt_io_paths = _stopped_receipt_io_view(paths, recovery_io_paths)

    def revalidate_runtime_generation() -> None:
        if _lock_custody is not None:
            _revalidate_generation_bound_controller_lock(paths, _lock_custody)

    revalidate_runtime_generation()
    raw_provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(recovery_io_paths["provenance"]),
    )
    stopped_restart = any(
        os.path.lexists(recovery_io_paths[name])
        for name in (
            "stopped_state_continuity",
            "stopped_state_restart_admission",
        )
    )
    if (
        not stopped_restart
        and os.path.lexists(recovery_io_paths["controller_status"])
    ):
        stopped_status_probe = _strict_json(
            Path(recovery_io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        )
        stopped_restart = (
            stopped_status_probe.get("lifecycle") == "stopped"
            and (
                (
                    stopped_status_probe.get("error") == ""
                    and type(
                        stopped_status_probe.get("scheduler_returncode")
                    )
                    is int
                    and stopped_status_probe.get("scheduler_returncode") == 0
                )
                or stopped_status_probe.get("error")
                == FAILED_START_STATUS_ERROR
            )
        )
    live_launch = _prepare_lgcvf_configured_board_live_launch(
        root=root,
        config_path=config_path,
        provenance=raw_provenance,
        stopped_restart=stopped_restart,
    )
    revalidate_runtime_generation()
    server: Any | None = None
    bootstrap_channel: socket.socket | None = None
    bootstrap_broker: _LgcvfStateOwnerBootstrapBroker | None = None
    previous_extension_environment: dict[str, str | None] = {}
    stop_requested = False
    prior_handlers: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    def stop_owner() -> Mapping[str, Any]:
        nonlocal server
        if server is None:
            return {"stopped": True, "already_stopped": True}
        owned_server = server
        server = None
        return owned_server.stop()

    def stop_bootstrap_broker() -> None:
        nonlocal bootstrap_broker, bootstrap_channel
        if bootstrap_broker is not None:
            owned_broker = bootstrap_broker
            owned_broker.stop()
            bootstrap_broker = None
            bootstrap_channel = None
            return
        if bootstrap_channel is not None:
            owned_channel = bootstrap_channel
            bootstrap_channel = None
            try:
                owned_channel.close()
            except OSError:
                pass

    try:
        # Receipt custody and owner startup begin before lane bootstrap.  Install
        # cooperative handlers first so an operator stop cannot bypass the
        # checkpoint/owner-stop/continuity finally path during that window.
        for signum in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[signum] = signal.signal(signum, request_stop)
        launch_home = Path(live_launch["launch_home"])
        extension_environment = {
            "HOME": str(launch_home),
            "IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME": str(launch_home),
            "XDG_CACHE_HOME": str(launch_home / ".cache" / "xdg"),
            "CUDA_CACHE_PATH": str(launch_home / ".cache" / "cuda"),
            "CUDA_CACHE_DISABLE": "1",
            BOARD_EXTENSION_INSTALL_POLICY_ENV: (
                BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
            ),
            STORE_GENERATION_ENV: SUCCESSOR_STORE_GENERATION,
        }
        previous_extension_environment.update(
            {name: os.environ.get(name) for name in extension_environment}
        )
        forbidden_loader_environment = {
            name
            for name in os.environ
            if name.startswith("LD_") or name == "GLIBC_TUNABLES"
        }
        if forbidden_loader_environment:
            raise SuccessorOperatorError(
                "LGCVF native owner inherited ambient loader authority"
            )
        os.environ.update(extension_environment)
        from ipfs_accelerate_py.llm_router import (
            preload_agent_supervisor_native_dependency,
        )

        preload_agent_supervisor_native_dependency(live_launch["native_launch"])
        if _lock_custody is not None:
            protected_completion = (
                _automatically_complete_protected_qualification_locked(
                    paths,
                    root=root,
                    lock_custody=_lock_custody,
                )
            )
            if protected_completion is not None:
                raise SuccessorOperatorError(
                    "protected LGCVF-113 qualification completed; restart the "
                    "successor controller against the new continuity receipt"
                )
        # Recovery may publish restart authority only after the configured
        # capsule and native runtime have passed their read-only preparation.
        # Existing receipts remain exact-source authority: later maintenance
        # descendants require a separately reviewed stop/reseal admission.
        revalidate_runtime_generation()
        recovery_status = (
            _strict_json(
                Path(recovery_io_paths["controller_status"]),
                expected_schema=CONTROLLER_STATUS_SCHEMA,
                require_private_owner=True,
            )
            if os.path.lexists(recovery_io_paths["controller_status"])
            else {}
        )
        failed_status_unbound = (
            recovery_status.get("error") == FAILED_START_STATUS_ERROR
            and "stopped_state_continuity_receipt_cid" not in recovery_status
            and "stopped_state_continuity_status_cid" not in recovery_status
        )
        if _lock_custody is not None:
            if failed_status_unbound:
                _recover_interrupted_failed_start_continuity(
                    paths,
                    root=root,
                    lock_custody=_lock_custody,
                    provenance=raw_provenance,
                )
            else:
                _restore_or_retire_stopped_restart_admission(receipt_io_paths)
                if recovery_status.get("error") == FAILED_START_STATUS_ERROR:
                    _recover_interrupted_failed_start_continuity(
                        paths,
                        root=root,
                        lock_custody=_lock_custody,
                        provenance=raw_provenance,
                    )
                else:
                    _recover_interrupted_stopped_state_continuity(
                        paths,
                        root=root,
                        lock_custody=_lock_custody,
                        provenance=raw_provenance,
                    )
        else:
            _restore_or_retire_stopped_restart_admission(receipt_io_paths)
        if stopped_restart and not os.path.lexists(
            recovery_io_paths["stopped_state_continuity"]
        ):
            raise SuccessorOperatorError(
                "stopped restart authority is unavailable after recovery"
            )
        revalidate_runtime_generation()
        sealed_admission = _verify_lgcvf_live_provenance_before_import_retarget(
            paths=paths,
            root=root,
            raw_provenance=raw_provenance,
            live_launch=live_launch,
            lock_custody=_lock_custody,
        )
        sealed_import_roots = tuple(sys.path[:2])
        provenance = sealed_admission["provenance"]
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            current_process_birth,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            configured_board_launch_plan,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
            establish_state_authority_process_boundary,
            harden_state_authority_process,
        )

        board = live_launch["board"]
        program = live_launch["program"]
        host = str(live_launch["host"])
        port = int(live_launch["port"])
        if program.store_generation != extension_environment[STORE_GENERATION_ENV]:
            raise SuccessorOperatorError(
                "configured board differs from the admitted live generation"
            )
        rendered_plan = configured_board_launch_plan(
            board,
            implement=implement,
            detach=False,
            duration_seconds=duration_seconds,
        )
        rendered_environment = rendered_plan.get("environment")
        if not isinstance(rendered_environment, Mapping):
            raise SuccessorOperatorError(
                "configured scheduler environment is unavailable"
            )
        expected_route_environment = {
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
            "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.6",
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
                "primary_quota_exhausted"
            ),
            "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
            "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
        }
        if any(
            rendered_environment.get(name) != value
            for name, value in expected_route_environment.items()
        ):
            raise SuccessorOperatorError(
                "configured scheduler did not render the reviewed ordered "
                "provider route"
            )
        owner_program_json = str(
            rendered_environment.get(DATABASE_PROGRAM_JSON_ENV) or ""
        ).strip()
        if not owner_program_json:
            raise SuccessorOperatorError(
                "configured scheduler did not render the database program"
            )
        previous_extension_environment.setdefault(
            DATABASE_PROGRAM_JSON_ENV,
            os.environ.get(DATABASE_PROGRAM_JSON_ENV),
        )
        os.environ[DATABASE_PROGRAM_JSON_ENV] = owner_program_json
        bootstrap_channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bootstrap_channel.bind(
            "\0ipfs-lgcvf-bootstrap-" + uuid.uuid4().hex
        )
        bootstrap_channel.listen(8)
        bootstrap_descriptor = bootstrap_channel.fileno()
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            validate_state_owner_bootstrap_listener,
        )

        validate_state_owner_bootstrap_listener(bootstrap_descriptor)
        scheduler_argv = [
            "--repo-root",
            str(root),
            "--config",
            str(config_path),
            "--configured-board-live-capsule-pin-json",
            str(live_launch["capsule_pin_json"]),
            "--configured-board-live-capsule-fd",
            str(live_launch["capsule"].descriptor),
            "--configured-board-live-admission-json",
            str(live_launch["admission_json"]),
            "--configured-board-live-native-launch-json",
            str(live_launch["native_launch_json"]),
            "--configured-board-live-native-fd",
            str(live_launch["native_launch"].descriptor.descriptor),
            "--state-owner-bootstrap-fd",
            str(bootstrap_descriptor),
            "--state-owner-bootstrap-store-id",
            str(program.store_id),
            "launch",
            "--foreground",
            "--duration-seconds",
            str(duration_seconds),
        ]
        if implement:
            scheduler_argv.append("--implement")
        from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
            build_lgcvf_configured_board_live_module_command,
        )

        command = build_lgcvf_configured_board_live_module_command(
            python_executable=sys.executable,
            capsule_pin_json=str(live_launch["capsule_pin_json"]),
            capsule_descriptor=live_launch["capsule"].descriptor,
            admission_json=str(live_launch["admission_json"]),
            native_launch_json=str(live_launch["native_launch_json"]),
            native_descriptor=live_launch["native_launch"].descriptor.descriptor,
            module_name=LGCVF_LIVE_SCHEDULER_MODULE,
            argv=scheduler_argv,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
            build_server,
        )

        # The owner-command dispatcher validates the logical generation
        # independently of the live integer server generation.
        # Establish the kernel boundary before Quack owner construction can
        # mint or stage an attach credential.  The post-mint call below also
        # verifies that the recognized credential-bearing path remains hard.
        establish_state_authority_process_boundary()
        paths["owner_state"].mkdir(mode=0o700, parents=True, exist_ok=True)
        _prepare_private_owner_socket(paths["owner_socket"])
        controller_birth = current_process_birth()
        server = build_server(
            database_path=paths["successor_database"],
            state_dir=paths["owner_state"],
            host=host,
            port=port,
            repository_id="repository:lgcvf-quack-successor",
            store_id=program.store_id,
            secret_handle=program.endpoint_secret_handle,
            migrate=datasets_profile_migration,
            typed_command_socket_path=paths["owner_socket"],
            allow_legacy_board_unstall=False,
        )
        if server.typed_command_socket_path() != paths["owner_socket"]:
            raise SuccessorOperatorError("owner did not retain its short socket path")
        # Preserve the last clean-stop projection evidence through every
        # read-only launch admission check.  Consume it only at the exact
        # boundary where a new owner may begin mutating the generation.
        revalidate_runtime_generation()
        expected_restart_receipt_cid = (
            str(sealed_admission.get("stopped_restart_receipt_cid") or "")
            if stopped_restart
            else ""
        )
        expected_restart_status_cid = (
            str(
                sealed_admission.get(
                    "stopped_restart_controller_status_cid"
                )
                or ""
            )
            if stopped_restart
            else ""
        )
        claimed_restart = _claim_stopped_state_restart_admission(
            receipt_io_paths,
            expected_restart=stopped_restart,
            expected_receipt_cid=expected_restart_receipt_cid,
            expected_controller_status_cid=expected_restart_status_cid,
        )
        if claimed_restart is not stopped_restart:
            raise SuccessorOperatorError(
                "stopped-state restart claim differs from launch admission"
            )
        revalidate_runtime_generation()
        identity = server.start()
        if identity.listen_uri != program.quack_endpoint:
            stop_owner()
            raise SuccessorOperatorError(
                "owner endpoint differs from scheduler program"
            )
        if server._vault is None:
            stop_owner()
            raise SuccessorOperatorError("owner token vault is unavailable")
        token = server._vault.resolve(identity.secret_handle)
        # Harden without copying the credential into the controller environment.
        harden_state_authority_process({TOKEN_ENV: token})
        token_path = paths["owner_state"] / (
            identity.secret_handle.replace(":", "_").replace("/", "_") + ".quack-token"
        )
        owner_status = server.status()
        if (
            token_path.exists()
            or token.encode("ascii") in _canonical_bytes(owner_status)
            or owner_status.get("legacy_board_unstall_enabled") is not False
        ):
            stop_owner()
            raise SuccessorOperatorError(
                "owner credential or typed unstall policy is unsafe"
            )
        execution_route_policy = _seal_lgcvf_execution_route_policy(
            server=server,
            program=program,
            identity=identity,
            controller_birth=controller_birth,
            owner_socket=paths["owner_socket"],
        )

        if any(token in item for item in command):
            stop_owner()
            raise SuccessorOperatorError("scheduler argv would contain the Quack token")
        environment = _child_environment(
            token=token,
            identity=identity,
            owner_state=paths["owner_state"],
            root=root,
            rendered_environment=rendered_environment,
            launch_home=launch_home,
        )
        paths["controller_log"].parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_handle = paths["controller_log"].open("ab")
        os.chmod(paths["controller_log"], 0o600)
        scheduler: subprocess.Popen[Any] | None = None
        scheduler_birth: Any | None = None
        clean_runtime_shutdown = False
        ready_status_published = False

        try:
            scheduler = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                pass_fds=tuple(
                    dict.fromkeys(
                        (*live_launch["pass_fds"], bootstrap_descriptor)
                    )
                ),
                start_new_session=True,
            )
            # Popen returns only after the child has crossed exec.  The child
            # owns inherited references; close the controller copies and drop
            # the disk capsule now so descriptor numbers cannot later be reused
            # and accidentally re-closed by the outer cleanup.
            _close_lgcvf_configured_board_live_launch(live_launch)
            live_launch = None
            scheduler_birth = _exact_birth(scheduler.pid)
            assert bootstrap_channel is not None
            bootstrap_broker = _LgcvfStateOwnerBootstrapBroker(
                channel=bootstrap_channel,
                descriptor=bootstrap_descriptor,
                server=server,
                scheduler_birth=scheduler_birth,
                endpoint=str(program.quack_endpoint),
                socket_path=paths["owner_socket"],
                store_id=str(program.store_id),
                execution_route_policy=execution_route_policy,
            )
            bootstrap_broker.start()
            bootstrap_deadline = (
                time.monotonic()
                + STATE_OWNER_BOOTSTRAP_READY_TIMEOUT_SECONDS
            )
            stable_signature: tuple[str, ...] = ()
            stable_since = 0.0
            while True:
                revalidate_runtime_generation()
                if stop_requested:
                    raise SuccessorOperatorError(
                        "controller stop requested before all lane daemons attached"
                    )
                if scheduler.poll() is not None:
                    raise SuccessorOperatorError(
                        "scheduler exited before all lane daemons attached"
                    )
                if bootstrap_broker.failure:
                    raise SuccessorOperatorError(
                        "lane state-owner bootstrap failed closed: "
                        + bootstrap_broker.failure
                    )
                if time.monotonic() >= bootstrap_deadline:
                    raise SuccessorOperatorError(
                        "lane state-owner bootstrap readiness timed out"
                    )
                observed_signature = bootstrap_broker.live_ready_signature
                if observed_signature != stable_signature:
                    stable_signature = observed_signature
                    stable_since = time.monotonic()
                if (
                    len(stable_signature)
                    == len(LGCVF_DATABASE_OWNER_SESSIONS)
                    and time.monotonic() - stable_since
                    >= STATE_OWNER_BOOTSTRAP_STABILITY_SECONDS
                ):
                    break
                server.service_mutation_inbox(max_requests=32)
                time.sleep(0.01)
            ready_status = _status_payload(
                lifecycle="ready",
                controller_birth=controller_birth.to_dict(),
                provenance_cid=str(provenance["receipt_cid"]),
                owner_identity=identity.to_dict(),
                scheduler_birth=scheduler_birth.to_dict(),
                projection_root=paths["projection_root"],
            )
            revalidate_runtime_generation()
            _write_status(paths["controller_status"], ready_status, token=token)
            ready_status_published = True
            started = time.monotonic()
            pump_error = ""
            while scheduler.poll() is None and not stop_requested:
                revalidate_runtime_generation()
                if (
                    bootstrap_broker is None
                    or bootstrap_broker.failure
                ):
                    pump_error = (
                        "state-owner bootstrap broker failed: "
                        + (
                            "missing"
                            if bootstrap_broker is None
                            else bootstrap_broker.failure
                        )
                    )
                    stop_requested = True
                    break
                if duration_seconds != float("inf") and (
                    time.monotonic() - started >= duration_seconds
                ):
                    stop_requested = True
                    break
                try:
                    server.service_mutation_inbox(max_requests=32)
                except Exception as exc:  # noqa: BLE001 - owner pump fails closed.
                    pump_error = f"{type(exc).__name__}: {exc}"
                    stop_requested = True
                    break
                time.sleep(0.01)
            if stop_requested and scheduler.poll() is None:
                _terminate_exact(
                    scheduler_birth,
                    grace_seconds=LGCVF_SCHEDULER_TREE_STOP_GRACE_SECONDS,
                    child_process=scheduler,
                )
            returncode = scheduler.wait(timeout=5.0)
            stop_bootstrap_broker()
            if pump_error:
                raise SuccessorOperatorError(
                    "mutation inbox pump failed: " + pump_error
                )
            clean_runtime_shutdown = True
        finally:
            active_runtime_failure = sys.exc_info()[1]
            if (
                scheduler is not None
                and scheduler.poll() is None
                and scheduler_birth is not None
            ):
                _terminate_exact(
                    scheduler_birth,
                    grace_seconds=LGCVF_SCHEDULER_TREE_STOP_GRACE_SECONDS,
                    child_process=scheduler,
                )
                scheduler.wait(timeout=5.0)
            stop_bootstrap_broker()
            log_handle.close()
            owner_checkpoint: Mapping[str, Any] = {}
            checkpoint_failure: Exception | None = None
            if clean_runtime_shutdown and server is not None:
                try:
                    owner_checkpoint = server.checkpoint()
                except Exception as exc:  # noqa: BLE001 - fail closed after stop.
                    checkpoint_failure = exc
            stop_receipt = stop_owner()
            credential_leak = bool(tuple(paths["owner_state"].glob("*.quack-token")))
            for surface in (
                paths["controller_log"],
                paths["controller_status"],
                paths["owner_state"] / "quack-state-server.status.json",
            ):
                credential_leak = credential_leak or _regular_file_contains(
                    surface,
                    token.encode("ascii"),
                )
            scheduler_returncode = (
                scheduler.returncode if scheduler is not None else None
            )
            stopped_error = (
                "attach_credential_persisted"
                if credential_leak
                else "owner_stop_failed"
                if stop_receipt.get("stopped") is not True
                else "owner_checkpoint_failed"
                if checkpoint_failure is not None
                else "unclean_controller_shutdown"
                if not clean_runtime_shutdown
                else "scheduler_exit_failed"
                if scheduler_returncode != 0
                else ""
            )
            stopped = _status_payload(
                lifecycle="stopped",
                controller_birth=controller_birth.to_dict(),
                provenance_cid=str(provenance["receipt_cid"]),
                owner_identity=identity.to_dict(),
                scheduler_birth=(
                    scheduler_birth.to_dict() if scheduler_birth is not None else {}
                ),
                scheduler_returncode=scheduler_returncode,
                error=stopped_error,
                projection_root=paths["projection_root"],
            )
            failed_start_reason = (
                _failed_start_reason_from_exception(active_runtime_failure)
                if not ready_status_published
                else ""
            )
            eligible_failed_start = (
                stopped_error == FAILED_START_STATUS_ERROR
                and failed_start_reason in FAILED_START_TRUSTED_RECOVERY_REASONS
                and not ready_status_published
                and not credential_leak
                and stop_receipt.get("stopped") is True
                and scheduler_birth is not None
                and type(scheduler_returncode) is int
                and _lock_custody is not None
            )
            if stopped_error and not eligible_failed_start:
                revalidate_runtime_generation()
                _write_status(
                    Path(recovery_io_paths["controller_status"]),
                    stopped,
                    token=token,
                )
            elif eligible_failed_start:
                anchored_status_written = False
                try:
                    _restore_lgcvf_stopped_candidate_import_boundary(
                        root=root,
                        sealed_import_roots=sealed_import_roots,
                    )
                    failed_anchors = _capture_failed_start_recovery_anchors(
                        paths,
                        root=root,
                        failed_status=stopped,
                        provenance=provenance,
                        failed_start_reason=failed_start_reason,
                        owner_stop=stop_receipt,
                        io_paths=recovery_io_paths,
                        lock_custody=_lock_custody,
                    )
                    stopped = _bind_failed_start_recovery_anchors_status(
                        stopped,
                        failed_anchors,
                    )
                    # Prior consumed authority remains untouched until this
                    # current-byte anchor is durable and has been reread.
                    revalidate_runtime_generation()
                    _write_status(
                        Path(recovery_io_paths["controller_status"]),
                        stopped,
                        token=token,
                    )
                    if _strict_json(
                        Path(recovery_io_paths["controller_status"]),
                        expected_schema=CONTROLLER_STATUS_SCHEMA,
                        require_private_owner=True,
                    ) != stopped:
                        raise SuccessorOperatorError(
                            "failed-start durable anchor status changed"
                        )
                    anchored_status_written = True
                    assert _lock_custody is not None
                    failed_continuity = (
                        _recover_interrupted_failed_start_continuity(
                            paths,
                            root=root,
                            lock_custody=_lock_custody,
                            provenance=provenance,
                            failed_start_reason=failed_start_reason,
                            _require_dead_controller_tree=False,
                        )
                    )
                    if not isinstance(failed_continuity, Mapping):
                        raise SuccessorOperatorError(
                            "failed-start finally did not publish continuity"
                        )
                except BaseException:
                    if not anchored_status_written:
                        revalidate_runtime_generation()
                        _write_status(
                            Path(recovery_io_paths["controller_status"]),
                            stopped,
                            token=token,
                        )
                    raise
            else:
                _restore_lgcvf_stopped_candidate_import_boundary(
                    root=root,
                    sealed_import_roots=sealed_import_roots,
                )
                recovery_anchors = _capture_stopped_recovery_anchors(
                    paths,
                    root=root,
                    stopped_status=stopped,
                    provenance=provenance,
                    io_paths=recovery_io_paths,
                    lock_custody=_lock_custody,
                )
                stopped = _bind_stopped_recovery_anchors_status(
                    stopped,
                    recovery_anchors,
                )
                # This anchored, unbound status is the durable recovery point.
                # If either following publication is interrupted, recovery can
                # replay only the exact stopped-time source/store/owner bytes.
                revalidate_runtime_generation()
                _write_status(
                    Path(recovery_io_paths["controller_status"]),
                    stopped,
                    token=token,
                )
                continuity = _write_stopped_state_continuity(
                    paths,
                    root=root,
                    stopped_status=stopped,
                    provenance=provenance,
                    owner_checkpoint=owner_checkpoint,
                    owner_stop=stop_receipt,
                    _io_paths=recovery_io_paths,
                )
                bound_stopped = _bind_stopped_state_continuity_status(
                    stopped, continuity
                )
                revalidate_runtime_generation()
                _write_status(
                    Path(recovery_io_paths["controller_status"]),
                    bound_stopped,
                    token=token,
                )
            token = ""
            if credential_leak:
                raise SuccessorOperatorError(
                    "raw Quack attach credential reached a persistent surface"
                )
            if checkpoint_failure is not None:
                raise SuccessorOperatorError(
                    "LGCVF owner clean-stop checkpoint failed"
                ) from checkpoint_failure
        return int(returncode)
    finally:
        try:
            stop_bootstrap_broker()
        except Exception as cleanup_exc:  # noqa: BLE001
            sys.stderr.write(
                "LGCVF bootstrap broker emergency stop failed: "
                f"{type(cleanup_exc).__name__}\n"
            )
        if server is not None:
            try:
                stop_owner()
            except Exception as cleanup_exc:  # noqa: BLE001
                sys.stderr.write(
                    "LGCVF owner emergency stop failed: "
                    f"{type(cleanup_exc).__name__}\n"
                )
        for name, previous in previous_extension_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        _close_lgcvf_configured_board_live_launch(live_launch)
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)


def controller_status(root: Path = ROOT) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    paths = _paths(root)
    status = _strict_json(
        paths["controller_status"],
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    birth = ProcessBirthIdentity.from_dict(status.get("controller_birth"))
    observed = owner_liveness(birth)
    projection = dict(status.get("ducklake_projection") or {})
    projection["receipt_present"] = paths["projection_receipt"].is_file()
    return {
        **status,
        "observed_controller_liveness": observed.value,
        "running": observed is OwnerLiveness.ALIVE
        and status.get("lifecycle") == "ready",
        "ducklake_projection": projection,
    }


def stop_controller(
    root: Path = ROOT, *, timeout_seconds: float = MAX_STOP_SECONDS
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )

    status = controller_status(root)
    birth = ProcessBirthIdentity.from_dict(status.get("controller_birth"))
    selected_timeout = float(timeout_seconds)
    if (
        not math.isfinite(selected_timeout)
        or selected_timeout < 1.0
        or selected_timeout > MAX_STOP_SECONDS
    ):
        raise SuccessorOperatorError(
            "controller stop timeout is outside the closed bound"
        )
    disposition = _terminate_exact(
        birth,
        grace_seconds=selected_timeout,
    )
    return {
        "stopped": True,
        "disposition": disposition,
        "controller_birth": birth.to_dict(),
    }


def _extension_preflight() -> dict[str, Any]:
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET autoinstall_known_extensions = false")
            connection.execute("SET autoload_known_extensions = false")
            loaded: dict[str, str] = {}
            for extension in ("quack", "ducklake", "httpfs"):
                connection.execute(f"LOAD {extension}")
                row = connection.execute(
                    "SELECT installed, loaded, extension_version FROM duckdb_extensions() "
                    "WHERE extension_name = ?",
                    [extension],
                ).fetchone()
                if row is None or row[0] is not True or row[1] is not True:
                    raise SuccessorOperatorError(f"{extension} is not preinstalled")
                loaded[extension] = str(row[2] or "")
        finally:
            connection.close()
    except Exception as exc:  # noqa: BLE001 - capability is typed unavailable.
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "automatic_installation_permitted": False,
        }
    return {
        "available": True,
        "extensions": loaded,
        "automatic_installation_permitted": False,
    }


def _controller_lock_is_held(path: Path) -> bool:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SuccessorOperatorError("controller lock cannot be inspected") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise SuccessorOperatorError("controller lock custody is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _validate_stopped_projection_native_provenance(
    paths: Mapping[str, Path],
    *,
    root: Path,
    receipt: Mapping[str, Any],
    final_continuity: Mapping[str, Any],
    _bootstrap_path: Path | None = None,
) -> None:
    """Replay immutable initial authority beneath an advanced clean board HEAD."""

    native_fields = {
        "schema",
        "issued_at",
        "admission_mode",
        "source_generation",
        "target_generation",
        "source_database",
        "target_database",
        "source_head",
        "source_tree",
        "source_forest_root",
        "datasets_head",
        "datasets_tree",
        "candidate_config_path",
        "candidate_config_sha256",
        "population_root",
        "plan_root_cid",
        "initial_projection",
        "materialized_projection",
        "bootstrap_receipt_cid",
        "bootstrap_verification_root",
        "target_initial_sha256",
        "target_coordination_initial_sha256",
        "target_execution_initial_sha256",
        "database_uuid",
        "schema_fingerprint",
        "catalog_fingerprint",
        "initial_projection_reset",
        "continuity_completion_records_imported",
        "source_database_statuses_read",
        "source_database_completion_records_imported",
        "quack_required_after_publish",
        "direct_multi_process_duckdb_permitted",
        "ducklake_projection_authoritative",
        "restart_requires_live_continuity_receipt",
        "live_continuity_receipt_implemented",
        "candidate_authored_validation",
        "validation_self_authority",
        "validation_completion_authoritative",
        "network_isolation_enforced",
        "model_provider_route",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "authoritative_for_release",
        "production_authorized",
        "receipt_cid",
    }

    def content_cid(field: str) -> bool:
        value = receipt.get(field)
        return (
            type(value) is str
            and re.fullmatch(r"b[a-z2-7]{60}", value) is not None
        )

    def sha256_pin(field: str) -> bool:
        value = receipt.get(field)
        return (
            type(value) is str
            and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
        )

    config, config_raw = _load_native_resume_config(root)
    source_head = str(receipt.get("source_head") or "")
    source_tree = str(receipt.get("source_tree") or "")
    datasets_head = str(receipt.get("datasets_head") or "")
    datasets_tree = str(receipt.get("datasets_tree") or "")
    if (
        set(receipt) != native_fields
        or receipt.get("source_generation") != NATIVE_RESUME_SOURCE_GENERATION
        or receipt.get("target_generation") != SUCCESSOR_STORE_GENERATION
        or receipt.get("source_database") != ""
        or receipt.get("target_database") != str(paths["successor_database"])
        or type(receipt.get("issued_at")) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            receipt["issued_at"],
        )
        is None
        or receipt.get("candidate_config_path")
        != DEFAULT_SUCCESSOR_CONFIG_RELATIVE.as_posix()
        or receipt.get("candidate_config_sha256")
        != "sha256:" + hashlib.sha256(config_raw).hexdigest()
        or receipt.get("initial_projection") != config.get("initial_projection")
        or not content_cid("source_forest_root")
        or not content_cid("population_root")
        or not content_cid("plan_root_cid")
        or not content_cid("bootstrap_receipt_cid")
        or not content_cid("bootstrap_verification_root")
        or not content_cid("schema_fingerprint")
        or not content_cid("catalog_fingerprint")
        or not content_cid("receipt_cid")
        or not sha256_pin("target_initial_sha256")
        or not sha256_pin("target_coordination_initial_sha256")
        or not sha256_pin("target_execution_initial_sha256")
        or type(receipt.get("database_uuid")) is not str
        or not receipt.get("database_uuid")
        or receipt.get("initial_projection_reset") is not True
        or receipt.get("continuity_completion_records_imported") is not False
        or receipt.get("source_database_statuses_read") is not False
        or receipt.get("source_database_completion_records_imported") is not False
        or receipt.get("quack_required_after_publish") is not True
        or receipt.get("direct_multi_process_duckdb_permitted") is not False
        or receipt.get("ducklake_projection_authoritative") is not False
        or receipt.get("restart_requires_live_continuity_receipt") is not True
        or receipt.get("live_continuity_receipt_implemented") is not False
        or receipt.get("candidate_authored_validation") is not True
        or receipt.get("validation_self_authority") is not False
        or receipt.get("validation_completion_authoritative") is not False
        or receipt.get("network_isolation_enforced") is not True
        or receipt.get("model_provider_route") != "none"
        or receipt.get("task_implementation_complete") is not False
        or receipt.get("test_qualification_complete") is not False
        or receipt.get("objective_complete") is not False
        or receipt.get("release_qualified") is not False
        or receipt.get("authoritative_for_release") is not False
        or receipt.get("production_authorized") is not False
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
        or re.fullmatch(r"[0-9a-f]{40}", datasets_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", datasets_tree) is None
    ):
        raise SuccessorOperatorError(
            "stopped-state initial provenance binding differs"
        )
    if _git_text(
        root,
        ("show", "-s", "--format=%T", source_head),
        noun="stopped-state initial source commit",
    ) != source_tree:
        raise SuccessorOperatorError(
            "stopped-state initial source commit/tree binding differs"
        )
    initial_gitlink = _git_text(
        root,
        ("ls-tree", source_head, "--", "ipfs_datasets_py"),
        noun="stopped-state initial nested gitlink",
    ).split(maxsplit=3)
    if (
        len(initial_gitlink) != 4
        or initial_gitlink[0] != "160000"
        or initial_gitlink[1] != "commit"
        or initial_gitlink[2] != datasets_head
        or initial_gitlink[3] != "ipfs_datasets_py"
    ):
        raise SuccessorOperatorError(
            "stopped-state initial nested gitlink binding differs"
        )
    _git_quiet(
        root,
        ("merge-base", "--is-ancestor", source_head, str(final_continuity["current_head"])),
        noun="stopped-state final source ancestry",
    )
    datasets = _contained(root, "ipfs_datasets_py")
    if _git_text(
        datasets,
        ("show", "-s", "--format=%T", datasets_head),
        noun="stopped-state initial nested commit",
    ) != datasets_tree:
        raise SuccessorOperatorError(
            "stopped-state initial nested commit/tree binding differs"
        )
    _git_quiet(
        datasets,
        (
            "merge-base",
            "--is-ancestor",
            datasets_head,
            str(final_continuity["datasets_head"]),
        ),
        noun="stopped-state final nested ancestry",
    )
    target_continuity = _target_source_continuity(
        root,
        source_head=source_head,
        source_tree=source_tree,
        config=config,
        observed_continuity=final_continuity,
    )
    if any(
        target_continuity.get(name) != value
        for name, value in final_continuity.items()
    ):
        raise SuccessorOperatorError(
            "stopped-state final source continuity differs"
        )

    bootstrap_path = _bootstrap_path or (
        paths["successor_database"].parent
        / "evidence"
        / "bootstrap"
        / "materialization.json"
    )
    bootstrap = _strict_json(
        bootstrap_path,
        expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
        require_private_owner=True,
    )
    _validate_native_bootstrap_receipt(
        bootstrap,
        config=config,
        database_paths={
            "control": SUCCESSOR_DATABASE_RELATIVE.as_posix(),
            "coordination": SUCCESSOR_DATABASE_RELATIVE.with_name(
                "control.coordination.duckdb"
            ).as_posix(),
            "execution": SUCCESSOR_DATABASE_RELATIVE.with_name(
                "control.execution.duckdb"
            ).as_posix(),
        },
        source_head=source_head,
        repository_tree_id="git-tree:" + source_tree,
        population_root=str(receipt["population_root"]),
        plan_root_cid=str(receipt["plan_root_cid"]),
        schema_fingerprint=str(receipt["schema_fingerprint"]),
        catalog_fingerprint=str(receipt["catalog_fingerprint"]),
    )
    verification = bootstrap.get("verification")
    if (
        not isinstance(verification, Mapping)
        or bootstrap.get("receipt_cid") != receipt.get("bootstrap_receipt_cid")
        or verification.get("verification_root")
        != receipt.get("bootstrap_verification_root")
        or bootstrap.get("population_root") != receipt.get("population_root")
        or bootstrap.get("plan_root_cid") != receipt.get("plan_root_cid")
    ):
        raise SuccessorOperatorError(
            "stopped-state bootstrap/provenance cross-binding differs"
        )
    control = verification.get("control")
    statuses = control.get("statuses") if isinstance(control, Mapping) else None
    if not isinstance(statuses, Mapping):
        raise SuccessorOperatorError(
            "stopped-state bootstrap task projection differs"
        )
    materialized_projection = _native_resume_materialized_projection(
        config,
        task_ids=list(statuses),
        completed_task_ids=[
            alias for alias, status in statuses.items() if status == "completed"
        ],
        todo_task_ids=[
            alias for alias, status in statuses.items() if status == "todo"
        ],
        blocked_task_ids=[
            alias for alias, status in statuses.items() if status == "blocked"
        ],
        ready_task_ids=list(control.get("ready_task_aliases") or ()),
    )
    if receipt.get("materialized_projection") != materialized_projection:
        raise SuccessorOperatorError(
            "stopped-state initial projection replay differs"
        )


def _observe_stopped_projection_source_continuity(
    root: Path,
    sealed: Mapping[str, Any],
    *,
    minimum_remote_head: str | None = None,
) -> dict[str, Any]:
    """Allow only monotonic remote catch-up beneath an exact stopped source."""

    observed = _observe_candidate_runtime_continuity(
        root,
        require_resolved_remote=False,
    )
    if set(observed) != set(sealed) or any(
        observed.get(field) != value
        for field, value in sealed.items()
        if field != "resolved_remote_head"
    ):
        raise SuccessorOperatorError(
            "stopped-state final source continuity differs"
        )
    sealed_remote = str(sealed.get("resolved_remote_head") or "")
    observed_remote = str(observed.get("resolved_remote_head") or "")
    current_head = str(observed.get("current_head") or "")
    minimum_remote = (
        sealed_remote if minimum_remote_head is None else minimum_remote_head
    )
    if any(
        re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in (sealed_remote, minimum_remote, observed_remote, current_head)
    ):
        raise SuccessorOperatorError(
            "stopped-state remote continuity is malformed"
        )
    checked_ancestors: set[str] = set()
    for ancestor in (sealed_remote, minimum_remote):
        if ancestor == observed_remote or ancestor in checked_ancestors:
            continue
        _git_quiet(
            root,
            ("merge-base", "--is-ancestor", ancestor, observed_remote),
            noun="stopped-state monotonic remote catch-up",
        )
        checked_ancestors.add(ancestor)
    if observed_remote != current_head:
        _git_quiet(
            root,
            ("merge-base", "--is-ancestor", observed_remote, current_head),
            noun="stopped-state remote/current source ancestry",
        )
    return observed


def _observe_failed_start_source_maintenance_descendant(
    root: Path,
    sealed: Mapping[str, Any],
    *,
    minimum_remote_head: str | None = None,
) -> dict[str, Any]:
    """Admit only one clean, strict descendant of a failed-start source."""

    observed = _observe_candidate_runtime_continuity(
        root,
        require_resolved_remote=False,
    )
    fixed_fields = {
        "approved_branch",
        "candidate_worktree_clean",
        "datasets_worktree_clean",
        "python_bytecode_quarantine",
    }
    if (
        set(observed) != set(sealed)
        or any(observed.get(field) != sealed.get(field) for field in fixed_fields)
        or observed.get("approved_branch") != APPROVED_BOARD_BRANCH
        or observed.get("candidate_worktree_clean") is not True
        or observed.get("datasets_worktree_clean") is not True
    ):
        raise SuccessorOperatorError(
            "failed-start source maintenance custody differs"
        )
    sealed_head = str(sealed.get("current_head") or "")
    observed_head = str(observed.get("current_head") or "")
    sealed_tree = str(sealed.get("current_tree") or "")
    observed_tree = str(observed.get("current_tree") or "")
    sealed_datasets_head = str(sealed.get("datasets_head") or "")
    observed_datasets_head = str(observed.get("datasets_head") or "")
    sealed_datasets_tree = str(sealed.get("datasets_tree") or "")
    observed_datasets_tree = str(observed.get("datasets_tree") or "")
    sealed_remote = str(sealed.get("resolved_remote_head") or "")
    observed_remote = str(observed.get("resolved_remote_head") or "")
    minimum_remote = (
        sealed_remote if minimum_remote_head is None else minimum_remote_head
    )
    if any(
        re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in (
            sealed_head,
            observed_head,
            sealed_tree,
            observed_tree,
            sealed_datasets_head,
            observed_datasets_head,
            sealed_datasets_tree,
            observed_datasets_tree,
            sealed_remote,
            observed_remote,
            minimum_remote,
        )
    ):
        raise SuccessorOperatorError(
            "failed-start source maintenance continuity is malformed"
        )
    if sealed_head == observed_head:
        raise SuccessorOperatorError(
            "failed-start continuity already matches the current source"
        )
    _git_quiet(
        root,
        ("merge-base", "--is-ancestor", sealed_head, observed_head),
        noun="failed-start source maintenance ancestry",
    )
    datasets = _contained(root, "ipfs_datasets_py")
    if sealed_datasets_head != observed_datasets_head:
        _git_quiet(
            datasets,
            (
                "merge-base",
                "--is-ancestor",
                sealed_datasets_head,
                observed_datasets_head,
            ),
            noun="failed-start nested source maintenance ancestry",
        )
    checked_remote_ancestors: set[str] = set()
    for ancestor in (sealed_remote, minimum_remote):
        if ancestor == observed_remote or ancestor in checked_remote_ancestors:
            continue
        _git_quiet(
            root,
            ("merge-base", "--is-ancestor", ancestor, observed_remote),
            noun="failed-start source maintenance remote ancestry",
        )
        checked_remote_ancestors.add(ancestor)
    if observed_remote != observed_head:
        _git_quiet(
            root,
            ("merge-base", "--is-ancestor", observed_remote, observed_head),
            noun="failed-start source maintenance remote/current ancestry",
        )
    return observed


def _load_projection_source_continuity(
    paths: Mapping[str, Path],
    *,
    root: Path,
    stopped_database_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    lock_custody: Mapping[str, Any] | None = None,
    _stopped_provenance: Mapping[str, Any] | None = None,
    _allow_failed_start_source_maintenance: bool = False,
) -> dict[str, Any]:
    """Authenticate one stopped state for non-authoritative projection only."""

    sealed_projection = stopped_database_snapshots is not None
    pinned_generation = lock_custody is not None
    if sealed_projection and not pinned_generation:
        raise SuccessorOperatorError(
            "stopped projection snapshot custody is incomplete"
        )
    provenance_receipt_path = paths["provenance"]
    continuity_receipt_path = paths["stopped_state_continuity"]
    controller_status_path = paths["controller_status"]
    bootstrap_receipt_path = (
        paths["successor_database"].parent
        / "evidence"
        / "bootstrap"
        / "materialization.json"
    )
    owner_status_path = (
        paths["owner_state"] / "quack-state-server.status.json"
    )
    owner_marker_path = paths["successor_database"].with_name(
        ".control.duckdb.state-owner.json"
    )
    bound_databases: Mapping[str, Path] | None = None
    pinned_generation_inventory: tuple[tuple[Any, ...], ...] | None = None
    pinned_bootstrap_sha256 = ""
    failed_superseded_snapshot: Mapping[str, Any] | None = None
    if pinned_generation:
        assert lock_custody is not None
        if sealed_projection:
            assert stopped_database_snapshots is not None
            _validate_stopped_database_snapshots(
                paths,
                lock_custody,
                stopped_database_snapshots,
            )
        provenance_receipt_path = _generation_bound_runtime_path(
            paths, lock_custody, provenance_receipt_path
        )
        continuity_receipt_path = _generation_bound_runtime_path(
            paths, lock_custody, continuity_receipt_path
        )
        controller_status_path = _generation_bound_runtime_path(
            paths, lock_custody, controller_status_path
        )
        bootstrap_receipt_path = _generation_bound_runtime_path(
            paths, lock_custody, bootstrap_receipt_path
        )
        owner_status_path = _generation_bound_runtime_path(
            paths, lock_custody, owner_status_path
        )
        owner_marker_path = _generation_bound_runtime_path(
            paths, lock_custody, owner_marker_path
        )
        bound_databases = {
            name: _generation_bound_runtime_path(paths, lock_custody, database)
            for name, database in _successor_state_databases(paths).items()
        }
        pinned_generation_inventory = _stopped_recovery_generation_inventory(
            paths,
            lock_custody,
        )
        pinned_bootstrap_sha256 = _sha256_regular_file(
            bootstrap_receipt_path,
            max_bytes=MAX_JSON_BYTES,
            noun="pinned stopped-state bootstrap receipt",
            require_private_owner=True,
        )
        provenance = _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=provenance_receipt_path,
        )
        if (
            _stopped_provenance is not None
            and provenance != dict(_stopped_provenance)
        ):
            raise SuccessorOperatorError(
                "pinned stopped provenance differs from native authorization"
            )
    elif _stopped_provenance is not None:
        provenance = dict(_stopped_provenance)
    elif os.path.lexists(continuity_receipt_path):
        provenance = _load_lgcvf_live_raw_provenance_receipt(paths)
    else:
        try:
            provenance = _load_provenance(paths, root=root)
        except SuccessorOperatorError as exc:
            if str(exc) not in {
                NATIVE_RESUME_LIVE_CONTINUITY_REQUIRED_ERROR,
                NATIVE_RESUME_PROVENANCE_BINDING_ERROR,
            }:
                raise
            # These are the only two expected consequences of successful owner
            # mutations: changed stores, or a clean board HEAD advanced by admitted
            # merges.  The projection-only path below independently replays the
            # immutable initial provenance and binds the exact final continuity.
            provenance = _load_lgcvf_live_raw_provenance_receipt(paths)
        else:
            return {
                "provenance": provenance,
                "receipt": {},
                "databases": _stopped_state_database_digests(paths),
                "admission_mode": INITIAL_PROVENANCE_PROJECTION_ADMISSION_MODE,
            }
    receipt = _strict_json(
        continuity_receipt_path,
        expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
        require_private_owner=True,
    )
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=provenance,
    )
    expected_fields = {
        "schema",
        "issued_at",
        "admission_mode",
        "target_generation",
        "source_provenance_cid",
        "controller_status_cid",
        "stop_evidence",
        "owner_status_sha256",
        "final_source_continuity",
        "databases",
        "controller_lock_held_at_issue",
        "live_wal_absent",
        "requires_stopped_checkpoint",
        "projection_only",
        "same_generation_restart_only",
        "restart_authority",
        "authoritative",
        "scheduling_authority",
        "completion_authority",
        "read_by_scheduler",
        "quack_endpoint_served",
        "production_authorized",
        "receipt_cid",
    }
    false_authority_fields = (
        "authoritative",
        "scheduling_authority",
        "completion_authority",
        "read_by_scheduler",
        "quack_endpoint_served",
        "production_authorized",
    )
    admission_mode = receipt.get("admission_mode")
    expected_checkpoint = (
        admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
    )
    if (
        set(receipt) != expected_fields
        or admission_mode
        not in {
            STOPPED_STATE_CONTINUITY_ADMISSION_MODE,
            FAILED_START_CONTINUITY_ADMISSION_MODE,
        }
        or receipt.get("target_generation") != SUCCESSOR_STORE_GENERATION
        or receipt.get("source_provenance_cid") != provenance.get("receipt_cid")
        or type(receipt.get("issued_at")) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            receipt["issued_at"],
        )
        is None
        or receipt.get("controller_lock_held_at_issue") is not True
        or receipt.get("live_wal_absent") is not True
        or receipt.get("requires_stopped_checkpoint")
        is not expected_checkpoint
        or receipt.get("projection_only") is not False
        or receipt.get("same_generation_restart_only") is not True
        or receipt.get("restart_authority") is not True
        or any(receipt.get(field) is not False for field in false_authority_fields)
    ):
        raise SuccessorOperatorError(
            "stopped-state continuity semantics differ"
        )
    final_continuity = receipt.get("final_source_continuity")
    if not isinstance(final_continuity, Mapping):
        raise SuccessorOperatorError(
            "stopped-state final source continuity differs"
        )
    source_observer = (
        _observe_failed_start_source_maintenance_descendant
        if _allow_failed_start_source_maintenance
        else _observe_stopped_projection_source_continuity
    )
    observed_continuity = source_observer(root, final_continuity)
    _validate_stopped_projection_native_provenance(
        paths,
        root=root,
        receipt=provenance,
        final_continuity=final_continuity,
        _bootstrap_path=bootstrap_receipt_path,
    )
    observed_bootstrap_sha256 = _sha256_regular_file(
        bootstrap_receipt_path,
        max_bytes=MAX_JSON_BYTES,
        noun="stopped-state bootstrap receipt",
        require_private_owner=True,
    )

    status = _strict_json(
        controller_status_path,
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    anchor_status = dict(status)
    anchor_status.pop("status_cid", None)
    linked_receipt_cid = anchor_status.pop(
        "stopped_state_continuity_receipt_cid", None
    )
    linked_status_cid = anchor_status.pop(
        "stopped_state_continuity_status_cid", None
    )
    anchor_status["status_cid"] = _content_id(anchor_status)
    owner_identity = anchor_status.get("owner_identity")
    controller_birth_raw = anchor_status.get("controller_birth")
    scheduler_birth_raw = anchor_status.get("scheduler_birth")
    expected_error = (
        ""
        if admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
        else FAILED_START_STATUS_ERROR
    )
    returncode_valid = (
        type(anchor_status.get("scheduler_returncode")) is int
        and (
            anchor_status.get("scheduler_returncode") == 0
            if admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
            else -255 <= anchor_status.get("scheduler_returncode") <= 255
        )
    )
    if (
        linked_receipt_cid != receipt.get("receipt_cid")
        or linked_status_cid != receipt.get("controller_status_cid")
        or anchor_status.get("status_cid") != linked_status_cid
        or anchor_status.get("lifecycle") != "stopped"
        or anchor_status.get("error") != expected_error
        or anchor_status.get("provenance_cid") != provenance.get("receipt_cid")
        or not returncode_valid
        or not isinstance(owner_identity, Mapping)
        or not isinstance(controller_birth_raw, Mapping)
        or not isinstance(scheduler_birth_raw, Mapping)
        or owner_identity.get("process_birth") != controller_birth_raw
        or owner_identity.get("database_uuid") != provenance.get("database_uuid")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            provenance.get("schema_fingerprint"),
        )
        or owner_identity.get("store_id") != SUCCESSOR_DATABASE_RELATIVE.as_posix()
        or owner_identity.get("secret_handle") != SECRET_HANDLE
    ):
        raise SuccessorOperatorError(
            "stopped-state controller status binding differs"
        )
    if admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE:
        _validate_unbound_stopped_controller_status(
            anchor_status,
            provenance=provenance,
        )
    else:
        _validate_unbound_failed_start_controller_status(
            anchor_status,
            provenance=provenance,
            require_dead=True,
        )

    stop_evidence = receipt.get("stop_evidence")
    owner_server_id = str(owner_identity.get("server_id") or "")
    if not isinstance(stop_evidence, Mapping):
        raise SuccessorOperatorError("stopped-state owner evidence differs")
    evidence_mode = stop_evidence.get("mode")
    if (
        admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
        and evidence_mode == STOPPED_STATE_LIVE_OWNER_EVIDENCE_MODE
    ):
        checkpoint = stop_evidence.get("owner_checkpoint")
        stopped = stop_evidence.get("owner_stop")
        if (
            set(stop_evidence)
            != {
                "mode",
                "owner_checkpoint",
                "owner_stop",
                "historical_owner_receipts_reconstructed",
            }
            or stop_evidence.get("historical_owner_receipts_reconstructed")
            is not False
            or not isinstance(checkpoint, Mapping)
            or set(checkpoint)
            != {"checkpointed", "server_id", "database_path", "at"}
            or checkpoint.get("checkpointed") is not True
            or checkpoint.get("server_id") != owner_server_id
            or checkpoint.get("database_path")
            != str(paths["successor_database"])
            or not isinstance(stopped, Mapping)
            or set(stopped) != {"stopped", "server_id", "at"}
            or stopped.get("stopped") is not True
            or stopped.get("server_id") != owner_server_id
        ):
            raise SuccessorOperatorError("stopped-state owner evidence differs")
    elif (
        admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
        and evidence_mode == STOPPED_STATE_RECOVERED_EVIDENCE_MODE
    ):
        recovered_at = stop_evidence.get("recovered_at")
        anchors = anchor_status.get("stopped_recovery_anchors")
        anchor_cid = (
            str(anchors.get("anchors_cid") or "")
            if isinstance(anchors, Mapping)
            else ""
        )
        expected_authorization_mode = (
            STOPPED_RECOVERY_DURABLE_ANCHOR_MODE
            if anchor_cid
            else STOPPED_RECOVERY_REVIEWED_LEGACY_MODE
        )
        if (
            set(stop_evidence)
            != {
                "mode",
                "recovered_at",
                "source_controller_status_cid",
                "recovery_preflight_cid",
                "recovery_authorization_mode",
                "durable_stopped_anchors_cid",
                "historical_owner_receipts_reconstructed",
            }
            or type(recovered_at) is not str
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
                recovered_at,
            )
            is None
            or recovered_at != receipt.get("issued_at")
            or stop_evidence.get("source_controller_status_cid")
            != receipt.get("controller_status_cid")
            or type(stop_evidence.get("recovery_preflight_cid")) is not str
            or not str(stop_evidence.get("recovery_preflight_cid") or "")
            or stop_evidence.get("recovery_authorization_mode")
            != expected_authorization_mode
            or stop_evidence.get("durable_stopped_anchors_cid")
            != anchor_cid
            or stop_evidence.get("historical_owner_receipts_reconstructed")
            is not False
        ):
            raise SuccessorOperatorError(
                "recovered stopped-state evidence differs"
            )
    elif admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE:
        raise SuccessorOperatorError("stopped-state owner evidence mode differs")
    observed_owner_status_sha256 = _stopped_owner_status_sha256(
        paths,
        controller_status=anchor_status,
        _status_path=owner_status_path,
        _marker_path=owner_marker_path,
    )
    if receipt.get("owner_status_sha256") != observed_owner_status_sha256:
        raise SuccessorOperatorError("stopped-state owner status digest differs")

    databases = receipt.get("databases")
    observed_databases = (
        _validate_stopped_database_snapshots(
            paths,
            lock_custody,
            stopped_database_snapshots,
        )
        if sealed_projection
        else _stopped_state_database_digests(
            paths,
            _database_paths=bound_databases,
        )
    )
    if (
        not isinstance(databases, Mapping)
        or set(databases) != {"control", "coordination", "execution"}
        or dict(databases) != observed_databases
    ):
        raise SuccessorOperatorError("stopped-state database binding differs")
    if admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE:
        anchors = _validate_stopped_recovery_anchors(
            anchor_status,
            provenance=provenance,
            final_continuity=final_continuity,
            databases=observed_databases,
            owner_status_sha256=observed_owner_status_sha256,
        )
    else:
        anchors = _validate_failed_start_recovery_anchors(
            paths,
            anchor_status,
            provenance=provenance,
            final_continuity=final_continuity,
            databases=observed_databases,
            owner_status_sha256=observed_owner_status_sha256,
            bootstrap_sha256=observed_bootstrap_sha256,
            require_dead=True,
        )
        if anchors is None:
            raise SuccessorOperatorError(
                "failed-start receipt lacks durable current-byte anchors"
            )
        _validate_failed_start_stop_evidence(
            receipt,
            failed_status=anchor_status,
            anchors=anchors,
        )
        failed_superseded_snapshot = anchors.get(
            "superseded_restart_admission"
        )
        _observe_failed_start_superseded_admission(
            paths,
            io_paths=_stopped_recovery_io_paths(paths, lock_custody),
            provenance=provenance,
            expected_snapshot=failed_superseded_snapshot,
        )
    if (
        admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
        and evidence_mode == STOPPED_STATE_LIVE_OWNER_EVIDENCE_MODE
        and anchors is None
    ):
        raise SuccessorOperatorError(
            "live-owner stopped receipt lacks durable recovery anchors"
        )
    if (
        admission_mode == STOPPED_STATE_CONTINUITY_ADMISSION_MODE
        and evidence_mode == STOPPED_STATE_RECOVERED_EVIDENCE_MODE
    ):
        reviewed_pins = {
            "target_generation": SUCCESSOR_STORE_GENERATION,
            "controller_status_cid": anchor_status["status_cid"],
            "controller_status": anchor_status,
            "source_provenance_cid": provenance["receipt_cid"],
            "source_continuity": dict(final_continuity),
            "databases": observed_databases,
            "owner_status_sha256": observed_owner_status_sha256,
            "durable_stopped_anchors_cid": (
                str(anchors["anchors_cid"]) if anchors is not None else ""
            ),
        }
        if stop_evidence.get("recovery_preflight_cid") != _content_id(
            {
                "schema": STOPPED_RECOVERY_PREFLIGHT_SCHEMA,
                "operation": STOPPED_RECOVERY_OPERATION,
                "reviewed_pins": reviewed_pins,
            }
        ):
            raise SuccessorOperatorError(
                "recovered stopped-state reviewed pins differ"
            )
    if sealed_projection:
        assert stopped_database_snapshots is not None
        control_snapshot = stopped_database_snapshots["control"]
        snapshot_descriptor = int(control_snapshot["snapshot_descriptor"])
        identity_path = Path(str(control_snapshot["snapshot_path"]))
        verification = _verify_profile(
            identity_path,
            sealed_descriptor=snapshot_descriptor,
        )
        database_identity = _database_identity(identity_path)
    else:
        identity_path = (
            paths["successor_database"]
            if bound_databases is None
            else bound_databases["control"]
        )
        verification = _verify_profile(
            identity_path,
            read_only=pinned_generation,
        )
        database_identity = _database_identity(identity_path)
    if (
        verification.get("schema_fingerprint")
        != provenance.get("schema_fingerprint")
        or verification.get("catalog_fingerprint")
        != provenance.get("catalog_fingerprint")
        or database_identity.get("database_uuid")
        != provenance.get("database_uuid")
        or database_identity.get("schema_fingerprint")
        != provenance.get("schema_fingerprint")
        or not _owner_schema_fingerprint_matches_canonical_cid(
            owner_identity.get("schema_fingerprint"),
            database_identity.get("schema_fingerprint"),
        )
    ):
        raise SuccessorOperatorError(
            "stopped-state database identity differs from provenance"
        )
    source_observer(
        root,
        final_continuity,
        minimum_remote_head=str(observed_continuity["resolved_remote_head"]),
    )
    if sealed_projection and (
        _validate_stopped_database_snapshots(
            paths,
            lock_custody,
            stopped_database_snapshots,
        )
        != observed_databases
    ):
        raise SuccessorOperatorError(
            "stopped-state database snapshot changed during admission"
        )
    if pinned_generation:
        assert lock_custody is not None
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        if (
            pinned_generation_inventory is None
            or _stopped_recovery_generation_inventory(paths, lock_custody)
            != pinned_generation_inventory
            or _load_lgcvf_live_raw_provenance_receipt(
                paths,
                _receipt_path=provenance_receipt_path,
            )
            != provenance
            or _strict_json(
                continuity_receipt_path,
                expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
                require_private_owner=True,
            )
            != receipt
            or _strict_json(
                controller_status_path,
                expected_schema=CONTROLLER_STATUS_SCHEMA,
                require_private_owner=True,
            )
            != status
            or _sha256_regular_file(
                bootstrap_receipt_path,
                max_bytes=MAX_JSON_BYTES,
                noun="pinned stopped-state bootstrap receipt",
                require_private_owner=True,
            )
            != pinned_bootstrap_sha256
            or _stopped_owner_status_sha256(
                paths,
                controller_status=anchor_status,
                _status_path=owner_status_path,
                _marker_path=owner_marker_path,
            )
            != observed_owner_status_sha256
            or (
                admission_mode == FAILED_START_CONTINUITY_ADMISSION_MODE
                and _observe_failed_start_superseded_admission(
                    paths,
                    io_paths=_stopped_recovery_io_paths(
                        paths,
                        lock_custody,
                    ),
                    provenance=provenance,
                    expected_snapshot=failed_superseded_snapshot,
                )
                != (
                    "absent"
                    if failed_superseded_snapshot is None
                    else "archived"
                )
            )
        ):
            raise SuccessorOperatorError(
                "pinned stopped-state evidence changed during admission"
            )
        if not sealed_projection and (
            _stopped_state_database_digests(
                paths,
                _database_paths=bound_databases,
            )
            != observed_databases
        ):
            raise SuccessorOperatorError(
                "pinned stopped-state databases changed during admission"
            )
    return {
        "provenance": provenance,
        "receipt": receipt,
        "controller_status": status,
        "databases": observed_databases,
        "observed_continuity": observed_continuity,
        "admission_mode": admission_mode,
    }


def _project_failed_start_source_maintenance_status(
    status: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a published failed start back to the legacy recovery boundary."""

    projected = dict(status)
    projected.pop("status_cid", None)
    projected.pop("stopped_state_continuity_receipt_cid", None)
    projected.pop("stopped_state_continuity_status_cid", None)
    projected.pop("failed_start_recovery_anchors", None)
    projected["status_cid"] = _content_id(projected)
    _validate_unbound_failed_start_controller_status(
        projected,
        provenance=provenance,
        require_dead=True,
    )
    return projected


def _failed_start_source_maintenance_superseded_snapshot(
    paths: Mapping[str, Path],
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the current receipt exactly as the recovery machine will."""

    snapshot = {
        "receipt": dict(receipt),
        "file_sha256": _sha256_regular_file(
            receipt_path,
            max_bytes=MAX_JSON_BYTES,
            noun="failed-start source maintenance receipt",
            require_private_owner=True,
        ),
        "archive_path": str(
            _failed_start_superseded_archive_path(paths, receipt)
        ),
    }
    validated = _validate_failed_start_superseded_admission_snapshot(
        paths,
        snapshot,
        provenance=provenance,
    )
    if validated is None:
        raise SuccessorOperatorError(
            "failed-start source maintenance superseded receipt is unavailable"
        )
    return validated


def _failed_start_source_maintenance_report(
    preflight: Mapping[str, Any],
    *,
    published_receipt_cid: str,
    published_controller_status_cid: str,
    already_resealed: bool = False,
) -> dict[str, Any]:
    reviewed_pins = preflight.get("reviewed_pins")
    if not isinstance(reviewed_pins, Mapping):
        raise SuccessorOperatorError(
            "failed-start source maintenance reviewed pins are malformed"
        )
    return {
        "schema": FAILED_START_SOURCE_MAINTENANCE_PREFLIGHT_SCHEMA,
        "operation": FAILED_START_SOURCE_MAINTENANCE_OPERATION,
        "observed_at": _utc_now(),
        "reviewed_pins": dict(reviewed_pins),
        "preflight_cid": str(preflight.get("preflight_cid") or ""),
        "published_stopped_state_continuity_receipt_cid": (
            published_receipt_cid
        ),
        "published_controller_status_cid": published_controller_status_cid,
        "already_resealed": already_resealed,
        "clean_monotonic_descendant_required": True,
        "controller_lock_held": True,
        "live_wal_absent": True,
        "restart_authority": False,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "production_authorized": False,
    }


def _failed_start_source_maintenance_preflight_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pin a published failed start and its exact clean descendant read-only."""

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    if not os.path.lexists(io_paths["stopped_state_continuity"]):
        raise SuccessorOperatorError(
            "published failed-start continuity is unavailable for maintenance"
        )
    if os.path.lexists(io_paths["stopped_state_restart_admission"]):
        raise SuccessorOperatorError(
            "failed-start source maintenance transition is already in progress"
        )
    generation_inventory = _stopped_recovery_generation_inventory(
        paths,
        lock_custody,
    )
    admitted = _load_projection_source_continuity(
        paths,
        root=root,
        lock_custody=lock_custody,
        _stopped_provenance=provenance,
        _allow_failed_start_source_maintenance=True,
    )
    observed_provenance = admitted.get("provenance")
    receipt = admitted.get("receipt")
    published_status = admitted.get("controller_status")
    current_continuity = admitted.get("observed_continuity")
    databases = admitted.get("databases")
    if (
        admitted.get("admission_mode")
        != FAILED_START_CONTINUITY_ADMISSION_MODE
        or not isinstance(observed_provenance, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(published_status, Mapping)
        or not isinstance(current_continuity, Mapping)
        or not isinstance(databases, Mapping)
    ):
        raise SuccessorOperatorError(
            "source maintenance requires published failed-start continuity"
        )
    projected_status = _project_failed_start_source_maintenance_status(
        published_status,
        provenance=observed_provenance,
    )
    bootstrap_path = Path(io_paths["bootstrap"])
    bootstrap_sha256 = _sha256_regular_file(
        bootstrap_path,
        max_bytes=MAX_JSON_BYTES,
        noun="failed-start source maintenance bootstrap receipt",
        require_private_owner=True,
    )
    _validate_stopped_projection_native_provenance(
        paths,
        root=root,
        receipt=observed_provenance,
        final_continuity=current_continuity,
        _bootstrap_path=bootstrap_path,
    )
    owner_status_sha256 = _stopped_owner_status_sha256(
        paths,
        controller_status=projected_status,
        _status_path=Path(io_paths["owner_status"]),
        _marker_path=Path(io_paths["owner_marker"]),
    )
    superseded = _failed_start_source_maintenance_superseded_snapshot(
        paths,
        receipt=receipt,
        receipt_path=Path(io_paths["stopped_state_continuity"]),
        provenance=observed_provenance,
    )
    reviewed_pins = _failed_start_recovery_reviewed_pins(
        failed_status=projected_status,
        provenance=observed_provenance,
        failed_start_reason=FAILED_START_REASON_LEGACY_UNCLASSIFIED,
        final_continuity=current_continuity,
        databases=databases,
        owner_status_sha256=owner_status_sha256,
        bootstrap_sha256=bootstrap_sha256,
        superseded_restart_admission=superseded,
        recovery_authorization_mode=FAILED_START_REVIEWED_LEGACY_MODE,
        owner_stop_receipt=None,
    )
    preflight = {
        "reviewed_pins": reviewed_pins,
        "preflight_cid": _failed_start_preflight_cid(reviewed_pins),
    }
    report = _failed_start_source_maintenance_report(
        preflight,
        published_receipt_cid=str(receipt["receipt_cid"]),
        published_controller_status_cid=str(published_status["status_cid"]),
    )
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    if (
        _stopped_recovery_generation_inventory(paths, lock_custody)
        != generation_inventory
        or _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        != dict(observed_provenance)
        or _strict_json(
            Path(io_paths["stopped_state_continuity"]),
            expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
            require_private_owner=True,
        )
        != dict(receipt)
        or _strict_json(
            Path(io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        )
        != dict(published_status)
        or _stopped_state_database_digests(
            paths,
            _database_paths=io_paths["databases"],
        )
        != dict(databases)
        or _stopped_owner_status_sha256(
            paths,
            controller_status=projected_status,
            _status_path=Path(io_paths["owner_status"]),
            _marker_path=Path(io_paths["owner_marker"]),
        )
        != owner_status_sha256
        or _sha256_regular_file(
            bootstrap_path,
            max_bytes=MAX_JSON_BYTES,
            noun="failed-start source maintenance bootstrap receipt",
            require_private_owner=True,
        )
        != bootstrap_sha256
        or _failed_start_source_maintenance_superseded_snapshot(
            paths,
            receipt=receipt,
            receipt_path=Path(io_paths["stopped_state_continuity"]),
            provenance=observed_provenance,
        )
        != superseded
        or _observe_stopped_projection_source_continuity(
            root,
            current_continuity,
        )
        != dict(current_continuity)
    ):
        raise SuccessorOperatorError(
            "failed-start source maintenance evidence changed during preflight"
        )
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    return report


def _validate_failed_start_source_maintenance_transition_preflight(
    paths: Mapping[str, Path],
    *,
    root: Path,
    provenance: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Prove a standard recovery preflight is the interrupted reseal."""

    pins = preflight.get("reviewed_pins")
    if not isinstance(pins, Mapping):
        raise SuccessorOperatorError(
            "failed-start source maintenance transition pins are malformed"
        )
    superseded = _validate_failed_start_superseded_admission_snapshot(
        paths,
        pins.get("superseded_restart_admission"),
        provenance=provenance,
    )
    prior_receipt = (
        superseded.get("receipt") if isinstance(superseded, Mapping) else None
    )
    current_continuity = pins.get("source_continuity")
    projected_status = pins.get("controller_status")
    if (
        preflight.get("preflight_cid") != _failed_start_preflight_cid(pins)
        or pins.get("failed_start_reason")
        != FAILED_START_REASON_LEGACY_UNCLASSIFIED
        or pins.get("recovery_authorization_mode")
        != FAILED_START_REVIEWED_LEGACY_MODE
        or pins.get("owner_stop_receipt") is not None
        or not isinstance(prior_receipt, Mapping)
        or prior_receipt.get("admission_mode")
        != FAILED_START_CONTINUITY_ADMISSION_MODE
        or not isinstance(current_continuity, Mapping)
        or not isinstance(projected_status, Mapping)
    ):
        raise SuccessorOperatorError(
            "failed-start source maintenance transition differs"
        )
    _validate_unbound_failed_start_controller_status(
        projected_status,
        provenance=provenance,
        require_dead=True,
    )
    observed = _observe_failed_start_source_maintenance_descendant(
        root,
        prior_receipt["final_source_continuity"],
    )
    if observed != dict(current_continuity):
        raise SuccessorOperatorError(
            "failed-start source maintenance reviewed source changed"
        )
    return superseded


def _completed_failed_start_source_maintenance(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any],
    reviewed_preflight_cid: str,
) -> dict[str, Any]:
    """Replay one completed reseal before reporting idempotent success."""

    admitted = _load_projection_source_continuity(
        paths,
        root=root,
        lock_custody=lock_custody,
        _stopped_provenance=provenance,
    )
    receipt = admitted.get("receipt")
    status = admitted.get("controller_status")
    if not isinstance(receipt, Mapping) or not isinstance(status, Mapping):
        raise SuccessorOperatorError(
            "completed failed-start source maintenance is unavailable"
        )
    anchor_status = dict(status)
    anchor_status.pop("status_cid", None)
    anchor_status.pop("stopped_state_continuity_receipt_cid", None)
    anchor_status.pop("stopped_state_continuity_status_cid", None)
    anchor_status["status_cid"] = _content_id(anchor_status)
    anchors = anchor_status.get("failed_start_recovery_anchors")
    evidence = receipt.get("stop_evidence")
    superseded = (
        anchors.get("superseded_restart_admission")
        if isinstance(anchors, Mapping)
        else None
    )
    prior_receipt = (
        superseded.get("receipt") if isinstance(superseded, Mapping) else None
    )
    if (
        admitted.get("admission_mode")
        != FAILED_START_CONTINUITY_ADMISSION_MODE
        or not isinstance(evidence, Mapping)
        or evidence.get("mode") != FAILED_START_REVIEWED_EVIDENCE_MODE
        or evidence.get("recovery_preflight_cid")
        != reviewed_preflight_cid
        or not isinstance(prior_receipt, Mapping)
        or prior_receipt.get("admission_mode")
        != FAILED_START_CONTINUITY_ADMISSION_MODE
    ):
        raise SuccessorOperatorError(
            "reviewed failed-start source maintenance result differs"
        )
    observed = _observe_failed_start_source_maintenance_descendant(
        root,
        prior_receipt["final_source_continuity"],
    )
    final_continuity = receipt.get("final_source_continuity")
    if not isinstance(final_continuity, Mapping) or any(
        observed.get(field) != value
        for field, value in final_continuity.items()
        if field != "resolved_remote_head"
    ):
        raise SuccessorOperatorError(
            "completed failed-start source maintenance source differs"
        )
    return {
        "schema": FAILED_START_SOURCE_MAINTENANCE_RESULT_SCHEMA,
        "resealed": True,
        "repeated": True,
        "preflight_cid": reviewed_preflight_cid,
        "superseded_stopped_state_continuity_receipt_cid": prior_receipt[
            "receipt_cid"
        ],
        "stopped_state_continuity_receipt_cid": receipt["receipt_cid"],
        "controller_status_cid": receipt["controller_status_cid"],
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "restart_authority": True,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "production_authorized": False,
    }


def _load_stopped_restart_admission(
    paths: Mapping[str, Path],
    *,
    root: Path,
    provenance: Mapping[str, Any],
    lock_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load exact stopped continuity as same-generation restart authority."""

    admitted = _load_projection_source_continuity(
        paths,
        root=root,
        lock_custody=lock_custody,
        _stopped_provenance=provenance,
    )
    receipt = admitted.get("receipt")
    if (
        admitted.get("admission_mode")
        not in {
            STOPPED_STATE_CONTINUITY_ADMISSION_MODE,
            FAILED_START_CONTINUITY_ADMISSION_MODE,
        }
        or not isinstance(receipt, Mapping)
        or receipt.get("restart_authority") is not True
        or receipt.get("same_generation_restart_only") is not True
        or receipt.get("target_generation") != SUCCESSOR_STORE_GENERATION
        or admitted.get("provenance") != dict(provenance)
    ):
        raise SuccessorOperatorError(
            "stopped-state same-generation restart authority differs"
        )
    status = admitted.get("controller_status")
    if not isinstance(status, Mapping):
        raise SuccessorOperatorError(
            "stopped-state restart controller status is unavailable"
        )
    return admitted


def _load_stopped_restart_provenance(
    paths: Mapping[str, Path],
    *,
    root: Path,
    provenance: Mapping[str, Any],
    lock_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return provenance only after exact stopped restart admission."""

    admitted = _load_stopped_restart_admission(
        paths,
        root=root,
        provenance=provenance,
        lock_custody=lock_custody,
    )
    return dict(admitted["provenance"])


def projection_preflight(
    root: Path = ROOT,
    *,
    _checkpoint_lock_held: bool = False,
    _locked_paths: Mapping[str, Path] | None = None,
    _lock_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (_locked_paths is None) != (_lock_custody is None):
        raise SuccessorOperatorError(
            "projection preflight generation custody is incomplete"
        )
    paths = dict(_locked_paths) if _locked_paths is not None else _paths(root)
    if _lock_custody is not None:
        _revalidate_generation_bound_controller_lock(paths, _lock_custody)
    lock_held = (
        True
        if _checkpoint_lock_held
        else _controller_lock_is_held(paths["controller_lock"])
    )
    running = lock_held and not _checkpoint_lock_held
    try:
        running = running or bool(controller_status(root).get("running"))
    except SuccessorOperatorError:
        pass
    capability = _extension_preflight()
    snapshot_capability = _stopped_snapshot_capacity_preflight(paths)
    source_admitted = False
    source_error = ""
    source_admission_mode = ""
    stopped_state_continuity_receipt_cid = ""
    stopped_controller_status_cid = ""
    if not running:
        try:
            continuity = _load_projection_source_continuity(paths, root=root)
            receipt = continuity["receipt"]
            source_admitted = True
            source_admission_mode = str(continuity["admission_mode"])
            stopped_state_continuity_receipt_cid = str(
                receipt.get("receipt_cid") or ""
            )
            stopped_controller_status_cid = str(
                receipt.get("controller_status_cid") or ""
            )
        except (OSError, RuntimeError, ValueError) as exc:
            source_error = f"{type(exc).__name__}: {exc}"
    projection_receipt_present = os.path.lexists(paths["projection_receipt"])
    projection_root_present = os.path.lexists(paths["projection_root"])
    return {
        "schema": PROJECTION_RECEIPT_SCHEMA,
        "valid": (
            capability.get("available") is True
            and snapshot_capability.get("available") is True
            and not running
            and source_admitted
            and source_admission_mode
            in {
                STOPPED_STATE_CONTINUITY_ADMISSION_MODE,
                FAILED_START_CONTINUITY_ADMISSION_MODE,
            }
            and not projection_receipt_present
            and not projection_root_present
        ),
        "projection_root": str(paths["projection_root"]),
        "projection_root_present": projection_root_present,
        "projection_receipt_present": projection_receipt_present,
        "control_catalog_path": str(paths["projection_root"] / "control.duckdb"),
        "ducklake_catalog_path": str(paths["projection_root"] / "lake.ducklake"),
        "ducklake_data_path": str(paths["projection_root"] / "lake-data"),
        "source_database": str(paths["successor_database"]),
        "controller_running": running,
        "controller_lock_held": lock_held,
        "source_database_present": paths["successor_database"].is_file(),
        "provenance_receipt_present": paths["provenance"].is_file(),
        "stopped_state_continuity_receipt_present": paths[
            "stopped_state_continuity"
        ].is_file(),
        "source_admitted": source_admitted,
        "source_admission_mode": source_admission_mode,
        "stopped_state_continuity_receipt_cid": (
            stopped_state_continuity_receipt_cid
        ),
        "stopped_controller_status_cid": stopped_controller_status_cid,
        "source_error": source_error,
        "requires_stopped_checkpoint": True,
        "capability": capability,
        "sealed_snapshot_capability": snapshot_capability,
        "authoritative": False,
        "restart_authority": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "read_by_scheduler": False,
        "quack_endpoint_served": False,
        "separate_projection_reason": (
            "BoardControlPlane owns a distinct DuckLake catalog but does not expose "
            "a qualified Quack state-owner endpoint; direct source-file reads are "
            "admitted only after the LGCVF owner stops"
        ),
    }


@contextlib.contextmanager
def _exclusive_projection_checkpoint(
    paths: Mapping[str, Path],
    *,
    _read_only_existing_lock: bool = False,
) -> Any:
    """Hold the controller lock so an owner cannot race a direct checkpoint."""

    custody = _open_generation_bound_controller_lock(
        paths,
        read_only_existing=_read_only_existing_lock,
    )
    handle = custody["lock_handle"]
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuccessorOperatorError(
                "LGCVF owner is active; refusing direct DuckLake checkpoint"
            ) from exc
        _revalidate_generation_bound_controller_lock(paths, custody)
        yield custody
        _revalidate_generation_bound_controller_lock(paths, custody)
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            _close_generation_bound_controller_lock(custody)


def project_ducklake_once(root: Path = ROOT) -> dict[str, Any]:
    paths = _paths(root)
    with _exclusive_projection_checkpoint(paths) as lock_custody:
        return _project_ducklake_once_locked(
            root,
            paths=paths,
            lock_custody=lock_custody,
        )


def _audit_task_history_connection(connection: Any) -> dict[str, Any]:
    """Audit lifecycle history without exporting task bodies or mutating state."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        canonical_json_bytes,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        MAX_BODY_BYTES,
        MAX_ID_BYTES,
        MAX_PLAN_PROJECTION_BYTES,
        MAX_PROJECTION_RECORDS,
        TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
    )

    task_population = connection.execute(
        "SELECT COUNT(*), "
        "COALESCE(MAX(octet_length(encode(task_cid))), 0) FROM tasks"
    ).fetchone()
    if task_population is None:
        raise SuccessorOperatorError("stopped task population is unavailable")
    task_count = int(task_population[0])
    if task_count > MAX_PROJECTION_RECORDS:
        raise SuccessorOperatorError(
            "stopped task population exceeds history-audit bound"
        )
    if int(task_population[1]) > MAX_ID_BYTES:
        raise SuccessorOperatorError(
            "stopped task identity exceeds history-audit bound"
        )
    history_population = connection.execute(
        "SELECT COUNT(*), COUNT(*) FILTER (WHERE t.task_cid IS NULL) "
        "FROM task_revisions AS r LEFT JOIN tasks AS t "
        "ON t.task_cid = r.task_cid"
    ).fetchone()
    if history_population is None:
        raise SuccessorOperatorError(
            "stopped task-history population is unavailable"
        )
    history_row_count = int(history_population[0])
    orphan_history_count = int(history_population[1])
    global_errors: list[str] = []
    if task_count < len(LGCVF_TASK_ALIASES):
        global_errors.append("task_population_below_initial_board")
    if history_row_count > MAX_PROJECTION_RECORDS:
        raise SuccessorOperatorError(
            "stopped task-history population exceeds history-audit bound"
        )
    if orphan_history_count:
        global_errors.append("orphan_task_revisions_present")
    task_cids = connection.execute(
        "SELECT task_cid FROM tasks ORDER BY task_cid LIMIT ?",
        [MAX_PROJECTION_RECORDS + 1],
    ).fetchall()
    if len(task_cids) != task_count:
        raise SuccessorOperatorError(
            "stopped task population changed during history audit"
    )
    audits: list[dict[str, Any]] = []
    valid_count = 0

    def strict_json_object(raw: str) -> Any:
        def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON field: {key}")
                result[key] = value
            return result

        def reject_constant(_value: str) -> Any:
            raise ValueError("non-finite JSON constant")

        def reject_float(_value: str) -> Any:
            raise ValueError("floating-point JSON number")

        return json.loads(
            raw,
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )

    for task_cid_row in task_cids:
        task_cid = str(task_cid_row[0] or "")
        errors: list[str] = []
        if not task_cid:
            errors.append("task_cid_empty")
        task_row = connection.execute(
            "SELECT "
            "CASE WHEN octet_length(encode(task_alias)) <= ? "
            "THEN task_alias END, octet_length(encode(task_alias)), "
            "CASE WHEN octet_length(encode(status)) <= ? THEN status END, "
            "octet_length(encode(status)), revision, "
            "CASE WHEN octet_length(encode(body_json)) <= ? THEN body_json END, "
            "octet_length(encode(body_json)), "
            "CASE WHEN octet_length(encode(updated_at)) <= ? THEN updated_at END, "
            "octet_length(encode(updated_at)) "
            "FROM tasks WHERE task_cid = ?",
            [MAX_ID_BYTES, MAX_ID_BYTES, MAX_BODY_BYTES, MAX_ID_BYTES, task_cid],
        ).fetchone()
        if task_row is None:
            raise SuccessorOperatorError(
                "stopped task population changed during history audit"
            )
        task_alias = str(task_row[0] or "")
        task_alias_bytes = int(task_row[1])
        task_status = str(task_row[2] or "")
        task_status_bytes = int(task_row[3])
        head_revision = task_row[4]
        task_body_json = task_row[5]
        task_body_byte_count = int(task_row[6])
        task_updated_at = str(task_row[7] or "")
        task_updated_at_bytes = int(task_row[8])
        if task_alias_bytes > MAX_ID_BYTES:
            errors.append("task_alias_exceeds_bound")
        elif not task_alias:
            errors.append("task_alias_empty")
        if task_status_bytes > MAX_ID_BYTES:
            errors.append("task_status_exceeds_bound")
        elif not task_status:
            errors.append("task_status_empty")
        if task_updated_at_bytes > MAX_ID_BYTES:
            errors.append("task_updated_at_exceeds_bound")
        elif not task_updated_at:
            errors.append("task_updated_at_empty")
        if (
            isinstance(head_revision, bool)
            or not isinstance(head_revision, int)
            or not 1 <= head_revision <= MAX_PROJECTION_RECORDS
        ):
            errors.append("head_revision_out_of_bounds")
        if not isinstance(task_body_json, str):
            errors.append(
                "task_body_exceeds_bound"
                if task_body_byte_count > MAX_BODY_BYTES
                else "task_body_not_encoded_json"
            )
            task_body_bytes = b""
            task_body: Any = None
        else:
            task_body_bytes = task_body_json.encode("utf-8")
            try:
                task_body = strict_json_object(task_body_json)
            except (TypeError, ValueError, RecursionError, OverflowError):
                task_body = None
            if not isinstance(task_body, dict):
                errors.append("task_body_not_object")
        completion_receipt = (
            task_body.get("completion_receipt")
            if isinstance(task_body, dict)
            else None
        )
        completion_receipt_operation_raw = (
            completion_receipt.get("operation")
            if isinstance(completion_receipt, dict)
            else None
        )
        completion_receipt_operation = (
            completion_receipt_operation_raw
            if isinstance(completion_receipt_operation_raw, str)
            and len(completion_receipt_operation_raw.encode("utf-8"))
            <= MAX_ID_BYTES
            else ""
        )
        if (
            completion_receipt_operation_raw is not None
            and not completion_receipt_operation
        ):
            errors.append("completion_receipt_operation_invalid")

        history_population = connection.execute(
            "SELECT COUNT(*), MIN(revision), MAX(revision), "
            "COALESCE(MAX(octet_length(encode(status))), 0), "
            "COALESCE(MAX(octet_length(encode(body_json))), 0), "
            "COALESCE(MAX(octet_length(encode(recorded_at))), 0) "
            "FROM task_revisions WHERE task_cid = ?",
            [task_cid],
        ).fetchone()
        if history_population is None:
            raise SuccessorOperatorError(
                "stopped task history population is unavailable"
            )
        history_count = int(history_population[0])
        history_first_revision = (
            int(history_population[1])
            if history_population[1] is not None
            else -1
        )
        history_last_revision = (
            int(history_population[2])
            if history_population[2] is not None
            else -1
        )
        history_status_bytes = int(history_population[3])
        history_body_bytes = int(history_population[4])
        history_recorded_at_bytes = int(history_population[5])
        if history_count > MAX_PROJECTION_RECORDS:
            errors.append("history_population_exceeds_bound")
        if history_status_bytes > MAX_ID_BYTES:
            errors.append("history_status_exceeds_bound")
        if history_body_bytes > MAX_BODY_BYTES:
            errors.append("history_body_exceeds_bound")
        if history_recorded_at_bytes > MAX_ID_BYTES:
            errors.append("history_recorded_at_exceeds_bound")
        if history_count and (
            history_first_revision != 1
            or history_last_revision != history_count
        ):
            errors.append("history_revision_gap_or_reorder")
        material_bytes = len(
            canonical_json_bytes(
                {
                    "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
                    "task_cid": task_cid,
                    "revisions": [],
                }
            )
        )
        tail_status = ""
        tail_body_json = ""
        tail_recorded_at = ""
        history_fields_bounded = (
            history_count <= MAX_PROJECTION_RECORDS
            and history_status_bytes <= MAX_ID_BYTES
            and history_body_bytes <= MAX_BODY_BYTES
            and history_recorded_at_bytes <= MAX_ID_BYTES
        )
        if history_count and history_fields_bounded:
            tail_row = connection.execute(
                "SELECT status, body_json, recorded_at FROM task_revisions "
                "WHERE task_cid = ? ORDER BY revision DESC LIMIT 1",
                [task_cid],
            ).fetchone()
            if tail_row is None:
                raise SuccessorOperatorError(
                    "stopped task history changed during history audit"
                )
            tail_status = str(tail_row[0] or "")
            tail_body_json = str(tail_row[1] or "")
            tail_recorded_at = str(tail_row[2] or "")
            prior_revision: int | None = None
            for index in range(1, history_count + 1):
                if prior_revision is None:
                    history_row = connection.execute(
                        "SELECT revision, status, body_json "
                        "FROM task_revisions WHERE task_cid = ? "
                        "ORDER BY revision LIMIT 1",
                        [task_cid],
                    ).fetchone()
                else:
                    history_row = connection.execute(
                        "SELECT revision, status, body_json "
                        "FROM task_revisions WHERE task_cid = ? AND revision > ? "
                        "ORDER BY revision LIMIT 1",
                        [task_cid, prior_revision],
                    ).fetchone()
                if history_row is None:
                    raise SuccessorOperatorError(
                        "stopped task history changed during history audit"
                    )
                revision = history_row[0]
                status = str(history_row[1] or "")
                body_json = history_row[2]
                if (
                    isinstance(revision, bool)
                    or not isinstance(revision, int)
                    or revision != index
                ):
                    errors.append("history_revision_gap_or_reorder")
                if not isinstance(body_json, str):
                    errors.append("history_body_not_encoded_json")
                    body: Any = None
                else:
                    try:
                        body = strict_json_object(body_json)
                    except (TypeError, ValueError, RecursionError, OverflowError):
                        body = None
                if not isinstance(body, dict):
                    errors.append("history_body_not_object")
                entry = {
                    "revision": revision,
                    "status": status,
                    "body": body if isinstance(body, dict) else {},
                }
                try:
                    entry_bytes = len(canonical_json_bytes(entry))
                except (TypeError, ValueError, RecursionError, OverflowError):
                    errors.append("history_body_not_canonicalizable")
                    entry_bytes = 0
                material_bytes += entry_bytes + (1 if index > 1 else 0)
                if isinstance(revision, int) and not isinstance(revision, bool):
                    prior_revision = revision
                if material_bytes > MAX_PLAN_PROJECTION_BYTES:
                    errors.append("history_projection_exceeds_byte_bound")
                    break
        if material_bytes > MAX_PLAN_PROJECTION_BYTES:
            errors.append("history_projection_exceeds_byte_bound")
        if not isinstance(head_revision, bool) and isinstance(head_revision, int):
            if history_count != head_revision:
                errors.append("history_count_differs_from_head")
        if (
            not history_count
            or tail_status != task_status
            or tail_body_json != (
                task_body_json if isinstance(task_body_json, str) else ""
            )
            or tail_recorded_at != task_updated_at
        ):
            errors.append("history_tail_differs_from_task")
        errors = sorted(set(errors))
        valid = not errors
        valid_count += int(valid)
        audits.append(
            {
                "task_cid": task_cid,
                "task_alias": task_alias,
                "status": task_status,
                "head_revision": (
                    int(head_revision)
                    if isinstance(head_revision, int)
                    and not isinstance(head_revision, bool)
                    else -1
                ),
                "history_count": history_count,
                "history_first_revision": history_first_revision,
                "history_last_revision": history_last_revision,
                # This bounded discriminator lets a stopped operator decide
                # which recovery contract applies without exporting the task
                # body or treating the audit as mutation authority.
                "completion_receipt_operation": (
                    completion_receipt_operation
                ),
                "task_body_sha256": hashlib.sha256(task_body_bytes).hexdigest(),
                "tail_body_sha256": hashlib.sha256(
                    tail_body_json.encode("utf-8")
                ).hexdigest(),
                "valid": valid,
                "errors": errors,
            }
        )
    return {
        "valid": not global_errors and valid_count == task_count,
        "errors": sorted(set(global_errors)),
        "task_count": task_count,
        "history_row_count": history_row_count,
        "orphan_history_count": orphan_history_count,
        "valid_task_count": valid_count,
        "invalid_task_count": task_count - valid_count,
        "tasks": audits,
    }


def stopped_task_history_audit(root: Path = ROOT) -> dict[str, Any]:
    """Audit an immutable stopped control snapshot under continuity custody."""

    paths = _paths(root)
    with _exclusive_projection_checkpoint(
        paths,
        _read_only_existing_lock=True,
    ) as lock_custody:
        with _sealed_stopped_database_snapshots(paths, lock_custody) as snapshots:
            continuity = _load_projection_source_continuity(
                paths,
                root=root,
                stopped_database_snapshots=snapshots,
                lock_custody=lock_custody,
            )
            control_snapshot = snapshots["control"]
            from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
                connect_duckdb_with_policy,
            )

            import duckdb

            try:
                connection = connect_duckdb_with_policy(
                    duckdb,
                    str(control_snapshot["snapshot_path"]),
                    read_only=True,
                )
                try:
                    audit = _audit_task_history_connection(connection)
                finally:
                    connection.close()
            except SuccessorOperatorError:
                raise
            except Exception as exc:
                raise SuccessorOperatorError(
                    "stopped task-history policy read failed: "
                    f"{type(exc).__name__}"
                ) from exc
            observed = _validate_stopped_database_snapshots(
                paths,
                lock_custody,
                snapshots,
            )
            if observed != continuity["databases"]:
                raise SuccessorOperatorError(
                    "stopped task-history source changed during audit"
                )
            body = {
                "schema": STOPPED_TASK_HISTORY_AUDIT_SCHEMA,
                "valid": bool(audit["valid"]),
                "authoritative": False,
                "mutation_authority": False,
                "source_admission_mode": continuity["admission_mode"],
                "source_provenance_cid": continuity["provenance"]["receipt_cid"],
                "stopped_state_continuity_receipt_cid": continuity["receipt"][
                    "receipt_cid"
                ],
                "control_database_sha256": observed["control"]["sha256"],
                "errors": audit["errors"],
                "task_count": audit["task_count"],
                "history_row_count": audit["history_row_count"],
                "orphan_history_count": audit["orphan_history_count"],
                "valid_task_count": audit["valid_task_count"],
                "invalid_task_count": audit["invalid_task_count"],
                "tasks": audit["tasks"],
            }
            return {**body, "audit_cid": _content_id(body)}


def _closed_json_object_bytes(raw: bytes, *, noun: str) -> dict[str, Any]:
    """Decode one bounded JSON object while rejecting duplicate/non-finite data."""

    def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, member in pairs:
            if name in value:
                raise ValueError(f"duplicate {noun} field")
            value[name] = member
        return value

    def reject_constant(_value: str) -> Any:
        raise ValueError(f"non-finite {noun} number")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(f"{noun} is malformed") from exc
    if not isinstance(value, dict):
        raise SuccessorOperatorError(f"{noun} root is not an object")
    return value


def _protected_qualification_stable_projection(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce the tracked qualifier's deliberately duration-free projection."""

    raw_suites = value.get("suites")
    if not isinstance(raw_suites, list) or any(
        not isinstance(item, Mapping) for item in raw_suites
    ):
        raise SuccessorOperatorError(
            "protected qualification suite projection is malformed"
        )
    suites = [
        {
            key: item.get(key)
            for key in (
                "schema",
                "suite_id",
                "manifest",
                "collected",
                "passed_count",
                "failed_count",
                "skipped_count",
                "xfailed_count",
                "xpassed_count",
                "error_count",
                "nodeids_cid",
                "exit_code",
                "passed",
                "isolation",
            )
        }
        for item in raw_suites
    ]
    return {
        key: value.get(key)
        for key in (
            "schema",
            "plan_cid",
            "predecessor_plan_cid",
            "cohort",
            "candidate_suites_are_self_authority",
            "independent_fixed_manifest_executed",
            "checkout_fingerprint_cid",
            "checkout_unchanged",
            "passed",
            "totals",
            "task_implementation_complete",
            "test_qualification_complete",
            "objective_complete",
            "release_qualified",
            "production_authorized",
            "production_authoritative",
            "limitations",
        )
    } | {"suites": suites}


def _validate_protected_qualification_result(
    value: Mapping[str, Any],
    *,
    noun: str,
) -> None:
    expected_fields = {
        "schema",
        "plan_cid",
        "predecessor_plan_cid",
        "cohort",
        "candidate_suites_are_self_authority",
        "independent_fixed_manifest_executed",
        "checkout_fingerprint_cid",
        "checkout_unchanged",
        "passed",
        "totals",
        "suites",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
        "limitations",
        "result_cid",
    }
    body = {name: member for name, member in value.items() if name != "result_cid"}
    if (
        set(value) != expected_fields
        or value.get("schema") != "lgcvf-independent-hermetic-qualification@1"
        or value.get("cohort") != "hermetic_local_execution"
        or value.get("result_cid") != _content_id(body)
        or value.get("candidate_suites_are_self_authority") is not False
        or value.get("independent_fixed_manifest_executed") is not True
        or value.get("checkout_unchanged") is not True
        or value.get("passed") is not True
        or value.get("test_qualification_complete") is not True
        or value.get("task_implementation_complete") is not False
        or value.get("objective_complete") is not False
        or value.get("release_qualified") is not False
        or value.get("production_authorized") is not False
        or value.get("production_authoritative") is not False
    ):
        raise SuccessorOperatorError(f"{noun} authority binding differs")
    _protected_qualification_stable_projection(value)


def _tracked_regular_file_pin(
    root: Path,
    relative: Path,
    *,
    noun: str,
    max_bytes: int,
) -> tuple[bytes, dict[str, str]]:
    path = _contained(root, relative)
    raw = _read_bounded_regular_file(path, max_bytes=max_bytes, noun=noun)
    observed_blob = _regular_git_blob_oid(path, noun=noun)
    expected_blob = _git_text(
        root,
        ("rev-parse", f"HEAD:{relative.as_posix()}"),
        noun=f"{noun} tracked blob",
    )
    if observed_blob != expected_blob or re.fullmatch(
        r"[0-9a-f]{40}", expected_blob
    ) is None:
        raise SuccessorOperatorError(f"{noun} differs from tracked HEAD")
    return raw, {
        "path": relative.as_posix(),
        "blob_oid": expected_blob,
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _run_protected_qualification_check(root: Path) -> dict[str, Any]:
    """Run and bind the tracked fixed-manifest qualifier without provider access."""

    validator_raw, validator_pin = _tracked_regular_file_pin(
        root,
        PROTECTED_QUALIFICATION_RELATIVE,
        noun="protected qualification validator",
        max_bytes=MAX_JSON_BYTES * 4,
    )
    artifact_raw, artifact_pin = _tracked_regular_file_pin(
        root,
        PROTECTED_QUALIFICATION_RESULT_RELATIVE,
        noun="protected qualification result",
        max_bytes=MAX_JSON_BYTES,
    )
    stored = _closed_json_object_bytes(
        artifact_raw,
        noun="stored protected qualification result",
    )
    expected_artifact_encoding = (
        json.dumps(stored, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    if artifact_raw != expected_artifact_encoding:
        raise SuccessorOperatorError(
            "stored protected qualification result encoding differs"
        )
    _validate_protected_qualification_result(
        stored,
        noun="stored protected qualification result",
    )
    environment = {
        "HOME": str(Path(tempfile.gettempdir()).resolve()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NO_COLOR": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    argv = [
        sys.executable,
        "-B",
        str(_contained(root, PROTECTED_QUALIFICATION_RELATIVE)),
        "--check",
    ]
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            env=environment,
            capture_output=True,
            check=False,
            timeout=PROTECTED_QUALIFICATION_CHECK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SuccessorOperatorError(
            "protected qualification replay could not complete"
        ) from exc
    if (
        len(completed.stdout) > MAX_JSON_BYTES
        or len(completed.stderr) > MAX_JSON_BYTES
        or completed.returncode != 0
    ):
        detail = completed.stderr[-1000:] or completed.stdout[-1000:]
        raise SuccessorOperatorError(
            "protected qualification replay failed: "
            + detail.decode("utf-8", errors="replace").strip()
        )
    replay = _closed_json_object_bytes(
        completed.stdout,
        noun="replayed protected qualification result",
    )
    if completed.stdout != (
        json.dumps(replay, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    ):
        raise SuccessorOperatorError(
            "replayed protected qualification result encoding differs"
        )
    _validate_protected_qualification_result(
        replay,
        noun="replayed protected qualification result",
    )
    stored_stable = _protected_qualification_stable_projection(stored)
    replay_stable = _protected_qualification_stable_projection(replay)
    if stored_stable != replay_stable:
        raise SuccessorOperatorError(
            "protected qualification stable projection differs"
        )
    recorded_at = _git_text(
        root,
        (
            "log",
            "-1",
            "--format=%cI",
            "HEAD",
            "--",
            PROTECTED_QUALIFICATION_RESULT_RELATIVE.as_posix(),
        ),
        noun="protected qualification recording time",
    ).replace("+00:00", "Z")
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z",
        recorded_at,
    ) is None:
        raise SuccessorOperatorError(
            "protected qualification recording time is malformed"
        )
    stable_cid = _content_id(stored_stable)
    return {
        "argv": [
            "python",
            PROTECTED_QUALIFICATION_RELATIVE.as_posix(),
            "--check",
        ],
        "validator_path": validator_pin["path"],
        "validator_blob_oid": validator_pin["blob_oid"],
        "validator_sha256": "sha256:" + hashlib.sha256(validator_raw).hexdigest(),
        "artifact_path": artifact_pin["path"],
        "artifact_blob_oid": artifact_pin["blob_oid"],
        "artifact_sha256": artifact_pin["sha256"],
        "stored_result_cid": stored["result_cid"],
        "stored_stable_projection_cid": stable_cid,
        "replay_stable_projection_cid": _content_id(replay_stable),
        "replay_exit_code": 0,
        "recorded_at": recorded_at,
        "passed": True,
        "test_qualification_complete": True,
        "independent_fixed_manifest_executed": True,
        "candidate_suites_are_self_authority": False,
        "task_implementation_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authoritative": False,
        "production_authorized": False,
    }


def _protected_qualification_completion_snapshot(
    connection: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the exact LGCVF-113 prestate from one immutable control snapshot."""

    task_rows = connection.execute(
        "SELECT task_alias, status, revision, body_json FROM tasks "
        "WHERE task_cid = ? LIMIT 2",
        [PROTECTED_QUALIFICATION_COMPLETION_TASK_CID],
    ).fetchall()
    if len(task_rows) != 1:
        raise SuccessorOperatorError(
            "protected qualification task is absent or ambiguous"
        )
    task_alias, status, revision, body_json = task_rows[0]
    if not isinstance(body_json, str):
        raise SuccessorOperatorError(
            "protected qualification task body is malformed"
        )
    prior_body = _closed_json_object_bytes(
        body_json.encode("utf-8"),
        noun="protected qualification task body",
    )
    if _canonical_bytes(prior_body).decode("utf-8") != body_json:
        raise SuccessorOperatorError(
            "protected qualification task body is not canonical"
        )
    body_sha256 = "sha256:" + hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    prior_receipt = prior_body.get("completion_receipt")
    if not isinstance(prior_receipt, Mapping):
        raise SuccessorOperatorError(
            "protected qualification prior receipt is absent"
        )
    history = [
        int(row[0])
        for row in connection.execute(
            "SELECT revision FROM task_revisions WHERE task_cid = ? "
            "ORDER BY revision",
            [PROTECTED_QUALIFICATION_COMPLETION_TASK_CID],
        ).fetchall()
    ]
    dependency_rows = connection.execute(
        "SELECT d.dependency_task_cid, t.task_alias, t.status, t.revision, "
        "t.body_json FROM task_dependencies AS d JOIN tasks AS t "
        "ON t.task_cid = d.dependency_task_cid WHERE d.task_cid = ? "
        "ORDER BY d.dependency_task_cid",
        [PROTECTED_QUALIFICATION_COMPLETION_TASK_CID],
    ).fetchall()
    dependencies = [
        {
            "task_cid": str(row[0]),
            "task_alias": str(row[1]),
            "status": str(row[2]),
            "revision": int(row[3]),
            "body_sha256": "sha256:"
            + hashlib.sha256(str(row[4]).encode("utf-8")).hexdigest(),
        }
        for row in dependency_rows
    ]
    expected_dependency_pairs = sorted(
        (
            cid,
            alias,
        )
        for alias, cid in PROTECTED_QUALIFICATION_COMPLETION_DEPENDENCIES.items()
    )
    if (
        str(task_alias) != PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS
        or str(status) != "retrying"
        or type(revision) is not int
        or revision != PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION
        or body_sha256 != PROTECTED_QUALIFICATION_COMPLETION_PRIOR_BODY_SHA256
        or prior_receipt.get("schema")
        != PROTECTED_QUALIFICATION_COMPLETION_PRIOR_RECEIPT_SCHEMA
        or prior_receipt.get("operation")
        != "database_claim_lost_sidecar_recovery"
        or history != [1]
        or [
            (item["task_cid"], item["task_alias"])
            for item in dependencies
        ]
        != expected_dependency_pairs
        or any(item["status"] != "completed" for item in dependencies)
    ):
        raise SuccessorOperatorError(
            "protected qualification task prestate differs"
        )
    task_pin = {
        "task_cid": PROTECTED_QUALIFICATION_COMPLETION_TASK_CID,
        "task_alias": PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS,
        "status": "retrying",
        "revision": PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION,
        "body_sha256": body_sha256,
        "prior_receipt_schema": prior_receipt["schema"],
        "prior_receipt_operation": prior_receipt["operation"],
        "prior_receipt_cid": _content_id(prior_receipt),
        "history_observed_revisions": history,
        "history_missing_revisions": list(
            range(2, PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION + 1)
        ),
        "gap_preserved": True,
        "repair_authority": False,
    }
    return (
        {
            "task": task_pin,
            "dependencies": dependencies,
            "dependency_binding_cid": _content_id(dependencies),
        },
        prior_body,
    )


def _protected_qualification_completion_preflight_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct deterministic review pins while all databases stay sealed."""

    with _sealed_stopped_database_snapshots(paths, lock_custody) as snapshots:
        continuity = _load_projection_source_continuity(
            paths,
            root=root,
            stopped_database_snapshots=snapshots,
            lock_custody=lock_custody,
        )
        source_continuity = _observe_candidate_runtime_continuity(
            root,
            require_resolved_remote=False,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            connect_duckdb_with_policy,
        )

        import duckdb

        connection = connect_duckdb_with_policy(
            duckdb,
            str(snapshots["control"]["snapshot_path"]),
            read_only=True,
        )
        try:
            state_pins, prior_body = _protected_qualification_completion_snapshot(
                connection
            )
        finally:
            connection.close()
        qualification = _run_protected_qualification_check(root)
        observed_databases = _validate_stopped_database_snapshots(
            paths,
            lock_custody,
            snapshots,
        )
        if (
            observed_databases != continuity["databases"]
            or _observe_candidate_runtime_continuity(
                root,
                require_resolved_remote=False,
            )
            != source_continuity
        ):
            raise SuccessorOperatorError(
                "protected qualification preflight source changed"
            )
        reviewed_pins = {
            "target_generation": SUCCESSOR_STORE_GENERATION,
            "source_provenance_cid": continuity["provenance"]["receipt_cid"],
            "stopped_state_continuity_receipt_cid": continuity["receipt"][
                "receipt_cid"
            ],
            "stopped_controller_status_cid": continuity["controller_status"][
                "status_cid"
            ],
            "source_continuity": source_continuity,
            "databases": observed_databases,
            **state_pins,
            "qualification": qualification,
        }
        binding = {
            "schema": PROTECTED_QUALIFICATION_COMPLETION_PREFLIGHT_SCHEMA,
            "operation": PROTECTED_QUALIFICATION_COMPLETION_OPERATION,
            "reviewed_pins": reviewed_pins,
        }
        report = {
            **binding,
            "observed_at": _utc_now(),
            "preflight_cid": _content_id(binding),
            "valid": True,
            "controller_lock_held": True,
            "owner_lock_held": False,
            "mutation_authority": False,
            "completion_authority": False,
            "scheduling_authority": False,
            "release_qualified": False,
            "production_authorized": False,
        }
        return report, prior_body


def protected_qualification_completion_preflight(
    root: Path = ROOT,
) -> dict[str, Any]:
    """Report exact qualification/task pins without mutating live state."""

    paths = _paths(root)
    with _exclusive_projection_checkpoint(
        paths,
        _read_only_existing_lock=True,
    ) as lock_custody:
        report, _prior_body = _protected_qualification_completion_preflight_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
        )
        return report


def _protected_qualification_completion_intent_path(
    paths: Mapping[str, Path],
    preflight_cid: str,
) -> Path:
    reviewed = str(preflight_cid or "").strip()
    if re.fullmatch(r"b[a-z2-7]{20,200}", reviewed) is None:
        raise SuccessorOperatorError(
            "protected qualification completion preflight CID is malformed"
        )
    return Path(paths["abandoned_owner_recovery_evidence"]) / (
        f"protected-qualification-completion-intent.{reviewed}.json"
    )


def _write_protected_qualification_completion_intent(
    paths: Mapping[str, Path],
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist exact reviewed authority before consuming restart custody."""

    reviewed = str(preflight.get("preflight_cid") or "")
    path = _protected_qualification_completion_intent_path(paths, reviewed)
    intent: dict[str, Any] = {
        "schema": PROTECTED_QUALIFICATION_COMPLETION_INTENT_SCHEMA,
        "issued_at": _utc_now(),
        "operation": PROTECTED_QUALIFICATION_COMPLETION_OPERATION,
        "preflight_cid": reviewed,
        "reviewed_pins": dict(preflight.get("reviewed_pins") or {}),
        "task_completion_scope": PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS,
        "objective_completion_authority": False,
        "scheduling_authority": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    intent["intent_cid"] = _content_id(intent)
    if os.path.lexists(path):
        existing = _strict_json(
            path,
            expected_schema=PROTECTED_QUALIFICATION_COMPLETION_INTENT_SCHEMA,
            require_private_owner=True,
        )
        repeated = dict(intent)
        repeated["issued_at"] = existing.get("issued_at")
        repeated["intent_cid"] = _content_id(
            {name: value for name, value in repeated.items() if name != "intent_cid"}
        )
        if existing != repeated:
            raise SuccessorOperatorError(
                "protected qualification completion intent differs"
            )
        return existing
    _atomic_json(path, intent, replace=False)
    if _strict_json(
        path,
        expected_schema=PROTECTED_QUALIFICATION_COMPLETION_INTENT_SCHEMA,
        require_private_owner=True,
    ) != intent:
        raise SuccessorOperatorError(
            "protected qualification completion intent changed"
        )
    return intent


def _observe_protected_qualification_completion_poststate(
    paths: Mapping[str, Path],
    *,
    lock_custody: Mapping[str, Any],
    reviewed_preflight_cid: str,
) -> dict[str, Any]:
    """Classify only exact pre- or post-command state through sealed snapshots."""

    with _sealed_stopped_database_snapshots(paths, lock_custody) as snapshots:
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            connect_duckdb_with_policy,
        )

        import duckdb

        connection = connect_duckdb_with_policy(
            duckdb,
            str(snapshots["control"]["snapshot_path"]),
            read_only=True,
        )
        try:
            rows = connection.execute(
                "SELECT task_alias, status, revision, body_json FROM tasks "
                "WHERE task_cid = ? LIMIT 2",
                [PROTECTED_QUALIFICATION_COMPLETION_TASK_CID],
            ).fetchall()
            history = [
                int(row[0])
                for row in connection.execute(
                    "SELECT revision FROM task_revisions WHERE task_cid = ? "
                    "ORDER BY revision",
                    [PROTECTED_QUALIFICATION_COMPLETION_TASK_CID],
                ).fetchall()
            ]
        finally:
            connection.close()
        _validate_stopped_database_snapshots(paths, lock_custody, snapshots)
    if len(rows) != 1 or not isinstance(rows[0][3], str):
        raise SuccessorOperatorError(
            "protected qualification completion poststate is ambiguous"
        )
    alias, status, revision, body_json = rows[0]
    body = _closed_json_object_bytes(
        body_json.encode("utf-8"),
        noun="protected qualification completion poststate body",
    )
    receipt = body.get("completion_receipt")
    if (
        str(alias) == PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS
        and str(status) == "completed"
        and type(revision) is int
        and revision == PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION + 1
        and history
        == [1, PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION + 1]
        and isinstance(receipt, Mapping)
        and receipt.get("operation")
        == PROTECTED_QUALIFICATION_COMPLETION_OPERATION
        and receipt.get("reviewed_preflight_cid") == reviewed_preflight_cid
        and receipt.get("receipt_cid")
        == _content_id(
            {name: value for name, value in receipt.items() if name != "receipt_cid"}
        )
    ):
        return {
            "completed": True,
            "task_revision": revision,
            "completion_receipt_cid": receipt["receipt_cid"],
            "history_revisions": history,
        }
    body_sha256 = "sha256:" + hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    if (
        str(alias) == PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS
        and str(status) == "retrying"
        and type(revision) is int
        and revision == PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION
        and history == [1]
        and body_sha256 == PROTECTED_QUALIFICATION_COMPLETION_PRIOR_BODY_SHA256
    ):
        return {
            "completed": False,
            "task_revision": revision,
            "completion_receipt_cid": "",
            "history_revisions": history,
        }
    raise SuccessorOperatorError(
        "protected qualification completion reached an unreviewed poststate"
    )


def _service_protected_qualification_completion_command(
    submit: Any,
) -> Any:
    """Submit on this thread through the client's bounded typed socket.

    The typed gateway is already serviced by its own owner threads.  Keeping
    this call synchronous is important: after it returns or raises there is no
    operator-created worker that can still be using ``client`` while the
    lifecycle finalizer closes the client, checkpoints, and stops the owner.
    """

    return submit()


def _checkpoint_and_stop_protected_qualification_owner(
    server: Any,
    identity: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Stop only after an exact checkpoint; otherwise preserve READY evidence."""

    owner_checkpoint = server.checkpoint()
    if (
        owner_checkpoint.get("checkpointed") is not True
        or owner_checkpoint.get("server_id") != identity.server_id
    ):
        raise SuccessorOperatorError(
            "protected qualification owner checkpoint differs"
        )
    owner_stop = server.stop()
    if (
        owner_stop.get("stopped") is not True
        or owner_stop.get("server_id") != identity.server_id
    ):
        raise SuccessorOperatorError(
            "protected qualification owner clean stop differs"
        )
    return owner_checkpoint, owner_stop


def _publish_protected_qualification_stopped_continuity(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    io_paths: Mapping[str, Any],
    provenance: Mapping[str, Any],
    identity: Any,
    controller_birth: Any,
    owner_checkpoint: Mapping[str, Any],
    owner_stop: Mapping[str, Any],
    reviewed_preflight_cid: str,
    owner_token: str,
    grant_token: str,
) -> dict[str, Any]:
    """Classify and publish exact stopped custody after one returned owner."""

    poststate = _observe_protected_qualification_completion_poststate(
        paths,
        lock_custody=lock_custody,
        reviewed_preflight_cid=reviewed_preflight_cid,
    )
    credential_leak = bool(tuple(paths["owner_state"].glob("*.quack-token")))
    for secret in (owner_token, grant_token):
        if not secret:
            continue
        for surface in (
            Path(io_paths["controller_status"]),
            Path(io_paths["owner_status"]),
        ):
            credential_leak = credential_leak or _regular_file_contains(
                surface,
                secret.encode("ascii"),
            )
    if credential_leak:
        raise SuccessorOperatorError(
            "protected qualification credential reached persistent state"
        )
    stopped = _status_payload(
        lifecycle="stopped",
        controller_birth=controller_birth.to_dict(),
        provenance_cid=str(provenance["receipt_cid"]),
        owner_identity=identity.to_dict(),
        scheduler_birth={},
        scheduler_returncode=0,
        error="",
        projection_root=paths["projection_root"],
    )
    stopped.pop("status_cid", None)
    stopped["protected_qualification_completion"] = {
        "schema": PROTECTED_QUALIFICATION_COMPLETION_STATUS_SCHEMA,
        "preflight_cid": reviewed_preflight_cid,
        "completion_receipt_cid": poststate["completion_receipt_cid"],
        "completed": poststate["completed"],
        "scheduling_attempted": False,
    }
    stopped["status_cid"] = _content_id(stopped)
    anchors = _capture_stopped_recovery_anchors(
        paths,
        root=root,
        stopped_status=stopped,
        provenance=provenance,
        io_paths=io_paths,
        lock_custody=lock_custody,
    )
    stopped = _bind_stopped_recovery_anchors_status(stopped, anchors)
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    _write_status(
        Path(io_paths["controller_status"]),
        stopped,
        token=owner_token,
    )
    continuity = _write_stopped_state_continuity(
        paths,
        root=root,
        stopped_status=stopped,
        provenance=provenance,
        owner_checkpoint=owner_checkpoint,
        owner_stop=owner_stop,
        _io_paths=io_paths,
    )
    final_status = _bind_stopped_state_continuity_status(
        stopped,
        continuity,
    )
    _write_status(
        Path(io_paths["controller_status"]),
        final_status,
        token=owner_token,
    )
    return {
        "poststate": poststate,
        "continuity": continuity,
        "final_status": final_status,
    }


def _complete_protected_qualification_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    reviewed_preflight_cid: str,
) -> dict[str, Any]:
    """Apply one exact typed validation/CAS/history transaction and reseal."""

    reviewed = str(reviewed_preflight_cid or "").strip()
    if re.fullmatch(r"b[a-z2-7]{20,200}", reviewed) is None:
        raise SuccessorOperatorError(
            "reviewed protected qualification preflight CID is required"
        )
    preflight, prior_body = _protected_qualification_completion_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
    )
    if preflight.get("preflight_cid") != reviewed:
        raise SuccessorOperatorError(
            "reviewed protected qualification preflight CID differs"
        )
    pins = preflight["reviewed_pins"]
    assert isinstance(pins, Mapping)
    task_pin = pins["task"]
    qualification = pins["qualification"]
    assert isinstance(task_pin, Mapping)
    assert isinstance(qualification, Mapping)
    intent = _write_protected_qualification_completion_intent(
        paths,
        preflight=preflight,
    )
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    receipt_paths = _stopped_receipt_io_view(paths, io_paths)
    provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    if provenance.get("receipt_cid") != pins["source_provenance_cid"]:
        raise SuccessorOperatorError(
            "protected qualification provenance changed after review"
        )
    live_launch: Mapping[str, Any] | None = None
    server: Any | None = None
    identity: Any | None = None
    client: Any | None = None
    grant: Any | None = None
    grant_token = ""
    owner_token = ""
    command_result: Any | None = None
    primary_error: BaseException | None = None
    primary_phase = ""
    command_attempted = False
    cleanup_errors: list[tuple[str, BaseException]] = []
    owner_checkpoint: Mapping[str, Any] = {}
    owner_stop: Mapping[str, Any] = {}
    publication: Mapping[str, Any] | None = None
    finalization_error: BaseException | None = None
    previous_environment: dict[str, str | None] = {}
    claimed = False
    start_attempted = False
    config_path = _contained(root, DEFAULT_SUCCESSOR_CONFIG_RELATIVE)
    try:
        live_launch = _prepare_lgcvf_configured_board_live_launch(
            root=root,
            config_path=config_path,
            provenance=provenance,
            stopped_restart=True,
        )
        if live_launch.get("continuity") != pins["source_continuity"]:
            raise SuccessorOperatorError(
                "protected qualification source changed after review"
            )
        launch_home = Path(str(live_launch["launch_home"]))
        extension_environment = {
            "HOME": str(launch_home),
            "IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME": str(launch_home),
            "XDG_CACHE_HOME": str(launch_home / ".cache" / "xdg"),
            "CUDA_CACHE_PATH": str(launch_home / ".cache" / "cuda"),
            "CUDA_CACHE_DISABLE": "1",
            BOARD_EXTENSION_INSTALL_POLICY_ENV: (
                BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
            ),
            STORE_GENERATION_ENV: SUCCESSOR_STORE_GENERATION,
        }
        previous_environment.update(
            {name: os.environ.get(name) for name in extension_environment}
        )
        if any(
            name.startswith("LD_") or name == "GLIBC_TUNABLES"
            for name in os.environ
        ):
            raise SuccessorOperatorError(
                "LGCVF protected completion inherited loader authority"
            )
        os.environ.update(extension_environment)
        from ipfs_accelerate_py.llm_router import (
            preload_agent_supervisor_native_dependency,
        )

        if "duckdb" not in sys.modules and "_duckdb" not in sys.modules:
            preload_agent_supervisor_native_dependency(live_launch["native_launch"])
        from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
            process_birth_id,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            current_process_birth,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            configured_board_launch_plan,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
            establish_state_authority_process_boundary,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
            build_server,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
            QuackStateClient,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
            TYPED_DATABASE_PROTECTED_QUALIFICATION_COMPLETION_CLIENT_ID,
            TYPED_DATABASE_PROTECTED_QUALIFICATION_COMPLETION_COMMAND,
            TYPED_STATE_OWNER_SOCKET_ENV,
            TYPED_STATE_OWNER_TOKEN_ENV,
        )

        board = live_launch["board"]
        program = live_launch["program"]
        rendered = configured_board_launch_plan(
            board,
            implement=False,
            detach=False,
            duration_seconds=1.0,
        ).get("environment")
        if not isinstance(rendered, Mapping):
            raise SuccessorOperatorError(
                "protected qualification database program is unavailable"
            )
        owner_program_json = str(
            rendered.get(DATABASE_PROGRAM_JSON_ENV) or ""
        ).strip()
        if not owner_program_json:
            raise SuccessorOperatorError(
                "protected qualification database program is unavailable"
            )
        previous_environment[DATABASE_PROGRAM_JSON_ENV] = os.environ.get(
            DATABASE_PROGRAM_JSON_ENV
        )
        os.environ[DATABASE_PROGRAM_JSON_ENV] = owner_program_json
        establish_state_authority_process_boundary()
        paths["owner_state"].mkdir(mode=0o700, parents=True, exist_ok=True)
        _prepare_private_owner_socket(paths["owner_socket"])
        controller_birth = current_process_birth()
        birth_id = process_birth_id(controller_birth)
        server = build_server(
            database_path=paths["successor_database"],
            state_dir=paths["owner_state"],
            host=str(live_launch["host"]),
            port=int(live_launch["port"]),
            repository_id="repository:lgcvf-quack-successor",
            store_id=program.store_id,
            secret_handle=program.endpoint_secret_handle,
            migrate=datasets_profile_migration,
            typed_command_socket_path=paths["owner_socket"],
            allow_legacy_board_unstall=False,
        )
        claimed = _claim_stopped_state_restart_admission(
            receipt_paths,
            expected_restart=True,
            expected_receipt_cid=str(
                pins["stopped_state_continuity_receipt_cid"]
            ),
            expected_controller_status_cid=str(
                pins["stopped_controller_status_cid"]
            ),
        )
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        # From this point forward the old receipt must never be restored.  A
        # failed start can already have opened/migrated the database or left
        # READY evidence for the abandoned-owner recovery path.
        start_attempted = True
        identity = server.start()
        try:
            if (
                identity.listen_uri != program.quack_endpoint
                or identity.store_id != program.store_id
                or identity.database_uuid != provenance.get("database_uuid")
                or not _owner_schema_fingerprint_matches_canonical_cid(
                    identity.schema_fingerprint,
                    provenance.get("schema_fingerprint"),
                )
                or server.typed_command_socket_path() != paths["owner_socket"]
                or server.status().get("legacy_board_unstall_enabled") is not False
            ):
                raise SuccessorOperatorError(
                    "protected qualification owner identity differs"
                )
            if server._vault is None:
                raise SuccessorOperatorError(
                    "protected qualification owner vault is unavailable"
                )
            owner_token = server._vault.resolve(identity.secret_handle)
            allowed_operations = (
                "whoami_metadata",
                "load_store_generation",
                "txn_load_generation",
                "txn_lookup_idempotency",
                "txn_advance_store_revision",
                "txn_record_idempotency",
                "executor_insert_validation_run",
                "executor_insert_validation_result",
                "executor_insert_validation_evidence",
                "executor_cas_task_status_receipt",
                "executor_insert_task_revision_history",
            )
            grant_token, grant = server.issue_typed_client_grant_record(
                client_id=(
                    TYPED_DATABASE_PROTECTED_QUALIFICATION_COMPLETION_CLIENT_ID
                ),
                process_birth_id=birth_id,
                allowed_operations=allowed_operations,
                allowed_command_operations=(
                    TYPED_DATABASE_PROTECTED_QUALIFICATION_COMPLETION_COMMAND,
                ),
                entity_scopes={
                    "task_cid": PROTECTED_QUALIFICATION_COMPLETION_TASK_CID,
                    "reviewed_preflight_cid": reviewed,
                },
                peer_pid=os.getpid(),
                ttl_seconds=(
                    PROTECTED_QUALIFICATION_COMMAND_GRANT_TTL_SECONDS
                ),
            )
            previous_environment[TYPED_STATE_OWNER_TOKEN_ENV] = os.environ.get(
                TYPED_STATE_OWNER_TOKEN_ENV
            )
            previous_environment[TYPED_STATE_OWNER_SOCKET_ENV] = os.environ.get(
                TYPED_STATE_OWNER_SOCKET_ENV
            )
            os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = grant_token
            os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = str(paths["owner_socket"])
            client = QuackStateClient(
                owner_id=(
                    TYPED_DATABASE_PROTECTED_QUALIFICATION_COMPLETION_CLIENT_ID
                ),
                store_id=str(program.store_id),
                process_birth_id=birth_id,
                connect_timeout_seconds=(
                    PROTECTED_QUALIFICATION_COMMAND_TIMEOUT_SECONDS
                ),
            )
            client.attach(
                str(program.quack_endpoint),
                server_id=identity.server_id,
            )
            os.environ.pop(TYPED_STATE_OWNER_TOKEN_ENV, None)
            os.environ.pop(TYPED_STATE_OWNER_SOCKET_ENV, None)
            reviewed_binding = {
                "schema": preflight["schema"],
                "operation": preflight["operation"],
                "reviewed_pins": pins,
            }
            command_attempted = True
            command_result = _service_protected_qualification_completion_command(
                lambda: client.complete_protected_qualification_legacy_gap(
                    task_cid=PROTECTED_QUALIFICATION_COMPLETION_TASK_CID,
                    task_alias=PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS,
                    expected_task_revision=int(task_pin["revision"]),
                    expected_prior_body_sha256=str(task_pin["body_sha256"]),
                    expected_prior_receipt_schema=str(
                        task_pin["prior_receipt_schema"]
                    ),
                    expected_prior_receipt_operation=str(
                        task_pin["prior_receipt_operation"]
                    ),
                    expected_prior_receipt_cid=str(
                        task_pin["prior_receipt_cid"]
                    ),
                    expected_history_revisions=list(
                        task_pin["history_observed_revisions"]
                    ),
                    expected_dependency_cids=[
                        str(item["task_cid"])
                        for item in pins["dependencies"]
                    ],
                    prior_body=prior_body,
                    reviewed_preflight_cid=reviewed,
                    reviewed_preflight=reviewed_binding,
                    qualification_result_cid=str(
                        qualification["stored_result_cid"]
                    ),
                    qualification_stable_projection_cid=str(
                        qualification["stored_stable_projection_cid"]
                    ),
                    qualification_artifact_sha256=str(
                        qualification["artifact_sha256"]
                    ),
                )
            )
        except BaseException as exc:  # noqa: BLE001
            primary_error = exc
            primary_phase = "command" if command_attempted else "owner setup"
        finally:
            if TYPED_STATE_OWNER_TOKEN_ENV in previous_environment:
                os.environ.pop(TYPED_STATE_OWNER_TOKEN_ENV, None)
            if TYPED_STATE_OWNER_SOCKET_ENV in previous_environment:
                os.environ.pop(TYPED_STATE_OWNER_SOCKET_ENV, None)
            if client is not None:
                try:
                    client.close()
                except BaseException as exc:  # noqa: BLE001
                    cleanup_errors.append(("client close", exc))
                client = None
            if grant is not None:
                try:
                    server.revoke_typed_client_grant(grant.grant_id)
                except BaseException as exc:  # noqa: BLE001
                    cleanup_errors.append(("grant revoke", exc))
                grant = None
            # Do not turn an uncheckpointed READY owner into unclassified
            # STOPPED residue.  Its durable READY/marker evidence is the input
            # to the next-process abandoned-owner recovery path.
            try:
                owner_checkpoint, owner_stop = (
                    _checkpoint_and_stop_protected_qualification_owner(
                        server,
                        identity,
                    )
                )
            except BaseException as exc:  # noqa: BLE001
                finalization_error = exc
            else:
                server = None
            if finalization_error is None:
                try:
                    publication = (
                        _publish_protected_qualification_stopped_continuity(
                            paths,
                            root=root,
                            lock_custody=lock_custody,
                            io_paths=io_paths,
                            provenance=provenance,
                            identity=identity,
                            controller_birth=controller_birth,
                            owner_checkpoint=owner_checkpoint,
                            owner_stop=owner_stop,
                            reviewed_preflight_cid=reviewed,
                            owner_token=owner_token,
                            grant_token=grant_token,
                        )
                    )
                except BaseException as exc:  # noqa: BLE001
                    finalization_error = exc
        owner_token = ""
        for phase, cleanup_error in cleanup_errors:
            if primary_error is not None:
                primary_error.add_note(
                    "protected qualification cleanup recovered from "
                    f"{phase}: {type(cleanup_error).__name__}"
                )
            else:
                sys.stderr.write(
                    "LGCVF protected completion cleanup recovered from "
                    f"{phase}: {type(cleanup_error).__name__}\n"
                )
        if finalization_error is not None:
            if primary_error is not None:
                primary_error.add_note(
                    "protected qualification stopped-state finalization also "
                    f"failed: {type(finalization_error).__name__}"
                )
                raise primary_error
            raise finalization_error
        if publication is None:
            raise SuccessorOperatorError(
                "protected qualification stopped-state publication is absent"
            )
        poststate = publication["poststate"]
        continuity = publication["continuity"]
        final_status = publication["final_status"]
        assert isinstance(poststate, Mapping)
        assert isinstance(continuity, Mapping)
        assert isinstance(final_status, Mapping)
        if not poststate["completed"]:
            if primary_error is not None:
                raise SuccessorOperatorError(
                    "protected qualification completion "
                    f"{primary_phase} failed safely"
                ) from primary_error
            raise SuccessorOperatorError(
                "protected qualification completion did not commit"
            )
        if primary_error is not None and primary_phase != "command":
            raise SuccessorOperatorError(
                "protected qualification reached a completed poststate after "
                "an owner setup failure"
            ) from primary_error
        if command_result is None and primary_error is None:
            raise SuccessorOperatorError(
                "protected qualification completion result is absent"
            )
        if command_result is not None and (
            command_result.outcome.value not in {"accepted", "idempotent_replay"}
            or command_result.changed
            is not (command_result.outcome.value == "accepted")
            or command_result.result.get("task_revision")
            != PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION + 1
            or command_result.result.get("completion_receipt_cid")
            != poststate["completion_receipt_cid"]
        ):
            raise SuccessorOperatorError(
                "protected qualification completion result differs"
            )
        return {
            "schema": PROTECTED_QUALIFICATION_COMPLETION_RESULT_SCHEMA,
            "operation": PROTECTED_QUALIFICATION_COMPLETION_OPERATION,
            "completed": True,
            "response_recovered": primary_error is not None,
            "preflight_cid": reviewed,
            "intent_cid": intent["intent_cid"],
            "completion_receipt_cid": poststate["completion_receipt_cid"],
            "task_alias": PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS,
            "task_revision": poststate["task_revision"],
            "history_revisions": poststate["history_revisions"],
            "stopped_state_continuity_receipt_cid": continuity["receipt_cid"],
            "controller_status_cid": final_status["status_cid"],
            "task_completion_effect": True,
            "objective_completion_authority": False,
            "scheduling_authority": False,
            "release_qualified": False,
            "production_authorized": False,
        }
    finally:
        if client is not None:
            try:
                client.close()
            except BaseException as cleanup_exc:  # noqa: BLE001
                sys.stderr.write(
                    "LGCVF protected completion client cleanup failed: "
                    f"{type(cleanup_exc).__name__}\n"
                )
        if grant is not None and server is not None:
            try:
                server.revoke_typed_client_grant(grant.grant_id)
            except BaseException as cleanup_exc:  # noqa: BLE001
                sys.stderr.write(
                    "LGCVF protected completion grant cleanup failed: "
                    f"{type(cleanup_exc).__name__}\n"
                )
        if server is not None and not start_attempted:
            try:
                server.stop()
            except BaseException as cleanup_exc:  # noqa: BLE001
                sys.stderr.write(
                    "LGCVF protected completion owner stop failed: "
                    f"{type(cleanup_exc).__name__}\n"
                )
        if claimed and not start_attempted:
            try:
                _restore_or_retire_stopped_restart_admission(receipt_paths)
            except BaseException as cleanup_exc:  # noqa: BLE001
                sys.stderr.write(
                    "LGCVF protected completion receipt restore failed: "
                    f"{type(cleanup_exc).__name__}\n"
                )
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        try:
            _close_lgcvf_configured_board_live_launch(live_launch)
        except BaseException as cleanup_exc:  # noqa: BLE001
            sys.stderr.write(
                "LGCVF protected completion capsule cleanup failed: "
                f"{type(cleanup_exc).__name__}\n"
            )


def complete_protected_qualification(
    root: Path = ROOT,
    *,
    reviewed_preflight_cid: str,
) -> dict[str, Any]:
    """Complete exact LGCVF-113 under one reviewed owner transaction."""

    paths = _paths(root)
    with _exclusive_projection_checkpoint(paths) as lock_custody:
        return _complete_protected_qualification_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
            reviewed_preflight_cid=reviewed_preflight_cid,
        )


def _recover_interrupted_protected_qualification_publication_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Finish only an anchored protected-completion continuity publication."""

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    status_path = Path(io_paths["controller_status"])
    if not os.path.lexists(status_path):
        return None
    status = _strict_json(
        status_path,
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    completion = status.get("protected_qualification_completion")
    if not isinstance(completion, Mapping):
        return None
    if (
        completion.get("schema")
        != PROTECTED_QUALIFICATION_COMPLETION_STATUS_SCHEMA
        or status.get("lifecycle") != "stopped"
        or status.get("error") != ""
        or type(status.get("scheduler_returncode")) is not int
        or status.get("scheduler_returncode") != 0
    ):
        raise SuccessorOperatorError(
            "interrupted protected qualification status differs"
        )
    linked_receipt = status.get("stopped_state_continuity_receipt_cid")
    linked_status = status.get("stopped_state_continuity_status_cid")
    if (linked_receipt is None) != (linked_status is None):
        raise SuccessorOperatorError(
            "interrupted protected qualification continuity links are partial"
        )
    if linked_receipt is not None:
        return None
    provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    receipt = _recover_interrupted_stopped_state_continuity(
        paths,
        root=root,
        lock_custody=lock_custody,
        provenance=provenance,
    )
    if not isinstance(receipt, Mapping):
        raise SuccessorOperatorError(
            "interrupted protected qualification continuity was not published"
        )
    return {
        "recovered": True,
        "completed": completion.get("completed") is True,
        "preflight_cid": completion.get("preflight_cid"),
        "stopped_state_continuity_receipt_cid": receipt["receipt_cid"],
        "controller_status_cid": receipt["controller_status_cid"],
    }


def _automatically_complete_protected_qualification_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Apply the one hard-coded safe completion whenever its exact gap recurs."""

    with _sealed_stopped_database_snapshots(paths, lock_custody) as snapshots:
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            connect_duckdb_with_policy,
        )

        import duckdb

        connection = connect_duckdb_with_policy(
            duckdb,
            str(snapshots["control"]["snapshot_path"]),
            read_only=True,
        )
        try:
            rows = connection.execute(
                "SELECT task_alias, status, revision, body_json FROM tasks "
                "WHERE task_cid = ? LIMIT 2",
                [PROTECTED_QUALIFICATION_COMPLETION_TASK_CID],
            ).fetchall()
            history = [
                int(row[0])
                for row in connection.execute(
                    "SELECT revision FROM task_revisions WHERE task_cid = ? "
                    "ORDER BY revision",
                    [PROTECTED_QUALIFICATION_COMPLETION_TASK_CID],
                ).fetchall()
            ]
        finally:
            connection.close()
        _validate_stopped_database_snapshots(paths, lock_custody, snapshots)
    if len(rows) != 1 or not isinstance(rows[0][3], str):
        return None
    alias, status, revision, body_json = rows[0]
    exact_candidate = (
        str(alias) == PROTECTED_QUALIFICATION_COMPLETION_TASK_ALIAS
        and str(status) == "retrying"
        and type(revision) is int
        and revision == PROTECTED_QUALIFICATION_COMPLETION_PRIOR_REVISION
        and history == [1]
        and "sha256:"
        + hashlib.sha256(body_json.encode("utf-8")).hexdigest()
        == PROTECTED_QUALIFICATION_COMPLETION_PRIOR_BODY_SHA256
    )
    if not exact_candidate:
        return None
    try:
        preflight, _prior_body = (
            _protected_qualification_completion_preflight_locked(
                paths,
                root=root,
                lock_custody=lock_custody,
            )
        )
    except SuccessorOperatorError as exc:
        # A near-miss retrying row is not the hard-coded gap.  Do not stall
        # successor launch on an uncompletable protected-qualification task.
        if str(exc) == "protected qualification task prestate differs":
            return None
        raise
    result = _complete_protected_qualification_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
        reviewed_preflight_cid=str(preflight["preflight_cid"]),
    )
    return {**result, "automatically_invoked": True}


def _private_regular_stat_pin(
    path: Path,
    *,
    noun: str,
    require_private_mode: bool = True,
) -> dict[str, Any]:
    """Bind one private regular recovery surface without exporting its bytes."""

    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or (
            require_private_mode
            and stat.S_IMODE(metadata.st_mode) & 0o077
        )
        or (
            not require_private_mode
            and (
                stat.S_IMODE(metadata.st_mode) & 0o600 != 0o600
                or stat.S_IMODE(metadata.st_mode) & 0o111
            )
        )
        or metadata.st_nlink != 1
    ):
        raise SuccessorOperatorError(f"{noun} custody is unsafe")
    return {
        "path": str(path),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": stat.S_IMODE(metadata.st_mode),
        "links": int(metadata.st_nlink),
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def _strict_owner_marker_json(path: Path) -> dict[str, Any]:
    """Read the exact pretty-JSON encoding used by ExclusiveOwnerLease."""

    raw = _read_bounded_regular_file(
        path,
        max_bytes=MAX_JSON_BYTES,
        noun="abandoned state-owner marker",
        require_private_owner=True,
    )

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate owner marker key")
            value[key] = item
        return value

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                separators=(",", ": "),
            )
            + "\n"
        ).encode("utf-8")
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(
            "abandoned state-owner marker is malformed"
        ) from exc
    if not isinstance(payload, dict) or raw != encoded:
        raise SuccessorOperatorError(
            "abandoned state-owner marker encoding differs"
        )
    return payload


def _require_abandoned_owner_lock_free(database: Path) -> dict[str, Any]:
    """Prove the stale owner's flock is free without creating or replacing it."""

    lock_path = database.with_name(f".{database.name}.state-owner.lock")
    pin = _private_regular_stat_pin(
        lock_path,
        noun="abandoned state-owner lock",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise SuccessorOperatorError(
            "abandoned state-owner lock is unreadable"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuccessorOperatorError(
                "abandoned state-owner lock is still held"
            ) from exc
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(descriptor)
    if _private_regular_stat_pin(
        lock_path,
        noun="abandoned state-owner lock",
    ) != pin:
        raise SuccessorOperatorError(
            "abandoned state-owner lock changed during liveness proof"
        )
    return pin


def _abandoned_owner_wal_pins(
    paths: Mapping[str, Path],
    *,
    database_paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    """Describe, but never read or remove, WALs left by the dead owner tree."""

    logical = _successor_state_databases(paths)
    if set(database_paths) != set(logical):
        raise SuccessorOperatorError(
            "abandoned owner database path custody is incomplete"
        )
    pins: dict[str, dict[str, Any]] = {}
    for name, database in logical.items():
        actual = Path(database_paths[name])
        wal = actual.with_name(actual.name + ".wal")
        if not os.path.lexists(wal):
            continue
        pin = _private_regular_stat_pin(
            wal,
            noun=f"abandoned {name} database WAL",
            require_private_mode=False,
        )
        pin["path"] = str(database.with_name(database.name + ".wal"))
        pins[name] = pin
    return pins


def _abandoned_owner_source_observation(
    root: Path,
    sealed: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Admit an exact source automatically or one reviewed descendant."""

    try:
        observed = _observe_stopped_projection_source_continuity(root, sealed)
    except SuccessorOperatorError as exc:
        if str(exc) != "stopped-state final source continuity differs":
            raise
        observed = _observe_failed_start_source_maintenance_descendant(
            root,
            sealed,
        )
        return "reviewed_descendant", observed
    return "exact_source", observed


def _abandoned_owner_recovery_preflight_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pin an owner crash that occurred after restart custody was consumed."""

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        OwnerMarker,
    )

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    generation_inventory = _stopped_recovery_generation_inventory(
        paths,
        lock_custody,
    )
    reviewed_inventory = [
        list(item)
        for item in generation_inventory
        if item[0] != paths["controller_lock"].name
    ]
    observed_provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    if provenance is not None and observed_provenance != dict(provenance):
        raise SuccessorOperatorError(
            "abandoned owner recovery provenance changed"
        )
    canonical = Path(io_paths["stopped_state_continuity"])
    admission = Path(io_paths["stopped_state_restart_admission"])
    canonical_present = os.path.lexists(canonical)
    admission_present = os.path.lexists(admission)
    if canonical_present is admission_present:
        raise SuccessorOperatorError(
            "abandoned owner recovery requires exactly one restart receipt"
        )
    receipt_path = canonical if canonical_present else admission
    receipt_custody = "published" if canonical_present else "consumed"
    receipt = _strict_json(
        receipt_path,
        expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
        require_private_owner=True,
    )
    _validate_stopped_continuity_receipt_shape(
        paths,
        receipt,
        provenance=observed_provenance,
    )
    status = _strict_json(
        Path(io_paths["controller_status"]),
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    source_status = dict(status)
    source_status.pop("status_cid", None)
    linked_receipt_cid = source_status.pop(
        "stopped_state_continuity_receipt_cid",
        None,
    )
    linked_status_cid = source_status.pop(
        "stopped_state_continuity_status_cid",
        None,
    )
    source_status["status_cid"] = _content_id(source_status)
    if (
        linked_receipt_cid != receipt.get("receipt_cid")
        or linked_status_cid != receipt.get("controller_status_cid")
        or source_status["status_cid"] != linked_status_cid
    ):
        raise SuccessorOperatorError(
            "abandoned owner restart receipt/status binding differs"
        )
    if source_status.get("error") == FAILED_START_STATUS_ERROR:
        _validate_unbound_failed_start_controller_status(
            source_status,
            provenance=observed_provenance,
            require_dead=True,
        )
    elif source_status.get("error") == "":
        _validate_unbound_stopped_controller_status(
            source_status,
            provenance=observed_provenance,
        )
    else:
        raise SuccessorOperatorError(
            "abandoned owner source status is not restart authority"
        )

    sealed_source = receipt.get("final_source_continuity")
    if not isinstance(sealed_source, Mapping):
        raise SuccessorOperatorError(
            "abandoned owner source continuity is malformed"
        )
    source_mode, observed_source = _abandoned_owner_source_observation(
        root,
        sealed_source,
    )

    owner_status_path = Path(io_paths["owner_status"])
    owner_status = _strict_json(
        owner_status_path,
        expected_schema=QUACK_STATE_SERVER_STATUS_SCHEMA,
        require_private_owner=True,
        verify_content_identity=False,
    )
    stale_identity = owner_status.get("identity")
    stale_birth_raw = (
        stale_identity.get("process_birth")
        if isinstance(stale_identity, Mapping)
        else None
    )
    try:
        stale_birth = ProcessBirthIdentity.from_dict(stale_birth_raw)
    except (TypeError, ValueError) as exc:
        raise SuccessorOperatorError(
            "abandoned owner process birth is malformed"
        ) from exc
    expected_marker_path = Path(io_paths["owner_marker"])
    if (
        not isinstance(stale_identity, Mapping)
        or stale_birth.to_dict() != stale_birth_raw
        or stale_identity.get("status") != "ready"
        or owner_status.get("lifecycle") != "ready"
        or owner_status.get("database_path")
        != str(paths["successor_database"])
        or owner_status.get("state_dir") != str(paths["owner_state"])
        or owner_status.get("store_id")
        != SUCCESSOR_DATABASE_RELATIVE.as_posix()
        or owner_status.get("secret_handle") != SECRET_HANDLE
        or owner_status.get("owner_marker_path")
        != str(paths["successor_database"].with_name(
            ".control.duckdb.state-owner.json"
        ))
        or owner_liveness(stale_birth) is not OwnerLiveness.DEAD
    ):
        raise SuccessorOperatorError(
            "abandoned owner ready projection is not exactly dead"
        )

    marker_payload = _strict_owner_marker_json(expected_marker_path)
    try:
        marker = OwnerMarker.from_dict(marker_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SuccessorOperatorError(
            "abandoned owner marker is malformed"
        ) from exc
    if (
        marker.to_dict() != marker_payload
        or marker.server_id != stale_identity.get("server_id")
        or marker.process_birth != stale_birth
        or marker.database_path != str(paths["successor_database"])
        or owner_liveness(marker.process_birth) is not OwnerLiveness.DEAD
    ):
        raise SuccessorOperatorError(
            "abandoned owner marker binding differs"
        )
    marker_pin = _private_regular_stat_pin(
        expected_marker_path,
        noun="abandoned state-owner marker",
    )
    marker_pin["sha256"] = _sha256_regular_file(
        expected_marker_path,
        max_bytes=MAX_JSON_BYTES,
        noun="abandoned state-owner marker",
        require_private_owner=True,
    )
    owner_lock_pin = _require_abandoned_owner_lock_free(
        paths["successor_database"]
    )
    wal_pins = _abandoned_owner_wal_pins(
        paths,
        database_paths=io_paths["databases"],
    )
    pins: dict[str, Any] = {
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "source_provenance_cid": observed_provenance["receipt_cid"],
        "receipt_custody": receipt_custody,
        "stopped_state_continuity_receipt_cid": receipt["receipt_cid"],
        "stopped_controller_status_cid": status["status_cid"],
        "source_status_cid": source_status["status_cid"],
        "source_mode": source_mode,
        "source_continuity": observed_source,
        "abandoned_owner_identity": dict(stale_identity),
        "owner_status_sha256": _sha256_regular_file(
            owner_status_path,
            max_bytes=MAX_JSON_BYTES,
            noun="abandoned owner status",
            require_private_owner=True,
        ),
        "owner_marker": marker_pin,
        "owner_lock": owner_lock_pin,
        "wal_surfaces": wal_pins,
        # Opening mutable controller custody may legitimately update only the
        # lock inode metadata between read-only review and execution.  Every
        # state/evidence surface remains pinned here and below.
        "generation_inventory": reviewed_inventory,
    }
    preflight_cid = _content_id(
        {
            "schema": ABANDONED_OWNER_RECOVERY_PREFLIGHT_SCHEMA,
            "operation": ABANDONED_OWNER_RECOVERY_OPERATION,
            "reviewed_pins": pins,
        }
    )
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    if (
        _stopped_recovery_generation_inventory(paths, lock_custody)
        != generation_inventory
        or _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        != observed_provenance
        or _strict_json(
            receipt_path,
            expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
            require_private_owner=True,
        )
        != receipt
        or _strict_json(
            Path(io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        )
        != status
        or _sha256_regular_file(
            owner_status_path,
            max_bytes=MAX_JSON_BYTES,
            noun="abandoned owner status",
            require_private_owner=True,
        )
        != pins["owner_status_sha256"]
        or _private_regular_stat_pin(
            expected_marker_path,
            noun="abandoned state-owner marker",
        )
        != {name: value for name, value in marker_pin.items() if name != "sha256"}
        or _abandoned_owner_wal_pins(
            paths,
            database_paths=io_paths["databases"],
        )
        != wal_pins
    ):
        raise SuccessorOperatorError(
            "abandoned owner recovery evidence changed during preflight"
        )
    return {
        "schema": ABANDONED_OWNER_RECOVERY_PREFLIGHT_SCHEMA,
        "operation": ABANDONED_OWNER_RECOVERY_OPERATION,
        "observed_at": _utc_now(),
        "reviewed_pins": pins,
        "preflight_cid": preflight_cid,
        "automatic_same_source_recovery": source_mode == "exact_source",
        "controller_lock_held": True,
        "owner_lock_held": False,
        "restart_authority": False,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "production_authorized": False,
    }


def abandoned_owner_recovery_preflight(root: Path = ROOT) -> dict[str, Any]:
    """Report exact dead-owner/WAL recovery pins without changing custody."""

    paths = _paths(root)
    with _exclusive_projection_checkpoint(
        paths,
        _read_only_existing_lock=True,
    ) as lock_custody:
        return _abandoned_owner_recovery_preflight_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
        )


def _abandoned_owner_recovery_intent_path(
    paths: Mapping[str, Path],
    preflight_cid: str,
) -> Path:
    reviewed = str(preflight_cid or "").strip()
    if re.fullmatch(r"b[a-z2-7]{20,200}", reviewed) is None:
        raise SuccessorOperatorError(
            "abandoned owner recovery preflight CID is malformed"
        )
    evidence = Path(paths["abandoned_owner_recovery_evidence"])
    return evidence / f"abandoned-owner-recovery-intent.{reviewed}.json"


def _write_abandoned_owner_recovery_intent(
    paths: Mapping[str, Path],
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist reviewed pins before any restart receipt or owner mutation."""

    reviewed = str(preflight.get("preflight_cid") or "")
    intent_path = _abandoned_owner_recovery_intent_path(paths, reviewed)
    intent: dict[str, Any] = {
        "schema": ABANDONED_OWNER_RECOVERY_INTENT_SCHEMA,
        "issued_at": _utc_now(),
        "operation": ABANDONED_OWNER_RECOVERY_OPERATION,
        "preflight_cid": reviewed,
        "reviewed_pins": dict(preflight.get("reviewed_pins") or {}),
        "automatic_same_source_recovery": (
            preflight.get("automatic_same_source_recovery") is True
        ),
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "production_authorized": False,
    }
    intent["intent_cid"] = _content_id(intent)
    if os.path.lexists(intent_path):
        existing = _strict_json(
            intent_path,
            expected_schema=ABANDONED_OWNER_RECOVERY_INTENT_SCHEMA,
            require_private_owner=True,
        )
        repeated = dict(intent)
        repeated["issued_at"] = existing.get("issued_at")
        repeated["intent_cid"] = _content_id(
            {name: value for name, value in repeated.items() if name != "intent_cid"}
        )
        if existing != repeated:
            raise SuccessorOperatorError(
                "abandoned owner recovery intent differs"
            )
        return existing
    _atomic_json(intent_path, intent, replace=False)
    if _strict_json(
        intent_path,
        expected_schema=ABANDONED_OWNER_RECOVERY_INTENT_SCHEMA,
        require_private_owner=True,
    ) != intent:
        raise SuccessorOperatorError(
            "abandoned owner recovery intent changed during publication"
        )
    return intent


def _recover_abandoned_owner_continuity_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
    reviewed_preflight_cid: str,
    _automatic: bool = False,
) -> dict[str, Any]:
    """Replay/checkpoint a dead READY owner, then publish current-byte authority."""

    reviewed = str(reviewed_preflight_cid or "").strip()
    if not reviewed:
        raise SuccessorOperatorError(
            "reviewed abandoned owner recovery preflight CID is required"
        )
    preflight = _abandoned_owner_recovery_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
    )
    if preflight.get("preflight_cid") != reviewed:
        raise SuccessorOperatorError(
            "reviewed abandoned owner recovery preflight CID differs"
        )
    if _automatic and preflight.get("automatic_same_source_recovery") is not True:
        raise SuccessorOperatorError(
            "abandoned owner source maintenance requires reviewed recovery"
        )
    pins = preflight.get("reviewed_pins")
    if not isinstance(pins, Mapping):
        raise SuccessorOperatorError(
            "abandoned owner recovery reviewed pins are malformed"
        )
    intent = _write_abandoned_owner_recovery_intent(
        paths,
        preflight=preflight,
    )
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    receipt_paths = _stopped_receipt_io_view(paths, io_paths)
    provenance = _load_lgcvf_live_raw_provenance_receipt(
        paths,
        _receipt_path=Path(io_paths["provenance"]),
    )
    config_path = _contained(root, DEFAULT_SUCCESSOR_CONFIG_RELATIVE)
    live_launch: Mapping[str, Any] | None = None
    server: Any | None = None
    identity: Any | None = None
    owner_checkpoint: Mapping[str, Any] = {}
    owner_stop: Mapping[str, Any] = {}
    token = ""
    previous_environment: dict[str, str | None] = {}
    try:
        # Seal and validate all current source/native bytes before consuming the
        # prior restart receipt.  This has no database or owner effect.
        live_launch = _prepare_lgcvf_configured_board_live_launch(
            root=root,
            config_path=config_path,
            provenance=provenance,
            stopped_restart=True,
        )
        if live_launch.get("continuity") != pins.get("source_continuity"):
            raise SuccessorOperatorError(
                "abandoned owner recovery source changed after review"
            )
        launch_home = Path(str(live_launch["launch_home"]))
        extension_environment = {
            "HOME": str(launch_home),
            "IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME": str(launch_home),
            "XDG_CACHE_HOME": str(launch_home / ".cache" / "xdg"),
            "CUDA_CACHE_PATH": str(launch_home / ".cache" / "cuda"),
            "CUDA_CACHE_DISABLE": "1",
            BOARD_EXTENSION_INSTALL_POLICY_ENV: (
                BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
            ),
            STORE_GENERATION_ENV: SUCCESSOR_STORE_GENERATION,
        }
        previous_environment.update(
            {name: os.environ.get(name) for name in extension_environment}
        )
        forbidden_loader_environment = {
            name
            for name in os.environ
            if name.startswith("LD_") or name == "GLIBC_TUNABLES"
        }
        if forbidden_loader_environment:
            raise SuccessorOperatorError(
                "LGCVF recovery owner inherited ambient loader authority"
            )
        os.environ.update(extension_environment)
        from ipfs_accelerate_py.llm_router import (
            preload_agent_supervisor_native_dependency,
        )

        preload_agent_supervisor_native_dependency(live_launch["native_launch"])
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            current_process_birth,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            configured_board_launch_plan,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
            establish_state_authority_process_boundary,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
            build_server,
        )

        board = live_launch["board"]
        program = live_launch["program"]
        rendered = configured_board_launch_plan(
            board,
            implement=False,
            detach=False,
            duration_seconds=1.0,
        ).get("environment")
        if not isinstance(rendered, Mapping):
            raise SuccessorOperatorError(
                "abandoned owner recovery database program is unavailable"
            )
        owner_program_json = str(
            rendered.get(DATABASE_PROGRAM_JSON_ENV) or ""
        ).strip()
        if not owner_program_json:
            raise SuccessorOperatorError(
                "abandoned owner recovery database program is unavailable"
            )
        previous_environment[DATABASE_PROGRAM_JSON_ENV] = os.environ.get(
            DATABASE_PROGRAM_JSON_ENV
        )
        os.environ[DATABASE_PROGRAM_JSON_ENV] = owner_program_json
        establish_state_authority_process_boundary()
        paths["owner_state"].mkdir(mode=0o700, parents=True, exist_ok=True)
        _prepare_private_owner_socket(paths["owner_socket"])
        controller_birth = current_process_birth()
        server = build_server(
            database_path=paths["successor_database"],
            state_dir=paths["owner_state"],
            host=str(live_launch["host"]),
            port=int(live_launch["port"]),
            repository_id="repository:lgcvf-quack-successor",
            store_id=program.store_id,
            secret_handle=program.endpoint_secret_handle,
            migrate=datasets_profile_migration,
            typed_command_socket_path=paths["owner_socket"],
            allow_legacy_board_unstall=False,
        )
        if pins.get("receipt_custody") == "published":
            if not _claim_stopped_state_restart_admission(
                receipt_paths,
                expected_restart=True,
                expected_receipt_cid=str(
                    pins["stopped_state_continuity_receipt_cid"]
                ),
                expected_controller_status_cid=str(
                    pins["stopped_controller_status_cid"]
                ),
            ):
                raise SuccessorOperatorError(
                    "abandoned owner recovery receipt was not claimed"
                )
        elif not os.path.lexists(io_paths["stopped_state_restart_admission"]):
            raise SuccessorOperatorError(
                "abandoned owner recovery consumed receipt is unavailable"
            )
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        identity = server.start()
        if (
            identity.listen_uri != program.quack_endpoint
            or identity.store_id != program.store_id
            or identity.database_uuid != provenance.get("database_uuid")
            or not _owner_schema_fingerprint_matches_canonical_cid(
                identity.schema_fingerprint,
                provenance.get("schema_fingerprint"),
            )
            or server.typed_command_socket_path() != paths["owner_socket"]
            or server.status().get("legacy_board_unstall_enabled") is not False
        ):
            raise SuccessorOperatorError(
                "recovery owner identity differs from the stopped generation"
            )
        if server._vault is None:
            raise SuccessorOperatorError(
                "recovery owner token vault is unavailable"
            )
        token = server._vault.resolve(identity.secret_handle)
        owner_checkpoint = server.checkpoint()
        owner_stop = server.stop()
        server = None
        if (
            owner_checkpoint.get("checkpointed") is not True
            or owner_checkpoint.get("server_id") != identity.server_id
            or owner_stop.get("stopped") is not True
            or owner_stop.get("server_id") != identity.server_id
        ):
            raise SuccessorOperatorError(
                "abandoned owner checkpoint/stop evidence differs"
            )
        credential_leak = bool(
            tuple(paths["owner_state"].glob("*.quack-token"))
        )
        for surface in (
            Path(io_paths["controller_status"]),
            Path(io_paths["owner_status"]),
        ):
            credential_leak = credential_leak or _regular_file_contains(
                surface,
                token.encode("ascii"),
            )
        if credential_leak:
            raise SuccessorOperatorError(
                "raw recovery-owner credential reached a persistent surface"
            )

        failed = _status_payload(
            lifecycle="stopped",
            controller_birth=controller_birth.to_dict(),
            provenance_cid=str(provenance["receipt_cid"]),
            owner_identity=identity.to_dict(),
            scheduler_birth={},
            scheduler_returncode=0,
            error=FAILED_START_STATUS_ERROR,
            projection_root=paths["projection_root"],
        )
        failed.pop("status_cid", None)
        abandoned_identity = pins.get("abandoned_owner_identity")
        abandoned_server_id = (
            str(abandoned_identity.get("server_id") or "")
            if isinstance(abandoned_identity, Mapping)
            else ""
        )
        failed["abandoned_owner_recovery"] = {
            "schema": ABANDONED_OWNER_RECOVERY_STATUS_SCHEMA,
            "preflight_cid": reviewed,
            "abandoned_owner_server_id": abandoned_server_id,
            "scheduling_attempted": False,
        }
        failed["status_cid"] = _content_id(failed)
        _validate_unbound_failed_start_controller_status(
            failed,
            provenance=provenance,
            require_dead=False,
        )
        anchors = _capture_failed_start_recovery_anchors(
            paths,
            root=root,
            failed_status=failed,
            provenance=provenance,
            failed_start_reason=(
                FAILED_START_REASON_ABANDONED_OWNER_RECOVERED
            ),
            owner_stop=owner_stop,
            io_paths=io_paths,
            lock_custody=lock_custody,
        )
        anchored = _bind_failed_start_recovery_anchors_status(
            failed,
            anchors,
        )
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        _write_status(
            Path(io_paths["controller_status"]),
            anchored,
            token=token,
        )
        continuity = _recover_interrupted_failed_start_continuity(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
            failed_start_reason=(
                FAILED_START_REASON_ABANDONED_OWNER_RECOVERED
            ),
            _require_dead_controller_tree=False,
        )
        if not isinstance(continuity, Mapping):
            raise SuccessorOperatorError(
                "abandoned owner recovery did not publish continuity"
            )
        final_status = _strict_json(
            Path(io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        )
        token = ""
        return {
            "schema": ABANDONED_OWNER_RECOVERY_RESULT_SCHEMA,
            "recovered": True,
            "repeated": False,
            "preflight_cid": reviewed,
            "intent_cid": intent["intent_cid"],
            "abandoned_owner_server_id": abandoned_server_id,
            "recovery_owner_server_id": identity.server_id,
            "stopped_state_continuity_receipt_cid": continuity["receipt_cid"],
            "controller_status_cid": final_status["status_cid"],
            "target_generation": SUCCESSOR_STORE_GENERATION,
            "restart_authority": True,
            "authoritative": False,
            "scheduling_authority": False,
            "completion_authority": False,
            "production_authorized": False,
        }
    finally:
        if server is not None:
            try:
                server.stop()
            except Exception as cleanup_exc:  # noqa: BLE001
                sys.stderr.write(
                    "LGCVF abandoned recovery owner stop failed: "
                    f"{type(cleanup_exc).__name__}\n"
                )
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        _close_lgcvf_configured_board_live_launch(live_launch)


def recover_abandoned_owner_continuity(
    root: Path = ROOT,
    *,
    reviewed_preflight_cid: str,
) -> dict[str, Any]:
    """Run the separately reviewed owner-only WAL/checkpoint recovery."""

    paths = _paths(root)
    with _exclusive_projection_checkpoint(paths) as lock_custody:
        return _recover_abandoned_owner_continuity_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
            reviewed_preflight_cid=reviewed_preflight_cid,
        )


def _automatically_recover_abandoned_owner_locked(
    paths: Mapping[str, Path],
    *,
    root: Path,
    lock_custody: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Repair only an exact-source dead READY owner before normal launch."""

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    receipt_surfaces = (
        Path(io_paths["stopped_state_continuity"]),
        Path(io_paths["stopped_state_restart_admission"]),
    )
    if sum(os.path.lexists(path) for path in receipt_surfaces) != 1:
        return None
    owner_status_path = Path(io_paths["owner_status"])
    owner_marker_path = Path(io_paths["owner_marker"])
    if not os.path.lexists(owner_status_path):
        return None
    owner_status = _strict_json(
        owner_status_path,
        expected_schema=QUACK_STATE_SERVER_STATUS_SCHEMA,
        require_private_owner=True,
        verify_content_identity=False,
    )
    owner_identity = owner_status.get("identity")
    looks_abandoned = (
        os.path.lexists(owner_marker_path)
        or owner_status.get("lifecycle") == "ready"
        or (
            isinstance(owner_identity, Mapping)
            and owner_identity.get("status") == "ready"
        )
    )
    if not looks_abandoned:
        return None
    preflight = _abandoned_owner_recovery_preflight_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
    )
    if preflight.get("automatic_same_source_recovery") is not True:
        raise SuccessorOperatorError(
            "abandoned state owner requires reviewed source-maintenance "
            "recovery; run abandoned-owner-recovery-preflight and use "
            f"preflight CID {preflight['preflight_cid']}"
        )
    return _recover_abandoned_owner_continuity_locked(
        paths,
        root=root,
        lock_custody=lock_custody,
        reviewed_preflight_cid=str(preflight["preflight_cid"]),
        _automatic=True,
    )


def stopped_recovery_preflight(root: Path = ROOT) -> dict[str, Any]:
    """Report exact legacy recovery pins without publishing authority."""

    paths = _paths(root)
    with _exclusive_projection_checkpoint(
        paths,
        _read_only_existing_lock=True,
    ) as lock_custody:
        io_paths = _stopped_recovery_io_paths(paths, lock_custody)
        if os.path.lexists(io_paths["stopped_state_continuity"]):
            raise SuccessorOperatorError(
                "stopped-state continuity is already published"
            )
        if os.path.lexists(io_paths["stopped_state_restart_admission"]):
            raise SuccessorOperatorError(
                "a consumed stopped-state restart admission remains"
            )
        provenance = _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        return _stopped_recovery_preflight_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
        )


def recover_stopped_continuity(
    root: Path = ROOT,
    *,
    reviewed_preflight_cid: str,
) -> dict[str, Any]:
    """Publish legacy recovery only from a separately reviewed exact preflight."""

    reviewed = str(reviewed_preflight_cid or "").strip()
    if not reviewed:
        raise SuccessorOperatorError(
            "reviewed stopped recovery preflight CID is required"
        )
    paths = _paths(root)
    with _exclusive_projection_checkpoint(paths) as lock_custody:
        io_paths = _stopped_recovery_io_paths(paths, lock_custody)
        if os.path.lexists(io_paths["stopped_state_continuity"]):
            raise SuccessorOperatorError(
                "stopped-state continuity is already published"
            )
        if os.path.lexists(io_paths["stopped_state_restart_admission"]):
            raise SuccessorOperatorError(
                "a consumed stopped-state restart admission remains"
            )
        provenance = _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        preflight = _stopped_recovery_preflight_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
        )
        if preflight.get("preflight_cid") != reviewed:
            raise SuccessorOperatorError(
                "reviewed stopped recovery preflight CID differs"
            )
        receipt = _recover_interrupted_stopped_state_continuity(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
            reviewed_preflight_cid=reviewed,
        )
        if not isinstance(receipt, Mapping):
            raise SuccessorOperatorError(
                "reviewed stopped recovery did not publish continuity"
            )
        return {
            "schema": STOPPED_RECOVERY_RESULT_SCHEMA,
            "recovered": True,
            "preflight_cid": reviewed,
            "stopped_state_continuity_receipt_cid": receipt["receipt_cid"],
            "controller_status_cid": receipt["controller_status_cid"],
            "target_generation": SUCCESSOR_STORE_GENERATION,
            "restart_authority": True,
            "authoritative": False,
            "scheduling_authority": False,
            "completion_authority": False,
            "production_authorized": False,
        }


def failed_start_source_maintenance_preflight(
    root: Path = ROOT,
) -> dict[str, Any]:
    """Report exact descendant-source reseal pins without changing custody."""

    paths = _paths(root)
    with _exclusive_projection_checkpoint(
        paths,
        _read_only_existing_lock=True,
    ) as lock_custody:
        io_paths = _stopped_recovery_io_paths(paths, lock_custody)
        provenance = _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        if os.path.lexists(io_paths["stopped_state_continuity"]):
            status = _strict_json(
                Path(io_paths["controller_status"]),
                expected_schema=CONTROLLER_STATUS_SCHEMA,
                require_private_owner=True,
            )
            linked = status.get("stopped_state_continuity_receipt_cid")
            linked_status = status.get("stopped_state_continuity_status_cid")
            if (linked is None) != (linked_status is None):
                raise SuccessorOperatorError(
                    "failed-start source maintenance status links are partial"
                )
            if linked is not None:
                return _failed_start_source_maintenance_preflight_locked(
                    paths,
                    root=root,
                    lock_custody=lock_custody,
                    provenance=provenance,
                )
            standard = _failed_start_recovery_preflight_locked(
                paths,
                root=root,
                lock_custody=lock_custody,
                provenance=provenance,
                _allow_continuity_receipt=True,
            )
        else:
            standard = _failed_start_recovery_preflight_locked(
                paths,
                root=root,
                lock_custody=lock_custody,
                provenance=provenance,
            )
        superseded = _validate_failed_start_source_maintenance_transition_preflight(
            paths,
            root=root,
            provenance=provenance,
            preflight=standard,
        )
        prior = superseded["receipt"]
        return _failed_start_source_maintenance_report(
            standard,
            published_receipt_cid=str(prior["receipt_cid"]),
            published_controller_status_cid=str(
                prior["controller_status_cid"]
            ),
        )


def reseal_failed_start_source_maintenance(
    root: Path = ROOT,
    *,
    reviewed_preflight_cid: str,
) -> dict[str, Any]:
    """Apply or resume only the separately reviewed descendant-source reseal."""

    reviewed = str(reviewed_preflight_cid or "").strip()
    if not reviewed:
        raise SuccessorOperatorError(
            "reviewed failed-start source maintenance preflight CID is required"
        )
    paths = _paths(root)
    with _exclusive_projection_checkpoint(paths) as lock_custody:
        io_paths = _stopped_recovery_io_paths(paths, lock_custody)
        receipt_paths = _stopped_receipt_io_view(paths, io_paths)
        provenance = _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )

        def exact_status() -> dict[str, Any]:
            status = _strict_json(
                Path(io_paths["controller_status"]),
                expected_schema=CONTROLLER_STATUS_SCHEMA,
                require_private_owner=True,
            )
            linked = status.get("stopped_state_continuity_receipt_cid")
            linked_status = status.get("stopped_state_continuity_status_cid")
            if (linked is None) != (linked_status is None):
                raise SuccessorOperatorError(
                    "failed-start source maintenance status links are partial"
                )
            return status

        def complete_standard_transition() -> dict[str, Any]:
            standard = _failed_start_recovery_preflight_locked(
                paths,
                root=root,
                lock_custody=lock_custody,
                provenance=provenance,
                _allow_continuity_receipt=os.path.lexists(
                    io_paths["stopped_state_continuity"]
                ),
            )
            _validate_failed_start_source_maintenance_transition_preflight(
                paths,
                root=root,
                provenance=provenance,
                preflight=standard,
            )
            if standard.get("preflight_cid") != reviewed:
                raise SuccessorOperatorError(
                    "reviewed failed-start source maintenance preflight CID differs"
                )
            _recover_interrupted_failed_start_continuity(
                paths,
                root=root,
                lock_custody=lock_custody,
                provenance=provenance,
                reviewed_preflight_cid=reviewed,
                failed_start_reason=FAILED_START_REASON_LEGACY_UNCLASSIFIED,
            )
            completed = _completed_failed_start_source_maintenance(
                paths,
                root=root,
                lock_custody=lock_custody,
                provenance=provenance,
                reviewed_preflight_cid=reviewed,
            )
            completed["repeated"] = False
            return completed

        if os.path.lexists(io_paths["stopped_state_continuity"]):
            published_receipt = _strict_json(
                Path(io_paths["stopped_state_continuity"]),
                expected_schema=STOPPED_STATE_CONTINUITY_SCHEMA,
                require_private_owner=True,
            )
            status = exact_status()
            evidence = published_receipt.get("stop_evidence")
            if (
                status.get("stopped_state_continuity_receipt_cid") is not None
                and isinstance(evidence, Mapping)
                and evidence.get("mode")
                == FAILED_START_REVIEWED_EVIDENCE_MODE
                and evidence.get("recovery_preflight_cid") == reviewed
            ):
                return _completed_failed_start_source_maintenance(
                    paths,
                    root=root,
                    lock_custody=lock_custody,
                    provenance=provenance,
                    reviewed_preflight_cid=reviewed,
                )
            if status.get("stopped_state_continuity_receipt_cid") is None:
                return complete_standard_transition()
        elif os.path.lexists(io_paths["stopped_state_restart_admission"]):
            status = exact_status()
            if status.get("stopped_state_continuity_receipt_cid") is not None:
                restored = _restore_or_retire_stopped_restart_admission(
                    receipt_paths
                )
                if restored != "restored_interrupted_claim":
                    raise SuccessorOperatorError(
                        "failed-start source maintenance claim recovery differs"
                    )
            else:
                return complete_standard_transition()
        else:
            status = exact_status()
            if status.get("stopped_state_continuity_receipt_cid") is not None:
                raise SuccessorOperatorError(
                    "failed-start source maintenance receipt custody is unavailable"
                )
            return complete_standard_transition()

        preflight = _failed_start_source_maintenance_preflight_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
        )
        if preflight.get("preflight_cid") != reviewed:
            raise SuccessorOperatorError(
                "reviewed failed-start source maintenance preflight CID differs"
            )
        pins = preflight.get("reviewed_pins")
        if not isinstance(pins, Mapping):
            raise SuccessorOperatorError(
                "failed-start source maintenance reviewed pins are malformed"
            )
        projected_status = pins.get("controller_status")
        if not isinstance(projected_status, Mapping):
            raise SuccessorOperatorError(
                "failed-start source maintenance projected status is malformed"
            )
        if not _claim_stopped_state_restart_admission(
            receipt_paths,
            expected_restart=True,
            expected_receipt_cid=str(
                preflight[
                    "published_stopped_state_continuity_receipt_cid"
                ]
            ),
            expected_controller_status_cid=str(
                preflight["published_controller_status_cid"]
            ),
        ):
            raise SuccessorOperatorError(
                "failed-start source maintenance receipt was not claimed"
            )
        _revalidate_generation_bound_controller_lock(paths, lock_custody)
        _write_status(Path(io_paths["controller_status"]), projected_status)
        if _strict_json(
            Path(io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        ) != dict(projected_status):
            raise SuccessorOperatorError(
                "failed-start source maintenance projected status changed"
            )
        return complete_standard_transition()


def failed_start_recovery_preflight(root: Path = ROOT) -> dict[str, Any]:
    """Report current failed-start pins without creating restart authority."""

    paths = _paths(root)
    with _exclusive_projection_checkpoint(
        paths,
        _read_only_existing_lock=True,
    ) as lock_custody:
        io_paths = _stopped_recovery_io_paths(paths, lock_custody)
        if os.path.lexists(io_paths["stopped_state_continuity"]):
            raise SuccessorOperatorError(
                "failed-start continuity is already published"
            )
        provenance = _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        return _failed_start_recovery_preflight_locked(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
        )


def recover_failed_start_continuity(
    root: Path = ROOT,
    *,
    reviewed_preflight_cid: str,
) -> dict[str, Any]:
    """Publish new current-byte authority after explicit legacy review."""

    reviewed = str(reviewed_preflight_cid or "").strip()
    if not reviewed:
        raise SuccessorOperatorError(
            "reviewed failed-start recovery preflight CID is required"
        )
    paths = _paths(root)
    with _exclusive_projection_checkpoint(paths) as lock_custody:
        io_paths = _stopped_recovery_io_paths(paths, lock_custody)
        provenance = _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(io_paths["provenance"]),
        )
        if not os.path.lexists(io_paths["stopped_state_continuity"]):
            preflight = _failed_start_recovery_preflight_locked(
                paths,
                root=root,
                lock_custody=lock_custody,
                provenance=provenance,
            )
            if preflight.get("preflight_cid") != reviewed:
                raise SuccessorOperatorError(
                    "reviewed failed-start recovery preflight CID differs"
                )
        receipt = _recover_interrupted_failed_start_continuity(
            paths,
            root=root,
            lock_custody=lock_custody,
            provenance=provenance,
            reviewed_preflight_cid=reviewed,
            failed_start_reason=FAILED_START_REASON_LEGACY_UNCLASSIFIED,
        )
        if not isinstance(receipt, Mapping):
            raise SuccessorOperatorError(
                "reviewed failed-start recovery did not publish continuity"
            )
        evidence = receipt.get("stop_evidence")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("mode") != FAILED_START_REVIEWED_EVIDENCE_MODE
            or evidence.get("recovery_preflight_cid") != reviewed
        ):
            raise SuccessorOperatorError(
                "reviewed failed-start recovery result binding differs"
            )
        return {
            "schema": FAILED_START_RECOVERY_RESULT_SCHEMA,
            "recovered": True,
            "preflight_cid": reviewed,
            "stopped_state_continuity_receipt_cid": receipt["receipt_cid"],
            "controller_status_cid": receipt["controller_status_cid"],
            "target_generation": SUCCESSOR_STORE_GENERATION,
            "restart_authority": True,
            "authoritative": False,
            "scheduling_authority": False,
            "completion_authority": False,
            "production_authorized": False,
        }


def _claim_projection_root(
    paths: Mapping[str, Path],
    lock_custody: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically create and pin the one non-authoritative projection root."""

    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    projection_root = paths["projection_root"]
    generation = paths["controller_lock"].parent
    if projection_root.parent != generation:
        raise SuccessorOperatorError("DuckLake projection escaped its generation")
    generation_descriptor = int(lock_custody["generation_descriptor"])
    try:
        os.mkdir(projection_root.name, mode=0o700, dir_fd=generation_descriptor)
    except FileExistsError as exc:
        raise SuccessorOperatorError(
            "refusing to reuse residual DuckLake projection root"
        ) from exc
    except OSError as exc:
        raise SuccessorOperatorError(
            "DuckLake projection root could not be claimed"
        ) from exc
    projection_descriptor = -1
    try:
        projection_descriptor = os.open(
            projection_root.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=generation_descriptor,
        )
        metadata = os.fstat(projection_descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise SuccessorOperatorError(
                "DuckLake projection root custody is unsafe"
            )
        custody = {
            "descriptor": projection_descriptor,
            "identity": _inode_identity(metadata),
            "logical_path": str(projection_root),
            "descriptor_path": f"/proc/self/fd/{projection_descriptor}",
        }
        _revalidate_projection_root(paths, lock_custody, custody)
        return custody
    except BaseException:
        if projection_descriptor >= 0:
            os.close(projection_descriptor)
        raise


def _revalidate_projection_root(
    paths: Mapping[str, Path],
    lock_custody: Mapping[str, Any],
    projection_custody: Mapping[str, Any],
) -> None:
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    descriptor = projection_custody.get("descriptor")
    if (
        type(descriptor) is not int
        or descriptor < 3
        or projection_custody.get("logical_path")
        != str(paths["projection_root"])
        or projection_custody.get("descriptor_path")
        != f"/proc/self/fd/{descriptor}"
    ):
        raise SuccessorOperatorError("DuckLake projection root binding differs")
    try:
        held = os.fstat(descriptor)
        named = os.stat(
            paths["projection_root"].name,
            dir_fd=int(lock_custody["generation_descriptor"]),
            follow_symlinks=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SuccessorOperatorError(
            "DuckLake projection root binding changed"
        ) from exc
    if (
        not stat.S_ISDIR(held.st_mode)
        or held.st_uid != os.geteuid()
        or stat.S_IMODE(held.st_mode) != 0o700
        or _inode_identity(held)
        != tuple(projection_custody.get("identity") or ())
        or _inode_identity(named) != _inode_identity(held)
    ):
        raise SuccessorOperatorError(
            "DuckLake projection root binding changed"
        )


def _validate_projection_root_outputs(
    projection_custody: Mapping[str, Any],
    *,
    board_namespace: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        board_database_path,
    )

    descriptor = int(projection_custody["descriptor"])
    expected = {
        "control.duckdb": "file",
        "lake.ducklake": "file",
        "lake-data": "directory",
    }
    for name, kind in expected.items():
        try:
            metadata = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise SuccessorOperatorError(
                "DuckLake projection output inventory differs"
            ) from exc
        if (
            metadata.st_uid != os.geteuid()
            or (kind == "file" and not stat.S_ISREG(metadata.st_mode))
            or (kind == "file" and metadata.st_size <= 0)
            or (kind == "directory" and not stat.S_ISDIR(metadata.st_mode))
        ):
            raise SuccessorOperatorError(
                "DuckLake projection output inventory differs"
            )
    for wal_name in ("control.duckdb.wal", "lake.ducklake.wal"):
        try:
            os.stat(wal_name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise SuccessorOperatorError(
                "DuckLake projection WAL custody cannot be inspected"
            ) from exc
        else:
            raise SuccessorOperatorError("DuckLake projection retained a live WAL")

    relative_board = board_database_path(Path(), board_namespace)
    if (
        relative_board.is_absolute()
        or len(relative_board.parts) != 2
        or relative_board.parts[0] != "boards"
    ):
        raise SuccessorOperatorError("DuckLake projection board path differs")
    boards_descriptor = -1
    try:
        boards_descriptor = os.open(
            "boards",
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
        boards_metadata = os.fstat(boards_descriptor)
        board_metadata = os.stat(
            relative_board.name,
            dir_fd=boards_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(boards_metadata.st_mode)
            or boards_metadata.st_uid != os.geteuid()
            or not stat.S_ISREG(board_metadata.st_mode)
            or board_metadata.st_uid != os.geteuid()
            or board_metadata.st_nlink != 1
            or board_metadata.st_size <= 0
        ):
            raise SuccessorOperatorError(
                "DuckLake projection board output custody differs"
            )
        try:
            os.stat(
                relative_board.name + ".wal",
                dir_fd=boards_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise SuccessorOperatorError("DuckLake projection retained a live WAL")
    except SuccessorOperatorError:
        raise
    except OSError as exc:
        raise SuccessorOperatorError(
            "DuckLake projection board output inventory differs"
        ) from exc
    finally:
        if boards_descriptor >= 0:
            os.close(boards_descriptor)


def _open_projection_plane(
    root: Path,
    projection_root: Path,
) -> Any:
    """Open every projection output through one pinned directory."""

    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        board_control_plane as board_module,
    )

    if re.fullmatch(r"/proc/self/fd/([1-9][0-9]*)", str(projection_root)) is None:
        raise SuccessorOperatorError("DuckLake projection storage is not pinned")
    return board_module.open_board_control_plane(
        root,
        root=projection_root,
        allow_extension_install=False,
    )


def _bind_projection_logical_paths(
    plane: Any,
    *,
    descriptor_root: Path,
    logical_root: Path,
    board_namespace: str,
) -> dict[str, Any]:
    """Relocate only durable path values after all projection I/O is pinned."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        board_database_path,
    )

    if (
        re.fullmatch(r"/proc/self/fd/([1-9][0-9]*)", str(descriptor_root)) is None
        or not logical_root.is_absolute()
        or str(getattr(plane, "root", "")) != str(descriptor_root)
        or getattr(plane, "ducklake_attached", False) is not True
    ):
        raise SuccessorOperatorError(
            "DuckLake projection logical-path binding is unavailable"
        )
    physical_board = board_database_path(descriptor_root, board_namespace)
    logical_board = board_database_path(logical_root, board_namespace)
    connection = plane._conn()
    changed = connection.execute(
        "UPDATE board_catalog SET duckdb_path = ? "
        "WHERE board_namespace = ? AND duckdb_path = ? "
        "RETURNING duckdb_path",
        [str(logical_board), board_namespace, str(physical_board)],
    ).fetchall()
    if len(changed) != 1 or str(changed[0][0]) != str(logical_board):
        raise SuccessorOperatorError(
            "DuckLake projection board logical path differs"
        )

    aggregate = plane.aggregate_boards()
    if aggregate.get("ducklake_attached") is not True:
        raise SuccessorOperatorError(
            "DuckLake projection could not persist the logical board path"
        )
    projected = connection.execute(
        "SELECT duckdb_path FROM lake.board_catalog "
        "WHERE board_namespace = ?",
        [board_namespace],
    ).fetchall()
    if len(projected) != 1 or str(projected[0][0]) != str(logical_board):
        raise SuccessorOperatorError(
            "DuckLake projection shadow board logical path differs"
        )

    physical_data = (descriptor_root / "lake-data").as_posix().rstrip("/") + "/"
    logical_data = (logical_root / "lake-data").as_posix().rstrip("/") + "/"
    relocated = connection.execute(
        "UPDATE __ducklake_metadata_lake.ducklake_metadata SET value = ? "
        "WHERE key = 'data_path' AND value = ? "
        "AND scope IS NULL AND scope_id IS NULL RETURNING value",
        [logical_data, physical_data],
    ).fetchall()
    if len(relocated) != 1 or str(relocated[0][0]) != logical_data:
        raise SuccessorOperatorError(
            "DuckLake projection data logical path differs"
        )
    return {
        "aggregate": aggregate,
        "logical_board_database": str(logical_board),
        "logical_data_path": logical_data,
    }


def _project_ducklake_once_locked(
    root: Path,
    *,
    paths: Mapping[str, Path],
    lock_custody: Mapping[str, Any],
) -> dict[str, Any]:
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    recovery_io_paths = _stopped_recovery_io_paths(paths, lock_custody)
    receipt_io_paths = _stopped_receipt_io_view(paths, recovery_io_paths)
    capability = _extension_preflight()
    if capability.get("available") is not True:
        raise SuccessorOperatorError("DuckLake projection preflight is not valid")
    bound_projection_receipt = _generation_bound_runtime_path(
        paths,
        lock_custody,
        paths["projection_receipt"],
    )
    if os.path.lexists(bound_projection_receipt):
        raise SuccessorOperatorError(
            "refusing to overwrite DuckLake projection receipt"
        )
    generation = paths["controller_lock"].parent
    if paths["projection_root"].parent != generation:
        raise SuccessorOperatorError("DuckLake projection escaped its generation")
    try:
        os.stat(
            paths["projection_root"].name,
            dir_fd=int(lock_custody["generation_descriptor"]),
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise SuccessorOperatorError(
            "DuckLake projection root residue cannot be inspected"
        ) from exc
    else:
        raise SuccessorOperatorError(
            "refusing to reuse residual DuckLake projection root"
        )
    _revalidate_generation_bound_controller_lock(paths, lock_custody)
    if os.path.lexists(recovery_io_paths["controller_status"]):
        raw_provenance = _load_lgcvf_live_raw_provenance_receipt(
            paths,
            _receipt_path=Path(recovery_io_paths["provenance"]),
        )
        recovery_status = _strict_json(
            Path(recovery_io_paths["controller_status"]),
            expected_schema=CONTROLLER_STATUS_SCHEMA,
            require_private_owner=True,
        )
        failed_unbound = (
            recovery_status.get("error") == FAILED_START_STATUS_ERROR
            and "stopped_state_continuity_receipt_cid" not in recovery_status
            and "stopped_state_continuity_status_cid" not in recovery_status
        )
        if failed_unbound:
            _recover_interrupted_failed_start_continuity(
                paths,
                root=root,
                lock_custody=lock_custody,
                provenance=raw_provenance,
            )
        else:
            _restore_or_retire_stopped_restart_admission(receipt_io_paths)
            if recovery_status.get("error") == FAILED_START_STATUS_ERROR:
                _recover_interrupted_failed_start_continuity(
                    paths,
                    root=root,
                    lock_custody=lock_custody,
                    provenance=raw_provenance,
                )
            elif not os.path.lexists(
                recovery_io_paths["stopped_state_continuity"]
            ):
                _recover_interrupted_stopped_state_continuity(
                    paths,
                    root=root,
                    lock_custody=lock_custody,
                    provenance=raw_provenance,
                )
    with _sealed_stopped_database_snapshots(paths, lock_custody) as snapshots:
        continuity = _load_projection_source_continuity(
            paths,
            root=root,
            stopped_database_snapshots=snapshots,
            lock_custody=lock_custody,
        )
        if (
            continuity.get("admission_mode")
            not in {
                STOPPED_STATE_CONTINUITY_ADMISSION_MODE,
                FAILED_START_CONTINUITY_ADMISSION_MODE,
            }
        ):
            raise SuccessorOperatorError(
                "DuckLake projection lost typed stopped-state continuity"
            )
        provenance = continuity["provenance"]
        stopped_state = continuity["receipt"]
        stopped_databases = continuity["databases"]
        source_digest = stopped_databases["control"]["sha256"]
        control_snapshot = snapshots["control"]
        source_path = str(control_snapshot["snapshot_path"])
        import duckdb

        source = duckdb.connect(source_path, read_only=True)
        try:
            columns = tuple(
                str(item[0])
                for item in source.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'tasks' ORDER BY ordinal_position"
                ).fetchall()
            )
            rows = source.execute(
                "SELECT * FROM tasks ORDER BY ordinal, task_cid"
            ).fetchall()
        finally:
            source.close()
        tasks: list[dict[str, Any]] = []
        for row in rows:
            record = {
                columns[index]: row[index] for index in range(len(columns))
            }
            body: dict[str, Any] = {}
            try:
                parsed = json.loads(str(record.get("body_json") or "{}"))
                if isinstance(parsed, dict):
                    body = parsed
            except json.JSONDecodeError:
                pass
            tasks.append(
                {
                    "task_id": str(
                        record.get("task_alias")
                        or record.get("task_cid")
                        or ""
                    ),
                    "status": str(record.get("status") or ""),
                    "title": str(body.get("title") or ""),
                    "depends_on": body.get("depends_on") or [],
                    "body": body,
                }
            )
        projection_custody = _claim_projection_root(paths, lock_custody)
        try:
            descriptor_projection_root = Path(
                str(projection_custody["descriptor_path"])
            )
            with _open_projection_plane(
                root,
                descriptor_projection_root,
            ) as plane:
                registration = plane.register_board(
                    "logic-governed-compositional-verification-fabric-history-shadow-v1",
                    source_path=str(paths["successor_database"]),
                    source_kind="duckdb-stopped-checkpoint-observation",
                    merge_target_branch=(
                        "agent/logic-governed-compositional-verification-fabric-v1"
                    ),
                    extra={
                        "authoritative": False,
                        "restart_authority": False,
                        "scheduling_authority": False,
                        "completion_authority": False,
                        "source_provenance_cid": provenance["receipt_cid"],
                    },
                    tasks=tasks,
                )
                if (
                    plane.backend != "ducklake+quack"
                    or not plane.ducklake_attached
                ):
                    raise SuccessorOperatorError(
                        "physical BoardControlPlane did not admit DuckLake + Quack"
                    )
                logical_paths = _bind_projection_logical_paths(
                    plane,
                    descriptor_root=descriptor_projection_root,
                    logical_root=paths["projection_root"],
                    board_namespace=registration["board_namespace"],
                )
                aggregate = logical_paths["aggregate"]
                backend = plane.backend
                extensions = {
                    "quack_loaded": plane.quack_loaded,
                    "ducklake_loaded": plane.ducklake_loaded,
                    "ducklake_attached": plane.ducklake_attached,
                }
            _revalidate_projection_root(
                paths,
                lock_custody,
                projection_custody,
            )
            _validate_projection_root_outputs(
                projection_custody,
                board_namespace=registration["board_namespace"],
            )
            if (
                _validate_stopped_database_snapshots(
                    paths,
                    lock_custody,
                    snapshots,
                )
                != stopped_databases
            ):
                raise SuccessorOperatorError(
                    "projection source changed during checkpoint"
                )
            receipt = {
                "schema": PROJECTION_RECEIPT_SCHEMA,
                "issued_at": _utc_now(),
                "projection_root": str(paths["projection_root"]),
                "control_catalog_path": str(
                    paths["projection_root"] / "control.duckdb"
                ),
                "ducklake_catalog_path": str(
                    paths["projection_root"] / "lake.ducklake"
                ),
                "ducklake_data_path": str(
                    paths["projection_root"] / "lake-data"
                ),
                "source_database": str(paths["successor_database"]),
                "source_sha256": source_digest,
                "source_provenance_cid": provenance["receipt_cid"],
                "source_admission_mode": continuity["admission_mode"],
                "source_stopped_state_continuity_receipt_cid": str(
                    stopped_state.get("receipt_cid") or ""
                ),
                "source_stopped_controller_status_cid": str(
                    stopped_state.get("controller_status_cid") or ""
                ),
                "board_namespace": registration["board_namespace"],
                "task_count": len(tasks),
                "backend": backend,
                "extensions": extensions,
                "aggregate": aggregate,
                "authoritative": False,
                "scheduling_authority": False,
                "completion_authority": False,
                "read_by_scheduler": False,
                "quack_endpoint_served": False,
                "requires_stopped_checkpoint": True,
                "production_authorized": False,
                "restart_authority": False,
            }
            receipt["receipt_cid"] = _content_id(receipt)
            receipt_relative = paths["projection_receipt"].relative_to(
                paths["controller_lock"].parent
            )
            bound_receipt_path = (
                Path(
                    f"/proc/self/fd/{int(lock_custody['generation_descriptor'])}"
                )
                / receipt_relative
            )
            _atomic_json(bound_receipt_path, receipt, replace=False)
            _revalidate_projection_root(
                paths,
                lock_custody,
                projection_custody,
            )
            if _strict_json(
                bound_receipt_path,
                expected_schema=PROJECTION_RECEIPT_SCHEMA,
                require_private_owner=True,
            ) != receipt:
                raise SuccessorOperatorError(
                    "DuckLake projection receipt binding differs"
                )
            _revalidate_projection_root(
                paths,
                lock_custody,
                projection_custody,
            )
            _validate_projection_root_outputs(
                projection_custody,
                board_namespace=registration["board_namespace"],
            )
            if (
                _validate_stopped_database_snapshots(
                    paths,
                    lock_custody,
                    snapshots,
                )
                != stopped_databases
            ):
                raise SuccessorOperatorError(
                    "projection source changed during receipt publication"
                )
            return receipt
        finally:
            os.close(int(projection_custody["descriptor"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    sealed = subparsers.add_parser("bootstrap-sealed-continuity")
    sealed.add_argument("--source-root", type=Path, required=True)
    sealed.add_argument("--control-sha256", required=True)
    sealed.add_argument("--coordination-sha256", required=True)
    sealed.add_argument("--execution-sha256", required=True)
    sealed.add_argument("--bootstrap-sha256", required=True)
    sealed.add_argument("--manifest-sha256", required=True)
    sealed.add_argument("--recovery-receipt-sha256", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument(
        "--config", type=Path, default=DEFAULT_SUCCESSOR_CONFIG_RELATIVE
    )
    launch.add_argument("--implement", action="store_true")
    launch.add_argument("--duration-seconds", type=float, default=float("inf"))
    subparsers.add_parser("status")
    stop = subparsers.add_parser("stop")
    stop.add_argument("--timeout-seconds", type=float, default=MAX_STOP_SECONDS)
    subparsers.add_parser("stopped-recovery-preflight")
    recover = subparsers.add_parser("recover-stopped-continuity")
    recover.add_argument("--reviewed-preflight-cid", required=True)
    subparsers.add_parser("failed-start-recovery-preflight")
    recover_failed = subparsers.add_parser(
        "recover-failed-start-continuity"
    )
    recover_failed.add_argument("--reviewed-preflight-cid", required=True)
    subparsers.add_parser("failed-start-source-maintenance-preflight")
    reseal_failed = subparsers.add_parser(
        "reseal-failed-start-source-maintenance"
    )
    reseal_failed.add_argument("--reviewed-preflight-cid", required=True)
    subparsers.add_parser("abandoned-owner-recovery-preflight")
    recover_abandoned = subparsers.add_parser(
        "recover-abandoned-owner-continuity"
    )
    recover_abandoned.add_argument("--reviewed-preflight-cid", required=True)
    subparsers.add_parser("projection-preflight")
    subparsers.add_parser("projection-once")
    subparsers.add_parser("history-audit")
    subparsers.add_parser("protected-qualification-completion-preflight")
    protected_completion = subparsers.add_parser(
        "complete-protected-qualification"
    )
    protected_completion.add_argument("--reviewed-preflight-cid", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.repo_root).resolve()
    try:
        if args.command == "bootstrap":
            result: Any = bootstrap_successor(root)
        elif args.command == "bootstrap-sealed-continuity":
            result = bootstrap_sealed_successor(
                root=root,
                source_root=Path(args.source_root),
                control_sha256=str(args.control_sha256),
                coordination_sha256=str(args.coordination_sha256),
                execution_sha256=str(args.execution_sha256),
                bootstrap_sha256=str(args.bootstrap_sha256),
                manifest_sha256=str(args.manifest_sha256),
                recovery_receipt_sha256=str(args.recovery_receipt_sha256),
            )
        elif args.command == "launch":
            config = Path(args.config)
            if not config.is_absolute():
                config = _contained(root, config)
            return run_successor(
                config,
                root=root,
                implement=bool(args.implement),
                duration_seconds=float(args.duration_seconds),
            )
        elif args.command == "status":
            result = controller_status(root)
        elif args.command == "stop":
            result = stop_controller(root, timeout_seconds=float(args.timeout_seconds))
        elif args.command == "stopped-recovery-preflight":
            result = stopped_recovery_preflight(root)
        elif args.command == "recover-stopped-continuity":
            result = recover_stopped_continuity(
                root,
                reviewed_preflight_cid=str(args.reviewed_preflight_cid),
            )
        elif args.command == "failed-start-recovery-preflight":
            result = failed_start_recovery_preflight(root)
        elif args.command == "recover-failed-start-continuity":
            result = recover_failed_start_continuity(
                root,
                reviewed_preflight_cid=str(args.reviewed_preflight_cid),
            )
        elif args.command == "failed-start-source-maintenance-preflight":
            result = failed_start_source_maintenance_preflight(root)
        elif args.command == "reseal-failed-start-source-maintenance":
            result = reseal_failed_start_source_maintenance(
                root,
                reviewed_preflight_cid=str(args.reviewed_preflight_cid),
            )
        elif args.command == "abandoned-owner-recovery-preflight":
            result = abandoned_owner_recovery_preflight(root)
        elif args.command == "recover-abandoned-owner-continuity":
            result = recover_abandoned_owner_continuity(
                root,
                reviewed_preflight_cid=str(args.reviewed_preflight_cid),
            )
        elif args.command == "projection-preflight":
            result = projection_preflight(root)
        elif args.command == "projection-once":
            result = project_ducklake_once(root)
        elif args.command == "history-audit":
            result = stopped_task_history_audit(root)
        elif args.command == "protected-qualification-completion-preflight":
            result = protected_qualification_completion_preflight(root)
        elif args.command == "complete-protected-qualification":
            result = complete_protected_qualification(
                root,
                reviewed_preflight_cid=str(args.reviewed_preflight_cid),
            )
        else:  # pragma: no cover - argparse closes this branch.
            parser.error("unsupported command")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": CONTROLLER_STATUS_SCHEMA,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
