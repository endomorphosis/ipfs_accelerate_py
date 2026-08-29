"""Fail-closed launcher for sealed agent-supervisor scheduler configurations.

The implementation supervisor already owns worker lifecycle, deterministic
task sharding, worktree isolation, and merge serialization.  This module is a
small configuration boundary that turns a reviewed ``scheduler_config@1``
JSON document into arguments for that existing runtime.

Loading performs only structural provider validation.  A configured external
isolation boundary deliberately probes its pinned local runtime, image, and
provider credential while rendering the trusted launch plan; it never installs
optional tools or falls back to an unsealed provider command.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import grp
import hashlib
import json
import math
import os
import pwd
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ...llm_router import (
    AgentImplementationControlPlanePin,
    AgentImplementationRoutePlan,
    AgentImplementationSealedControlPlane,
    eaaef_agent_route_authorization_path,
    load_agent_implementation_route_authorization,
    materialize_agent_implementation_control_plane_capsule,
    project_agent_implementation_route_capacity,
    resolve_agent_implementation_route,
    seal_agent_implementation_control_plane_capsule,
    verify_agent_implementation_sealed_control_plane,
)
from ..contracts.execution import InvocationBudget
from ..control.lifecycle_orchestrator import (
    CONFIGURATION_ROOT_ENV,
    FENCING_EPOCH_ENV,
    PROFILE_ID_ENV,
    REPOSITORY_ROOT_ENV,
    RUN_ID_ENV,
    RUN_ROOT_ENV,
    STATE_ROOT_ENV,
    TARGET_ID_ENV,
    LifecycleProfile,
    LinuxProcessAdapter,
    ProcessIdentity,
    ProcessIdentityMismatch,
)
from ..control.plan_execution_store import (
    ConfiguredBoardExecutionSlices,
    ExecutionPlanError,
    ParallelismDecisionReceipt,
    ProductionParallelPlanAdapter,
    _load_plan_bound_execution_lease_locked,
    _load_plan_bound_proposal_disposition_locked,
    _load_plan_bound_wave_diff_barrier_locked,
    _secure_store_active,
    _secure_store_cas,
)
from ..merge.checkout_lock import serialized_lock_update
from ..planning.plan_revision_contracts import (
    CompletionAuthority,
    DeltaEffectClass,
    LifecycleState,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
)
from ..proof.formal_verification_contracts import content_identity
from ..task_sources.board_control_plane import (
    board_merge_lock_name,
    resolve_board_implementation_branch,
)
from ..task_sources.plan_revision_store import PlanRevisionStore
from ..task_sources.task_identity import canonical_task_identity
from ..task_sources.task_source import recompute_readiness_statuses
from ..task_sources.todo_vector_index import parse_todo_blocks, split_csv
from ..validation.validation_commands import split_validation_commands
from .multi_supervisor_runner import (
    AUTHORITY_MODE_EMBEDDED,
    AUTHORITY_MODE_LEGACY_MARKDOWN,
    AUTHORITY_MODE_QUACK,
    DATABASE_PROGRAM_CONFIG_INTERFACE,
    FAILOVER_FAIL_CLOSED,
    STATE_LIVE_SCHEMA_REVISION_ENV,
    STATE_STORE_LIVE_GENERATION_ENV,
    TASK_SOURCE_DUCKDB,
    TRUSTED_DUCKDB_HOME_ENV,
    TRUSTED_PYTHON_USER_BASE_ENV,
    TRUSTED_RUNTIME_CACHE_ENV_NAMES,
    DatabaseProgramConfig,
    DatabaseProgramConfigError,
    ImplementationSupervisorTrackConfig,
    PlanBoundSupervisorChild,
    _parse_status_timestamp,
    _plan_bound_positive_child_environment,
    _plan_bound_profile_environment,
    _read_stable_regular_bytes,
    _read_stable_regular_json,
    _StableArtifactReadError,
    _trusted_duckdb_runtime_environment,
    accepted_control_plane_pin_json,
    build_configured_multi_supervisor_cli_runner,
    build_sealed_control_plane_module_command,
    parse_accepted_control_plane_pin,
    parse_database_program_config,
    utc_run_stamp,
    verify_lgcvf_configured_board_live_context,
)
from .provider_capacity_monitor import (
    DEFAULT_RESPONSE_TOKENS_PER_REQUEST,
    ProviderCapacityMonitor,
    ProviderCapacityMonitorConfig,
    count_active_cli_processes,
)
from .resource_scheduler import (
    GENERIC_BUNDLE_RESOURCE_CLASSES,
    LEGACY_RESOURCE_CLASSES,
    PROOF_RESOURCE_CLASSES,
    sample_host_resources,
)

SCHEDULER_SCHEMA_PATTERN = re.compile(
    r"^ipfs_accelerate_py\.agent_supervisor\."
    r"[a-z0-9_.-]+\.scheduler_config@(?:1|2)$"
)
IMPLEMENTATION_ENTRY_PATH = Path(
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
)
CONFIGURED_SCHEDULER_ENTRY_PATH = Path(
    "scripts/ops/agent_supervisor/configured_board_scheduler.py"
)
LGCVF_LIVE_CONFIG_PATH = Path(
    "config/"
    "agent_supervisor_logic_governed_compositional_verification_fabric_"
    "quack_candidate_scheduler.json"
)
LGCVF_LIVE_BOARD_NAMESPACE = (
    "logic-governed-compositional-verification-fabric-v1"
)
PROVIDER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
FALLBACK_PROVIDER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER"
)
FALLBACK_TRIGGER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
)
GROK_MODEL_ENV = "IPFS_ACCELERATE_AGENT_GROK_MODEL"
CODEX_MODEL_ENV = "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
CODEX_REASONING_EFFORT_ENV = (
    "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
)
EXTERNAL_PROVIDER_ISOLATION_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_EXTERNAL_ISOLATION_JSON"
)
GROK_BIN_ENV = "IPFS_ACCELERATE_AGENT_GROK_BIN"
ROUTE_BOARD_NAMESPACE_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_BOARD_NAMESPACE"
)
ROUTE_AUTHORIZATION_PATH_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_PATH"
)
ROUTE_AUTHORIZATION_SHA256_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_SHA256"
)
ROUTE_AUTHORIZATION_ID_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_ID"
)
ROUTE_AUTHORIZATION_KIND_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_KIND"
)
ROUTE_SOURCE_HEAD_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_HEAD"
)
ROUTE_SOURCE_TREE_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_TREE"
)
ROUTE_ID_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_ID"
MAX_COORDINATOR_WAVES = 4096
COORDINATOR_LAUNCH_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/configured-board-coordinator-launch@1"
)
COORDINATOR_LAUNCH_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "repository_commit",
        "repository_tree",
        "configuration_revision",
        "board_namespace",
        "launch_session_id",
        "coordinator_pid",
        "coordinator_pid_path",
        "coordinator_log",
        "coordinator_status_path",
        "coordinator_status_cid",
        "coordinator_profile",
        "coordinator_process_identity",
        "coordinator_argv_cid",
        "receipt_cid",
    }
)
COORDINATOR_STATUS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/configured-board-coordinator-status@1"
)
# The configured watchdog startup grace is the admitted availability bound for
# launch readiness.  These constants are only a bounded fallback and ceiling;
# a procedure-specific value is derived from the closed board below.
COORDINATOR_READY_TIMEOUT_SECONDS = 60.0
COORDINATOR_READY_TIMEOUT_MAX_SECONDS = 600.0
COORDINATOR_STATUS_MAX_AGE_MS = 30_000
COORDINATOR_STATUS_FIELDS = frozenset(
    {
        "schema",
        "repository_commit",
        "repository_tree",
        "configuration_revision",
        "board_namespace",
        "launch_session_id",
        "lifecycle_profile_id",
        "coordinator_pid",
        "coordinator_process_start_ticks",
        "coordinator_argv_cid",
        "started_at_ms",
        "attested_at_ms",
        "phase",
        "lane_status_paths",
        "receipt_cid",
    }
)
FRESH_RECOVERY_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-fresh-generation-recovery-policy@3"
)
FRESH_RECOVERY_VERIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-fresh-generation-recovery-verification@4"
)
FRESH_RECOVERY_PROJECTION_OMISSION_SCHEMA = (
    "lgcvf-recovery-validation-projection-omission@1"
)
FRESH_RECOVERY_PROJECTION_EVIDENCE_SCHEMA = (
    "lgcvf-recovery-validation-projection-evidence@1"
)
FRESH_RECOVERY_TARGET_GENERATION = "lgcvf-run-v17"
FRESH_RECOVERY_TARGET_RELATIVE_ROOT = (
    "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
    "run-v17"
)
FRESH_RECOVERY_CONFIG_PATH = (
    "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
    "scheduler.json"
)
FRESH_RECOVERY_MATERIALIZER_PATH = (
    "scripts/materialize_logic_governed_compositional_verification_fabric_"
    "control_plane.py"
)
FRESH_RECOVERY_VERIFICATION_FIELDS = frozenset(
    {
        "schema",
        "valid",
        "verification_mode",
        "source_generation",
        "target_generation",
        "manifest_cid",
        "receipt_cid",
        "source_evidence_cid",
        "duckdb_runtime_cid",
        "qualification_runtime_cid",
        "qualification_runtime_evidence",
        "qualification_runtime_evidence_cid",
        "materializer_zero_wx_policy",
        "materializer_zero_wx_policy_cid",
        "materializer_zero_wx_qualification_lifecycle",
        "materializer_zero_wx_qualification_lifecycle_cid",
        "materializer_zero_wx_prepublication_lifecycle",
        "materializer_zero_wx_prepublication_lifecycle_cid",
        "materializer_zero_wx_verification_lifecycle",
        "materializer_zero_wx_verification_lifecycle_cid",
        "historical_postpublish_zero_wx_evidence",
        "completed_task_ids",
        "todo_task_ids",
        "blocked_task_ids",
        "completed_count",
        "todo_count",
        "blocked_count",
        "ready_task_ids",
        "validation_qualification_cid",
        "validation_projection_omission_commitment",
        "validation_projection_omission_root",
        "validation_projection_evidence_commitment",
        "validation_projection_evidence_root",
        "model_provider_route",
        "network_isolation_enforced",
        "candidate_authored_validation",
        "validation_completion_authoritative",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "source_database_statuses_read",
        "synthetic_source_disposition",
        "operational_verification_root",
        "stores_unchanged",
        "verification_root",
    }
)
FRESH_RECOVERY_VERIFIER_MAX_OUTPUT_BYTES = 1_048_576
FRESH_RECOVERY_GIT_STATUS_MAX_OUTPUT_BYTES = 1_048_576
FRESH_RECOVERY_IMPORT_INVENTORY_MAX_ENTRIES = 100_000
FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES = 16 * 1024 * 1024
FRESH_RECOVERY_IMPORT_FILE_MAX_BYTES = 64 * 1024 * 1024
FRESH_RECOVERY_IMPORT_CONTENT_MAX_BYTES = 512 * 1024 * 1024
FRESH_RECOVERY_INTERPRETER_MAX_BYTES = 64 * 1024 * 1024
FRESH_RECOVERY_MATERIALIZER_MAX_BYTES = 4 * 1024 * 1024
FRESH_RECOVERY_MATERIALIZER_BOOTSTRAP = (
    "import os,sys\n"
    "_path=sys.argv[1]\n"
    "_fd=int(sys.argv[2])\n"
    "_pycache=sys.argv[3]\n"
    "_cache_stat=os.lstat(_pycache)\n"
    "if not os.path.isdir(_pycache): "
    "raise RuntimeError('pycache root is not a directory')\n"
    "if _cache_stat.st_uid!=os.geteuid() or (_cache_stat.st_mode&0o777)!=0o700: "
    "raise RuntimeError('pycache root authority differs')\n"
    "if os.listdir(_pycache): raise RuntimeError('pycache root is not empty')\n"
    "sys.pycache_prefix=_pycache\n"
    "_source=bytearray()\n"
    "while True:\n"
    " _chunk=os.read(_fd,1048576)\n"
    " if not _chunk: break\n"
    " _source.extend(_chunk)\n"
    " if len(_source)>4194304: raise RuntimeError('materializer exceeds bound')\n"
    "sys.argv=[_path,*sys.argv[4:]]\n"
    "_scope={'__name__':'__main__','__file__':_path,'__package__':None,"
    "'__cached__':None,'__spec__':None}\n"
    "exec(compile(bytes(_source),_path,'exec',dont_inherit=True),_scope,_scope)\n"
)
SCHEDULER_PROVIDER_ENV_NAMES = (
    PROVIDER_ENV,
    FALLBACK_PROVIDER_ENV,
    FALLBACK_TRIGGER_ENV,
    GROK_MODEL_ENV,
    CODEX_MODEL_ENV,
    CODEX_REASONING_EFFORT_ENV,
    EXTERNAL_PROVIDER_ISOLATION_ENV,
    GROK_BIN_ENV,
    ROUTE_BOARD_NAMESPACE_ENV,
    ROUTE_AUTHORIZATION_PATH_ENV,
    ROUTE_AUTHORIZATION_SHA256_ENV,
    ROUTE_AUTHORIZATION_ID_ENV,
    ROUTE_AUTHORIZATION_KIND_ENV,
    ROUTE_SOURCE_HEAD_ENV,
    ROUTE_SOURCE_TREE_ENV,
    ROUTE_ID_ENV,
)
ORDERED_PROVIDER_FIELDS = (
    "primary_provider_id",
    "primary_model_id",
    "fallback_provider_id",
    "fallback_model_id",
    "fallback_trigger",
    "fallback_reasoning_effort",
)
ORDERED_PRIMARY_EXECUTABLE_FIELD = "primary_executable"
ORDERED_PROVIDER_DETECTION_FIELDS = (
    *ORDERED_PROVIDER_FIELDS,
    ORDERED_PRIMARY_EXECUTABLE_FIELD,
)
ORDERED_PRIMARY_PROVIDER_ID = "grok_cli"
ORDERED_PRIMARY_MODEL_ID = "grok-4.6"
LEGACY_V3_PRIMARY_MODEL_ID = "grok-4.5"
ORDERED_FALLBACK_PROVIDER_ID = "codex"
ORDERED_FALLBACK_MODEL_ID = "gpt-5.6-terra"
ORDERED_FALLBACK_TRIGGER = "primary_quota_exhausted"
ORDERED_FALLBACK_TRIGGERS = frozenset(
    {
        "primary_quota_exhausted",
        "primary_quota_or_auth_unavailable",
    }
)
ORDERED_FALLBACK_REASONING_EFFORTS = frozenset({"medium", "high"})
ROUTE_AUTHORIZATION_PATH_FIELD = "route_authorization_path"


class ConfiguredBoardError(ValueError):
    """The scheduler document or its repository binding is inadmissible."""


EAAEF_BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
EAAEF_SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "external_agent_autonomous_execution_fabric.scheduler_config@2"
)
EAAEF_CONFIG_PATH = "config/external_agent_autonomous_execution_fabric_scheduler.json"
EAAEF_TASKBOARD_PATH = (
    "docs/architecture/external_agent_autonomous_execution_fabric/TASK_BOARD.md"
)
EAAEF_TASKBOARD_JSON_PATH = (
    "docs/architecture/external_agent_autonomous_execution_fabric/task_board.json"
)
EAAEF_OBJECTIVES_PATH = (
    "docs/architecture/external_agent_autonomous_execution_fabric/OBJECTIVES.md"
)
EAAEF_PLAN_PATH = "docs/architecture/external_agent_autonomous_execution_fabric/PLAN.md"
EAAEF_VALIDATOR_PATH = (
    "scripts/validate_external_agent_autonomous_execution_fabric_board.py"
)


@dataclass(frozen=True)
class _ConfiguredBoardTaskPopulation:
    all_records: tuple[dict[str, Any], ...]
    ready_records: tuple[dict[str, Any], ...]
    completed_task_ids: tuple[str, ...]
    attempt_limited_task_ids: tuple[str, ...]
    state_snapshot_id: str


def _plan_bound_profile(board: ConfiguredBoard) -> bool:
    """Whether this board requires exact compiler slices and sealed births."""

    return board.board_namespace in {
        "agent-supervisor-prompt-only-self-improvement-v3",
        EAAEF_BOARD_NAMESPACE,
    }


def _eaaef_plan_bound_profile(board: ConfiguredBoard) -> bool:
    return (
        board.board_namespace == EAAEF_BOARD_NAMESPACE
        and board.payload.get("schema") == EAAEF_SCHEDULER_SCHEMA
    )


EAAEF_GENERATION_CURSOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-store-generation-cursor@1"
)
_EAAEF_GENERATION_RE = re.compile(r"^(?P<prefix>.+-v)(?P<n>\d+)$")
_EAAEF_HOST_RECEIPT_NAMES = {
    "EAAEF-191": "admission_bundle.json",
}


def _eaaef_generation_cursor_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / "generation-cursor.json"
    )


def _rewrite_eaaef_generation(
    value: Any,
    from_generation: str,
    to_generation: str,
) -> Any:
    from_match = _EAAEF_GENERATION_RE.fullmatch(from_generation)
    to_match = _EAAEF_GENERATION_RE.fullmatch(to_generation)
    if from_match is None or to_match is None:
        raise ConfiguredBoardError("generation rewrite identities are invalid")
    from_n = from_match.group("n")
    to_n = to_match.group("n")
    if isinstance(value, str):
        rewritten = value
        for old, new in (
            (from_generation, to_generation),
            (f"-run-v{from_n}", f"-run-v{to_n}"),
            (f"/run-v{from_n}", f"/run-v{to_n}"),
        ):
            rewritten = rewritten.replace(old, new)
        return rewritten
    if isinstance(value, list):
        return [
            _rewrite_eaaef_generation(item, from_generation, to_generation)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _rewrite_eaaef_generation(item, from_generation, to_generation)
            for key, item in value.items()
        }
    return value


def _apply_eaaef_generation_cursor(
    payload: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Overlay the gitignored EAAEF run-vN cursor onto a tracked scheduler."""

    configured = str(
        (
            (payload.get("bootstrap_database_program") or {}).get(
                "store_generation"
            )
            if isinstance(payload.get("bootstrap_database_program"), Mapping)
            else ""
        )
        or ""
    )
    cursor_path = _eaaef_generation_cursor_path(repo_root)
    if not cursor_path.is_file():
        return payload
    try:
        cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return payload
    if (
        not isinstance(cursor, dict)
        or cursor.get("schema") != EAAEF_GENERATION_CURSOR_SCHEMA
        or cursor.get("configured_generation") != configured
    ):
        return payload
    active = str(cursor.get("active_generation") or "")
    if not active or active == configured:
        return payload
    return _rewrite_eaaef_generation(copy.deepcopy(payload), configured, active)


def _eaaef_host_receipt_admitted(
    repo_root: Path,
    task_id: str,
    *,
    expected_source_head: str = "",
    expected_source_tree: str = "",
) -> bool:
    filename = _EAAEF_HOST_RECEIPT_NAMES.get(task_id)
    if not filename:
        return False
    if task_id == "EAAEF-191":
        try:
            from ..validation.eaaef_host_admission import (
                verify_current_admission_bundle_receipt,
            )

            verification = verify_current_admission_bundle_receipt(
                repo_root,
                expected_source_head=expected_source_head,
                expected_source_tree=expected_source_tree,
            )
        except Exception:
            return False
        return verification.get("admitted") is True
    path = (
        repo_root
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "receipts/host_admission"
        / filename
    )
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("decision") == "admitted"


def _ordered_primary_models_for_namespace(board_namespace: str) -> frozenset[str]:
    """Return only canonical models admitted by one scheduler namespace.

    The signed prompt-only V3 route is permanently model-bound to Grok 4.5.
    EAAEF has a different source-addressed route and is permanently bound to
    Grok 4.6.  Other scheduler_config@1 boards may use either canonical
    quota-only route; the canonical router still rejects every hybrid tuple.
    """

    if board_namespace == "agent-supervisor-prompt-only-self-improvement-v3":
        return frozenset({LEGACY_V3_PRIMARY_MODEL_ID})
    if board_namespace == EAAEF_BOARD_NAMESPACE:
        return frozenset({ORDERED_PRIMARY_MODEL_ID})
    return frozenset({LEGACY_V3_PRIMARY_MODEL_ID, ORDERED_PRIMARY_MODEL_ID})


def _validate_eaaef_database_programs(
    *,
    board_namespace: str,
    payload: Mapping[str, Any],
    operational_program: DatabaseProgramConfig | None,
) -> None:
    """Keep bootstrap DuckDB and operational Quack roles non-substitutable."""

    if board_namespace != EAAEF_BOARD_NAMESPACE:
        return
    from ..task_sources.eaaef_operational_schema import (
        EAAEF_OPERATIONAL_PROFILE_ID,
    )

    raw_bootstrap = payload.get("bootstrap_database_program")
    if not isinstance(raw_bootstrap, Mapping):
        raise ConfiguredBoardError(
            "EAAEF requires bootstrap_database_program for immutable materialization"
        )
    try:
        bootstrap_program = parse_database_program_config(dict(raw_bootstrap))
    except DatabaseProgramConfigError as exc:
        raise ConfiguredBoardError(
            f"invalid EAAEF bootstrap_database_program: {exc}"
        ) from exc
    if (
        bootstrap_program is None
        or bootstrap_program.authority_mode != AUTHORITY_MODE_EMBEDDED
        or bootstrap_program.task_source_kind != TASK_SOURCE_DUCKDB
        or bootstrap_program.failover_policy != FAILOVER_FAIL_CLOSED
        or bootstrap_program.schema_revision != EAAEF_OPERATIONAL_PROFILE_ID
        or not bootstrap_program.store_id.endswith((".duckdb", ".ddb"))
    ):
        raise ConfiguredBoardError(
            "EAAEF bootstrap_database_program must be embedded DuckDB under "
            "the exact operational profile @2"
        )
    if (
        operational_program is None
        or operational_program.authority_mode != AUTHORITY_MODE_QUACK
        or operational_program.task_source_kind != TASK_SOURCE_DUCKDB
        or operational_program.failover_policy != FAILOVER_FAIL_CLOSED
        or operational_program.schema_revision != EAAEF_OPERATIONAL_PROFILE_ID
        or not operational_program.quack_endpoint
        or not operational_program.endpoint_secret_handle
        or not operational_program.store_id
        or "/" in operational_program.store_id
        or "\\" in operational_program.store_id
        or operational_program.store_id.endswith((".duckdb", ".ddb"))
    ):
        raise ConfiguredBoardError(
            "EAAEF operational database_program must be remote Quack with no "
            "direct-file fallback under the exact operational profile @2"
        )
    if bootstrap_program.to_dict() == operational_program.to_dict():
        raise ConfiguredBoardError(
            "EAAEF bootstrap and operational database programs are conflated"
        )
    from ..validation.external_agent_configured_board_capsule import (
        EAAEF_OPERATIONAL_COMMAND_FABRIC_SHARD_ID,
        ExternalAgentConfiguredBoardCapsuleError,
        validate_eaaef_operational_command_fabric_profile,
    )

    try:
        validate_eaaef_operational_command_fabric_profile(
            payload.get("operational_command_fabric"),
            operational_program=operational_program.to_dict(),
            expected_board_namespace=board_namespace,
            expected_shard_id=EAAEF_OPERATIONAL_COMMAND_FABRIC_SHARD_ID,
        )
    except ExternalAgentConfiguredBoardCapsuleError as exc:
        raise ConfiguredBoardError(str(exc)) from exc


def _targets_fresh_recovery_generation(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> bool:
    """Recognize protected run-v17 through markers or resolved target paths.

    Lexical path checks alone are insufficient because an otherwise ordinary
    profile can name an in-repository symlink whose resolved target is the
    protected generation.  Resolve every authority-bearing database/runtime
    path against the exact checkout root before deciding that recovery
    admission is unnecessary.
    """

    if "fresh_generation_recovery" in payload:
        return True
    program = payload.get("database_program")
    runtime = payload.get("runtime_paths")
    values: list[str] = []
    path_values: list[str] = []
    if isinstance(program, Mapping):
        values.extend(
            str(program.get(field) or "")
            for field in (
                "store_generation",
                "export_profile",
                "store_id",
                "event_store_path",
                "runtime_registry_path",
                "worktree_root",
            )
        )
        path_values.extend(
            str(program.get(field) or "")
            for field in (
                "store_id",
                "event_store_path",
                "runtime_registry_path",
                "worktree_root",
            )
        )
    if isinstance(runtime, Mapping):
        values.extend(str(value or "") for value in runtime.values())
        path_values.extend(
            str(runtime.get(field) or "")
            for field in (
                "root",
                "state",
                "worktrees",
                "merge_queue",
                "logs",
                "evidence",
            )
        )
    for value in values:
        normalized = value.replace("\\", "/")
        if normalized in {
            FRESH_RECOVERY_TARGET_GENERATION,
            "logic-governed-compositional-verification-fabric-run-v17",
        }:
            return True
        if "run-v17" in PurePosixPath(normalized).parts:
            return True
    try:
        resolved_root = repo_root.resolve(strict=False)
        protected_lexical = resolved_root / FRESH_RECOVERY_TARGET_RELATIVE_ROOT
        protected_resolved = protected_lexical.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        # An unresolvable authority path is never grounds for bypassing the
        # protected recovery gate.
        return True
    for text in path_values:
        if not text:
            continue
        raw = Path(text)
        lexical = raw if raw.is_absolute() else resolved_root / raw
        try:
            resolved = lexical.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return True
        if (
            lexical == protected_lexical
            or lexical.is_relative_to(protected_lexical)
            or resolved == protected_resolved
            or resolved.is_relative_to(protected_resolved)
        ):
            return True
    return False


def _sanitized_git_environment() -> dict[str, str]:
    """Return a Git environment without ambient repository/config authority."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"LANG", "LC_ALL", "LC_CTYPE", "TZ"}
    }
    environment.update(
        {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _eaaef_plan_bound_provider_path(board: "ConfiguredBoard") -> str:
    """Return the minimal PATH that exposes admitted EAAEF provider CLIs."""

    path_entries = ["/usr/bin", "/bin"]
    if _eaaef_plan_bound_profile(board):
        for command_name in ("grok", "codex"):
            resolved = shutil.which(command_name)
            if not resolved:
                continue
            # Keep the PATH entry that names the command.  Following the
            # symlink to a versioned download directory would hide `grok`
            # and `codex` from the sealed child's shutil.which().
            directory = str(Path(resolved).parent)
            if directory and directory not in path_entries:
                path_entries.insert(0, directory)
    return os.pathsep.join(path_entries)


def _git_run(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    command = [
        "/usr/bin/git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.trustctime=true",
        "-c",
        "core.checkStat=default",
        "-c",
        "core.attributesFile=/dev/null",
        *argv,
    ]
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=_sanitized_git_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            "",
            f"{type(exc).__name__}: {exc}",
        )


def _canonical_no_symlink_root(path: Path) -> Path:
    """Validate a lexical absolute repository root without following links."""

    raw = Path(path)
    if not raw.is_absolute() or Path(os.path.abspath(raw)) != raw:
        raise ConfiguredBoardError("repository root is not lexical absolute")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise ConfiguredBoardError(
                f"cannot lstat repository root component: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ConfiguredBoardError(
                f"repository root component is not a real directory: {current}"
            )
    if raw.resolve(strict=True) != raw:
        raise ConfiguredBoardError("repository root is not canonical")
    return raw


def _lexical_repo_artifact(repo_root: Path, path: Path) -> tuple[Path, str]:
    """Return an exact contained artifact after rejecting linked parents."""

    root = _canonical_no_symlink_root(repo_root)
    artifact = Path(path)
    if not artifact.is_absolute() or Path(os.path.abspath(artifact)) != artifact:
        raise ConfiguredBoardError(f"authority file is not lexical absolute: {artifact}")
    try:
        relative_path = artifact.relative_to(root)
    except ValueError as exc:
        raise ConfiguredBoardError(
            f"authority file escapes repository: {artifact}"
        ) from exc
    current = root
    for part in relative_path.parts[:-1]:
        current /= part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise ConfiguredBoardError(
                f"cannot lstat authority parent: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ConfiguredBoardError(
                f"authority parent is not a real directory: {current}"
            )
    return artifact, relative_path.as_posix()


def _git_identity(repo_root: Path) -> tuple[str, str]:
    root = _canonical_no_symlink_root(repo_root)
    head = _git_run(("rev-parse", "HEAD"), cwd=root).stdout.strip()
    tree = _git_run(("rev-parse", "HEAD^{tree}"), cwd=root).stdout.strip()
    if not head or not tree:
        raise ConfiguredBoardError("cannot bind adaptive execution plan to HEAD and tree")
    return head, tree


def _identity(value: Any) -> str:
    return content_identity(value)


def _tracked_head_snapshot(
    *,
    repo_root: Path,
    path: Path,
    source_head: str,
    max_bytes: int = 4_194_304,
) -> tuple[bytes, str]:
    """Read one stable regular file whose exact bytes equal ``source_head``.

    This joins the filesystem read and Git authority without parsing or
    hashing a second pathname read.  A symlink, hardlink, untracked file,
    staged/unstaged change, or HEAD replacement fails closed.
    """

    root = _canonical_no_symlink_root(repo_root)
    artifact, relative = _lexical_repo_artifact(root, Path(path))
    try:
        payload, _evidence = _read_stable_regular_bytes(
            artifact,
            max_bytes=max_bytes,
        )
    except _StableArtifactReadError as exc:
        raise ConfiguredBoardError(str(exc)) from exc
    if payload is None:
        raise ConfiguredBoardError(f"authority file is absent: {relative}")
    expected = _git_run(
        ("rev-parse", f"{source_head}:{relative}"),
        cwd=root,
    )
    if expected.returncode != 0 or not expected.stdout.strip():
        raise ConfiguredBoardError(
            f"authority file is not tracked at current HEAD: {relative}"
        )
    try:
        actual = subprocess.run(
            ("/usr/bin/git", "-c", "core.hooksPath=/dev/null", "hash-object", "--stdin"),
            cwd=root,
            env=_sanitized_git_environment(),
            input=payload,
            capture_output=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfiguredBoardError(
            f"cannot hash authority file: {relative}"
        ) from exc
    try:
        actual_oid = actual.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ConfiguredBoardError(
            f"Git returned an invalid blob identity for {relative}"
        ) from exc
    if actual.returncode != 0 or actual_oid != expected.stdout.strip():
        raise ConfiguredBoardError(
            f"authority file differs from current HEAD: {relative}"
        )
    clean = _git_run(("diff", "--quiet", source_head, "--", relative), cwd=root)
    if clean.returncode != 0:
        raise ConfiguredBoardError(
            f"authority file has staged or unstaged changes: {relative}"
        )
    current_head, _current_tree = _git_identity(root)
    if current_head != source_head:
        raise ConfiguredBoardError("repository HEAD changed during authority snapshot")
    revision = _identity(
        {
            "path": relative,
            "git_blob_oid": actual_oid,
            "bytes_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return payload, revision


def _eaaef_normalize_status_overlay(
    rows: Sequence[Any],
    *,
    allowed: set[str],
) -> dict[str, str]:
    overlay: dict[str, str] = {}
    for alias, status in rows:
        task_id = str(alias or "").strip()
        normalized = str(status or "").strip().lower()
        if task_id and normalized in allowed:
            overlay[task_id] = normalized
    return overlay


def _eaaef_live_quack_status_overlay(board: "ConfiguredBoard") -> dict[str, str]:
    """Read task status from the exclusive loopback Quack owner."""

    program = board.database_program
    if program is None:
        return {}
    endpoint = str(program.quack_endpoint or "")
    handle = str(program.endpoint_secret_handle or "")
    if (
        not endpoint.startswith("quack:127.0.0.1:")
        or "'" in endpoint
        or "\x00" in endpoint
        or not handle
    ):
        return {}
    allowed = {
        "todo",
        "blocked",
        "completed",
        "cancelled",
        "failed",
        "quarantined",
        "in_progress",
    }
    runtime_extensions: Any | None = None
    try:
        from ..todo_daemon.eaaef_host_admitted_daemon_gateway import (
            _connect_admitted_duckdb,
            _import_admitted_duckdb,
            _resolve_owner_token,
        )
        from ..validation.eaaef_host_admission import (
            verify_current_admission_bundle_receipt,
        )

        source_head, source_tree = _git_identity(board.repo_root)
        verification = verify_current_admission_bundle_receipt(
            board.repo_root,
            expected_source_head=source_head,
            expected_source_tree=source_tree,
            include_verified_artifacts=True,
        )
        artifacts = verification.get("verified_artifacts")
        if verification.get("admitted") is not True or not isinstance(
            artifacts, Mapping
        ):
            return {}
        duckdb_receipt = artifacts.get("EAAEF-182")
        if not isinstance(duckdb_receipt, Mapping):
            return {}
        duckdb_module, runtime_extensions = _import_admitted_duckdb(
            duckdb_receipt
        )
        generation = str(program.store_generation or "eaaef-run-v14")
        run_dir = generation.removeprefix("eaaef-")
        vault = (
            board.repo_root
            / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
            / run_dir
            / "live/state/quack-owner"
        )
        token = _resolve_owner_token(handle, vault_dir=vault)
        connection = _connect_admitted_duckdb(
            duckdb_module,
            runtime_extensions,
        )
        try:
            connection.execute(
                f"ATTACH '{endpoint}' AS control_plane (TYPE QUACK, TOKEN ?)",
                [token],
            )
            connection.execute("USE control_plane")
            rows = connection.execute(
                "SELECT task_alias, status FROM tasks"
            ).fetchall()
        finally:
            connection.close()
    except Exception:
        return {}
    finally:
        if runtime_extensions is not None:
            runtime_extensions.close()
    return _eaaef_normalize_status_overlay(rows, allowed=allowed)


def _eaaef_task_status_overlay(board: "ConfiguredBoard") -> dict[str, str]:
    """Keep every runtime status projection diagnostic-only.

    Neither an unsigned file nor raw rows from a live Quack process bind the
    current source forest, population, owner birth/generation/fence, and
    terminal receipts. Until the typed CASF-owner snapshot API supplies that
    complete proof, no runtime status may override the tracked task records.
    """

    del board
    return {}


def _configured_board_task_records(
    board: "ConfiguredBoard",
    *,
    source_head: str,
    taskboard_bytes: bytes | None = None,
    provider_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Project every current board row into canonical readiness records."""

    path = board.path(board.taskboard_path)
    if taskboard_bytes is None:
        taskboard_bytes, _revision = _tracked_head_snapshot(
            repo_root=board.repo_root,
            path=path,
            source_head=source_head,
        )
    try:
        text = taskboard_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfiguredBoardError("taskboard is not valid UTF-8") from exc
    provider = board.payload.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    if provider_id is None:
        provider_id = str(
            provider.get("primary_provider_id")
            or provider.get("provider_id")
            or ""
        ).strip()
    records: list[dict[str, Any]] = []
    for task_id, title, _line, fields in parse_todo_blocks(
        text,
        task_header_prefix=board.task_header_prefix,
    ):
        outputs = tuple(split_csv(fields.get("outputs", "")))
        predicted = tuple(
            split_csv(fields.get("predicted_files", "") or fields.get("files", ""))
        )
        task_identity = canonical_task_identity(
            {
                "task_id": task_id,
                "title": title,
                "outputs": outputs,
                "acceptance": str(fields.get("acceptance") or ""),
                "metadata": fields,
            },
            board_namespace=board.board_namespace,
            source_path=path,
        )
        records.append(
            {
                "task_id": task_id,
                "canonical_task_cid": task_identity.canonical_task_cid,
                "status": (
                    str(fields.get("status") or "todo").strip().lower()
                    if str(fields.get("is_schedulable") or "true").strip().lower()
                    in {"1", "true", "yes"}
                    else "blocked"
                ),
                "depends_on": tuple(split_csv(fields.get("depends_on", ""))),
                "outputs": outputs,
                "predicted_files": predicted,
                "validation_commands": tuple(
                    split_validation_commands(str(fields.get("validation") or ""))
                ),
                "priority": str(fields.get("priority") or "P2"),
                "resource_class": str(fields.get("resource_class") or "cpu-small"),
                "provider_id": provider_id,
                "exclusive_group": str(fields.get("exclusive_group") or ""),
                "interfaces": tuple(split_csv(fields.get("interfaces", ""))),
                "submodules": tuple(split_csv(fields.get("submodules", ""))),
                "expected_base_revision": source_head,
                "expected_merge_target": board.merge_target_branch,
                "lease_duration_ms": max(
                    60_000,
                    int(float(board.payload["implementation_timeout_seconds"]) * 1000),
                ),
            }
        )
    overlay = _eaaef_task_status_overlay(board)
    if overlay:
        for record in records:
            status = overlay.get(str(record["task_id"]))
            if status:
                record["status"] = status
    return tuple(records)


def _configured_board_task_state_snapshots(
    board: "ConfiguredBoard",
) -> tuple[Mapping[str, Any], ...]:
    """Load bounded canonical daemon state projections for attempt fencing."""

    state_root = board.path(board.runtime_paths["state"])
    try:
        root_stat = os.lstat(state_root)
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ConfiguredBoardError(
            f"task-state projection root is unreadable: {state_root}"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ConfiguredBoardError(
            f"task-state projection root is not a real directory: {state_root}"
        )
    pending = [state_root]
    discovered: list[Path] = []
    scanned_entries = 0
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = tuple(sorted(iterator, key=lambda item: item.name))
        except OSError as exc:
            raise ConfiguredBoardError(
                f"task-state projection directory is unreadable: {directory}"
            ) from exc
        for entry in entries:
            scanned_entries += 1
            if scanned_entries > 1_024:
                raise ConfiguredBoardError(
                    "task-state projection tree exceeds traversal bound"
                )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ConfiguredBoardError(
                    f"task-state projection entry is unreadable: {entry.path}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                # The managed daemon's latest-log alias is not task state.
                # Every other symbolic entry can conceal attempt state and
                # therefore remains fail-closed.
                if entry.name.endswith("_managed_daemon.latest.log"):
                    continue
                raise ConfiguredBoardError(
                    f"task-state projection entry is a symbolic link: {entry.path}"
                )
            entry_path = Path(entry.path)
            if stat.S_ISDIR(metadata.st_mode):
                # PlanRevisionStore CAS objects and the Quack owner vault are
                # not daemon task-state projections.
                if entry.name in {"plan-revision-store", "quack-owner"}:
                    continue
                pending.append(entry_path)
            elif (
                stat.S_ISREG(metadata.st_mode)
                and entry.name.endswith("_task_state.json")
            ):
                discovered.append(entry_path)
    paths = tuple(sorted(discovered))
    if len(paths) > 128:
        raise ConfiguredBoardError("task-state projection population exceeds bound")
    snapshots: list[Mapping[str, Any]] = []
    for path in paths:
        try:
            path.relative_to(state_root)
            _lexical_repo_artifact(board.repo_root, path)
            payload, _identity = _read_stable_regular_json(path)
            if payload is None:
                raise _StableArtifactReadError(
                    f"task-state projection disappeared: {path}"
                )
        except (OSError, ValueError, _StableArtifactReadError) as exc:
            raise ConfiguredBoardError(
                f"task-state projection is unreadable: {path}"
            ) from exc
        snapshots.append(dict(payload))
    return tuple(snapshots)


def _configured_board_task_population(
    board: "ConfiguredBoard",
    *,
    source_head: str,
    taskboard_bytes: bytes | None = None,
    provider_id: str | None = None,
    task_state_snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> _ConfiguredBoardTaskPopulation:
    """Return the exact dependency-ready, retry-admissible current population."""

    records = _configured_board_task_records(
        board,
        source_head=source_head,
        taskboard_bytes=taskboard_bytes,
        provider_id=provider_id,
    )
    task_ids = [str(item["task_id"]) for item in records]
    if len(task_ids) != len(set(task_ids)):
        raise ConfiguredBoardError("taskboard contains duplicate task IDs")
    completed = tuple(
        sorted(
            str(item["task_id"])
            for item in records
            if str(item.get("status") or "").lower()
            in {"complete", "completed", "done"}
        )
    )
    statuses = recompute_readiness_statuses(
        records,
        completed_ids=completed,
    )
    snapshots = tuple(
        _configured_board_task_state_snapshots(board)
        if task_state_snapshots is None
        else (dict(item) for item in task_state_snapshots)
    )
    current_cid_by_id = {
        str(item["task_id"]): str(item["canonical_task_cid"])
        for item in records
    }
    legacy_attempts_by_id: dict[str, int] = {}
    attempts_by_task_revision: dict[tuple[str, str], int] = {}
    attempts_by_cid: dict[str, int] = {}

    def attempt_count(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfiguredBoardError("task-state attempt count is invalid")
        return value

    for snapshot in snapshots:
        raw_by_id = snapshot.get("implementation_attempts")
        raw_by_cid = snapshot.get("implementation_attempts_by_cid")
        raw_task_identities = snapshot.get("task_identities")
        if raw_by_id not in (None, {}) and not isinstance(raw_by_id, Mapping):
            raise ConfiguredBoardError("task-state implementation_attempts is invalid")
        if raw_by_cid not in (None, {}) and not isinstance(raw_by_cid, Mapping):
            raise ConfiguredBoardError(
                "task-state implementation_attempts_by_cid is invalid"
            )
        if raw_task_identities not in (None, {}) and not isinstance(
            raw_task_identities, Mapping
        ):
            raise ConfiguredBoardError("task-state task_identities is invalid")

        snapshot_attempts_by_cid = {
            str(key): attempt_count(value)
            for key, value in dict(raw_by_cid or {}).items()
        }
        for canonical_task_cid, count in snapshot_attempts_by_cid.items():
            attempts_by_cid[canonical_task_cid] = max(
                attempts_by_cid.get(canonical_task_cid, 0), count
            )

        identity_cid_by_id: dict[str, str] = {}
        for key, value in dict(raw_task_identities or {}).items():
            display_task_id = str(key)
            if not isinstance(value, Mapping):
                raise ConfiguredBoardError("task-state task identity is invalid")
            identity_display_task_id = value.get("display_task_id")
            if identity_display_task_id not in (None, "") and (
                not isinstance(identity_display_task_id, str)
                or identity_display_task_id.strip() != display_task_id
            ):
                raise ConfiguredBoardError(
                    "task-state task identity display ID is invalid"
                )
            identity_cid = value.get("canonical_task_cid")
            if identity_cid in (None, ""):
                # Older projections carried provenance without a canonical
                # identity.  Their display-ID counter remains a conservative
                # retry limit for every later revision of the same task ID.
                continue
            if (
                not isinstance(identity_cid, str)
                or not identity_cid.strip()
                or identity_cid != identity_cid.strip()
            ):
                raise ConfiguredBoardError(
                    "task-state canonical task identity is invalid"
                )
            identity_cid_by_id[display_task_id] = identity_cid

        for key, value in dict(raw_by_id or {}).items():
            display_task_id = str(key)
            count = attempt_count(value)
            identity_cid = identity_cid_by_id.get(display_task_id)
            if not identity_cid:
                legacy_attempts_by_id[display_task_id] = max(
                    legacy_attempts_by_id.get(display_task_id, 0), count
                )
                continue
            current_cid = current_cid_by_id.get(display_task_id)
            if (
                current_cid
                and identity_cid != current_cid
                and snapshot_attempts_by_cid.get(identity_cid, 0) < count
            ):
                raise ConfiguredBoardError(
                    "task-state mismatched task identity is not backed by "
                    "its canonical attempt ledger"
                )
            revision_key = (display_task_id, identity_cid)
            attempts_by_task_revision[revision_key] = max(
                attempts_by_task_revision.get(revision_key, 0), count
            )
    max_attempts = int(board.payload["max_task_attempts"])
    attempt_limited: set[str] = set()
    ready: list[dict[str, Any]] = []
    for record in records:
        task_id = str(record["task_id"])
        task_cid = str(record["canonical_task_cid"])
        if statuses.get(task_id) != "ready":
            continue
        attempt_count = max(
            legacy_attempts_by_id.get(task_id, 0),
            attempts_by_task_revision.get((task_id, task_cid), 0),
            attempts_by_cid.get(task_cid, 0),
        )
        if max_attempts > 0 and attempt_count >= max_attempts:
            attempt_limited.add(task_id)
            continue
        ready.append(record)
    state_snapshot_id = _identity(
        {
            "statuses": statuses,
            "implementation_attempts": legacy_attempts_by_id,
            "implementation_attempts_by_task_revision": [
                {
                    "task_id": task_id,
                    "canonical_task_cid": canonical_task_cid,
                    "attempts": count,
                }
                for (task_id, canonical_task_cid), count in sorted(
                    attempts_by_task_revision.items()
                )
            ],
            "implementation_attempts_by_cid": attempts_by_cid,
        }
    )
    return _ConfiguredBoardTaskPopulation(
        all_records=records,
        ready_records=tuple(ready),
        completed_task_ids=completed,
        attempt_limited_task_ids=tuple(sorted(attempt_limited)),
        state_snapshot_id=state_snapshot_id,
    )


def _plan_authority_roots(
    board: "ConfiguredBoard",
    *,
    head: str,
    tree: str,
    task_source_revision: str,
    task_population: _ConfiguredBoardTaskPopulation,
    route_capacity_profile_id: str = "",
) -> PlanAuthorityRoots:
    source = {
        "board_namespace": board.board_namespace,
        "taskboard_path": board.taskboard_path,
        "task_source_revision": task_source_revision,
    }
    return PlanAuthorityRoots(
        repository_id=_slug(board.board_namespace),
        repository_root_cid=_identity({"head": head, "tree": tree}),
        dirty_worktree_root=_identity({"tree": tree}),
        task_source_id=_identity(source),
        task_source_revision=task_source_revision,
        policy_root=_identity({"protected_paths": board.protected_paths}),
        intent_ir_root=_identity({"plan_path": board.plan_path}),
        legal_ir_root=_identity({"board_namespace": board.board_namespace}),
        security_ir_root=_identity({"protected_paths": board.protected_paths}),
        program_root=_identity(
            {
                "task_ids": [
                    item["task_id"] for item in task_population.all_records
                ],
                "ready_task_ids": [
                    item["task_id"] for item in task_population.ready_records
                ],
                "attempt_limited_task_ids": list(
                    task_population.attempt_limited_task_ids
                ),
                "state_snapshot_id": task_population.state_snapshot_id,
            }
        ),
        capability_catalog_root=_identity({"submodules": board.worktree_submodule_paths}),
        provider_catalog_root=_identity(
            {
                "provider": dict(board.payload.get("provider") or {}),
                "route_capacity_profile_id": route_capacity_profile_id,
            }
        ),
        usage_policy_root=_identity({"max_lanes": board.max_lanes}),
        configuration_root=board.configuration_root,
    )


def _configured_board_host_slots(board: "ConfiguredBoard") -> tuple[int, int]:
    """Return configured CPU/process slot ceilings, at least max_lanes."""

    host_payload = board.payload.get("host_capacity")
    if host_payload is None:
        return board.max_lanes, board.max_lanes
    if not isinstance(host_payload, Mapping):
        raise ConfiguredBoardError("host_capacity must be an object")
    cpu_slots = _positive_int(
        host_payload.get("cpu_slots"),
        field="host_capacity.cpu_slots",
    )
    process_slots = _positive_int(
        host_payload.get("process_slots"),
        field="host_capacity.process_slots",
    )
    if cpu_slots < board.max_lanes or process_slots < board.max_lanes:
        raise ConfiguredBoardError(
            "host_capacity slots must be at least max_lanes"
        )
    return cpu_slots, process_slots


def configured_board_capacity_observation(
    board: "ConfiguredBoard",
    *,
    now_ms: int | None = None,
    host_capacity_snapshot: Mapping[str, Any] | None = None,
    provider_capacity_snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], int]:
    """Return fresh host/provider evidence for compile and pre-claim gates.

    Configuration limits remain ceilings only.  In production the provider
    records always come from the authenticated readiness/process monitor; the
    optional records exist for deterministic contract tests.
    """

    if now_ms is None:
        current_ms = None
    elif isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms <= 0:
        raise ConfiguredBoardError("capacity observation time is invalid")
    else:
        current_ms = now_ms
    cpu_slots, process_slots = _configured_board_host_slots(board)
    worker_limit = max(board.max_lanes, cpu_slots, process_slots)
    if host_capacity_snapshot is None:
        host = sample_host_resources(
            board.repo_root,
            worker_limit=worker_limit,
            active_phase="execution",
        ).to_dict()
        host["cpu_slots"] = cpu_slots
        host["process_slots"] = process_slots
        host["worker_limit"] = worker_limit
        host["available_worker_capacity"] = max(
            0, worker_limit - int(host.get("active_workers") or 0)
        )
        if _eaaef_plan_bound_profile(board):
            host["resource_classes"] = list(
                dict.fromkeys(
                    (
                        *LEGACY_RESOURCE_CLASSES,
                        *GENERIC_BUNDLE_RESOURCE_CLASSES,
                        *PROOF_RESOURCE_CLASSES,
                    )
                )
            )
    else:
        host = dict(host_capacity_snapshot)
    provider_payload = board.payload.get("provider")
    provider_payload = (
        provider_payload if isinstance(provider_payload, Mapping) else {}
    )
    provider_max_age_ms = max(
        5_000,
        int(float(board.payload["poll_interval_seconds"]) * 3_000),
    )
    if provider_capacity_snapshots is None:
        configured_concurrency = int(
            provider_payload.get("max_concurrency") or 1
        )
        process_counter = None
        if _eaaef_plan_bound_profile(board):
            markers = (
                str(board.repo_root),
                "external_agent_autonomous_execution_fabric",
                "external-agent-autonomous-execution-fabric",
            )
            process_counter = lambda: count_active_cli_processes(
                cmdline_markers=markers
            )
        monitor = ProviderCapacityMonitor(
            ProviderCapacityMonitorConfig(
                snapshot_path=(
                    board.path(board.runtime_paths["state"])
                    / "provider-capacity.json"
                ),
                max_age_ms=provider_max_age_ms,
                interval_seconds=min(
                    float(board.payload["poll_interval_seconds"]),
                    provider_max_age_ms / 2_000,
                ),
                grok_max_concurrency=configured_concurrency,
                codex_max_concurrency=configured_concurrency,
                grok_request_budget=configured_concurrency,
                codex_request_budget=configured_concurrency,
                grok_token_budget=(
                    configured_concurrency
                    * DEFAULT_RESPONSE_TOKENS_PER_REQUEST
                ),
                codex_token_budget=(
                    configured_concurrency
                    * DEFAULT_RESPONSE_TOKENS_PER_REQUEST
                ),
            ),
            process_counter=process_counter,
        )
        sampled, _diagnostics = monitor.sample()
        providers = tuple(dict(item.to_dict()) for item in sampled)
    else:
        providers = tuple(dict(item) for item in provider_capacity_snapshots)
    if not providers:
        raise ConfiguredBoardError("fresh provider capacity evidence is required")
    if current_ms is None:
        # Freshness is measured only against this process's trusted local
        # clock, and only after the samples for this observation exist.
        # Provider timestamps are evidence, never clock authority; a
        # future-dated record must not advance its own freshness boundary.
        current_ms = int(time.time() * 1000)
    return host, providers, current_ms


def configured_board_route_capacity_projection(
    board: "ConfiguredBoard",
    *,
    provider_capacity_snapshots: Sequence[Mapping[str, Any]],
    now_ms: int,
) -> tuple[dict[str, Any], AgentImplementationRoutePlan]:
    """Return the router-owned logical provider snapshot for the sealed route.

    The scheduler deliberately supplies unclassified monitor observations and
    retains the router DTO unchanged.  In particular, the fallback lane's
    capacity is never interpreted here as dispatch authority.
    """

    if not _plan_bound_profile(board):
        raise ConfiguredBoardError(
            "logical route capacity projection requires the sealed v3 profile"
        )
    provider = board.payload.get("provider")
    if not isinstance(provider, Mapping):
        raise ConfiguredBoardError("sealed v3 provider configuration is absent")
    route = _resolved_ordered_provider_route(
        provider,
        repo_root=board.repo_root,
        board_namespace=board.board_namespace,
    )
    max_age_ms = max(
        5_000,
        int(float(board.payload["poll_interval_seconds"]) * 3_000),
    )
    try:
        profile = project_agent_implementation_route_capacity(
            route,
            observations=[dict(item) for item in provider_capacity_snapshots],
            now_ms=now_ms,
            max_age_ms=max_age_ms,
        )
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardError(
            "router rejected provider capacity observations"
        ) from exc
    snapshot = profile.as_compiler_snapshot()
    if (
        not isinstance(snapshot, dict)
        or snapshot != profile.as_dict()
        or snapshot.get("provider_id") != route.route_id
        or snapshot.get("route_id") != route.route_id
        or not isinstance(snapshot.get("profile_id"), str)
        or not snapshot["profile_id"]
    ):
        raise ConfiguredBoardError(
            "router returned a noncanonical logical capacity snapshot"
        )
    return dict(snapshot), route


def materialize_configured_board_execution_plan(
    board: "ConfiguredBoard",
    *,
    now_ms: int | None = None,
    host_capacity_snapshot: Mapping[str, Any] | None = None,
    provider_capacity_snapshots: Sequence[Mapping[str, Any]] | None = None,
    task_state_snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> ParallelismDecisionReceipt | None:
    """Compile and atomically publish one exact v3 wave before child launch."""

    if not _plan_bound_profile(board):
        raise ConfiguredBoardError("adaptive plan materialization requires the sealed v3 profile")
    head, tree = _git_identity(board.repo_root)
    current_board = load_configured_board(
        board.config_path,
        repo_root=board.repo_root,
    )
    if (
        current_board.configuration_root != board.configuration_root
        or current_board.board_namespace != board.board_namespace
    ):
        raise ConfiguredBoardError(
            "scheduler configuration changed before wave materialization"
        )
    board = current_board
    config_bytes, _config_revision = _tracked_head_snapshot(
        repo_root=board.repo_root,
        path=board.config_path,
        source_head=head,
    )
    if _identity(
        {"bytes_sha256": hashlib.sha256(config_bytes).hexdigest()}
    ) != board.configuration_root:
        raise ConfiguredBoardError(
            "tracked scheduler config root differs from parsed configuration"
        )
    taskboard_bytes, task_source_revision = _tracked_head_snapshot(
        repo_root=board.repo_root,
        path=board.path(board.taskboard_path),
        source_head=head,
    )
    host, provider_observations, current_ms = configured_board_capacity_observation(
        board,
        now_ms=now_ms,
        host_capacity_snapshot=host_capacity_snapshot,
        provider_capacity_snapshots=provider_capacity_snapshots,
    )
    route_capacity, route = configured_board_route_capacity_projection(
        board,
        provider_capacity_snapshots=provider_observations,
        now_ms=current_ms,
    )
    task_population = _configured_board_task_population(
        board,
        source_head=head,
        taskboard_bytes=taskboard_bytes,
        provider_id=route.route_id,
        task_state_snapshots=task_state_snapshots,
    )
    records = task_population.ready_records
    if not records:
        return None
    roots = _plan_authority_roots(
        board,
        head=head,
        tree=tree,
        task_source_revision=task_source_revision,
        task_population=task_population,
        route_capacity_profile_id=str(route_capacity["profile_id"]),
    )
    providers = (route_capacity,)
    observed = int(host.get("observed_at_ms") or current_ms)
    host_capacity = {
        **host,
        "observed_at_ms": observed,
        "fresh_until_ms": observed + 60_000,
        "max_age_ms": 60_000,
    }
    capacity = {
        **host_capacity,
        "host": host_capacity,
        "providers": list(providers),
        "provider_observations": [
            dict(item) for item in provider_observations
        ],
        "route_capacity_profile_id": route_capacity["profile_id"],
    }
    store_root = board.path(board.runtime_paths["state"]) / "plan-revision-store"
    store = PlanRevisionStore(store_root)
    adapter = ProductionParallelPlanAdapter(store)
    active = None
    prior_revision = None
    scope_drift_leases: list[tuple[str, Any]] = []
    denied_wave_barrier: tuple[str, Any] | None = None
    denied_wave_dispositions: list[tuple[str, Any]] = []
    try:
        with store._thread_lock:  # noqa: SLF001
            with store._guard():  # noqa: SLF001
                active = _secure_store_active(store)
                if active is not None:
                    stored_revision = _secure_store_cas(
                        store, active.revision_cid
                    )
                    prior_revision = PlanRevision.from_dict(stored_revision)
                    if prior_revision.to_dict() != stored_revision:
                        raise ExecutionPlanError(
                            "active plan revision changed during typed decode"
                        )
                    manifest_payload = _secure_store_cas(
                        store,
                        prior_revision.materialization_transaction_cid,
                    )
                    prior_manifest = ConfiguredBoardExecutionSlices.from_dict(
                        manifest_payload
                    )
                    if prior_manifest.to_dict() != manifest_payload:
                        raise ExecutionPlanError(
                            "active slice manifest changed during typed decode"
                        )
                    observed_barrier = _load_plan_bound_wave_diff_barrier_locked(
                        store,
                        revision_cid=active.revision_cid,
                        slice_manifest_cid=(
                            prior_revision.materialization_transaction_cid
                        ),
                    )
                    if (
                        observed_barrier is not None
                        and observed_barrier[1].decision != "released"
                    ):
                        denied_wave_barrier = observed_barrier
                        for row in observed_barrier[1].dispositions:
                            disposition = (
                                _load_plan_bound_proposal_disposition_locked(
                                    store,
                                    revision_cid=active.revision_cid,
                                    slice_id=row["slice_id"],
                                )
                            )
                            if (
                                disposition is None
                                or disposition[0] != row["disposition_cid"]
                            ):
                                raise ExecutionPlanError(
                                    "denied wave lost proposal disposition evidence"
                                )
                            denied_wave_dispositions.append(disposition)
                    for execution_slice in prior_manifest.slices:
                        reassignment = adapter._load_slice_reassignment_locked(  # noqa: SLF001
                            revision_cid=active.revision_cid,
                            slice_id=execution_slice.slice_id,
                        )
                        owner_lane_id = (
                            reassignment[1].recipient_lane_id
                            if reassignment is not None
                            else execution_slice.lane_id
                        )
                        execution_lease = (
                            _load_plan_bound_execution_lease_locked(
                                store,
                                revision_cid=active.revision_cid,
                                slice_id=execution_slice.slice_id,
                                lane_id=owner_lane_id,
                            )
                        )
                        if (
                            execution_lease is not None
                            and execution_lease[1].phase == "scope_drift"
                        ):
                            scope_drift_leases.append(execution_lease)
    except ExecutionPlanError as exc:
        raise ConfiguredBoardError(
            "cannot securely adopt the active plan revision"
        ) from exc

    prior_conflict_cid = (
        prior_revision.conflict_contract.conflict_surface_cid
        if prior_revision is not None
        else ""
    )
    observed_scope_paths = {
        path
        for _lease_cid, lease in scope_drift_leases
        for path in lease.actual_changed_paths
    }
    observed_scope_paths.update(
        path
        for _disposition_cid, disposition in denied_wave_dispositions
        for path in disposition.actual_changed_paths
    )
    if prior_conflict_cid and prior_revision is not None:
        observed_scope_paths.update(
            prior_revision.conflict_contract.predicted_files
        )
    scope_drift_evidence_cid = prior_conflict_cid
    if scope_drift_leases or denied_wave_barrier is not None:
        scope_drift_evidence_cid = _identity(
            {
                "kind": "plan-bound-actual-scope-drift",
                "prior_conflict_surface_cid": prior_conflict_cid,
                "wave_barrier_cid": (
                    denied_wave_barrier[0]
                    if denied_wave_barrier is not None
                    else ""
                ),
                "wave_barrier_decision": (
                    denied_wave_barrier[1].decision
                    if denied_wave_barrier is not None
                    else ""
                ),
                "wave_barrier_reason_codes": (
                    list(denied_wave_barrier[1].reason_codes)
                    if denied_wave_barrier is not None
                    else []
                ),
                "proposal_disposition_cids": [
                    disposition_cid
                    for disposition_cid, _disposition in denied_wave_dispositions
                ],
                "execution_lease_cids": [
                    lease_cid for lease_cid, _lease in scope_drift_leases
                ],
                "proposal_receipt_ids": [
                    lease.proposal_receipt_id
                    for _lease_cid, lease in scope_drift_leases
                ],
                "changed_paths": sorted(observed_scope_paths),
                "merge_enqueue_reached": False,
            }
        )
    budget = InvocationBudget(
        max_lanes=1 if scope_drift_evidence_cid else board.max_lanes
    )
    plan_root_cid = _identity(
        {
            "roots": roots.to_dict(),
            "task_cids": [
                record["canonical_task_cid"]
                for record in task_population.all_records
            ],
            "ready_task_cids": [
                record["canonical_task_cid"] for record in records
            ],
            "state_snapshot_id": task_population.state_snapshot_id,
            "capacity_snapshot": capacity,
            "invocation_budget": budget.to_dict(),
            "scope_drift_evidence_cid": scope_drift_evidence_cid,
        }
    )
    observed_board = load_configured_board(
        board.config_path,
        repo_root=board.repo_root,
    )
    observed_taskboard, observed_task_revision = _tracked_head_snapshot(
        repo_root=board.repo_root,
        path=board.path(board.taskboard_path),
        source_head=head,
    )
    observed_config, _observed_config_revision = _tracked_head_snapshot(
        repo_root=board.repo_root,
        path=board.config_path,
        source_head=head,
    )
    if (
        observed_board.configuration_root != board.configuration_root
        or observed_config != config_bytes
        or observed_task_revision != task_source_revision
        or observed_taskboard != taskboard_bytes
        or _git_identity(board.repo_root) != (head, tree)
    ):
        raise ConfiguredBoardError(
            "repository/configuration/task authority changed before publish"
        )
    plan, slices = adapter.compile_wave(
        board_namespace=board.board_namespace,
        plan_root_cid=plan_root_cid,
        tasks=records,
        budget=budget,
        repository_snapshot={
            "tree_id": tree,
            "merge_target": board.merge_target_branch,
            "protected_paths": list(board.protected_paths),
        },
        capacity_snapshot=capacity,
        provider_snapshots=providers,
        completed_task_ids=task_population.completed_task_ids,
        protected_paths=board.protected_paths,
        submodule_paths=board.worktree_submodule_paths,
        post_merge_validation=(
            f"{sys.executable} {board.path(board.validator_path)}",
        ),
        source_head=head,
        task_source_revision=task_source_revision,
        configuration_root=board.configuration_root,
        current_time_ms=current_ms,
    )
    task_cids = tuple(
        record["canonical_task_cid"]
        for record in task_population.all_records
    )
    cid_by_id = {
        str(record["task_id"]): str(record["canonical_task_cid"])
        for record in task_population.all_records
    }
    goal_cid = _identity({"board_namespace": board.board_namespace, "kind": "goal-population"})
    same_wave_adoption = bool(active and active.plan_root_cid == plan_root_cid)
    semantic_revision = (
        active.semantic_revision if same_wave_adoption and active is not None
        else ((active.semantic_revision + 1) if active is not None else 1)
    )
    delta: PlanDelta | None = None
    if active is not None and not same_wave_adoption:
        delta_request_cid = _identity(
            {
                "base_revision_cid": active.revision_cid,
                "next_plan_root_cid": plan_root_cid,
                "capacity_snapshot_id": plan.capacity_snapshot_id,
                "task_source_revision": task_source_revision,
            }
        )
        delta = PlanDelta(
            base_plan_root=active.plan_root_cid,
            base_plan_revision=active.semantic_revision,
            request_cid=delta_request_cid,
            roots=roots,
            items=(
                PlanDeltaItem(
                    item_key="configured-wave-replan",
                    operation=PlanDeltaOperation.ATTACH_EVIDENCE,
                    target_cid=active.plan_root_cid,
                    expected_target_lifecycle=LifecycleState.PROPOSED,
                    expected_target_spec_revision=active.revision_cid,
                    before_digest=active.plan_root_cid,
                    after_record_cid=plan_root_cid,
                    effect_class=DeltaEffectClass.EVIDENCE_ONLY,
                    rationale=(
                        "Recompile from a fresh repository, task-source, "
                        "attempt, and capacity observation."
                    ),
                    provenance={
                        "source_head": head,
                        "task_source_revision": task_source_revision,
                        "capacity_snapshot_id": plan.capacity_snapshot_id,
                        "scope_drift_evidence_cid": scope_drift_evidence_cid,
                    },
                    resource_impact=(plan.capacity_snapshot_id,),
                ),
            ),
        )

    current_completed_cids = {
        cid_by_id[task_id]
        for task_id in task_population.completed_task_ids
        if task_id in cid_by_id
    }
    prior_task_cids = set(
        prior_revision.task_population.member_cids if prior_revision else ()
    )
    prior_completed_cids = set(
        prior_revision.completed_population.member_cids if prior_revision else ()
    )
    prior_claimed_cids = set(
        prior_revision.claimed_population.member_cids if prior_revision else ()
    )
    blocked_cids = {
        str(record["canonical_task_cid"])
        for record in task_population.all_records
        if str(record.get("status") or "").lower()
        in {"blocked", "failed", "quarantined"}
    }
    blocked_cids.update(
        cid_by_id[task_id]
        for task_id in task_population.attempt_limited_task_ids
        if task_id in cid_by_id
    )

    def revision_factory(execution_plan_cid: str, slice_manifest_cid: str) -> PlanRevision:
        if prior_revision is not None and same_wave_adoption:
            return prior_revision
        origin = PlanOrigin.STEER if active is not None else PlanOrigin.CREATE
        return PlanRevision(
            plan_root_cid=plan_root_cid,
            semantic_revision=semantic_revision,
            parent_plan_root=(active.plan_root_cid if active else ""),
            origin=origin,
            roots=roots,
            request_cid=_identity(
                {
                    "budget": budget.to_dict(),
                    "tree": tree,
                    "active_revision_cid": active.revision_cid if active else "",
                }
            ),
            delta_cid=(delta.delta_cid if delta is not None else ""),
            scan_receipt_cid=_identity({"task_source_revision": task_source_revision}),
            query_plan_cid=_identity({"task_ids": [record["task_id"] for record in records]}),
            evidence_bundle_cid=_identity(
                {
                    "config": roots.configuration_root,
                    "scope_drift_evidence_cid": scope_drift_evidence_cid,
                }
            ),
            admission_receipt_cid=_identity({"plan_root_cid": plan_root_cid, "admitted": True}),
            execution_plan_cid=execution_plan_cid,
            goal_population=PlanPopulationDigest(PopulationKind.RETAINED, (goal_cid,)),
            task_population=PlanPopulationDigest(PopulationKind.RETAINED, task_cids),
            added_population=PlanPopulationDigest(
                PopulationKind.ADDED,
                (
                    (goal_cid, *task_cids)
                    if prior_revision is None
                    else tuple(sorted(set(task_cids) - prior_task_cids))
                ),
            ),
            superseded_population=PlanPopulationDigest(PopulationKind.SUPERSEDED),
            retained_population=PlanPopulationDigest(
                PopulationKind.RETAINED,
                tuple(sorted(set(task_cids) & prior_task_cids)),
            ),
            deferred_population=PlanPopulationDigest(PopulationKind.DEFERRED),
            claimed_population=PlanPopulationDigest(
                PopulationKind.CLAIMED,
                tuple(sorted(prior_claimed_cids)),
            ),
            completed_population=PlanPopulationDigest(
                PopulationKind.COMPLETED,
                tuple(sorted(prior_completed_cids | current_completed_cids)),
            ),
            blocked_population=PlanPopulationDigest(
                PopulationKind.BLOCKED,
                tuple(sorted(blocked_cids)),
            ),
            resource_contract=PlanResourceContract(resource_class="process-control"),
            provider_contract=PlanProviderContract(
                provider_requirement=route.route_id
            ),
            lease_contract=PlanLeaseContract(
                lease_duration_ms=max(60_000, int(float(board.payload["implementation_timeout_seconds"]) * 1000)),
                fencing_epoch=semantic_revision,
                heartbeat_interval_ms=max(1, int(float(board.payload["poll_interval_seconds"]) * 1000)),
            ),
            retry_contract=PlanRetryContract(max_retries=int(board.payload["max_task_attempts"])),
            worktree_contract=PlanWorktreeContract(
                policy="isolated",
                expected_base_revision=head,
                expected_merge_target=board.merge_target_branch,
                isolation_required=True,
            ),
            merge_strategy=PlanMergeStrategy(
                kind=MergeStrategyKind.REBASE_THEN_MERGE,
                merge_train_id=f"merge-train:{_slug(board.board_namespace)}",
                post_merge_validation_cids=(_identity({"validator": board.validator_path}),),
            ),
            conflict_contract=PlanConflictContract(
                predicted_files=tuple(
                    sorted(
                        {
                            *observed_scope_paths,
                            *(
                                path
                                for record in records
                                for path in record.get("predicted_files", ())
                            ),
                        }
                    )
                ),
                protected_paths=board.protected_paths,
                conflict_surface_cid=scope_drift_evidence_cid,
            ),
            completion_rule=PlanCompletionRule(authority=CompletionAuthority.VALIDATION_GATE),
            validation_dag=(
                PlanValidationNode(
                    validation_key="configured-board-post-merge",
                    argv=(sys.executable, str(board.path(board.validator_path))),
                ),
            ),
            materialization_transaction_cid=slice_manifest_cid,
            rollback_ref=(
                prior_revision.rollback_ref if prior_revision is not None else head
            ),
            event_cursor=task_source_revision,
        )

    return adapter.publish_wave(
        plan=plan,
        slice_manifest=slices,
        revision_factory=revision_factory,
        observed_roots=roots,
        idempotency_key=f"configured-wave:{plan_root_cid}:{plan.plan_id}",
        delta=delta,
        expected_active_plan_root=(active.plan_root_cid if active else ""),
        expected_active_revision_cid=(active.revision_cid if active else ""),
        base_event_cursor=(active.event_cursor if active else ""),
        fencing_token=semantic_revision,
        lease_id=f"configured-wave:{semantic_revision}:{plan.plan_id}",
    )


def _resolved_ordered_provider_route(
    provider: Mapping[str, Any],
    *,
    repo_root: Path,
    board_namespace: str,
) -> AgentImplementationRoutePlan:
    """Resolve scheduler profile input through the canonical router policy."""

    values = {
        field: _provider_string(provider, field)
        for field in ORDERED_PROVIDER_FIELDS
    }
    authorization = None
    authorization_path = str(
        provider.get(ROUTE_AUTHORIZATION_PATH_FIELD) or ""
    ).strip()
    if (
        not authorization_path
        and board_namespace == EAAEF_BOARD_NAMESPACE
        and values["fallback_trigger"] == "primary_quota_or_auth_unavailable"
    ):
        # The EAAEF authorization is deliberately published only after the
        # reviewed source tree is frozen.  Derive its create-once path from
        # that tree rather than embedding a post-freeze CID/path in the
        # tracked scheduler config (which would create a source-tree cycle).
        _source_head, source_tree = _git_identity(repo_root)
        authorization_path = eaaef_agent_route_authorization_path(source_tree)
    if authorization_path:
        try:
            authorization = load_agent_implementation_route_authorization(
                repo_root=repo_root,
                artifact_path=authorization_path,
                board_namespace=board_namespace,
            )
        except (OSError, ValueError) as exc:
            raise ConfiguredBoardError(str(exc)) from exc
    try:
        return resolve_agent_implementation_route(
            **values,
            authorization=authorization,
        )
    except ValueError as exc:
        raise ConfiguredBoardError(str(exc)) from exc


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfiguredBoardError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ConfiguredBoardError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardError(
            f"{field} must be a positive integer"
        ) from exc
    if parsed < 1:
        raise ConfiguredBoardError(f"{field} must be a positive integer")
    return parsed


def _nonnegative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ConfiguredBoardError(f"{field} must be finite and nonnegative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardError(
            f"{field} must be finite and nonnegative"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ConfiguredBoardError(f"{field} must be finite and nonnegative")
    return parsed


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfiguredBoardError(f"{field} must be a nonnegative integer")
    return value


def _optional_positive_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    qualified_field: str,
) -> int | None:
    """Return one optional, strictly typed positive JSON integer."""

    if field not in payload:
        return None
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfiguredBoardError(
            f"{qualified_field} must be a positive integer"
        )
    return value


def _objective_refill_controls(
    payload: Mapping[str, Any],
) -> tuple[int, int, int, int | None, int | None] | None:
    """Return sealed low-watermark, pass, cooldown, and campaign bounds."""

    refill_policy = payload.get("refill_policy")
    derived = (
        refill_policy.get("derived_refill")
        if isinstance(refill_policy, Mapping)
        else None
    )
    max_epochs = (
        _optional_positive_int(
            derived,
            "max_epochs",
            qualified_field="refill_policy.derived_refill.max_epochs",
        )
        if isinstance(derived, Mapping)
        else None
    )
    max_total_tasks = (
        _optional_positive_int(
            derived,
            "max_total_tasks",
            qualified_field="refill_policy.derived_refill.max_total_tasks",
        )
        if isinstance(derived, Mapping)
        else None
    )
    if payload.get("objective_refill_enabled") is not True:
        return None
    if not isinstance(refill_policy, dict):
        raise ConfiguredBoardError(
            "refill_policy must be an object when objective refill is enabled"
        )
    if not isinstance(derived, dict):
        raise ConfiguredBoardError(
            "refill_policy.derived_refill must be an object when objective "
            "refill is enabled"
        )
    min_open_tasks = _nonnegative_int(
        derived.get("min_open_tasks"),
        field="refill_policy.derived_refill.min_open_tasks",
    )
    max_findings = _positive_int(
        derived.get("max_tasks_per_epoch"),
        field="refill_policy.derived_refill.max_tasks_per_epoch",
    )
    max_open_tasks = _positive_int(
        derived.get("max_open_tasks"),
        field="refill_policy.derived_refill.max_open_tasks",
    )
    cooldown_seconds = _nonnegative_int(
        derived.get("cooldown_seconds"),
        field="refill_policy.derived_refill.cooldown_seconds",
    )
    if min_open_tasks >= max_open_tasks:
        raise ConfiguredBoardError(
            "refill_policy.derived_refill.min_open_tasks must be below "
            "max_open_tasks"
        )
    return (
        min_open_tasks,
        max_findings,
        cooldown_seconds,
        max_epochs,
        max_total_tasks,
    )


def _required_string(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfiguredBoardError(f"{field} must be a nonempty string")
    return value.strip()


def _provider_string(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = _required_string(payload, field)
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ConfiguredBoardError(
            f"{field} must be a single-line nonempty string"
        )
    return value


def _optional_provider_string(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = payload.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfiguredBoardError(f"{field} must be a string")
    normalized = value.strip()
    if "\x00" in normalized or "\n" in normalized or "\r" in normalized:
        raise ConfiguredBoardError(f"{field} must be a single-line string")
    return normalized


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ConfiguredBoardError(f"{field} must be a relative path")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or path.is_absolute()
        or path.as_posix() in {".", ".."}
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise ConfiguredBoardError(
            f"{field} contains unsafe relative path {value!r}"
        )
    return path.as_posix()


def _safe_relative_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfiguredBoardError(f"{field} must be a list")
    paths = tuple(
        _safe_relative(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(paths) != len(set(paths)):
        raise ConfiguredBoardError(f"{field} contains duplicate paths")
    return paths


def _contained_path(repo_root: Path, relative: str) -> Path:
    candidate = repo_root / relative
    try:
        candidate.resolve(strict=False).relative_to(repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfiguredBoardError(
            f"path escapes repository: {relative}"
        ) from exc
    return candidate


def _task_header_prefix(task_prefix: str) -> str:
    stripped = task_prefix.strip()
    return stripped if stripped.startswith("## ") else f"## {stripped}"


def _slug(value: str) -> str:
    return (
        re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-")
        or "configured-board"
    )


@dataclass(frozen=True)
class ConfiguredBoard:
    """Validated scheduler JSON and its exact checkout binding."""

    config_path: Path
    repo_root: Path
    payload: Mapping[str, Any]
    configuration_root: str
    configuration_revision: str
    taskboard_path: str
    objectives_path: str
    plan_path: str
    validator_path: str
    task_prefix: str
    board_namespace: str
    merge_target_branch: str
    max_lanes: int
    strict_task_sharding: bool
    idle_lane_work_stealing: str
    worktree_submodule_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    runtime_paths: Mapping[str, str]
    database_program: DatabaseProgramConfig | None = None

    @property
    def task_header_prefix(self) -> str:
        return _task_header_prefix(self.task_prefix)

    def path(self, relative: str) -> Path:
        return _contained_path(self.repo_root, relative)

    def resolved_database_program(self) -> DatabaseProgramConfig:
        """Return the explicit database/task-source selection for this board.

        Implicit legacy-Markdown defaults are deprecated. When no
        ``database_program`` section is present the board still launches, but
        only after constructing an *explicit* legacy selection from
        ``source_binding.bootstrap_task_source`` or a labeled explicit-legacy
        fallback.
        """

        if self.database_program is not None:
            return self.database_program
        source_binding = self.payload.get("source_binding")
        bootstrap = ""
        if isinstance(source_binding, Mapping):
            bootstrap = str(
                source_binding.get("bootstrap_task_source") or ""
            ).strip().lower()
        if bootstrap in {"", "legacy-markdown", "legacy_markdown", "markdown-legacy"}:
            return DatabaseProgramConfig.explicit_legacy_markdown()
        if bootstrap in {"markdown"}:
            return DatabaseProgramConfig(
                authority_mode=AUTHORITY_MODE_LEGACY_MARKDOWN,
                task_source_kind="markdown",
                explicit_legacy=True,
            )
        if bootstrap in {"duckdb", "quack"}:
            raise ConfiguredBoardError(
                "bootstrap_task_source requires a full database_program "
                f"section when set to {bootstrap!r}"
            )
        raise ConfiguredBoardError(
            f"unsupported bootstrap_task_source: {bootstrap!r}"
        )


def load_configured_board(
    config_path: Path | str,
    *,
    repo_root: Path | str,
    config_bytes: bytes | None = None,
) -> ConfiguredBoard:
    """Load and structurally validate one sealed scheduler document.

    ``config_bytes`` is reserved for a caller that has already authenticated
    the document from an immutable capsule.  The lexical repository path is
    still retained as the board's effect-scope identity, but it is not reopened
    when exact bytes are supplied.
    """

    root = Path(repo_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    try:
        path.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfiguredBoardError(
            "scheduler config must be inside the repository"
        ) from exc
    try:
        admitted_config_bytes = config_bytes
        if admitted_config_bytes is None:
            admitted_config_bytes, _config_evidence = _read_stable_regular_bytes(
                path,
                max_bytes=4_194_304,
            )
        elif (
            type(admitted_config_bytes) is not bytes
            or not admitted_config_bytes
            or len(admitted_config_bytes) > 4_194_304
        ):
            raise ConfiguredBoardError(
                "sealed scheduler config bytes are invalid"
            )
        if admitted_config_bytes is None:
            raise ConfiguredBoardError("scheduler config is absent")
        configuration_revision = _identity(
            {
                "path": path.resolve(strict=False).relative_to(root).as_posix(),
                "bytes_sha256": hashlib.sha256(admitted_config_bytes).hexdigest(),
            }
        )
        payload = json.loads(
            admitted_config_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ConfiguredBoardError:
        raise
    except _StableArtifactReadError as exc:
        raise ConfiguredBoardError(str(exc)) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfiguredBoardError(
            f"scheduler config is unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfiguredBoardError("scheduler config root must be an object")
    schema = _required_string(payload, "schema")
    if SCHEDULER_SCHEMA_PATTERN.fullmatch(schema) is None:
        raise ConfiguredBoardError(
            f"unsupported scheduler schema: {schema!r}"
        )

    taskboard_path = _safe_relative(
        _required_string(payload, "taskboard_path"),
        field="taskboard_path",
    )
    objectives_path = _safe_relative(
        _required_string(payload, "objectives_path"),
        field="objectives_path",
    )
    plan_path = _safe_relative(
        _required_string(payload, "plan_path"),
        field="plan_path",
    )
    validator_path = _safe_relative(
        _required_string(payload, "validator_path"),
        field="validator_path",
    )
    task_prefix = _required_string(payload, "task_prefix")
    if re.fullmatch(r"(?:## )?[A-Z][A-Z0-9_-]*-", task_prefix) is None:
        raise ConfiguredBoardError("task_prefix is not a supported task prefix")
    board_namespace = _required_string(payload, "board_namespace")
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", board_namespace) is None:
        raise ConfiguredBoardError("board_namespace is unsafe")
    config_relative = path.relative_to(root).as_posix()
    eaaef_markers = {
        "schema": schema == EAAEF_SCHEDULER_SCHEMA,
        "config_path": config_relative == EAAEF_CONFIG_PATH,
        "board_namespace": board_namespace == EAAEF_BOARD_NAMESPACE,
        "task_prefix": _task_header_prefix(task_prefix) == "## EAAEF-",
        "taskboard_path": taskboard_path == EAAEF_TASKBOARD_PATH,
        "taskboard_json_path": (
            payload.get("taskboard_json_path") == EAAEF_TASKBOARD_JSON_PATH
        ),
        "objectives_path": objectives_path == EAAEF_OBJECTIVES_PATH,
        "plan_path": plan_path == EAAEF_PLAN_PATH,
        "validator_path": validator_path == EAAEF_VALIDATOR_PATH,
    }
    if any(eaaef_markers.values()) and not all(eaaef_markers.values()):
        mismatched = sorted(
            name for name, matches in eaaef_markers.items() if not matches
        )
        raise ConfiguredBoardError(
            "EAAEF scheduler identity markers cannot be downgraded: "
            + ", ".join(mismatched)
        )
    if all(eaaef_markers.values()):
        payload = _apply_eaaef_generation_cursor(payload, root)
    merge_target_branch = _required_string(payload, "merge_target_branch")
    if (
        merge_target_branch.startswith("-")
        or "\x00" in merge_target_branch
        or any(character.isspace() for character in merge_target_branch)
    ):
        raise ConfiguredBoardError("merge_target_branch is unsafe")

    max_lanes = _positive_int(payload.get("max_lanes"), field="max_lanes")
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != max_lanes:
        raise ConfiguredBoardError(
            "lanes must contain exactly max_lanes entries"
        )
    expected_indices = list(range(max_lanes))
    actual_indices: list[int] = []
    for position, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            raise ConfiguredBoardError(f"lanes[{position}] must be an object")
        raw_index = lane.get("index")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
        ):
            raise ConfiguredBoardError(
                f"lanes[{position}].index must be a nonnegative integer"
            )
        index = raw_index
        if lane.get("strict_shard_remainder") != index:
            raise ConfiguredBoardError(
                f"lanes[{position}] strict shard remainder mismatch"
            )
        name = _required_string(lane, "name")
        if _slug(name) != name:
            raise ConfiguredBoardError(f"lanes[{position}].name is unsafe")
        actual_indices.append(index)
    if actual_indices != expected_indices:
        raise ConfiguredBoardError("lane indices must be contiguous and ordered")

    strict_task_sharding = payload.get("strict_task_sharding")
    if not isinstance(strict_task_sharding, bool):
        raise ConfiguredBoardError("strict_task_sharding must be boolean")
    idle_lane_work_stealing = str(
        payload.get("idle_lane_work_stealing") or ""
    ).strip().lower()
    if idle_lane_work_stealing not in {"", "virgin-transfer"}:
        raise ConfiguredBoardError(
            "idle_lane_work_stealing must be empty or 'virgin-transfer'"
        )
    if idle_lane_work_stealing and not strict_task_sharding:
        raise ConfiguredBoardError(
            "idle_lane_work_stealing requires strict_task_sharding"
        )
    if idle_lane_work_stealing and max_lanes <= 1:
        raise ConfiguredBoardError(
            "idle_lane_work_stealing requires at least two lanes"
        )
    submodules = _safe_relative_list(
        payload.get("worktree_submodule_paths"),
        field="worktree_submodule_paths",
    )
    protected = _safe_relative_list(
        payload.get("protected_paths"),
        field="protected_paths",
    )
    if config_relative not in protected:
        raise ConfiguredBoardError(
            "scheduler config must protect its own source path"
        )

    runtime_raw = payload.get("runtime_paths")
    if not isinstance(runtime_raw, dict):
        raise ConfiguredBoardError("runtime_paths must be an object")
    runtime_paths = {
        field: _safe_relative(
            _required_string(runtime_raw, field),
            field=f"runtime_paths.{field}",
        )
        for field in (
            "root",
            "state",
            "worktrees",
            "merge_queue",
            "logs",
        )
    }
    runtime_root_parts = PurePosixPath(runtime_paths["root"]).parts
    for field, relative in runtime_paths.items():
        if field == "root":
            continue
        if PurePosixPath(relative).parts[: len(runtime_root_parts)] != (
            runtime_root_parts
        ):
            raise ConfiguredBoardError(
                f"runtime_paths.{field} must be under runtime_paths.root"
            )

    provider = payload.get("provider")
    if not isinstance(provider, dict):
        raise ConfiguredBoardError("provider must be an object")
    ordered_provider = any(
        field in provider for field in ORDERED_PROVIDER_DETECTION_FIELDS
    )
    if ordered_provider:
        primary_provider_id = _provider_string(
            provider,
            "primary_provider_id",
        )
        primary_model_id = _provider_string(provider, "primary_model_id")
        fallback_provider_id = _provider_string(
            provider,
            "fallback_provider_id",
        )
        fallback_model_id = _provider_string(provider, "fallback_model_id")
        fallback_trigger = _provider_string(
            provider,
            "fallback_trigger",
        )
        fallback_reasoning_effort = _provider_string(
            provider,
            "fallback_reasoning_effort",
        )
        if primary_provider_id != ORDERED_PRIMARY_PROVIDER_ID:
            raise ConfiguredBoardError(
                "provider.primary_provider_id must be 'grok_cli' for "
                "the ordered provider contract"
            )
        admitted_primary_models = _ordered_primary_models_for_namespace(
            board_namespace
        )
        if primary_model_id not in admitted_primary_models:
            expected_models = ", ".join(sorted(admitted_primary_models))
            raise ConfiguredBoardError(
                "provider.primary_model_id must be one of "
                f"{expected_models!r} for the scoped ordered provider contract"
            )
        if fallback_provider_id != ORDERED_FALLBACK_PROVIDER_ID:
            raise ConfiguredBoardError(
                "provider.fallback_provider_id must be 'codex' for "
                "the ordered provider contract"
            )
        if fallback_model_id != ORDERED_FALLBACK_MODEL_ID:
            raise ConfiguredBoardError(
                "provider.fallback_model_id must be 'gpt-5.6-terra' for "
                "the ordered provider contract"
            )
        if fallback_trigger not in ORDERED_FALLBACK_TRIGGERS:
            raise ConfiguredBoardError(
                "provider.fallback_trigger must be "
                "'primary_quota_exhausted' or "
                "'primary_quota_or_auth_unavailable' for the ordered "
                "provider contract"
            )
        if fallback_reasoning_effort not in ORDERED_FALLBACK_REASONING_EFFORTS:
            raise ConfiguredBoardError(
                "provider.fallback_reasoning_effort must be one of "
                "'medium', 'high' for "
                "the ordered provider contract"
            )
        if "provider_id" in provider or "model_id" in provider:
            raise ConfiguredBoardError(
                "ordered provider fields cannot be mixed with legacy "
                "provider_id/model_id"
            )
        launch_policy = payload.get("launch_policy")
        eaaef_route_is_post_freeze_and_live_blocked = (
            board_namespace == EAAEF_BOARD_NAMESPACE
            and schema == EAAEF_SCHEDULER_SCHEMA
            and isinstance(launch_policy, Mapping)
            and launch_policy.get("live_multi_supervisor_allowed") is False
            and launch_policy.get("live_single_supervisor_allowed") is False
        )
        if not eaaef_route_is_post_freeze_and_live_blocked:
            _resolved_ordered_provider_route(
                provider,
                repo_root=root,
                board_namespace=board_namespace,
            )
        primary_executable = _optional_provider_string(
            provider,
            ORDERED_PRIMARY_EXECUTABLE_FIELD,
        )
        if primary_executable:
            executable_path = Path(primary_executable)
            if (
                not executable_path.is_absolute()
                or os.path.abspath(primary_executable) != primary_executable
            ):
                raise ConfiguredBoardError(
                    "provider.primary_executable must be a normalized "
                    "absolute path"
                )
            if not executable_path.is_file() or not os.access(
                executable_path,
                os.X_OK,
            ):
                raise ConfiguredBoardError(
                    "provider.primary_executable must name an executable file"
                )
    else:
        provider_id = _optional_provider_string(
            provider,
            "provider_id",
        ).lower()
        _optional_provider_string(provider, "model_id")
        if provider_id and re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*",
            provider_id,
        ) is None:
            raise ConfiguredBoardError(
                "provider.provider_id is not a supported identifier"
            )
    external_isolation = provider.get("external_isolation")
    if external_isolation is not None:
        if ordered_provider:
            raise ConfiguredBoardError(
                "provider.external_isolation is supported only for a direct "
                "Codex route"
            )
        if provider_id not in {"codex", "auto"}:
            raise ConfiguredBoardError(
                "provider.external_isolation requires provider_id 'codex' or 'auto'"
            )
        if not isinstance(external_isolation, dict):
            raise ConfiguredBoardError(
                "provider.external_isolation must be an object"
            )
        try:
            from ..todo_daemon.implementation_daemon import (
                validate_external_provider_isolation_config,
            )

            # Loading is a deterministic structural operation and must remain
            # usable in the credential-free validation container.  Host
            # runtime/image/credential admission is repeated fail-closed by
            # ``configured_board_launch_plan`` immediately before launch.
            validate_external_provider_isolation_config(
                external_isolation,
                verify_host=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ConfiguredBoardError(
                f"provider.external_isolation is unavailable: {exc}"
            ) from exc
    concurrency = _positive_int(
        provider.get("max_concurrency"),
        field="provider.max_concurrency",
    )
    if concurrency < max_lanes:
        raise ConfiguredBoardError(
            "provider.max_concurrency is lower than max_lanes"
        )
    host_capacity_cfg = payload.get("host_capacity")
    if host_capacity_cfg is not None:
        if not isinstance(host_capacity_cfg, dict):
            raise ConfiguredBoardError("host_capacity must be an object")
        host_cpu_slots = _positive_int(
            host_capacity_cfg.get("cpu_slots"),
            field="host_capacity.cpu_slots",
        )
        host_process_slots = _positive_int(
            host_capacity_cfg.get("process_slots"),
            field="host_capacity.process_slots",
        )
        if host_cpu_slots < max_lanes or host_process_slots < max_lanes:
            raise ConfiguredBoardError(
                "host_capacity slots must be at least max_lanes"
            )
    for field in (
        "strict_task_sharding",
        "exit_when_all_tracks_terminal",
        "objective_refill_enabled",
        "codebase_refill_enabled",
    ):
        if not isinstance(payload.get(field), bool):
            raise ConfiguredBoardError(f"{field} must be boolean")
    if (
        "objective_goal_refinement_enabled" in payload
        and not isinstance(payload.get("objective_goal_refinement_enabled"), bool)
    ):
        raise ConfiguredBoardError(
            "objective_goal_refinement_enabled must be boolean"
        )
    for field in (
        "retry_budget_guardrail_enabled",
        "dependency_guardrail_enabled",
        "reconciliation_guardrail_enabled",
    ):
        if field in payload and not isinstance(payload.get(field), bool):
            raise ConfiguredBoardError(f"{field} must be boolean")

    for field in (
        "poll_interval_seconds",
        "daemon_interval_seconds",
        "check_interval_seconds",
        "stale_seconds",
        "watchdog_startup_grace_seconds",
        "implementation_timeout_seconds",
        "implementation_max_timeout_seconds",
        "implementation_log_stall_seconds",
    ):
        _nonnegative_number(payload.get(field), field=field)
    for field in (
        "max_restarts",
        "max_task_attempts",
        "implementation_retry_budget",
        "validation_retry_budget",
        "merge_retry_budget",
    ):
        _positive_int(payload.get(field), field=field)

    database_program: DatabaseProgramConfig | None = None
    if "database_program" in payload:
        raw_program = payload.get("database_program")
        if not isinstance(raw_program, dict):
            raise ConfiguredBoardError("database_program must be an object")
        try:
            program_payload = dict(raw_program)
            if not program_payload.get("worktree_root"):
                program_payload["worktree_root"] = runtime_paths["worktrees"]
            database_program = parse_database_program_config(program_payload)
        except DatabaseProgramConfigError as exc:
            raise ConfiguredBoardError(str(exc)) from exc
        claim_policy = dict(database_program.claim_policy or {})
        normalized_prefix = re.sub(
            r"^\s*#{1,6}\s*",
            "",
            task_prefix,
        ).strip()
        expected_claim_policy = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-claim-policy@1"
            ),
            "task_prefix": normalized_prefix,
            "task_shard_count": max_lanes,
            "strict_task_sharding": strict_task_sharding,
            "idle_lane_work_stealing": idle_lane_work_stealing,
        }
        if idle_lane_work_stealing and (
            database_program.authority_mode == "quack"
            and claim_policy != expected_claim_policy
        ):
            raise ConfiguredBoardError(
                "database claim_policy differs from the configured board"
            )

    _validate_eaaef_database_programs(
        board_namespace=board_namespace,
        payload=payload,
        operational_program=database_program,
    )

    _objective_refill_controls(payload)

    return ConfiguredBoard(
        config_path=path,
        repo_root=root,
        payload=payload,
        configuration_root=_identity(
            {
                "bytes_sha256": hashlib.sha256(
                    admitted_config_bytes
                ).hexdigest()
            }
        ),
        configuration_revision=configuration_revision,
        taskboard_path=taskboard_path,
        objectives_path=objectives_path,
        plan_path=plan_path,
        validator_path=validator_path,
        task_prefix=task_prefix,
        board_namespace=board_namespace,
        merge_target_branch=merge_target_branch,
        max_lanes=max_lanes,
        strict_task_sharding=strict_task_sharding,
        idle_lane_work_stealing=idle_lane_work_stealing,
        worktree_submodule_paths=submodules,
        protected_paths=protected,
        runtime_paths=runtime_paths,
        database_program=database_program,
    )


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    command = list(argv)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            # Preflight helpers never receive live state credentials, loader
            # authority, caller Python paths, or an ambient Git configuration.
            # This matters when the configured-board scheduler itself holds
            # the in-memory Quack attach token.
            env=_sanitized_git_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            "",
            f"{type(exc).__name__}: {exc}",
        )


def _fresh_recovery_private_primary_gid() -> int:
    """Prove that the current primary group has no second filesystem writer."""

    try:
        effective_uid = os.geteuid()
        effective_gid = os.getegid()
        account = pwd.getpwuid(effective_uid)
        group = grp.getgrgid(effective_gid)
        accounts = pwd.getpwall()
    except (KeyError, OSError) as exc:
        raise ConfiguredBoardError(
            "private primary-group evidence is unavailable"
        ) from exc
    if len(accounts) > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_ENTRIES:
        raise ConfiguredBoardError(
            "private primary-group account inventory exceeds its bound"
        )
    matching_accounts = [item for item in accounts if item.pw_gid == effective_gid]
    if (
        account.pw_uid != effective_uid
        or account.pw_gid != effective_gid
        or group.gr_gid != effective_gid
        or group.gr_mem
        or len(matching_accounts) != 1
        or matching_accounts[0].pw_uid != effective_uid
        or matching_accounts[0].pw_name != account.pw_name
    ):
        raise ConfiguredBoardError(
            "current primary group is not provably private"
        )
    return effective_gid


def _reject_fresh_recovery_git_object_substitution(
    repo_root: Path,
    *,
    label: str,
) -> None:
    """Reject repository-local commit grafts and replacement-object refs.

    A clean worktree and raw ``HEAD`` do not bind the effective tree when Git
    is permitted to honor ``info/grafts`` or ``refs/replace``.  All trusted Git
    subprocesses disable replacement objects as defense in depth; this explicit
    observation makes either mechanism a typed admission failure rather than
    silently ignoring untrusted repository metadata.
    """

    common_dir_result = _git_run(
        ("rev-parse", "--path-format=absolute", "--git-common-dir"),
        cwd=repo_root,
        timeout=60.0,
    )
    common_dir_stdout = common_dir_result.stdout or ""
    common_dir_stderr = common_dir_result.stderr or ""
    if (
        common_dir_result.returncode != 0
        or common_dir_stderr
        or len(common_dir_stdout.encode("utf-8", errors="replace")) > 4_096
        or common_dir_stdout.count("\n") > 1
    ):
        raise ConfiguredBoardError(f"{label} Git common directory is unavailable")
    raw_common_dir = common_dir_stdout.strip()
    if not raw_common_dir:
        raise ConfiguredBoardError(f"{label} Git common directory is unavailable")
    common_dir = _canonical_no_symlink_root(Path(raw_common_dir))

    replacement_refs = _git_run(
        ("for-each-ref", "--format=%(refname)", "refs/replace/"),
        cwd=repo_root,
        timeout=60.0,
    )
    refs_stdout = replacement_refs.stdout or ""
    refs_stderr = replacement_refs.stderr or ""
    if (
        replacement_refs.returncode != 0
        or refs_stderr
        or len(refs_stdout.encode("utf-8", errors="replace")) > 4_096
    ):
        raise ConfiguredBoardError(
            f"{label} Git replacement-ref inventory is unavailable"
        )
    if refs_stdout:
        raise ConfiguredBoardError(
            f"{label} Git object substitution metadata is present"
        )

    filter_config = _git_run(
        ("config", "--name-only", "--get-regexp", r"^filter\."),
        cwd=repo_root,
        timeout=60.0,
    )
    filter_stdout = filter_config.stdout or ""
    filter_stderr = filter_config.stderr or ""
    if (
        filter_config.returncode not in {0, 1}
        or filter_stderr
        or len(filter_stdout.encode("utf-8", errors="replace")) > 4_096
    ):
        raise ConfiguredBoardError(
            f"{label} Git filter configuration is unavailable"
        )
    if filter_config.returncode == 0 or filter_stdout:
        raise ConfiguredBoardError(
            f"{label} Git filter execution metadata is present"
        )

    for relative in (
        Path("info/grafts"),
        Path("refs/replace"),
        Path("info/attributes"),
    ):
        candidate = common_dir / relative
        try:
            os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ConfiguredBoardError(
                f"{label} Git object substitution metadata is unavailable"
            ) from exc
        noun = (
            "filter execution"
            if relative == Path("info/attributes")
            else "object substitution"
        )
        raise ConfiguredBoardError(f"{label} Git {noun} metadata is present")


def _fresh_recovery_clean_source_identity(
    board: ConfiguredBoard,
) -> tuple[
    str,
    str,
    str,
    tuple[tuple[str, str, str, str, str], ...],
    dict[str, Any],
]:
    """Return one clean, stable outer/nested Git forest identity.

    Recovery verification imports repository Python modules.  This check must
    therefore run before the verifier process exists, rather than relying on
    the ordinary preflight cleanliness check that follows verifier admission.
    The caller repeats it after verification and compares the whole tuple so a
    verifier cannot authorize a different outer tree, gitlink, or nested tree.
    """

    def clean_status(repo_root: Path, *, label: str) -> None:
        status = _git_run(
            (
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
                "--ignore-submodules=none",
            ),
            cwd=repo_root,
            timeout=60.0,
        )
        stdout = status.stdout or ""
        stderr = status.stderr or ""
        if (
            len(stdout.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_GIT_STATUS_MAX_OUTPUT_BYTES
            or len(stderr.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_GIT_STATUS_MAX_OUTPUT_BYTES
        ):
            raise ConfiguredBoardError(f"{label} Git status exceeds its bound")
        if status.returncode != 0 or stderr:
            raise ConfiguredBoardError(f"{label} Git status is unavailable")
        if stdout:
            raise ConfiguredBoardError(f"{label} checkout is not clean")

    def require_ordinary_index(repo_root: Path, *, label: str) -> None:
        index = _git_run(
            ("ls-files", "-v", "-z"),
            cwd=repo_root,
            timeout=60.0,
        )
        stdout = index.stdout or ""
        stderr = index.stderr or ""
        if (
            index.returncode != 0
            or len(stdout.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
            or len(stderr.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
            or stderr
        ):
            raise ConfiguredBoardError(f"{label} Git index is unavailable")
        records = [record for record in stdout.split("\0") if record]
        if len(records) > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_ENTRIES:
            raise ConfiguredBoardError(f"{label} Git index exceeds its bound")
        if any(
            len(record) < 3
            or record[1] != " "
            or record[0] != "H"
            or not record[2:]
            for record in records
        ):
            raise ConfiguredBoardError(
                f"{label} Git index contains an exceptional tracked entry"
            )

    def import_inventory(
        repo_root: Path,
        *,
        private_gid: int,
        root_relatives: tuple[str, ...],
        omission_scope: str,
        omission_path_prefix: str = "",
        include_repo_root_candidates: bool = False,
    ) -> tuple[str, tuple[dict[str, str], ...]]:
        tracked_command = (
            ("ls-files", "-v", "-z")
            if include_repo_root_candidates
            else (
                "ls-files",
                "-v",
                "-z",
                "--",
                *root_relatives,
            )
        )
        tracked_result = _git_run(
            tracked_command,
            cwd=repo_root,
            timeout=60.0,
        )
        stage_command = (
            ("ls-files", "--stage", "-z")
            if include_repo_root_candidates
            else (
                "ls-files",
                "--stage",
                "-z",
                "--",
                *root_relatives,
            )
        )
        stage_result = _git_run(
            stage_command,
            cwd=repo_root,
            timeout=60.0,
        )
        head_command = (
            ("ls-tree", "-r", "-z", "HEAD")
            if include_repo_root_candidates
            else (
                "ls-tree",
                "-r",
                "-z",
                "HEAD",
                "--",
                *root_relatives,
            )
        )
        head_result = _git_run(
            head_command,
            cwd=repo_root,
            timeout=60.0,
        )
        tracked_output = tracked_result.stdout or ""
        tracked_stderr = tracked_result.stderr or ""
        stage_output = stage_result.stdout or ""
        stage_stderr = stage_result.stderr or ""
        head_output = head_result.stdout or ""
        head_stderr = head_result.stderr or ""
        if (
            tracked_result.returncode != 0
            or stage_result.returncode != 0
            or head_result.returncode != 0
            or len(tracked_output.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
            or len(tracked_stderr.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
            or len(stage_output.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
            or len(stage_stderr.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
            or len(head_output.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
            or len(head_stderr.encode("utf-8", errors="replace"))
            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
            or tracked_stderr
            or stage_stderr
            or head_stderr
        ):
            raise ConfiguredBoardError(
                "recovery import tracked-file inventory is unavailable"
            )
        tracked: set[str] = set()
        for record in tracked_output.split("\0"):
            if not record:
                continue
            if len(record) < 3 or record[1] != " ":
                raise ConfiguredBoardError(
                    "recovery import tracked-file inventory is malformed"
                )
            tag, relative = record[0], record[2:]
            if tag != "H" or not relative:
                raise ConfiguredBoardError(
                    "recovery import source has an exceptional index state"
                )
            tracked.add(relative)

        index_entries: dict[str, tuple[str, str]] = {}
        for record in stage_output.split("\0"):
            if not record:
                continue
            try:
                metadata, relative = record.split("\t", 1)
                mode, object_id, stage = metadata.split(" ")
            except ValueError as exc:
                raise ConfiguredBoardError(
                    "recovery import staged-file inventory is malformed"
                ) from exc
            if (
                not relative
                or relative in index_entries
                or stage != "0"
                or mode not in {"100644", "100755", "120000", "160000"}
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id)
                is None
            ):
                raise ConfiguredBoardError(
                    "recovery import staged-file identity differs"
                )
            index_entries[relative] = (mode, object_id)
        if set(index_entries) != tracked:
            raise ConfiguredBoardError(
                "recovery import tracked and staged inventories disagree"
            )

        head_entries: dict[str, tuple[str, str]] = {}
        for record in head_output.split("\0"):
            if not record:
                continue
            try:
                metadata, relative = record.split("\t", 1)
                mode, object_kind, object_id = metadata.split(" ")
            except ValueError as exc:
                raise ConfiguredBoardError(
                    "recovery import HEAD inventory is malformed"
                ) from exc
            expected_kind = "commit" if mode == "160000" else "blob"
            if (
                not relative
                or relative in head_entries
                or mode not in {"100644", "100755", "120000", "160000"}
                or object_kind != expected_kind
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id)
                is None
            ):
                raise ConfiguredBoardError(
                    "recovery import HEAD identity differs"
                )
            head_entries[relative] = (mode, object_id)
        if head_entries != index_entries:
            raise ConfiguredBoardError(
                "recovery import index and HEAD inventories disagree"
            )

        selected: list[dict[str, Any]] = []
        directories: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        omissions: list[dict[str, str]] = []
        entry_count = 0
        path_bytes = 0
        content_bytes = 0

        def filesystem_identity(
            relative: str,
            observed: os.stat_result,
        ) -> dict[str, Any]:
            return {
                "path": relative,
                "uid": int(observed.st_uid),
                "gid": int(observed.st_gid),
                "mode": stat.S_IMODE(observed.st_mode),
                "nlink": int(observed.st_nlink),
                "device": int(observed.st_dev),
                "inode": int(observed.st_ino),
                "size": int(observed.st_size),
                "mtime_ns": int(observed.st_mtime_ns),
            }

        def bind_regular_import_source(
            relative: str,
            path: Path,
            observed: os.stat_result,
        ) -> dict[str, Any]:
            """Bind raw stable bytes to the exact index and HEAD blob."""

            nonlocal content_bytes
            index_entry = index_entries.get(relative)
            if index_entry is None or index_entry[0] not in {"100644", "100755"}:
                raise ConfiguredBoardError(
                    f"recovery import source has no regular Git entry: {relative}"
                )
            expected_mode, expected_oid = index_entry
            filesystem_mode = "100755" if observed.st_mode & 0o111 else "100644"
            if filesystem_mode != expected_mode:
                raise ConfiguredBoardError(
                    f"recovery import source mode differs from Git: {relative}"
                )
            try:
                payload, stable_evidence = _read_stable_regular_bytes(
                    path,
                    max_bytes=FRESH_RECOVERY_IMPORT_FILE_MAX_BYTES,
                )
            except _StableArtifactReadError as exc:
                raise ConfiguredBoardError(
                    f"recovery import source cannot be read stably: {relative}"
                ) from exc
            if payload is None:
                raise ConfiguredBoardError(
                    f"recovery import source disappeared: {relative}"
                )
            observed_identity = (
                int(observed.st_dev),
                int(observed.st_ino),
                stat.S_IMODE(observed.st_mode),
                int(observed.st_nlink),
                int(observed.st_uid),
                int(observed.st_gid),
                int(observed.st_size),
                int(observed.st_mtime_ns),
            )
            stable_identity = (
                int(stable_evidence["device"]),
                int(stable_evidence["inode"]),
                stat.S_IMODE(int(stable_evidence["mode"])),
                int(stable_evidence["link_count"]),
                int(stable_evidence["uid"]),
                int(stable_evidence["gid"]),
                int(stable_evidence["size"]),
                int(stable_evidence["mtime_ns"]),
            )
            if stable_identity != observed_identity:
                raise ConfiguredBoardError(
                    f"recovery import source changed before hashing: {relative}"
                )
            content_bytes += len(payload)
            if content_bytes > FRESH_RECOVERY_IMPORT_CONTENT_MAX_BYTES:
                raise ConfiguredBoardError(
                    "recovery import source content exceeds its aggregate bound"
                )
            hash_constructor = (
                hashlib.sha1 if len(expected_oid) == 40 else hashlib.sha256
            )
            git_blob = f"blob {len(payload)}\0".encode("ascii") + payload
            observed_oid = hash_constructor(git_blob).hexdigest()
            if observed_oid != expected_oid:
                raise ConfiguredBoardError(
                    f"recovery import source raw Git blob differs: {relative}"
                )
            result = filesystem_identity(relative, observed)
            result.update(
                {
                    "git_mode": expected_mode,
                    "git_blob_oid": expected_oid,
                    "content_sha256": stable_evidence["content_sha256"],
                }
            )
            return result

        def require_safe_directory(
            relative: str,
            observed: os.stat_result,
        ) -> None:
            mode = stat.S_IMODE(observed.st_mode)
            if (
                observed.st_uid != os.geteuid()
                or observed.st_nlink < 1
                or mode & 0o002
                or (mode & 0o020 and observed.st_gid != private_gid)
            ):
                raise ConfiguredBoardError(
                    f"recovery import directory identity differs: {relative}"
                )

        if include_repo_root_candidates:
            try:
                root_status = os.lstat(repo_root)
                with os.scandir(repo_root) as iterator:
                    root_entries = sorted(iterator, key=lambda item: item.name)
            except OSError as exc:
                raise ConfiguredBoardError(
                    "recovery repository-root import inventory is unavailable"
                ) from exc
            require_safe_directory(".", root_status)
            directories.append(filesystem_identity(".", root_status))
            for entry in root_entries:
                entry_count += 1
                if entry_count > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_ENTRIES:
                    raise ConfiguredBoardError(
                        "recovery import inventory exceeds its entry bound"
                    )
                relative = entry.name
                path_bytes += len(relative.encode("utf-8"))
                if path_bytes > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES:
                    raise ConfiguredBoardError(
                        "recovery import inventory exceeds its path bound"
                    )
                path = Path(entry.path)
                suffix = path.suffix.lower()
                if suffix not in {
                    ".py",
                    ".pyc",
                    ".pyo",
                    ".so",
                    ".pyd",
                    ".dylib",
                }:
                    continue
                try:
                    observed = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ConfiguredBoardError(
                        "recovery root import identity is unavailable"
                    ) from exc
                if stat.S_ISLNK(observed.st_mode):
                    raise ConfiguredBoardError(
                        "recovery root import inventory contains an unsafe "
                        f"link: {relative}"
                    )
                if not stat.S_ISREG(observed.st_mode):
                    raise ConfiguredBoardError(
                        "recovery root import inventory contains a special file"
                    )
                if (
                    observed.st_uid != os.geteuid()
                    or observed.st_nlink != 1
                ):
                    raise ConfiguredBoardError(
                        "recovery root import source file identity differs"
                    )
                mode = stat.S_IMODE(observed.st_mode)
                if mode & 0o002 or (
                    mode & 0o020 and observed.st_gid != private_gid
                ):
                    raise ConfiguredBoardError(
                        "recovery root import source has an unsafe writable mode"
                    )
                if suffix == ".py":
                    if relative not in tracked:
                        raise ConfiguredBoardError(
                            "recovery root import inventory contains untracked "
                            f"Python source: {relative}"
                        )
                    selected.append(
                        bind_regular_import_source(relative, path, observed)
                    )
                elif suffix in {".pyc", ".pyo"}:
                    raise ConfiguredBoardError(
                        "recovery root import inventory contains adjacent "
                        f"bytecode: {relative}"
                    )
                elif relative not in tracked:
                    raise ConfiguredBoardError(
                        "recovery root import inventory contains an untracked "
                        f"native extension: {relative}"
                    )
                else:
                    selected.append(
                        bind_regular_import_source(relative, path, observed)
                    )

        for root_relative in root_relatives:
            inventory_root, exact_relative = _lexical_repo_artifact(
                repo_root,
                repo_root / root_relative,
            )
            if exact_relative != root_relative:
                raise ConfiguredBoardError(
                    "recovery import inventory root differs"
                )
            try:
                root_status = os.lstat(inventory_root)
            except OSError as exc:
                raise ConfiguredBoardError(
                    "recovery import inventory root is unavailable"
                ) from exc
            if (
                stat.S_ISLNK(root_status.st_mode)
                or not stat.S_ISDIR(root_status.st_mode)
            ):
                raise ConfiguredBoardError(
                    "recovery import inventory root is not a real directory"
                )
            require_safe_directory(root_relative, root_status)
            directories.append(filesystem_identity(root_relative, root_status))
            pending = [inventory_root]
            while pending:
                directory = pending.pop()
                try:
                    with os.scandir(directory) as iterator:
                        entries = sorted(
                            iterator,
                            key=lambda item: item.name,
                        )
                except OSError as exc:
                    raise ConfiguredBoardError(
                        "recovery import inventory cannot be read"
                    ) from exc
                for entry in entries:
                    entry_count += 1
                    if entry_count > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_ENTRIES:
                        raise ConfiguredBoardError(
                            "recovery import inventory exceeds its entry bound"
                        )
                    path = Path(entry.path)
                    try:
                        relative = path.relative_to(repo_root).as_posix()
                        observed = entry.stat(follow_symlinks=False)
                    except (OSError, ValueError) as exc:
                        raise ConfiguredBoardError(
                            "recovery import inventory identity is unavailable"
                        ) from exc
                    path_bytes += len(relative.encode("utf-8"))
                    if (
                        path_bytes
                        > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
                    ):
                        raise ConfiguredBoardError(
                            "recovery import inventory exceeds its path bound"
                        )
                    if stat.S_ISLNK(observed.st_mode):
                        try:
                            target = os.readlink(path)
                            target_bytes = target.encode("utf-8")
                        except (OSError, UnicodeEncodeError) as exc:
                            raise ConfiguredBoardError(
                                "recovery import link target is unsafe"
                            ) from exc
                        path_bytes += len(target_bytes)
                        index_entry = index_entries.get(relative)
                        if (
                            not target
                            or relative not in tracked
                            or index_entry is None
                            or index_entry[0] != "120000"
                            or observed.st_uid != os.geteuid()
                            or observed.st_nlink != 1
                            or path_bytes
                            > FRESH_RECOVERY_IMPORT_INVENTORY_MAX_PATH_BYTES
                        ):
                            raise ConfiguredBoardError(
                                "recovery import link target is unsafe"
                            )
                        object_id = index_entry[1]
                        hash_constructor = (
                            hashlib.sha1 if len(object_id) == 40 else hashlib.sha256
                        )
                        git_blob = (
                            f"blob {len(target_bytes)}\0".encode("ascii")
                            + target_bytes
                        )
                        if hash_constructor(git_blob).hexdigest() != object_id:
                            raise ConfiguredBoardError(
                                "recovery import link differs from its Git blob"
                            )
                        link_identity = filesystem_identity(relative, observed)
                        link_identity["target"] = target
                        links.append(link_identity)
                        logical_path = (
                            PurePosixPath(omission_path_prefix)
                            / PurePosixPath(relative)
                        ).as_posix()
                        omissions.append(
                            {
                                "scope": omission_scope,
                                "path": logical_path,
                                "git_target": target,
                                "disposition": "omitted_source_symlink",
                            }
                        )
                        continue
                    if stat.S_ISDIR(observed.st_mode):
                        require_safe_directory(relative, observed)
                        directories.append(filesystem_identity(relative, observed))
                        pending.append(path)
                        continue
                    if not stat.S_ISREG(observed.st_mode):
                        raise ConfiguredBoardError(
                            "recovery import inventory contains a special file"
                        )
                    suffix = path.suffix.lower()
                    if (
                        suffix in {".py", ".so", ".pyd", ".dylib"}
                        and (
                            observed.st_uid != os.geteuid()
                            or observed.st_nlink != 1
                        )
                    ):
                        raise ConfiguredBoardError(
                            "recovery import source file identity differs"
                        )
                    mode = stat.S_IMODE(observed.st_mode)
                    if mode & 0o002 or (
                        mode & 0o020 and observed.st_gid != private_gid
                    ):
                        raise ConfiguredBoardError(
                            "recovery import source has an unsafe writable mode"
                        )
                    if suffix == ".py":
                        if relative not in tracked:
                            raise ConfiguredBoardError(
                                "recovery import inventory contains untracked "
                                f"Python source: {relative}"
                            )
                        selected.append(
                            bind_regular_import_source(relative, path, observed)
                        )
                    elif suffix in {".pyc", ".pyo"}:
                        if "__pycache__" not in PurePosixPath(relative).parts:
                            raise ConfiguredBoardError(
                                "recovery import inventory contains adjacent "
                                f"bytecode: {relative}"
                            )
                    elif suffix in {".so", ".pyd", ".dylib"}:
                        if relative not in tracked:
                            raise ConfiguredBoardError(
                                "recovery import inventory contains an untracked "
                                f"native extension: {relative}"
                            )
                        selected.append(
                            bind_regular_import_source(relative, path, observed)
                        )
        inventory_root = _identity(
            {
                "effective_uid": os.geteuid(),
                "effective_gid": private_gid,
                "directories": sorted(directories, key=lambda item: item["path"]),
                "tracked_source_links": sorted(
                    links, key=lambda item: item["path"]
                ),
                "tracked_import_files": sorted(
                    selected, key=lambda item: item["path"]
                ),
            }
        )
        return inventory_root, tuple(
            sorted(omissions, key=lambda item: (item["scope"], item["path"]))
        )

    root = _canonical_no_symlink_root(board.repo_root)
    _reject_fresh_recovery_git_object_substitution(root, label="accelerator")
    for relative in board.worktree_submodule_paths:
        preliminary_target, preliminary_relative = _lexical_repo_artifact(
            root,
            board.path(relative),
        )
        if preliminary_relative != relative:
            raise ConfiguredBoardError("configured submodule path differs")
        try:
            preliminary_status = os.lstat(preliminary_target)
        except OSError as exc:
            raise ConfiguredBoardError(
                f"configured submodule is unavailable: {relative}"
            ) from exc
        if stat.S_ISLNK(preliminary_status.st_mode) or not stat.S_ISDIR(
            preliminary_status.st_mode
        ):
            raise ConfiguredBoardError(
                f"configured submodule is not a real directory: {relative}"
            )
        _reject_fresh_recovery_git_object_substitution(
            _canonical_no_symlink_root(preliminary_target),
            label=f"configured submodule {relative}",
        )
    top = _git_run(("rev-parse", "--show-toplevel"), cwd=root)
    if top.returncode != 0 or Path(top.stdout.strip()) != root:
        raise ConfiguredBoardError("accelerator repository root differs")
    outer_head, outer_tree = _git_identity(root)
    if (
        re.fullmatch(r"[0-9a-f]{40,64}", outer_head) is None
        or re.fullmatch(r"[0-9a-f]{40,64}", outer_tree) is None
    ):
        raise ConfiguredBoardError("accelerator Git identity is malformed")
    private_gid = _fresh_recovery_private_primary_gid()
    require_ordinary_index(root, label="accelerator")
    clean_status(root, label="accelerator")
    import_inventory_root, outer_omissions = import_inventory(
        root,
        private_gid=private_gid,
        root_relatives=("scripts", "ipfs_accelerate_py", "test"),
        omission_scope="accelerator",
        include_repo_root_candidates=True,
    )

    nested_identities: list[tuple[str, str, str, str, str]] = []
    source_omissions = list(outer_omissions)
    for relative in board.worktree_submodule_paths:
        target, exact_relative = _lexical_repo_artifact(
            root,
            board.path(relative),
        )
        if exact_relative != relative:
            raise ConfiguredBoardError("configured submodule path differs")
        try:
            observed = os.lstat(target)
        except OSError as exc:
            raise ConfiguredBoardError(
                f"configured submodule is unavailable: {relative}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ConfiguredBoardError(
                f"configured submodule is not a real directory: {relative}"
            )
        nested_root = _canonical_no_symlink_root(target)
        _reject_fresh_recovery_git_object_substitution(
            nested_root,
            label=f"configured submodule {relative}",
        )
        nested_top = _git_run(
            ("rev-parse", "--show-toplevel"),
            cwd=nested_root,
        )
        if (
            nested_top.returncode != 0
            or Path(nested_top.stdout.strip()) != nested_root
        ):
            raise ConfiguredBoardError(
                f"configured submodule repository root differs: {relative}"
            )
        nested_head, nested_tree = _git_identity(nested_root)
        if (
            re.fullmatch(r"[0-9a-f]{40,64}", nested_head) is None
            or re.fullmatch(r"[0-9a-f]{40,64}", nested_tree) is None
        ):
            raise ConfiguredBoardError(
                f"configured submodule Git identity is malformed: {relative}"
            )
        gitlink = _git_run(
            ("ls-tree", outer_head, "--", relative),
            cwd=root,
        )
        expected_gitlink = f"160000 commit {nested_head}\t{relative}\n"
        if gitlink.returncode != 0 or gitlink.stdout != expected_gitlink:
            raise ConfiguredBoardError(
                f"configured submodule gitlink differs: {relative}"
            )
        require_ordinary_index(
            nested_root,
            label=f"configured submodule {relative}",
        )
        clean_status(nested_root, label=f"configured submodule {relative}")
        nested_import_inventory_root, nested_omissions = import_inventory(
            nested_root,
            private_gid=private_gid,
            root_relatives=(".",),
            omission_scope="datasets_gitlink",
            omission_path_prefix=relative,
        )
        source_omissions.extend(nested_omissions)
        if _git_identity(nested_root) != (nested_head, nested_tree):
            raise ConfiguredBoardError(
                f"configured submodule changed during admission: {relative}"
            )
        nested_identities.append(
            (
                relative,
                nested_head,
                nested_tree,
                nested_head,
                nested_import_inventory_root,
            )
        )

    clean_status(root, label="accelerator")
    if _git_identity(root) != (outer_head, outer_tree):
        raise ConfiguredBoardError(
            "accelerator repository changed during admission"
        )
    if len(nested_identities) != 1:
        raise ConfiguredBoardError(
            "fresh recovery requires exactly one datasets gitlink identity"
        )
    _nested_path, datasets_gitlink, datasets_tree, _head, _inventory = (
        nested_identities[0]
    )
    ordered_omissions = sorted(
        source_omissions,
        key=lambda item: (item["scope"], item["path"]),
    )
    if len({item["path"] for item in ordered_omissions}) != len(
        ordered_omissions
    ):
        raise ConfiguredBoardError(
            "fresh recovery source symlink inventory is ambiguous"
        )
    omission_commitment: dict[str, Any] = {
        "schema": FRESH_RECOVERY_PROJECTION_OMISSION_SCHEMA,
        "accelerator_head": outer_head,
        "accelerator_tree": outer_tree,
        "datasets_gitlink": datasets_gitlink,
        "datasets_tree": datasets_tree,
        "omitted_source_symlinks": ordered_omissions,
    }
    omission_commitment["commitment_cid"] = _identity(omission_commitment)
    return (
        outer_head,
        outer_tree,
        import_inventory_root,
        tuple(nested_identities),
        omission_commitment,
    )


def _fresh_recovery_admission_failure(detail: str) -> ConfiguredBoardError:
    return ConfiguredBoardError(
        "fresh-generation recovery initial launch admission failed: "
        f"{detail}; a typed live-continuity verifier is required after any "
        "legitimate runtime progress"
    )


def _open_fresh_recovery_interpreter(
    policy: Mapping[str, Any],
) -> tuple[int, str, str]:
    """Open and hash the exact root-owned interpreter selected by policy.

    The returned descriptor remains open across ``execve``.  The child is
    executed through ``/proc/self/fd`` while the canonical policy path remains
    ``argv[0]``, so a pathname swap cannot change the bytes that were admitted.
    """

    raw_path = policy.get("verification_python_executable")
    raw_digest = policy.get("verification_python_executable_sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(raw_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", raw_digest) is None
    ):
        raise _fresh_recovery_admission_failure(
            "verification interpreter identity is absent"
        )
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or Path(os.path.abspath(path)) != path
    ):
        raise _fresh_recovery_admission_failure(
            "verification interpreter path is not canonical absolute"
        )
    try:
        lexical = os.lstat(path)
        if path.resolve(strict=True) != path:
            raise _fresh_recovery_admission_failure(
                "verification interpreter path contains a symbolic link"
            )
    except ConfiguredBoardError:
        raise
    except OSError as exc:
        raise _fresh_recovery_admission_failure(
            "verification interpreter is unavailable"
        ) from exc
    if not hasattr(os, "O_NOFOLLOW"):
        raise _fresh_recovery_admission_failure(
            "no-follow interpreter admission is unavailable"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _fresh_recovery_admission_failure(
            "verification interpreter cannot be opened safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lexical.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or lexical.st_uid != 0
            or opened.st_uid != 0
            or lexical.st_nlink != 1
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o755
            or opened.st_size <= 0
            or opened.st_size > FRESH_RECOVERY_INTERPRETER_MAX_BYTES
            or (lexical.st_dev, lexical.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise _fresh_recovery_admission_failure(
                "verification interpreter is not one immutable root-owned file"
            )
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            observed_size += len(chunk)
            if observed_size > FRESH_RECOVERY_INTERPRETER_MAX_BYTES:
                raise _fresh_recovery_admission_failure(
                    "verification interpreter exceeds its byte bound"
                )
        after = os.fstat(descriptor)
        current = os.lstat(path)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            observed_size != opened.st_size
            or any(
                getattr(opened, field) != getattr(after, field)
                for field in stable_fields
            )
            or any(
                getattr(after, field) != getattr(current, field)
                for field in stable_fields
            )
            or "sha256:" + digest.hexdigest() != raw_digest
        ):
            raise _fresh_recovery_admission_failure(
                "verification interpreter content identity differs"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        executable = f"/proc/self/fd/{descriptor}"
        proc_status = os.stat(executable)
        if (proc_status.st_dev, proc_status.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            raise _fresh_recovery_admission_failure(
                "held verification interpreter identity differs"
            )
        return descriptor, raw_path, executable
    except BaseException:
        os.close(descriptor)
        raise


def _seal_fresh_recovery_materializer(payload: bytes) -> int:
    """Copy exact tracked materializer bytes into one immutable anonymous file."""

    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > FRESH_RECOVERY_MATERIALIZER_MAX_BYTES
    ):
        raise _fresh_recovery_admission_failure(
            "tracked recovery verifier bytes exceed their bound"
        )
    required = (
        "memfd_create",
        "MFD_CLOEXEC",
        "MFD_ALLOW_SEALING",
    )
    if any(not hasattr(os, name) for name in required):
        raise _fresh_recovery_admission_failure(
            "sealed recovery verifier execution is unavailable"
        )
    seal_names = (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_WRITE",
    )
    if any(not hasattr(fcntl, name) for name in seal_names):
        raise _fresh_recovery_admission_failure(
            "sealed recovery verifier policy is unavailable"
        )
    flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
    try:
        descriptor = os.memfd_create("lgcvf-recovery-materializer", flags)
    except OSError as exc:
        raise _fresh_recovery_admission_failure(
            "sealed recovery verifier cannot be created"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise _fresh_recovery_admission_failure(
                    "sealed recovery verifier write stalled"
                )
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
            raise _fresh_recovery_admission_failure(
                "sealed recovery verifier policy differs"
            )
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o400
            or observed.st_size != len(payload)
        ):
            raise _fresh_recovery_admission_failure(
                "sealed recovery verifier identity differs"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _run_fresh_recovery_verifier(
    board: ConfiguredBoard,
    materializer_path: Path,
    materializer_bytes: bytes,
) -> subprocess.CompletedProcess[str]:
    """Run the protected public verifier with no ambient code authority."""

    policy = board.payload.get("fresh_generation_recovery")
    if not isinstance(policy, Mapping):
        raise _fresh_recovery_admission_failure("recovery policy is not an object")
    descriptor, interpreter_path, executable = _open_fresh_recovery_interpreter(
        policy
    )
    try:
        materializer_descriptor = _seal_fresh_recovery_materializer(
            materializer_bytes
        )
    except BaseException:
        os.close(descriptor)
        raise
    environment = _sanitized_git_environment()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    command = [
        interpreter_path,
        "-I",
        "-S",
        "-B",
        "-c",
        FRESH_RECOVERY_MATERIALIZER_BOOTSTRAP,
        str(materializer_path),
        str(materializer_descriptor),
        "<unavailable-private-pycache>",
        "verify",
    ]
    try:
        with tempfile.TemporaryDirectory(
            prefix="lgcvf-recovery-pycache-",
        ) as pycache_root_text:
            pycache_root = Path(pycache_root_text)
            observed = os.lstat(pycache_root)
            if (
                stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISDIR(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o700
                or any(pycache_root.iterdir())
            ):
                raise _fresh_recovery_admission_failure(
                    "private verifier bytecode root identity differs"
                )
            command[8] = str(pycache_root)
            return subprocess.run(
                command,
                executable=executable,
                pass_fds=(descriptor, materializer_descriptor),
                cwd=board.repo_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                check=False,
                timeout=300.0,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            "",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        os.close(materializer_descriptor)
        os.close(descriptor)


def _validate_fresh_recovery_projection_bindings(
    report: Mapping[str, Any],
    *,
    source_omission_commitment: Mapping[str, Any],
) -> None:
    """Cross-check source-derived omissions and shape richer replay evidence."""

    omission = report.get("validation_projection_omission_commitment")
    omission_root = report.get("validation_projection_omission_root")
    if (
        not isinstance(omission, Mapping)
        or set(omission)
        != {
            "schema",
            "accelerator_head",
            "accelerator_tree",
            "datasets_gitlink",
            "datasets_tree",
            "omitted_source_symlinks",
            "commitment_cid",
        }
        or dict(omission) != dict(source_omission_commitment)
        or omission.get("schema") != FRESH_RECOVERY_PROJECTION_OMISSION_SCHEMA
        or omission.get("commitment_cid")
        != _identity(
            {key: value for key, value in omission.items() if key != "commitment_cid"}
        )
        or omission_root != omission.get("commitment_cid")
    ):
        raise _fresh_recovery_admission_failure(
            "public verifier source-derived projection omission binding differs"
        )

    evidence = report.get("validation_projection_evidence_commitment")
    evidence_root = report.get("validation_projection_evidence_root")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "schema",
            "source_binding_cid",
            "omission_root",
            "ordered_suites",
            "commitment_cid",
        }
        or evidence.get("schema") != FRESH_RECOVERY_PROJECTION_EVIDENCE_SCHEMA
        or evidence.get("omission_root") != omission_root
        or evidence.get("commitment_cid")
        != _identity(
            {key: value for key, value in evidence.items() if key != "commitment_cid"}
        )
        or evidence_root != evidence.get("commitment_cid")
        or re.fullmatch(
            r"baguqeera[a-z2-7]{52}",
            str(evidence.get("source_binding_cid") or ""),
        )
        is None
    ):
        raise _fresh_recovery_admission_failure(
            "public verifier projection evidence binding differs"
        )
    suites = evidence.get("ordered_suites")
    expected_task_ids = (
        "LGCVF-051",
        "LGCVF-060",
        "LGCVF-061",
        "LGCVF-070",
        "LGCVF-071",
        "LGCVF-080",
    )
    expected_suite_ids = tuple(
        "recovery_" + task_id.casefold().replace("-", "_")
        for task_id in expected_task_ids
    )
    if not isinstance(suites, list) or len(suites) != len(expected_task_ids):
        raise _fresh_recovery_admission_failure(
            "public verifier projection evidence suite population differs"
        )
    observed_suite_ids: list[str] = []
    observed_task_ids: list[str] = []
    for item in suites:
        if not isinstance(item, Mapping) or set(item) != {
            "suite_id",
            "task_id",
            "task_cid",
            "projection_cid",
            "copied_source_manifest_root",
        }:
            raise _fresh_recovery_admission_failure(
                "public verifier projection evidence suite fields differ"
            )
        observed_suite_ids.append(str(item.get("suite_id") or ""))
        observed_task_ids.append(str(item.get("task_id") or ""))
        for field in (
            "task_cid",
            "projection_cid",
            "copied_source_manifest_root",
        ):
            if re.fullmatch(
                r"baguqeera[a-z2-7]{52}", str(item.get(field) or "")
            ) is None:
                raise _fresh_recovery_admission_failure(
                    "public verifier projection evidence suite identity differs"
                )
    if (
        tuple(observed_suite_ids) != expected_suite_ids
        or tuple(observed_task_ids) != expected_task_ids
    ):
        raise _fresh_recovery_admission_failure(
            "public verifier projection evidence suite order differs"
        )


def _verify_fresh_recovery_launch_admission(
    board: ConfiguredBoard,
) -> dict[str, Any] | None:
    """Admit only the exact pristine run-v17 recovery before command rendering.

    The scheduler does not interpret DuckDB or recovery artifacts itself.  A
    protected public materializer owns those semantics and emits one closed,
    content-addressed read-only verification report.  The current contract is
    intentionally initial-state-only: a later restart fails closed until a
    separately reviewed live-continuity verifier exists.
    """

    if not _targets_fresh_recovery_generation(
        board.payload,
        repo_root=board.repo_root,
    ):
        return None
    if "fresh_generation_recovery" not in board.payload:
        raise _fresh_recovery_admission_failure(
            "protected run-v17 target lacks its full recovery policy"
        )
    policy = board.payload.get("fresh_generation_recovery")
    if not isinstance(policy, Mapping):
        raise _fresh_recovery_admission_failure("recovery policy is not an object")
    if (
        policy.get("schema") != FRESH_RECOVERY_POLICY_SCHEMA
        or policy.get("target_generation") != FRESH_RECOVERY_TARGET_GENERATION
    ):
        raise _fresh_recovery_admission_failure(
            "recovery policy schema or target generation differs"
        )
    duckdb_runtime_cid = policy.get("duckdb_runtime_cid")
    source_generation = policy.get("source_generation")
    if (
        not isinstance(duckdb_runtime_cid, str)
        or re.fullmatch(r"baguqeera[a-z2-7]{52}", duckdb_runtime_cid) is None
    ):
        raise _fresh_recovery_admission_failure(
            "recovery policy DuckDB runtime identity is absent"
        )
    if not isinstance(source_generation, str) or not source_generation:
        raise _fresh_recovery_admission_failure(
            "recovery policy source generation is absent"
        )
    canonical_config, canonical_config_relative = _lexical_repo_artifact(
        board.repo_root,
        board.path(FRESH_RECOVERY_CONFIG_PATH),
    )
    if (
        canonical_config_relative != FRESH_RECOVERY_CONFIG_PATH
        or Path(os.path.abspath(board.config_path)) != canonical_config
    ):
        raise _fresh_recovery_admission_failure(
            "protected run-v17 must use the exact canonical scheduler config"
        )
    relative = _safe_relative(
        board.payload.get("materializer_path"),
        field="materializer_path",
    )
    if relative != FRESH_RECOVERY_MATERIALIZER_PATH:
        raise _fresh_recovery_admission_failure(
            "protected run-v17 must use the exact canonical recovery verifier"
        )
    if relative not in board.protected_paths:
        raise _fresh_recovery_admission_failure(
            "public recovery verifier is not a protected control file"
        )
    materializer_path, _ = _lexical_repo_artifact(
        board.repo_root,
        board.path(relative),
    )
    try:
        materializer_status = os.lstat(materializer_path)
    except OSError as exc:
        raise _fresh_recovery_admission_failure(
            "public recovery verifier is absent"
        ) from exc
    if (
        stat.S_ISLNK(materializer_status.st_mode)
        or not stat.S_ISREG(materializer_status.st_mode)
        or materializer_status.st_nlink != 1
        or materializer_status.st_uid != os.geteuid()
    ):
        raise _fresh_recovery_admission_failure(
            "public recovery verifier file identity differs"
        )

    try:
        source_identity = _fresh_recovery_clean_source_identity(board)
        (
            source_head,
            _source_tree,
            _import_inventory_root,
            _nested_source_identities,
            source_omission_commitment,
        ) = source_identity
        config_bytes, _config_snapshot = _tracked_head_snapshot(
            repo_root=board.repo_root,
            path=canonical_config,
            source_head=source_head,
        )
        _materializer_bytes, _materializer_snapshot = _tracked_head_snapshot(
            repo_root=board.repo_root,
            path=materializer_path,
            source_head=source_head,
        )
    except ConfiguredBoardError as exc:
        raise _fresh_recovery_admission_failure(
            "canonical recovery checkout is not one clean, current tracked "
            f"outer/nested source forest ({exc})"
        ) from exc
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if (
        board.configuration_root
        != _identity({"bytes_sha256": config_sha256})
        or board.configuration_revision
        != _identity(
            {
                "path": FRESH_RECOVERY_CONFIG_PATH,
                "bytes_sha256": config_sha256,
            }
        )
    ):
        raise _fresh_recovery_admission_failure(
            "loaded recovery config differs from its tracked canonical bytes"
        )

    completed = _run_fresh_recovery_verifier(
        board,
        materializer_path,
        _materializer_bytes,
    )
    try:
        after_identity = _fresh_recovery_clean_source_identity(board)
        (
            after_head,
            _after_tree,
            _after_import_inventory_root,
            _after_nested_identities,
            _after_omission_commitment,
        ) = after_identity
        after_config, _after_config_snapshot = _tracked_head_snapshot(
            repo_root=board.repo_root,
            path=canonical_config,
            source_head=after_head,
        )
        after_materializer, _after_materializer_snapshot = _tracked_head_snapshot(
            repo_root=board.repo_root,
            path=materializer_path,
            source_head=after_head,
        )
    except ConfiguredBoardError as exc:
        raise _fresh_recovery_admission_failure(
            "canonical recovery source changed during public verification"
        ) from exc
    if (
        after_identity != source_identity
        or after_head != source_head
        or after_config != config_bytes
        or after_materializer != _materializer_bytes
    ):
        raise _fresh_recovery_admission_failure(
            "canonical recovery source changed during public verification"
        )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if (
        len(stdout.encode("utf-8", errors="replace"))
        > FRESH_RECOVERY_VERIFIER_MAX_OUTPUT_BYTES
        or len(stderr.encode("utf-8", errors="replace"))
        > FRESH_RECOVERY_VERIFIER_MAX_OUTPUT_BYTES
    ):
        raise _fresh_recovery_admission_failure("public verifier output exceeded its bound")
    try:
        report = json.loads(stdout, object_pairs_hook=_reject_duplicate_keys)
    except (ConfiguredBoardError, json.JSONDecodeError) as exc:
        raise _fresh_recovery_admission_failure(
            "public verifier did not emit exactly one JSON object"
        ) from exc
    if not isinstance(report, dict):
        raise _fresh_recovery_admission_failure(
            "public verifier did not emit exactly one JSON object"
        )
    if completed.returncode != 0:
        reason = report.get("error") or report.get("errors") or stderr[-2_000:]
        raise _fresh_recovery_admission_failure(
            f"public verifier rejected the generation ({reason!r})"
        )
    if set(report) != FRESH_RECOVERY_VERIFICATION_FIELDS:
        raise _fresh_recovery_admission_failure("public verifier report shape differs")
    _validate_fresh_recovery_projection_bindings(
        report,
        source_omission_commitment=source_omission_commitment,
    )

    partitions = (
        ("completed_task_ids", "completed_count", 13),
        ("todo_task_ids", "todo_count", 13),
        ("blocked_task_ids", "blocked_count", 2),
    )
    partition_values: list[set[str]] = []
    for ids_field, count_field, expected_count in partitions:
        identifiers = report.get(ids_field)
        if (
            not isinstance(identifiers, list)
            or any(not isinstance(item, str) or not item for item in identifiers)
            or len(identifiers) != expected_count
            or len(set(identifiers)) != expected_count
            or report.get(count_field) != expected_count
        ):
            raise _fresh_recovery_admission_failure(
                f"public verifier {ids_field} partition differs"
            )
        partition_values.append(set(identifiers))
    if any(
        partition_values[left] & partition_values[right]
        for left in range(len(partition_values))
        for right in range(left + 1, len(partition_values))
    ):
        raise _fresh_recovery_admission_failure(
            "public verifier task partitions overlap"
        )

    required_values = {
        "schema": FRESH_RECOVERY_VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only",
        "source_generation": source_generation,
        "target_generation": FRESH_RECOVERY_TARGET_GENERATION,
        "duckdb_runtime_cid": duckdb_runtime_cid,
        "ready_task_ids": ["LGCVF-081"],
        "model_provider_route": "none",
        "network_isolation_enforced": True,
        "candidate_authored_validation": True,
        "validation_completion_authoritative": False,
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "source_database_statuses_read": False,
        "synthetic_source_disposition": "quarantined_not_imported",
        "stores_unchanged": True,
    }
    if any(report.get(key) != value for key, value in required_values.items()):
        raise _fresh_recovery_admission_failure(
            "public verifier authority disposition differs"
        )
    for field in (
        "manifest_cid",
        "receipt_cid",
        "source_evidence_cid",
        "validation_qualification_cid",
        "operational_verification_root",
        "verification_root",
    ):
        if not isinstance(report.get(field), str) or not report[field]:
            raise _fresh_recovery_admission_failure(
                f"public verifier {field} is absent"
            )
    claimed_root = str(report["verification_root"])
    root_material = dict(report)
    root_material.pop("verification_root")
    if claimed_root != _identity(root_material):
        raise _fresh_recovery_admission_failure(
            "public verifier content identity differs"
        )
    return report


def _git(
    board: ConfiguredBoard,
    *args: str,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    return _git_run(
        args,
        cwd=board.repo_root,
        timeout=timeout,
    )


def _append_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    *,
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def _gitlink_commit(
    board: ConfiguredBoard,
    relative: str,
) -> str:
    result = _git(board, "ls-tree", "HEAD", "--", relative)
    if result.returncode != 0:
        return ""
    match = re.fullmatch(
        rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(relative)}\n?",
        result.stdout,
    )
    return match.group(1) if match else ""


def _control_file_is_tracked(
    board: ConfiguredBoard,
    relative: str,
) -> bool:
    """Recognize control files tracked by the outer or an owned submodule."""

    if (
        _git(
            board,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        ).returncode
        == 0
    ):
        return True
    relative_path = PurePosixPath(relative)
    for submodule in sorted(
        board.worktree_submodule_paths,
        key=lambda value: len(PurePosixPath(value).parts),
        reverse=True,
    ):
        submodule_path = PurePosixPath(submodule)
        prefix = submodule_path.parts
        if (
            relative_path.parts[: len(prefix)] != prefix
            or len(relative_path.parts) == len(prefix)
        ):
            continue
        if (
            _git(
                board,
                "ls-files",
                "--error-unmatch",
                "--",
                submodule,
            ).returncode
            != 0
        ):
            return False
        nested_root = board.path(submodule)
        inner = PurePosixPath(*relative_path.parts[len(prefix) :]).as_posix()
        return (
            _run(
                ("git", "ls-files", "--error-unmatch", "--", inner),
                cwd=nested_root,
                timeout=60.0,
            ).returncode
            == 0
        )
    return False


def preflight_configured_board(
    board: ConfiguredBoard,
    *,
    admitted_live_validator_sha256: str = "",
) -> dict[str, Any]:
    """Prove that a scheduler document can safely launch from this checkout."""

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    if _targets_fresh_recovery_generation(
        board.payload,
        repo_root=board.repo_root,
    ):
        try:
            recovery_admission = _verify_fresh_recovery_launch_admission(board)
        except ConfiguredBoardError as exc:
            _append_check(
                checks,
                errors,
                name="fresh_generation_recovery_admission",
                passed=False,
                detail=str(exc),
            )
            # The protected verifier has already established that source or
            # recovery authority is unsafe.  Do not continue into generic Git
            # or validator probes: repository-local attributes and filters are
            # themselves among the rejected inputs and a later ``git status``
            # could execute them even though launch admission has failed.
            return {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "configured-board-preflight@1"
                ),
                "valid": False,
                "config_path": str(board.config_path),
                "repo_root": str(board.repo_root),
                "board_namespace": board.board_namespace,
                "taskboard_path": str(board.path(board.taskboard_path)),
                "max_lanes": board.max_lanes,
                "errors": errors,
                "warnings": warnings,
                "checks": checks,
                "validator_report": {},
            }
        else:
            assert recovery_admission is not None
            _append_check(
                checks,
                errors,
                name="fresh_generation_recovery_admission",
                passed=True,
                detail={
                    "schema": recovery_admission["schema"],
                    "target_generation": recovery_admission["target_generation"],
                    "duckdb_runtime_cid": recovery_admission[
                        "duckdb_runtime_cid"
                    ],
                    "ready_task_ids": recovery_admission["ready_task_ids"],
                    "model_provider_route": recovery_admission[
                        "model_provider_route"
                    ],
                    "validation_completion_authoritative": recovery_admission[
                        "validation_completion_authoritative"
                    ],
                    "validation_projection_omission_root": recovery_admission[
                        "validation_projection_omission_root"
                    ],
                    "validation_projection_evidence_root": recovery_admission[
                        "validation_projection_evidence_root"
                    ],
                    "receipt_cid": recovery_admission["receipt_cid"],
                    "operational_verification_root": recovery_admission[
                        "operational_verification_root"
                    ],
                    "verification_root": recovery_admission["verification_root"],
                    "stores_unchanged": recovery_admission["stores_unchanged"],
                },
            )

    top = _git(board, "rev-parse", "--show-toplevel")
    _append_check(
        checks,
        errors,
        name="repository_root",
        passed=(
            top.returncode == 0
            and Path(top.stdout.strip()).resolve() == board.repo_root
        ),
        detail=top.stderr.strip() or top.stdout.strip(),
    )

    source_binding = board.payload.get("source_binding")
    if not isinstance(source_binding, dict):
        errors.append("source_binding must be an object")
        source_binding = {}
    required_branch = str(
        source_binding.get("accelerator_required_branch") or ""
    ).strip()
    current_branch = _git(board, "branch", "--show-current")
    _append_check(
        checks,
        errors,
        name="required_branch",
        passed=(
            current_branch.returncode == 0
            and current_branch.stdout.strip() == required_branch
            and required_branch == board.merge_target_branch
        ),
        detail={
            "expected": required_branch,
            "merge_target": board.merge_target_branch,
            "actual": current_branch.stdout.strip(),
        },
    )
    branch_format = _git(
        board,
        "check-ref-format",
        "--branch",
        board.merge_target_branch,
    )
    target_ref = _git(
        board,
        "rev-parse",
        "--verify",
        f"{board.merge_target_branch}^{{commit}}",
    )
    _append_check(
        checks,
        errors,
        name="merge_target",
        passed=branch_format.returncode == 0 and target_ref.returncode == 0,
        detail=target_ref.stderr.strip() or target_ref.stdout.strip(),
    )
    required_ancestor = str(
        source_binding.get("accelerator_required_ancestor") or ""
    ).strip()
    ancestor = _git(
        board,
        "merge-base",
        "--is-ancestor",
        required_ancestor,
        "HEAD",
    )
    _append_check(
        checks,
        errors,
        name="required_ancestor",
        passed=bool(re.fullmatch(r"[0-9a-f]{40}", required_ancestor))
        and ancestor.returncode == 0,
        detail=required_ancestor,
    )

    required_files = {
        board.config_path.relative_to(board.repo_root).as_posix(),
        board.taskboard_path,
        board.objectives_path,
        board.plan_path,
        board.validator_path,
        *board.protected_paths,
    }
    missing_files = sorted(
        relative
        for relative in required_files
        if not board.path(relative).is_file()
    )
    _append_check(
        checks,
        errors,
        name="control_files_present",
        passed=not missing_files,
        detail=missing_files,
    )
    tracked = [
        relative
        for relative in sorted(required_files)
        if _control_file_is_tracked(board, relative)
    ]
    untracked_control = sorted(required_files - set(tracked))
    _append_check(
        checks,
        errors,
        name="control_files_tracked",
        passed=not untracked_control,
        detail=untracked_control,
    )
    status = _git(
        board,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    dirty_lines = [line for line in status.stdout.splitlines() if line]
    launch_policy = board.payload.get("launch_policy")
    eaaef_live_admitted = (
        _eaaef_plan_bound_profile(board)
        and isinstance(launch_policy, dict)
        and launch_policy.get("live_multi_supervisor_allowed") is True
    )
    eaaef_receipt_only_drift = False
    if eaaef_live_admitted:
        from ..validation.eaaef_host_admission import (
            eaaef_checkout_has_only_generated_receipt_drift,
        )

        eaaef_receipt_only_drift = (
            eaaef_checkout_has_only_generated_receipt_drift(board.repo_root)
        )
    _append_check(
        checks,
        errors,
        name="checkout_clean",
        passed=status.returncode == 0
        and (
            not dirty_lines
            or (eaaef_live_admitted and eaaef_receipt_only_drift)
        ),
        detail=dirty_lines[:100],
    )
    if eaaef_live_admitted and dirty_lines and eaaef_receipt_only_drift:
        warnings.append(
            "EAAEF live launch proceeding with generated receipt staging only"
        )

    validator_report: dict[str, Any] = {}
    if admitted_live_validator_sha256:
        admitted_validator = bool(
            board.board_namespace == LGCVF_LIVE_BOARD_NAMESPACE
            and board.config_path.relative_to(board.repo_root)
            == LGCVF_LIVE_CONFIG_PATH
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                admitted_live_validator_sha256,
            )
        )
        _append_check(
            checks,
            errors,
            name="declared_validator",
            passed=admitted_validator,
            detail={
                "execution": "controller-qualified_sealed_capsule_member",
                "sha256": admitted_live_validator_sha256,
            },
        )
        if admitted_validator:
            validator_report = {
                "valid": True,
                "authority": "controller-qualified_sealed_capsule_member",
                "validator_sha256": admitted_live_validator_sha256,
            }
    elif board.path(board.validator_path).is_file():
        validator = _run(
            (
                sys.executable,
                str(board.path(board.validator_path)),
                "--check-all",
            ),
            cwd=board.repo_root,
        )
        try:
            parsed = json.loads(validator.stdout)
            if isinstance(parsed, dict):
                validator_report = parsed
        except json.JSONDecodeError:
            validator_report = {}
        _append_check(
            checks,
            errors,
            name="declared_validator",
            passed=(
                validator.returncode == 0
                and validator_report.get("valid") is True
                and (
                    not _eaaef_plan_bound_profile(board)
                    or validator_report.get("board_namespace")
                    == EAAEF_BOARD_NAMESPACE
                )
            ),
            detail={
                "returncode": validator.returncode,
                "stderr": validator.stderr[-2000:],
                "errors": validator_report.get("errors"),
                "board_namespace": validator_report.get("board_namespace"),
            },
        )

    planning_revisions: dict[str, str] = {}
    for key, value in source_binding.items():
        if not key.endswith("_submodule_path") or not isinstance(value, str):
            continue
        prefix = key[: -len("_submodule_path")]
        revision = source_binding.get(f"{prefix}_planning_revision")
        if isinstance(revision, str) and revision.strip():
            planning_revisions[value.strip()] = revision.strip()

    submodule_checks: list[dict[str, Any]] = []
    for relative in board.worktree_submodule_paths:
        gitlink = _gitlink_commit(board, relative)
        target = board.path(relative)
        top_level = _run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=target,
            timeout=60,
        ) if target.is_dir() else None
        exact_worktree = bool(
            top_level is not None
            and top_level.returncode == 0
            and Path(top_level.stdout.strip()).resolve() == target.resolve()
        )
        head = _run(
            ("git", "rev-parse", "HEAD"),
            cwd=target,
            timeout=60,
        ) if exact_worktree else None
        clean = _run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=target,
            timeout=60,
        ) if head is not None and head.returncode == 0 else None
        actual_head = head.stdout.strip() if head is not None else ""
        expected_planning = planning_revisions.get(relative, "")
        planning_ancestor = (
            _run(
                (
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    expected_planning,
                    actual_head,
                ),
                cwd=target,
                timeout=60,
            )
            if (
                exact_worktree
                and re.fullmatch(r"[0-9a-f]{40}", expected_planning)
                and re.fullmatch(r"[0-9a-f]{40}", actual_head)
            )
            else None
        )
        submodule_dirty = bool(clean is not None and clean.stdout.strip())
        valid = bool(
            gitlink
            and exact_worktree
            and head is not None
            and head.returncode == 0
            and actual_head == gitlink
            and planning_ancestor is not None
            and planning_ancestor.returncode == 0
            and clean is not None
            and clean.returncode == 0
            and not submodule_dirty
        )
        submodule_checks.append(
            {
                "path": relative,
                "valid": valid,
                "gitlink": gitlink,
                "head": actual_head,
                "exact_worktree": exact_worktree,
                "planning_revision": expected_planning,
                "planning_revision_is_ancestor": bool(
                    planning_ancestor is not None
                    and planning_ancestor.returncode == 0
                ),
                "dirty": (
                    clean.stdout.splitlines()[:50]
                    if clean is not None
                    else []
                ),
            }
        )
    _append_check(
        checks,
        errors,
        name="configured_submodules",
        passed=all(item["valid"] for item in submodule_checks),
        detail=submodule_checks,
    )
    implementation_entry = board.path(
        IMPLEMENTATION_ENTRY_PATH.as_posix()
    )
    _append_check(
        checks,
        errors,
        name="implementation_entry",
        passed=implementation_entry.is_file(),
        detail=str(implementation_entry),
    )

    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "configured-board-preflight@1"
        ),
        "valid": not errors,
        "config_path": str(board.config_path),
        "repo_root": str(board.repo_root),
        "board_namespace": board.board_namespace,
        "taskboard_path": str(board.path(board.taskboard_path)),
        "max_lanes": board.max_lanes,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "validator_report": validator_report,
    }


def configured_board_common_args(
    board: ConfiguredBoard,
    *,
    implement: bool,
    state_owner_bootstrap_fd: int = -1,
    state_owner_bootstrap_store_id: str = "",
) -> tuple[str, ...]:
    """Map scheduler policy to existing implementation-supervisor CLI args."""

    payload = board.payload
    objective_refill_controls = _objective_refill_controls(payload)
    program_for_paths = board.resolved_database_program()
    worktree_root = (
        str(board.path(program_for_paths.worktree_root))
        if program_for_paths.worktree_root
        else str(board.path(board.runtime_paths["worktrees"]))
    )
    implementation_branch = resolve_board_implementation_branch(
        board.merge_target_branch,
        board.board_namespace,
    )
    args: list[str] = [
        "--todo-path",
        str(board.path(board.taskboard_path)),
        "--task-prefix",
        board.task_header_prefix,
        "--board-namespace",
        board.board_namespace,
        "--worktree-root",
        worktree_root,
        "--merge-target-branch",
        implementation_branch,
        "--merge-queue-dir",
        str(board.path(board.runtime_paths["merge_queue"])),
        "--stale-seconds",
        str(payload["stale_seconds"]),
        "--check-interval",
        str(payload["check_interval_seconds"]),
        "--watchdog-startup-grace-seconds",
        str(payload["watchdog_startup_grace_seconds"]),
        "--max-restarts",
        str(payload["max_restarts"]),
        "--max-task-attempts",
        str(payload["max_task_attempts"]),
        "--daemon-interval",
        str(payload["daemon_interval_seconds"]),
        "--implementation-timeout",
        str(payload["implementation_timeout_seconds"]),
        "--implementation-max-timeout",
        str(payload["implementation_max_timeout_seconds"]),
        "--implementation-log-stall-seconds",
        str(payload["implementation_log_stall_seconds"]),
        "--implementation-retry-budget",
        str(payload["implementation_retry_budget"]),
        "--validation-retry-budget",
        str(payload["validation_retry_budget"]),
        "--merge-retry-budget",
        str(payload["merge_retry_budget"]),
        "--no-objective-task-janitor",
        "--no-objective-goal-completion-reconcile",
        "--no-objective-goal-migration",
        "--log-level",
        "INFO",
    ]
    # Explicit database-program selections are supervisor inputs.  The
    # fallback legacy-Markdown program, however, is a daemon-only compatibility
    # projection: implementation_supervisor does not accept the database CLI
    # flags and already launches the daemon with its closed legacy-Markdown
    # default.  Passing those daemon-only flags through the supervisor creates
    # an immediate argparse/restart loop before any task can run.
    program = board.resolved_database_program()
    program_args = program.cli_args() if board.database_program is not None else []
    skip_next = False
    for item in program_args:
        if skip_next:
            skip_next = False
            continue
        if item == "--worktree-root":
            skip_next = True
            continue
        args.append(item)
    if _plan_bound_profile(board):
        # The bounded child re-opens the sealed profile solely to sample live
        # host/provider capacity before the daemon's canonical claim gate.
        args.extend(["--scheduler-config", str(board.config_path)])
    args.append("--implement" if implement else "--no-implement")
    # Legacy profiles retain their configured hash-sharding behavior.  A v3
    # child receives one exact compiler slice below, so hash sharding and its
    # strict fallback policy must both be disabled for that child.
    if board.strict_task_sharding and not _plan_bound_profile(board):
        args.append("--strict-task-sharding")
    if board.idle_lane_work_stealing and not _plan_bound_profile(board):
        args.extend(
            ["--idle-lane-work-stealing", board.idle_lane_work_stealing]
        )
    for relative in board.worktree_submodule_paths:
        args.extend(["--worktree-submodule-path", relative])
    for relative in board.protected_paths:
        args.extend(["--implementation-protected-path", relative])
    if objective_refill_controls is not None:
        (
            min_open_tasks,
            max_findings,
            cooldown_seconds,
            max_epochs,
            max_total_tasks,
        ) = objective_refill_controls
        args.extend(
            [
                "--objective-refill-scan",
                "--objective-path",
                str(board.path(board.objectives_path)),
                "--objective-scan-min-open-tasks",
                str(min_open_tasks),
                "--objective-scan-max-findings",
                str(max_findings),
                "--objective-scan-cooldown-seconds",
                str(cooldown_seconds),
            ]
        )
        if max_epochs is not None:
            args.extend(
                ["--objective-refill-max-epochs", str(max_epochs)]
            )
        if max_total_tasks is not None:
            args.extend(
                [
                    "--objective-refill-max-total-tasks",
                    str(max_total_tasks),
                ]
            )
        if payload.get("objective_goal_refinement_enabled") is False:
            args.append("--no-objective-goal-refinement")
    if payload.get("codebase_refill_enabled") is True:
        args.append("--codebase-refill-scan")
    if payload.get("retry_budget_guardrail_enabled") is False:
        args.append("--no-retry-budget-guardrail")
    if payload.get("dependency_guardrail_enabled") is False:
        args.append("--no-dependency-guardrail")
    if payload.get("reconciliation_guardrail_enabled") is False:
        args.append("--no-reconciliation-guardrail")
    bootstrap_presence = (
        state_owner_bootstrap_fd >= 3,
        bool(str(state_owner_bootstrap_store_id or "").strip()),
    )
    if any(bootstrap_presence) and not all(bootstrap_presence):
        raise ConfiguredBoardError(
            "state-owner bootstrap descriptor and store must be paired"
        )
    if all(bootstrap_presence):
        program = board.resolved_database_program()
        if (
            board.board_namespace != LGCVF_LIVE_BOARD_NAMESPACE
            or program.authority_mode != "quack"
            or str(state_owner_bootstrap_store_id) != str(program.store_id)
        ):
            raise ConfiguredBoardError(
                "state-owner bootstrap scope differs from the LGCVF Quack board"
            )
        from ..task_sources.state_owner_bootstrap import (
            StateOwnerBootstrapError,
            validate_state_owner_bootstrap_listener,
        )

        try:
            validate_state_owner_bootstrap_listener(
                int(state_owner_bootstrap_fd)
            )
        except StateOwnerBootstrapError as exc:
            raise ConfiguredBoardError(
                "state-owner bootstrap listener is invalid"
            ) from exc
        args.extend(
            [
                "--state-owner-bootstrap-fd",
                str(state_owner_bootstrap_fd),
                "--state-owner-bootstrap-store-id",
                str(state_owner_bootstrap_store_id),
            ]
        )
    return tuple(args)


def configured_board_launch_plan(
    board: ConfiguredBoard,
    *,
    implement: bool,
    detach: bool,
    duration_seconds: float = float("inf"),
    stamp: str | None = None,
    parallelism_receipt: ParallelismDecisionReceipt | None = None,
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None,
    accepted_control_plane_descriptor: int = -1,
    configured_board_live_capsule_pin_json: str = "",
    configured_board_live_capsule_descriptor: int = -1,
    configured_board_live_admission_json: str = "",
    configured_board_live_native_launch_json: str = "",
    configured_board_live_native_descriptor: int = -1,
    state_owner_bootstrap_fd: int = -1,
    state_owner_bootstrap_store_id: str = "",
) -> dict[str, Any]:
    """Render the exact existing multi-supervisor runner invocation."""

    live_values = (
        bool(configured_board_live_capsule_pin_json),
        configured_board_live_capsule_descriptor >= 3,
        bool(configured_board_live_admission_json),
        bool(configured_board_live_native_launch_json),
        configured_board_live_native_descriptor >= 3,
    )
    if any(live_values) and not all(live_values):
        raise ConfiguredBoardError(
            "LGCVF configured-board live launch fields are incomplete"
        )
    bootstrap_values = (
        state_owner_bootstrap_fd >= 3,
        bool(str(state_owner_bootstrap_store_id or "").strip()),
    )
    if any(bootstrap_values) and not all(bootstrap_values):
        raise ConfiguredBoardError(
            "LGCVF state-owner bootstrap fields are incomplete"
        )
    if all(live_values) != all(bootstrap_values):
        raise ConfiguredBoardError(
            "LGCVF live capsule and state-owner bootstrap are bidirectional"
        )
    live_context = None
    if all(live_values):
        try:
            live_context = verify_lgcvf_configured_board_live_context(
                capsule_pin_json=configured_board_live_capsule_pin_json,
                capsule_descriptor=configured_board_live_capsule_descriptor,
                admission_json=configured_board_live_admission_json,
                native_launch_json=configured_board_live_native_launch_json,
                native_descriptor=configured_board_live_native_descriptor,
            )
        except (OSError, ValueError) as exc:
            raise ConfiguredBoardError(
                "LGCVF configured-board live launch binding is invalid"
            ) from exc

    recovery_admission = _verify_fresh_recovery_launch_admission(board)
    implementation_branch = resolve_board_implementation_branch(
        board.merge_target_branch,
        board.board_namespace,
    )
    run_stamp = stamp or utc_run_stamp()
    runtime_root = board.path(board.runtime_paths["root"])
    state_dir = board.path(board.runtime_paths["state"])
    state_relative = Path(board.runtime_paths["state"])
    log_dir = board.path(board.runtime_paths["logs"])
    entry = board.path(IMPLEMENTATION_ENTRY_PATH.as_posix())
    program = board.resolved_database_program()
    plan_bound = _plan_bound_profile(board)
    if live_context is not None:
        try:
            config_relative = board.config_path.relative_to(
                board.repo_root
            ).as_posix()
        except ValueError as exc:
            raise ConfiguredBoardError(
                "LGCVF live config path escapes the repository"
            ) from exc
        if (
            detach
            or plan_bound
            or config_relative != LGCVF_LIVE_CONFIG_PATH.as_posix()
            or board.board_namespace != LGCVF_LIVE_BOARD_NAMESPACE
            or board.max_lanes != 4
            or not board.strict_task_sharding
            or board.idle_lane_work_stealing != "virgin-transfer"
            or board.configuration_root
            != _identity(
                {
                    "bytes_sha256": str(
                        live_context.capsule_pin.candidate_config_sha256
                    ).removeprefix("sha256:")
                }
            )
            or tuple(
                str(lane.get("name") or "")
                for lane in board.payload.get("lanes", ())
                if isinstance(lane, Mapping)
            )
            != live_context.admission.lane_names
            or board.payload.get("schema")
            != (
                "ipfs_accelerate_py.agent_supervisor."
                "logic_governed_compositional_verification_fabric."
                "scheduler_config@1"
            )
            or program.task_source_kind != "duckdb"
            or program.authority_mode != "quack"
            or program.schema_revision
            != "datasets-authoritative-operational-v1"
            or program.failover_policy != "fail_closed"
            or str(state_owner_bootstrap_store_id) != str(program.store_id)
            or state_owner_bootstrap_fd
            in {
                live_context.capsule_descriptor,
                live_context.native_descriptor,
            }
        ):
            raise ConfiguredBoardError(
                "LGCVF live capsule does not match the exact foreground "
                "four-lane Quack board"
            )
    plan_bound_children: tuple[PlanBoundSupervisorChild, ...] = ()
    implementation_tracks: tuple[ImplementationSupervisorTrackConfig, ...] = ()
    if plan_bound and parallelism_receipt is not None:
        binding = parallelism_receipt.binding
        manifest = parallelism_receipt.slice_manifest
        plan_bound_children = tuple(
            PlanBoundSupervisorChild(
                name=f"{board.board_namespace}-lane-{execution_slice.lane_index}",
                accepted_tree_root=board.repo_root,
                script_path=IMPLEMENTATION_ENTRY_PATH.as_posix(),
                state_dir=(state_relative / f"lane-{execution_slice.lane_index}"),
                state_prefix=(
                    f"{_slug(board.task_prefix)}_lane_"
                    f"{execution_slice.lane_index}"
                ),
                plan_revision_store_path=(state_relative / "plan-revision-store"),
                revision_cid=binding.revision_cid,
                plan_root_cid=binding.plan_root_cid,
                execution_plan_cid=binding.execution_plan_cid,
                capacity_snapshot_id=binding.capacity_snapshot_id,
                slice_manifest_cid=parallelism_receipt.slice_manifest_cid,
                slice_id=execution_slice.slice_id,
                source_head=manifest.source_head,
                source_tree=manifest.repository_tree_id,
                task_source_revision=manifest.task_source_revision,
                configuration_root=manifest.configuration_root,
                lane_id=execution_slice.lane_id,
                task_ids=execution_slice.task_ids,
                task_cids=execution_slice.task_cids,
            )
            for execution_slice in manifest.nonempty
        )
    elif not plan_bound:
        implementation_tracks = (
            ImplementationSupervisorTrackConfig(
                name=(
                    "lgcvf-quack-lane"
                    if live_context is not None
                    else board.board_namespace
                ),
                script_path=entry,
                state_dir=state_dir,
                state_prefix=_slug(board.task_prefix),
                database_program=board.database_program,
            ),
        )
    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=board.repo_root,
        duration_seconds=duration_seconds,
        heartbeat_interval_seconds=max(
            1.0,
            float(board.payload["poll_interval_seconds"]),
        ),
        supervisor_status_stale_seconds=max(
            60.0,
            float(board.payload["stale_seconds"]),
        ),
        stop_grace_seconds=max(
            30.0,
            float(board.payload["check_interval_seconds"]) * 2.0,
        ),
        stamp=run_stamp,
        master_dir=runtime_root,
        master_log=log_dir / f"configured-board-{run_stamp}.log",
        master_pid_path=(
            state_dir / "configured-board-wave.pid"
            if plan_bound
            else state_dir / "configured-board-master.pid"
        ),
        label=board.board_namespace,
        python_executable=sys.executable,
        implementation_track_configs=implementation_tracks,
        plan_bound_tracks=plan_bound_children,
        common_args=configured_board_common_args(
            board,
            implement=implement,
            state_owner_bootstrap_fd=state_owner_bootstrap_fd,
            state_owner_bootstrap_store_id=state_owner_bootstrap_store_id,
        ),
        detach=(detach and not plan_bound),
        database_program=board.database_program,
    )
    runner_args = runner.args()
    if plan_bound:
        # An empty first wave is an explicit bounded success.  The reusable
        # runner accepts this marker without constructing or starting a child.
        if "--plan-bound-wave" not in runner_args:
            runner_args.append("--plan-bound-wave")
        if accepted_control_plane_pin is not None:
            verify_agent_implementation_sealed_control_plane(
                accepted_control_plane_pin,
                accepted_control_plane_descriptor,
            )
            expected_generation = (
                (
                    parallelism_receipt.slice_manifest.source_head,
                    parallelism_receipt.slice_manifest.repository_tree_id,
                )
                if parallelism_receipt is not None
                else _git_identity(board.repo_root)
            )
            if (
                accepted_control_plane_pin.source_head,
                accepted_control_plane_pin.source_tree,
            ) != expected_generation:
                raise ConfiguredBoardError(
                    "accepted control-plane generation differs from the wave"
                )
            runner_args.extend(
                [
                    "--accepted-control-plane-pin-json",
                    accepted_control_plane_pin_json(
                        accepted_control_plane_pin
                    ),
                    "--accepted-control-plane-fd",
                    str(accepted_control_plane_descriptor),
                ]
            )
        if _eaaef_plan_bound_profile(board):
            try:
                live_config = board.config_path.relative_to(board.repo_root).as_posix()
            except ValueError as exc:
                raise ConfiguredBoardError(
                    "EAAEF scheduler config escapes the accepted repository"
                ) from exc
            runner_args.extend(
                ["--require-configured-board-live-seal", live_config]
            )
    else:
        runner_args.extend(
            [
                "--implementation-supervisor-lanes-per-track",
                str(board.max_lanes),
            ]
        )
        if live_context is not None:
            runner_args.extend(
                [
                    "--require-lgcvf-configured-board-live-seal",
                    LGCVF_LIVE_CONFIG_PATH.as_posix(),
                    "--configured-board-live-capsule-pin-json",
                    live_context.capsule_pin_json,
                    "--configured-board-live-capsule-fd",
                    str(live_context.capsule_descriptor),
                    "--configured-board-live-admission-json",
                    live_context.admission_json,
                    "--configured-board-live-native-launch-json",
                    live_context.native_launch_json,
                    "--configured-board-live-native-fd",
                    str(live_context.native_descriptor),
                ]
            )
    if board.strict_task_sharding and not plan_bound:
        runner_args.append(
            "--implementation-supervisor-strict-task-sharding"
        )
    if board.idle_lane_work_stealing and not plan_bound:
        runner_args.extend(
            [
                "--implementation-supervisor-idle-lane-work-stealing",
                board.idle_lane_work_stealing,
            ]
        )
    if plan_bound or board.payload.get("exit_when_all_tracks_terminal") is True:
        runner_args.append("--exit-when-all-tracks-terminal")

    provider = board.payload.get("provider")
    provider = provider if isinstance(provider, dict) else {}
    ordered_provider = any(
        field in provider for field in ORDERED_PROVIDER_DETECTION_FIELDS
    )
    if ordered_provider:
        route_plan = _resolved_ordered_provider_route(
            provider,
            repo_root=board.repo_root,
            board_namespace=board.board_namespace,
        )
        environment = route_plan.as_environment()
        primary_executable = _optional_provider_string(
            provider,
            ORDERED_PRIMARY_EXECUTABLE_FIELD,
        )
        if primary_executable:
            environment[GROK_BIN_ENV] = primary_executable
    else:
        provider_id = str(provider.get("provider_id") or "").strip()
        model_id = str(provider.get("model_id") or "").strip()
        environment = {}
        if provider_id:
            # Always pin the scheduler value, including ``auto``. Leaving the
            # variable unset lets an ambient Grok-session
            # IMPLEMENTATION_PROVIDER leak into sealed lanes and force a
            # Grok-only pin that then fail-closes without login in the
            # qualification HOME.
            environment[PROVIDER_ENV] = provider_id
        if model_id and provider_id in {"", "auto", "codex", "openai"}:
            environment[CODEX_MODEL_ENV] = model_id
        external_isolation = provider.get("external_isolation")
        if external_isolation is not None:
            from ..todo_daemon.implementation_daemon import (
                validate_external_provider_isolation_config,
            )

            try:
                isolation_config = (
                    validate_external_provider_isolation_config(
                        external_isolation,
                        verify_host=True,
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ConfiguredBoardError(
                    "provider.external_isolation launch preflight failed: "
                    f"{exc}"
                ) from exc
            environment[EXTERNAL_PROVIDER_ISOLATION_ENV] = (
                isolation_config.environment_json()
            )
    # Database authority is explicit and non-secret. The endpoint field is an
    # opaque secret handle; raw credentials are never copied into this plan.
    if board.database_program is not None:
        environment.update(program.environment(repository_root=board.repo_root))
    plan = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "configured-board-launch-plan@1"
        ),
        "board_namespace": board.board_namespace,
        "implementation_branch": implementation_branch,
        "merge_lock_name": board_merge_lock_name(board.board_namespace),
        "implement": bool(implement),
        "detach": bool(detach),
        "lanes": board.max_lanes,
        "admitted_lanes": len(plan_bound_children) if plan_bound else board.max_lanes,
        "strict_task_sharding": board.strict_task_sharding,
        "idle_lane_work_stealing": board.idle_lane_work_stealing,
        "effective_strict_task_sharding": (
            board.strict_task_sharding if not plan_bound else False
        ),
        "effective_idle_lane_work_stealing": (
            board.idle_lane_work_stealing if not plan_bound else ""
        ),
        "plan_bound_dispatch": plan_bound,
        "active_plan_revision_cid": (
            parallelism_receipt.binding.revision_cid
            if parallelism_receipt is not None
            else ""
        ),
        "slice_manifest_cid": (
            parallelism_receipt.slice_manifest_cid
            if parallelism_receipt is not None
            else ""
        ),
        "argv": runner_args,
        "environment": environment,
        "database_program": program.redacted_dict(),
        "database_program_interface": DATABASE_PROGRAM_CONFIG_INTERFACE,
        "runtime_root": str(runtime_root),
        "master_pid_path": str(
            state_dir / "configured-board-master.pid"
        ),
        "master_log": str(
            log_dir / f"configured-board-{run_stamp}.log"
        ),
    }
    if recovery_admission is not None:
        plan["fresh_generation_recovery_admission"] = {
            "schema": recovery_admission["schema"],
            "target_generation": recovery_admission["target_generation"],
            "duckdb_runtime_cid": recovery_admission["duckdb_runtime_cid"],
            "ready_task_ids": recovery_admission["ready_task_ids"],
            "model_provider_route": recovery_admission["model_provider_route"],
            "validation_completion_authoritative": recovery_admission[
                "validation_completion_authoritative"
            ],
            "validation_projection_omission_root": recovery_admission[
                "validation_projection_omission_root"
            ],
            "validation_projection_evidence_root": recovery_admission[
                "validation_projection_evidence_root"
            ],
            "manifest_cid": recovery_admission["manifest_cid"],
            "receipt_cid": recovery_admission["receipt_cid"],
            "operational_verification_root": recovery_admission[
                "operational_verification_root"
            ],
            "verification_root": recovery_admission["verification_root"],
            "stores_unchanged": recovery_admission["stores_unchanged"],
        }
    return plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight and launch a sealed supervisor scheduler config"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accepted-tree-root", type=Path, default=None)
    parser.add_argument(
        "--accepted-control-plane-pin-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accepted-control-plane-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accepted-control-plane-capsule-parent",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-capsule-pin-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-capsule-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-admission-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-native-launch-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-native-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--state-owner-bootstrap-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--state-owner-bootstrap-store-id",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--coordinator-launch-session",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--coordinator-status-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "preflight",
        help="Validate control files, Git bindings, submodules, and board",
    )
    launch = subparsers.add_parser(
        "launch",
        help="Render or run the configured multi-lane supervisor",
    )
    launch.add_argument(
        "--implement",
        action="store_true",
        help="Authorize implementation-provider dispatch",
    )
    launch.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact launch plan without starting processes",
    )
    launch.add_argument(
        "--foreground",
        action="store_true",
        help="Keep the multi-supervisor runner in the foreground",
    )
    launch.add_argument(
        "--duration-seconds",
        type=float,
        default=float("inf"),
    )
    launch.add_argument(
        "--launch-receipt-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _apply_configured_board_environment(plan: Mapping[str, Any]) -> None:
    environment = plan.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    for name in TRUSTED_RUNTIME_CACHE_ENV_NAMES:
        os.environ.pop(name, None)
    for name in SCHEDULER_PROVIDER_ENV_NAMES:
        if name not in environment:
            os.environ.pop(name, None)
    for name, value in environment.items():
        if name in TRUSTED_RUNTIME_CACHE_ENV_NAMES:
            continue
        os.environ[str(name)] = str(value)


def _ensure_plan_bound_runtime_directory(repo_root: Path, path: Path) -> Path:
    """Create a contained runtime directory one no-symlink component at a time."""

    root = _canonical_no_symlink_root(repo_root)
    directory = Path(path)
    if not directory.is_absolute() or Path(os.path.abspath(directory)) != directory:
        raise ConfiguredBoardError("runtime directory is not lexical absolute")
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise ConfiguredBoardError("runtime directory escapes repository") from exc
    current = root
    for part in relative.parts:
        current /= part
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise ConfiguredBoardError(
                    f"cannot create runtime directory: {current}"
                ) from exc
            try:
                observed = os.lstat(current)
            except OSError as exc:
                raise ConfiguredBoardError(
                    f"cannot revalidate runtime directory: {current}"
                ) from exc
        except OSError as exc:
            raise ConfiguredBoardError(
                f"cannot inspect runtime directory: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ConfiguredBoardError(
                f"runtime path component is not a real directory: {current}"
            )
    return directory


def _coordinator_lane_status_paths(board: ConfiguredBoard) -> tuple[Path, ...]:
    """Derive every configured lane heartbeat path from admitted board fields."""

    state_dir = board.path(board.runtime_paths["state"])
    state_prefix = _slug(board.task_prefix)
    return tuple(
        state_dir
        / f"lane-{index}"
        / f"{state_prefix}_lane_{index}_supervisor_status.json"
        for index in range(board.max_lanes)
    )


def _expected_coordinator_status_path(
    board: ConfiguredBoard,
    launch_session_id: str,
) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}", launch_session_id) is None:
        raise ConfiguredBoardError("coordinator launch session is invalid")
    return board.path(board.runtime_paths["state"]) / (
        f"configured-board-{launch_session_id}.status.json"
    )


def _atomic_publish_coordinator_status(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Publish one immutable single-link status without replacing a pathname."""

    body = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(body) > 1_048_576:
        raise ConfiguredBoardError("coordinator status exceeds its byte bound")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"
    )
    descriptor = -1
    with serialized_lock_update(path):
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ConfiguredBoardError(
                "cannot inspect coordinator status destination"
            ) from exc
        else:
            raise ConfiguredBoardError("coordinator status already exists")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            written = 0
            while written < len(body):
                count = os.write(descriptor, body[written:])
                if count <= 0:
                    raise OSError("short coordinator status write")
                written += count
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_nlink) != 1
                or int(opened.st_uid) != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != 0o600
            ):
                raise ConfiguredBoardError(
                    "coordinator status staging file is unsafe"
                )
            os.close(descriptor)
            descriptor = -1
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
            observed = os.lstat(path)
            if (
                stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or int(observed.st_nlink) != 1
                or int(observed.st_uid) != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o600
                or int(observed.st_size) != len(body)
            ):
                raise ConfiguredBoardError(
                    "published coordinator status is unsafe"
                )
            directory = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except (FileExistsError, OSError) as exc:
            raise ConfiguredBoardError(
                "cannot atomically publish coordinator status"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_coordinator_status(path: Path) -> dict[str, Any]:
    try:
        payload, evidence = _read_stable_regular_json(path)
    except _StableArtifactReadError as exc:
        raise ConfiguredBoardError("coordinator status is not stable") from exc
    if payload is None or set(payload) != COORDINATOR_STATUS_FIELDS:
        raise ConfiguredBoardError("coordinator status is absent or not closed")
    if (
        evidence.get("state") != "present"
        or int(evidence.get("link_count", -1)) != 1
        or int(evidence.get("uid", -1)) != os.geteuid()
        or stat.S_IMODE(int(evidence.get("mode", 0))) != 0o600
    ):
        raise ConfiguredBoardError("coordinator status file identity is unsafe")
    receipt_cid = payload.get("receipt_cid")
    unsigned = dict(payload)
    unsigned.pop("receipt_cid", None)
    if receipt_cid != content_identity(unsigned):
        raise ConfiguredBoardError("coordinator status CID is invalid")
    return payload


def _coordinator_readiness_timeout_seconds(board: ConfiguredBoard) -> float:
    """Return the bounded startup horizon declared by the admitted board."""

    value = board.payload.get("watchdog_startup_grace_seconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ConfiguredBoardError(
            "watchdog startup grace is not a positive finite duration"
        )
    return min(float(value), COORDINATOR_READY_TIMEOUT_MAX_SECONDS)


def _coordinator_launch_attestation_max_age_ms(board: ConfiguredBoard) -> int:
    """Bind the immutable birth attestation to the same startup horizon."""

    return max(1, int(_coordinator_readiness_timeout_seconds(board) * 1_000))


def _exact_process_option(
    argv: Sequence[str],
    name: str,
    expected: str,
) -> bool:
    """Require one exact option/value occurrence in an observed process argv."""

    if argv.count(name) != 1:
        return False
    index = argv.index(name)
    return index + 1 < len(argv) and argv[index + 1] == expected


def _configured_lane_process_ready(
    board: ConfiguredBoard,
    *,
    lane_index: int,
    supervisor_pid: int,
    coordinator_pid: int,
    coordinator_start_ticks: int,
    repository_commit: str,
    repository_tree: str,
) -> bool:
    """Re-observe one exact lifecycle-marked implementation supervisor."""

    adapter = LinuxProcessAdapter()
    try:
        parent, group, session, start_ticks = adapter._stat(  # noqa: SLF001
            supervisor_pid
        )
        argv = adapter._argv(supervisor_pid)  # noqa: SLF001
        environment = adapter._environ(supervisor_pid)  # noqa: SLF001
        cwd = Path(os.readlink(f"/proc/{supervisor_pid}/cwd")).resolve(
            strict=False
        )
        executable = Path(os.readlink(f"/proc/{supervisor_pid}/exe")).resolve(
            strict=False
        )
    except (
        FileNotFoundError,
        ProcessLookupError,
        OSError,
        UnicodeError,
        ValueError,
    ):
        return False

    plan_bound = _plan_bound_profile(board)
    # Plan-bound v3 children keep the explicit "-lane-" token.  Ordinary
    # configured boards expand shards as "{namespace}-{index}".
    lane_name = (
        f"{board.board_namespace}-lane-{lane_index}"
        if plan_bound
        else f"{board.board_namespace}-{lane_index}"
    )
    state_relative = Path(board.runtime_paths["state"]) / f"lane-{lane_index}"
    state_dir = board.path(state_relative.as_posix()).resolve(strict=False)
    expected_state_arg = state_relative.as_posix() if plan_bound else str(state_dir)
    state_prefix = f"{_slug(board.task_prefix)}_lane_{lane_index}"
    expected_run_id = (
        "multi-supervisor:"
        + hashlib.sha256(
            f"{board.repo_root.resolve()}:{lane_name}".encode("utf-8")
        ).hexdigest()
    )
    expected_markers = {
        RUN_ID_ENV: expected_run_id,
        TARGET_ID_ENV: f"supervisor-track:{lane_name}",
        REPOSITORY_ROOT_ENV: str(board.repo_root.resolve()),
        STATE_ROOT_ENV: str(state_dir),
        RUN_ROOT_ENV: str(state_dir / "lifecycle-runs" / lane_name),
        FENCING_EPOCH_ENV: "0",
    }
    if (
        parent != coordinator_pid
        or group != supervisor_pid
        or session != supervisor_pid
        or start_ticks < coordinator_start_ticks
        or cwd != board.repo_root.resolve()
        or executable != Path(sys.executable).resolve()
        or not argv
        or Path(argv[0]).resolve(strict=False)
        != Path(sys.executable).resolve(strict=False)
        or any(environment.get(name) != value for name, value in expected_markers.items())
        or re.fullmatch(r"sha256:[0-9a-f]{64}", environment.get(PROFILE_ID_ENV, ""))
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            environment.get(CONFIGURATION_ROOT_ENV, ""),
        )
        is None
        or not _exact_process_option(
            argv, "--todo-path", str(board.path(board.taskboard_path))
        )
        or not _exact_process_option(
            argv, "--task-prefix", board.task_header_prefix
        )
        or not _exact_process_option(argv, "--state-dir", expected_state_arg)
        or not _exact_process_option(argv, "--state-prefix", state_prefix)
    ):
        return False
    if plan_bound:
        return bool(
            len(argv) > 9
            and argv[1:3] == ("-I", "-c")
            and argv[6]
            == (
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_supervisor"
            )
            and argv[7]
            == "sha256:" + hashlib.sha256(argv[3].encode("utf-8")).hexdigest()
            and re.fullmatch(r"sha256:[0-9a-f]{64}", argv[8]) is not None
            and _exact_process_option(
                argv, "--plan-bound-accepted-tree-root", str(board.repo_root)
            )
            and _exact_process_option(
                argv, "--plan-bound-source-head", repository_commit
            )
            and _exact_process_option(
                argv, "--plan-bound-source-tree", repository_tree
            )
            and _exact_process_option(argv, "--task-shard-count", "1")
            and _exact_process_option(argv, "--task-shard-index", "0")
        )
    return bool(
        len(argv) > 2
        and Path(argv[1]).resolve(strict=False)
        == board.path(IMPLEMENTATION_ENTRY_PATH.as_posix())
        and _exact_process_option(
            argv, "--task-shard-count", str(board.max_lanes)
        )
        and _exact_process_option(argv, "--task-shard-index", str(lane_index))
    )


def _lane_statuses_ready(
    board: ConfiguredBoard,
    paths: Sequence[Path],
    *,
    started_at_ms: int,
    now_ms: int,
    coordinator_pid: int,
    coordinator_start_ticks: int,
    repository_commit: str,
    repository_tree: str,
) -> bool:
    """Require every configured lane's fresh status and exact live identity."""

    lane_pids: set[int] = set()
    for lane_index, path in enumerate(paths):
        try:
            payload, evidence = _read_stable_regular_json(path)
        except _StableArtifactReadError:
            return False
        if payload is None or (
            evidence.get("state") != "present"
            or int(evidence.get("link_count", -1)) != 1
            or int(evidence.get("uid", -1)) != os.geteuid()
            or stat.S_IMODE(int(evidence.get("mode", 0))) != 0o600
            or payload.get("schema")
            != (
                "ipfs_accelerate_py.agent_supervisor."
                "todo_implementation_supervisor.supervisor"
            )
            or str(payload.get("status") or "")
            not in {
                "starting",
                "running",
                "restarting",
                "agentic_maintenance_started",
            }
            or payload.get("repo_root") != str(board.repo_root)
            or payload.get("task_prefix") != board.task_header_prefix
            or payload.get("state_prefix")
            != f"{_slug(board.task_prefix)}_lane_{lane_index}"
        ):
            return False
        updated_at = _parse_status_timestamp(payload.get("updated_at"))
        try:
            supervisor_pid = int(payload.get("supervisor_pid") or 0)
        except (TypeError, ValueError):
            return False
        if (
            updated_at is None
            or supervisor_pid < 2
            or supervisor_pid in lane_pids
        ):
            return False
        updated_at_ms = int(updated_at.timestamp() * 1000)
        if (
            updated_at_ms < started_at_ms
            or updated_at_ms > now_ms + 5_000
            or now_ms - updated_at_ms > COORDINATOR_STATUS_MAX_AGE_MS
        ):
            return False
        if not _configured_lane_process_ready(
            board,
            lane_index=lane_index,
            supervisor_pid=supervisor_pid,
            coordinator_pid=coordinator_pid,
            coordinator_start_ticks=coordinator_start_ticks,
            repository_commit=repository_commit,
            repository_tree=repository_tree,
        ):
            return False
        lane_pids.add(supervisor_pid)
    return True


def _publish_coordinator_launch_attestation(
    board: ConfiguredBoard,
    *,
    launch_session_id: str,
    status_path: Path,
) -> dict[str, Any]:
    expected_path = _expected_coordinator_status_path(board, launch_session_id)
    if status_path != expected_path:
        raise ConfiguredBoardError("coordinator status path differs from its session")
    head, tree = _git_identity(board.repo_root)
    adapter = LinuxProcessAdapter()
    try:
        _parent, _group, _session, started = adapter._stat(os.getpid())  # noqa: SLF001
        argv = adapter._argv(os.getpid())  # noqa: SLF001
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfiguredBoardError(
            "cannot observe coordinator process birth"
        ) from exc
    now_ms = int(time.time() * 1000)
    unsigned = {
        "schema": COORDINATOR_STATUS_SCHEMA,
        "repository_commit": head,
        "repository_tree": tree,
        "configuration_revision": board.configuration_revision,
        "board_namespace": board.board_namespace,
        "launch_session_id": launch_session_id,
        "lifecycle_profile_id": str(os.environ.get(PROFILE_ID_ENV) or ""),
        "coordinator_pid": os.getpid(),
        "coordinator_process_start_ticks": started,
        "coordinator_argv_cid": content_identity({"argv": list(argv)}),
        "started_at_ms": now_ms,
        "attested_at_ms": now_ms,
        "phase": "launch_attested",
        "lane_status_paths": [
            str(path) for path in _coordinator_lane_status_paths(board)
        ],
    }
    payload = {**unsigned, "receipt_cid": content_identity(unsigned)}
    _atomic_publish_coordinator_status(status_path, payload)
    return payload


def _bind_foreground_wave_pid(plan: dict[str, Any], board: ConfiguredBoard) -> None:
    """Keep the child runner PID distinct from the outer coordinator marker."""

    argv = plan.get("argv")
    if not isinstance(argv, list) or argv.count("--master-pid-path") != 1:
        raise ConfiguredBoardError("coordinator runner master PID binding is ambiguous")
    index = argv.index("--master-pid-path")
    if index + 1 >= len(argv):
        raise ConfiguredBoardError("coordinator runner master PID binding is incomplete")
    wave_path = board.path(board.runtime_paths["state"]) / "configured-board-wave.pid"
    argv[index + 1] = str(wave_path)
    plan["master_pid_path"] = str(wave_path)


def _prepare_coordinator_lane_status_permissions(board: ConfiguredBoard) -> None:
    """Prepare exact owner-private lane directories and status projections."""

    for path in _coordinator_lane_status_paths(board):
        lane_directory = _ensure_plan_bound_runtime_directory(
            board.repo_root,
            path.parent,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise ConfiguredBoardError(
                "private lane directory admission requires no-follow access"
            )
        flags |= nofollow
        try:
            directory_descriptor = os.open(lane_directory, flags)
        except OSError as exc:
            raise ConfiguredBoardError(
                "cannot open configured lane directory safely"
            ) from exc
        try:
            opened_directory = os.fstat(directory_descriptor)
            observed_directory = os.lstat(lane_directory)
            if (
                not stat.S_ISDIR(opened_directory.st_mode)
                or stat.S_ISLNK(observed_directory.st_mode)
                or int(opened_directory.st_uid) != os.geteuid()
                or int(observed_directory.st_uid) != os.geteuid()
                or (int(opened_directory.st_dev), int(opened_directory.st_ino))
                != (int(observed_directory.st_dev), int(observed_directory.st_ino))
                or stat.S_IMODE(opened_directory.st_mode) != 0o700
                or stat.S_IMODE(observed_directory.st_mode) != 0o700
            ):
                raise ConfiguredBoardError(
                    "configured lane directory must be an exact owner-private directory"
                )
        finally:
            os.close(directory_descriptor)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ConfiguredBoardError(
                "cannot inspect pre-existing lane status safely"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            observed = os.lstat(path)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_nlink) != 1
                or int(opened.st_uid) != os.geteuid()
                or stat.S_ISLNK(observed.st_mode)
                or (int(opened.st_dev), int(opened.st_ino))
                != (int(observed.st_dev), int(observed.st_ino))
            ):
                raise ConfiguredBoardError(
                    "pre-existing lane status is not a safe owned file"
                )
            os.fchmod(descriptor, 0o600)
            private = os.fstat(descriptor)
            if stat.S_IMODE(private.st_mode) != 0o600:
                raise ConfiguredBoardError(
                    "pre-existing lane status could not be made private"
                )
        finally:
            os.close(descriptor)


def _open_plan_bound_coordinator_log(log_path: Path):
    """Open one append-only log without following or accepting hardlinks."""

    path = Path(log_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ConfiguredBoardError(
            "cannot open detached coordinator log safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        observed = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or int(opened.st_uid) != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (int(opened.st_dev), int(opened.st_ino))
            != (int(observed.st_dev), int(observed.st_ino))
            or stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or int(observed.st_nlink) != 1
            or int(observed.st_uid) != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise ConfiguredBoardError(
                "detached coordinator log is not a stable single-link file"
            )
        return os.fdopen(descriptor, "ab", closefd=True)
    except Exception:
        os.close(descriptor)
        raise


def _reserve_coordinator_pid_projection(pid_path: Path) -> tuple[int, tuple[int, int]]:
    """Exclusively reserve a no-follow PID artifact before process creation."""

    path = Path(pid_path)
    with serialized_lock_update(path):
        try:
            existing = os.lstat(path)
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise ConfiguredBoardError(
                "cannot inspect detached coordinator PID projection"
            ) from exc
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                reason = "symbolic link"
            elif not stat.S_ISREG(existing.st_mode):
                reason = "non-regular file"
            elif int(existing.st_nlink) != 1:
                reason = "hardlinked file"
            else:
                reason = "existing owned file"
            raise ConfiguredBoardError(
                "detached coordinator PID projection is an unsafe " + reason
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise ConfiguredBoardError(
                "cannot exclusively reserve detached coordinator PID projection"
            ) from exc
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or int(opened.st_uid) != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise ConfiguredBoardError(
                "detached coordinator PID reservation is not a single-link file"
            )
        return descriptor, (int(opened.st_dev), int(opened.st_ino))


def _publish_reserved_coordinator_pid(
    pid_path: Path,
    descriptor: int,
    reserved_identity: tuple[int, int],
    pid: int,
) -> None:
    """Publish an exact PID only while the reserved pathname still owns the fd."""

    payload = f"{int(pid)}\n".encode("ascii")
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short PID projection write")
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        observed = os.lstat(pid_path)
        if (
            (int(opened.st_dev), int(opened.st_ino)) != reserved_identity
            or (int(observed.st_dev), int(observed.st_ino))
            != reserved_identity
            or stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or int(observed.st_nlink) != 1
            or int(opened.st_uid) != os.geteuid()
            or int(observed.st_uid) != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or stat.S_IMODE(observed.st_mode) != 0o600
            or int(observed.st_size) != len(payload)
        ):
            raise ConfiguredBoardError(
                "detached coordinator PID projection changed during publication"
            )
    except OSError as exc:
        raise ConfiguredBoardError(
            "cannot publish detached coordinator PID projection"
        ) from exc


def _repair_unreaped_coordinator_pid_projection(
    pid_path: Path,
    descriptor: int,
    reserved_identity: tuple[int, int],
    pid: int,
) -> None:
    """Repair the reserved projection with the exact known unreaped PID.

    This is used only after process-group termination failed.  It never opens
    a replacement pathname for writing: the original exclusive descriptor and
    its device/inode identity remain the authority boundary.
    """

    try:
        opened = os.fstat(descriptor)
        observed = os.lstat(pid_path)
        if (
            (int(opened.st_dev), int(opened.st_ino)) != reserved_identity
            or (int(observed.st_dev), int(observed.st_ino))
            != reserved_identity
            or stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or int(observed.st_nlink) != 1
            or int(opened.st_uid) != os.geteuid()
            or int(observed.st_uid) != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise ConfiguredBoardError(
                "cannot preserve unreaped coordinator PID in a changed projection"
            )
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _publish_reserved_coordinator_pid(
            pid_path,
            descriptor,
            reserved_identity,
            pid,
        )
    except OSError as exc:
        raise ConfiguredBoardError(
            f"cannot preserve unreaped coordinator PID {int(pid)}"
        ) from exc


def _publish_foreground_unreaped_coordinator_pid(
    board: ConfiguredBoard,
    pid: int,
) -> Path:
    """Create one secure recovery projection for an unreaped foreground PID."""

    state_dir = _ensure_plan_bound_runtime_directory(
        board.repo_root,
        board.path(board.runtime_paths["state"]),
    )
    pid_path = state_dir / f"configured-board-unreaped-{int(pid)}.pid"
    descriptor, identity = _reserve_coordinator_pid_projection(pid_path)
    try:
        _publish_reserved_coordinator_pid(
            pid_path,
            descriptor,
            identity,
            pid,
        )
    except BaseException:
        _remove_reserved_coordinator_pid(pid_path, identity)
        raise
    finally:
        os.close(descriptor)
    return pid_path


def _remove_reserved_coordinator_pid(
    pid_path: Path,
    reserved_identity: tuple[int, int],
) -> None:
    """Remove only the still-identical empty reservation after launch failure."""

    with serialized_lock_update(pid_path):
        try:
            observed = os.lstat(pid_path)
        except FileNotFoundError:
            return
        if (
            (int(observed.st_dev), int(observed.st_ino)) == reserved_identity
            and stat.S_ISREG(observed.st_mode)
            and int(observed.st_nlink) == 1
            and int(observed.st_uid) == os.geteuid()
            and stat.S_IMODE(observed.st_mode) == 0o600
        ):
            pid_path.unlink()


def _materialize_plan_bound_control_plane(
    board: ConfiguredBoard,
) -> tuple[
    AgentImplementationControlPlanePin,
    AgentImplementationSealedControlPlane,
    Path,
]:
    """Seal one clean accepted HEAD outside the candidate repository."""

    accepted_tree_root = Path(__file__).absolute().parents[3]
    if board.repo_root != accepted_tree_root:
        raise ConfiguredBoardError(
            "plan-bound coordinator repo root is not the accepted module tree"
        )
    source_head, source_tree = _git_identity(accepted_tree_root)
    if _eaaef_plan_bound_profile(board):
        # EAAEF authority is signed only after the tracked source/configuration
        # is frozen.  The tracked config therefore names stable registry paths,
        # never the post-freeze receipt CIDs (which would form a cryptographic
        # fixed-point cycle).  Re-open and verify those create-once records
        # before sealing the exact accepted archive for this coordinator.
        live_seal = board.payload.get("configured_board_live_seal")
        if not isinstance(live_seal, Mapping):
            raise ConfiguredBoardError(
                "EAAEF configured_board_live_seal is absent"
            )
        from ..validation.external_agent_bootstrap_admission import (
            ExternalAgentBootstrapAdmissionError,
            external_agent_bootstrap_admission_relative_path,
            verify_external_agent_bootstrap_admission,
        )
        from ..validation.external_agent_configured_board_capsule import (
            ExternalAgentConfiguredBoardCapsuleError,
            _read_stable_repo_json,
            external_agent_configured_board_launch_capsule_relative_path,
            verify_external_agent_configured_board_live_seal,
        )

        try:
            registry_prefix = str(live_seal.get("authority_registry_prefix") or "")
            admission_path = external_agent_bootstrap_admission_relative_path(
                source_head,
                registry_prefix=registry_prefix,
            )
            admission_payload, _admission_evidence = _read_stable_repo_json(
                board.repo_root,
                admission_path.as_posix(),
                noun="bootstrap admission receipt",
            )
            admission = verify_external_agent_bootstrap_admission(
                admission_payload,
                trusted_operator_dids=tuple(
                    live_seal.get("trusted_operator_dids") or ()
                ),
                trusted_security_reviewer_dids=tuple(
                    live_seal.get("trusted_security_reviewer_dids") or ()
                ),
                now_ms=int(time.time() * 1000),
            )
            capsule_path = external_agent_configured_board_launch_capsule_relative_path(
                source_head,
                str(admission["plan_root_cid"]),
                registry_prefix=registry_prefix,
            )
            capsule_payload, _capsule_evidence = _read_stable_repo_json(
                board.repo_root,
                capsule_path.as_posix(),
                noun="configured-board launch capsule",
            )
            raw_pin = capsule_payload["accepted_control_plane_pin"]
            if not isinstance(raw_pin, dict):
                raise TypeError("pin is not an object")
            pin = AgentImplementationControlPlanePin(**raw_pin)
            sealed = None
            try:
                verify_external_agent_configured_board_live_seal(
                    live_seal,
                    repo_root=board.repo_root,
                    configuration_root=board.configuration_root,
                    expected_source_head=source_head,
                    expected_source_tree=source_tree,
                    accepted_control_plane_pin=pin,
                    now_ms=int(time.time() * 1000),
                )
                sealed = seal_agent_implementation_control_plane_capsule(pin)
                verify_agent_implementation_sealed_control_plane(
                    pin, sealed.descriptor
                )
            except (
                ExternalAgentConfiguredBoardCapsuleError,
                OSError,
                ValueError,
            ) as exc:
                if sealed is not None:
                    try:
                        os.close(sealed.descriptor)
                    except OSError:
                        pass
                raise ConfiguredBoardError(
                    f"EAAEF configured-board live seal rejected: {exc}"
                ) from exc
            return pin, sealed, Path(pin.capsule_root).parent
        except (
            ExternalAgentBootstrapAdmissionError,
            ExternalAgentConfiguredBoardCapsuleError,
            ConfiguredBoardError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            if not _eaaef_host_receipt_admitted(
                board.repo_root,
                "EAAEF-191",
                expected_source_head=source_head,
                expected_source_tree=source_tree,
            ):
                if isinstance(exc, ConfiguredBoardError):
                    raise
                raise ConfiguredBoardError(
                    "EAAEF configured-board capsule has no canonical pin"
                ) from exc
            # Independently signed EAAEF-191 admits launch while create-once
            # bootstrap admission/capsule receipts remain unpublished.
    capsule_parent = Path(
        tempfile.mkdtemp(prefix="asref-configured-control-plane-")
    )
    try:
        pin = materialize_agent_implementation_control_plane_capsule(
            source_root=accepted_tree_root,
            capsule_parent=capsule_parent,
            source_head=source_head,
            source_tree=source_tree,
            allow_dirty_worktree=(
                _eaaef_plan_bound_profile(board)
                and _eaaef_host_receipt_admitted(
                    board.repo_root,
                    "EAAEF-191",
                    expected_source_head=source_head,
                    expected_source_tree=source_tree,
                )
            ),
        )
        sealed = seal_agent_implementation_control_plane_capsule(pin)
        if (
            pin.source_head != source_head
            or pin.source_tree != source_tree
            or verify_agent_implementation_sealed_control_plane(
                pin,
                sealed.descriptor,
            )
            != sealed.executable_path
        ):
            raise ConfiguredBoardError(
                "accepted control-plane capsule identity drifted"
            )
        return pin, sealed, capsule_parent
    except BaseException:
        try:
            shutil.rmtree(capsule_parent)
        except OSError:
            pass
        raise


def _plan_bound_coordinator_module_argv(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
    pin: AgentImplementationControlPlanePin,
    sealed: AgentImplementationSealedControlPlane,
    capsule_parent: Path,
    launch_session_id: str = "",
    coordinator_status_path: Path | None = None,
) -> list[str]:
    argv = [
        "--repo-root",
        str(board.repo_root),
        "--config",
        str(board.config_path),
        "--accepted-tree-root",
        str(board.repo_root),
        "--accepted-control-plane-pin-json",
        accepted_control_plane_pin_json(pin),
        "--accepted-control-plane-fd",
        str(sealed.descriptor),
        "--accepted-control-plane-capsule-parent",
        str(capsule_parent),
    ]
    if launch_session_id or coordinator_status_path is not None:
        if (
            re.fullmatch(r"[0-9a-f]{64}", launch_session_id) is None
            or coordinator_status_path is None
        ):
            raise ConfiguredBoardError(
                "coordinator launch session binding is incomplete"
            )
        argv.extend(
            [
                "--coordinator-launch-session",
                launch_session_id,
                "--coordinator-status-path",
                str(coordinator_status_path),
            ]
        )
    argv.extend(
        [
            "launch",
            "--foreground",
            "--duration-seconds",
            str(duration_seconds),
        ]
    )
    if implement:
        argv.append("--implement")
    return argv


def _cleanup_plan_bound_control_plane(
    pin: AgentImplementationControlPlanePin,
    capsule_parent: Path,
) -> None:
    """Remove only the uniquely-created private capsule parent after fencing."""

    parent = Path(capsule_parent)
    capsule = Path(pin.capsule_root)
    if (
        not parent.is_absolute()
        or parent.parent != Path(tempfile.gettempdir())
        or not parent.name.startswith("asref-configured-control-plane-")
        or capsule.parent != parent
    ):
        return
    try:
        for entry in parent.rglob("*"):
            observed = os.lstat(entry)
            if stat.S_ISLNK(observed.st_mode) or int(observed.st_uid) != os.geteuid():
                return
        directories = sorted(
            (entry for entry in parent.rglob("*") if entry.is_dir()),
            key=lambda entry: len(entry.parts),
            reverse=True,
        )
        for directory in directories:
            os.chmod(directory, 0o700)
        os.chmod(parent, 0o700)
        shutil.rmtree(parent)
    except OSError:
        return


def _plan_bound_coordinator_environment(
    board: ConfiguredBoard | None = None,
) -> dict[str, str]:
    """Retain only locale and exact live database identities across reseal."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if name
        in {
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TZ",
            STATE_STORE_LIVE_GENERATION_ENV,
            STATE_LIVE_SCHEMA_REVISION_ENV,
            TRUSTED_DUCKDB_HOME_ENV,
            TRUSTED_PYTHON_USER_BASE_ENV,
        }
    }
    for name in (
        STATE_STORE_LIVE_GENERATION_ENV,
        STATE_LIVE_SCHEMA_REVISION_ENV,
    ):
        value = str(environment.get(name, "") or "")
        if value and (
            re.fullmatch(r"[0-9]{1,20}", value) is None
            or int(value) > 2**63 - 1
        ):
            raise ConfiguredBoardError(
                f"plan-bound coordinator {name} is not a bounded identity"
            )
    trusted_home = str(environment.get(TRUSTED_DUCKDB_HOME_ENV, "") or "")
    if trusted_home:
        try:
            environment.update(
                _trusted_duckdb_runtime_environment(
                    os.environ,
                    repository_root=Path(__file__).absolute().parents[3],
                )
            )
        except ValueError as exc:
            raise ConfiguredBoardError(
                "plan-bound coordinator trusted DuckDB HOME is invalid"
            ) from exc
    else:
        environment.pop(TRUSTED_PYTHON_USER_BASE_ENV, None)
        for name in TRUSTED_RUNTIME_CACHE_ENV_NAMES:
            environment.pop(name, None)
    environment["PATH"] = (
        _eaaef_plan_bound_provider_path(board)
        if board is not None
        else "/usr/bin:/bin"
    )
    return environment


def _require_plan_bound_process_launch_policy(
    board: ConfiguredBoard,
    *,
    implement: bool,
) -> None:
    """Fail before process birth when a configured board prohibits live launch."""

    raw_policy = board.payload.get("launch_policy")
    if raw_policy is None:
        if _eaaef_plan_bound_profile(board):
            raise ConfiguredBoardError(
                "EAAEF configured-board live launch requires an explicit "
                "launch_policy authority boundary"
            )
        return
    if not isinstance(raw_policy, Mapping):
        raise ConfiguredBoardError("launch_policy must be an object")
    if _eaaef_plan_bound_profile(board):
        expected_policy_fields = {
            "blockers",
            "bypass_prohibited",
            "dry_run_allowed",
            "live_multi_supervisor_allowed",
            "live_single_supervisor_allowed",
            "materialize_allowed",
            "verify_allowed",
        }
        if set(raw_policy) != expected_policy_fields:
            raise ConfiguredBoardError(
                "EAAEF launch_policy fields do not match the closed authority "
                "contract"
            )
        for field in expected_policy_fields - {"blockers"}:
            if type(raw_policy.get(field)) is not bool:
                raise ConfiguredBoardError(
                    f"EAAEF launch_policy.{field} must be boolean"
                )
    if raw_policy.get("bypass_prohibited") is not True:
        raise ConfiguredBoardError(
            "configured-board live launch requires bypass_prohibited=true"
        )
    if raw_policy.get("live_multi_supervisor_allowed") is not True:
        raise ConfiguredBoardError(
            "configured-board live multi-supervisor launch is prohibited by policy"
        )
    raw_blockers = raw_policy.get("blockers")
    if not isinstance(raw_blockers, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_blockers
    ):
        raise ConfiguredBoardError(
            "launch_policy.blockers must be a list of nonempty strings"
        )
    if raw_blockers:
        raise ConfiguredBoardError(
            "configured-board live launch retains policy blockers: "
            + "; ".join(raw_blockers)
        )
    if not implement:
        return
    container_policy = board.payload.get("container_policy")
    if not isinstance(container_policy, Mapping):
        raise ConfiguredBoardError(
            "implementation launch requires a container_policy object"
        )
    if container_policy.get("live_dispatch_allowed") is not True:
        raise ConfiguredBoardError(
            "implementation launch is prohibited by container live-dispatch policy"
        )
    if str(container_policy.get("bootstrap_image_status") or "") != "admitted":
        raise ConfiguredBoardError(
            "implementation launch requires an admitted immutable worker image"
        )


def _terminate_plan_bound_coordinator(process: subprocess.Popen[bytes]) -> None:
    """Boundedly terminate and reap the coordinator's dedicated process group."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2.0)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired as exc:
        if process.poll() is None:
            raise ConfiguredBoardError(
                "configured-board coordinator process-group "
                f"{int(process.pid)} remained live after SIGKILL and could not "
                "be reaped"
            ) from exc


def _launch_foreground_plan_bound_coordinator(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
) -> int:
    _require_plan_bound_process_launch_policy(board, implement=implement)
    pin, sealed, capsule_parent = _materialize_plan_bound_control_plane(board)
    process: subprocess.Popen[bytes] | None = None
    preserve_capsule_for_unreaped_process = False
    try:
        command = build_sealed_control_plane_module_command(
            python_executable=sys.executable,
            pin=pin,
            descriptor=sealed.descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.runtime."
                "configured_board_scheduler"
            ),
            argv=_plan_bound_coordinator_module_argv(
                board,
                implement=implement,
                duration_seconds=duration_seconds,
                pin=pin,
                sealed=sealed,
                capsule_parent=capsule_parent,
            ),
        )
        environment = _plan_bound_coordinator_environment(board)
        process = subprocess.Popen(
            command,
            cwd=board.repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(sealed.descriptor,),
        )
        return int(process.wait())
    except BaseException as exc:
        if process is not None:
            try:
                _terminate_plan_bound_coordinator(process)
            except ConfiguredBoardError as termination_error:
                preserve_capsule_for_unreaped_process = True
                exc.add_note(str(termination_error))
                try:
                    recovery_path = _publish_foreground_unreaped_coordinator_pid(
                        board,
                        process.pid,
                    )
                    exc.add_note(
                        "unreaped coordinator recovery projection: "
                        f"{recovery_path}; preserved capsule: {capsule_parent}"
                    )
                except (ConfiguredBoardError, OSError) as recovery_error:
                    exc.add_note(
                        "unreaped coordinator recovery projection failed: "
                        f"{recovery_error}; preserved capsule: {capsule_parent}"
                    )
        raise
    finally:
        os.close(sealed.descriptor)
        if not preserve_capsule_for_unreaped_process:
            _cleanup_plan_bound_control_plane(pin, capsule_parent)


def _launch_detached_plan_bound_coordinator(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
) -> dict[str, Any]:
    """Detach the outer coordinator, never an individual finite wave."""

    _require_plan_bound_process_launch_policy(board, implement=implement)
    state_dir = _ensure_plan_bound_runtime_directory(
        board.repo_root,
        board.path(board.runtime_paths["state"]),
    )
    log_dir = _ensure_plan_bound_runtime_directory(
        board.repo_root,
        board.path(board.runtime_paths["logs"]),
    )
    stamp = utc_run_stamp()
    log_path = log_dir / f"configured-board-{stamp}.log"
    pid_path = state_dir / "configured-board-master.pid"
    accepted_tree_root = Path(__file__).absolute().parents[3]
    if board.repo_root != accepted_tree_root:
        raise ConfiguredBoardError(
            "detached coordinator repo root is not the accepted module tree"
        )
    entry = accepted_tree_root / CONFIGURED_SCHEDULER_ENTRY_PATH
    _lexical_repo_artifact(accepted_tree_root, pid_path)
    source_head, _source_tree = _git_identity(accepted_tree_root)
    for authority_path in (
        entry,
        board.config_path,
        board.path(board.taskboard_path),
    ):
        _tracked_head_snapshot(
            repo_root=accepted_tree_root,
            path=authority_path,
            source_head=source_head,
        )
    descriptor, reserved_identity = _reserve_coordinator_pid_projection(
        pid_path
    )
    process: subprocess.Popen[bytes] | None = None
    pin: AgentImplementationControlPlanePin | None = None
    sealed: AgentImplementationSealedControlPlane | None = None
    capsule_parent: Path | None = None
    try:
        pin, sealed, capsule_parent = _materialize_plan_bound_control_plane(
            board
        )
        command = build_sealed_control_plane_module_command(
            python_executable=sys.executable,
            pin=pin,
            descriptor=sealed.descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.runtime."
                "configured_board_scheduler"
            ),
            argv=_plan_bound_coordinator_module_argv(
                board,
                implement=implement,
                duration_seconds=duration_seconds,
                pin=pin,
                sealed=sealed,
                capsule_parent=capsule_parent,
            ),
        )
        with _open_plan_bound_coordinator_log(log_path) as stream:
            launch_environment = _plan_bound_coordinator_environment(board)
            process = subprocess.Popen(
                command,
                cwd=accepted_tree_root,
                env=launch_environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(sealed.descriptor,),
            )
        _publish_reserved_coordinator_pid(
            pid_path,
            descriptor,
            reserved_identity,
            process.pid,
        )
    except BaseException as exc:
        fenced = True
        if process is not None:
            fenced = _fence_exact_coordinator_group(
                process,
                observed_start_ticks=0,
            )
        if not fenced:
            exc.add_note(
                "detached coordinator failure could not be exactly fenced; "
                "preserving PID projection and control-plane capsule"
            )
            assert process is not None
            try:
                _repair_unreaped_coordinator_pid_projection(
                    pid_path,
                    descriptor,
                    reserved_identity,
                    process.pid,
                )
            except ConfiguredBoardError as projection_error:
                exc.add_note(str(projection_error))
            raise
        _remove_reserved_coordinator_pid(pid_path, reserved_identity)
        if capsule_parent is not None:
            try:
                shutil.rmtree(capsule_parent)
            except OSError:
                pass
        raise
    finally:
        os.close(descriptor)
        if sealed is not None:
            os.close(sealed.descriptor)
    assert process is not None
    return {
        "coordinator_pid": process.pid,
        "coordinator_pid_path": str(pid_path),
        "coordinator_log": str(log_path),
    }


def _fence_exact_coordinator_group(
    process: subprocess.Popen[bytes],
    *,
    observed_start_ticks: int,
) -> bool:
    """Fence the exact unreaped child handle without group-signal races."""

    if process.poll() is not None:
        return True
    # Popen retains the exact, unreaped direct-child relationship.  When an
    # observed birth is available, require it before signaling.  Signaling a
    # process group after a separate /proc observation would introduce a
    # mutable-membership and PGID-reuse race, so use only the exact child
    # handle and let the coordinator's bounded shutdown fence its own lanes.
    if observed_start_ticks > 0:
        try:
            _parent, _group, _session, start_ticks = LinuxProcessAdapter._stat(  # noqa: SLF001
                process.pid
            )
        except (OSError, UnicodeError, ValueError):
            # The unreaped Popen handle is still an exact direct-child
            # identity even when procfs cannot be sampled during cleanup.
            pass
        else:
            if start_ticks != observed_start_ticks:
                return False
    try:
        process.terminate()
        process.wait(timeout=35.0)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                return False
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return False
    return process.poll() is not None


def _launch_detached_receipt_coordinator(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
) -> dict[str, Any]:
    """Launch one lifecycle-bound coordinator and admit all lane heartbeats."""

    state_dir = _ensure_plan_bound_runtime_directory(
        board.repo_root,
        board.path(board.runtime_paths["state"]),
    )
    log_dir = _ensure_plan_bound_runtime_directory(
        board.repo_root,
        board.path(board.runtime_paths["logs"]),
    )
    launch_session_id = secrets.token_hex(32)
    status_path = _expected_coordinator_status_path(board, launch_session_id)
    log_path = log_dir / (
        f"configured-board-{utc_run_stamp()}-{launch_session_id}.log"
    )
    pid_path = state_dir / "configured-board-master.pid"
    accepted_tree_root = Path(__file__).absolute().parents[3]
    if board.repo_root != accepted_tree_root:
        raise ConfiguredBoardError(
            "receipt coordinator repo root is not the accepted module tree"
        )
    entry = accepted_tree_root / CONFIGURED_SCHEDULER_ENTRY_PATH
    _lexical_repo_artifact(accepted_tree_root, pid_path)
    head, tree = _git_identity(accepted_tree_root)
    for authority_path in (
        entry,
        board.config_path,
        board.path(board.taskboard_path),
    ):
        _tracked_head_snapshot(
            repo_root=accepted_tree_root,
            path=authority_path,
            source_head=head,
        )
    try:
        os.lstat(status_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ConfiguredBoardError(
            "cannot inspect coordinator status destination"
        ) from exc
    else:
        raise ConfiguredBoardError("coordinator status destination already exists")

    descriptor, reserved_identity = _reserve_coordinator_pid_projection(pid_path)
    process: subprocess.Popen[bytes] | None = None
    sealed: AgentImplementationSealedControlPlane | None = None
    capsule_parent: Path | None = None
    process_identity: ProcessIdentity | None = None
    observed_start_ticks = 0
    try:
        pin, sealed, capsule_parent = _materialize_plan_bound_control_plane(board)
        command = build_sealed_control_plane_module_command(
            python_executable=sys.executable,
            pin=pin,
            descriptor=sealed.descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.runtime."
                "configured_board_scheduler"
            ),
            argv=_plan_bound_coordinator_module_argv(
                board,
                implement=implement,
                duration_seconds=duration_seconds,
                pin=pin,
                sealed=sealed,
                capsule_parent=capsule_parent,
                launch_session_id=launch_session_id,
                coordinator_status_path=status_path,
            ),
        )
        base_environment = _plan_bound_coordinator_environment(board)
        readiness_timeout_seconds = _coordinator_readiness_timeout_seconds(board)
        launch_attestation_max_age_ms = (
            _coordinator_launch_attestation_max_age_ms(board)
        )
        profile = LifecycleProfile(
            target_id=f"configured-board-coordinator:{board.board_namespace}",
            run_id=f"configured-board:{board.board_namespace}:{launch_session_id}",
            configuration_root=board.configuration_revision,
            repository_root=str(board.repo_root),
            state_root=str(state_dir),
            run_root=str(state_dir),
            argv=tuple(command),
            cwd=str(board.repo_root),
            environment=_plan_bound_profile_environment(base_environment),
            health_path=str(status_path),
            health_stale_ms=launch_attestation_max_age_ms,
        )
        launch_environment = _plan_bound_positive_child_environment(
            profile.launch_environment(0)
        )
        with _open_plan_bound_coordinator_log(log_path) as stream:
            process = subprocess.Popen(
                command,
                cwd=accepted_tree_root,
                env=launch_environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(sealed.descriptor,),
            )
        identity_deadline = time.monotonic() + 10.0
        adapter = LinuxProcessAdapter()
        while time.monotonic() < identity_deadline:
            if process.poll() is not None:
                raise ConfiguredBoardError(
                    "coordinator exited before process identity admission"
                )
            try:
                _parent, group, session, observed_start_ticks = (
                    adapter._stat(process.pid)  # noqa: SLF001
                )
                candidate = adapter._identity(process.pid, profile)  # noqa: SLF001
            except (
                FileNotFoundError,
                ProcessLookupError,
                ProcessIdentityMismatch,
                OSError,
                UnicodeError,
                ValueError,
            ):
                time.sleep(0.02)
                continue
            if (
                group == process.pid
                and session == process.pid
                and candidate.argv == profile.argv
                and candidate.cwd == str(board.repo_root)
            ):
                process_identity = candidate
                break
            time.sleep(0.02)
        if process_identity is None:
            raise ConfiguredBoardError(
                "coordinator process identity did not become admissible"
            )
        _publish_reserved_coordinator_pid(
            pid_path,
            descriptor,
            reserved_identity,
            process.pid,
        )
        status: dict[str, Any] | None = None
        expected_lane_paths = _coordinator_lane_status_paths(board)
        readiness_deadline = time.monotonic() + readiness_timeout_seconds
        while time.monotonic() < readiness_deadline:
            if process.poll() is not None:
                raise ConfiguredBoardError(
                    "coordinator exited before lane readiness admission"
                )
            try:
                candidate_status = _read_coordinator_status(status_path)
            except ConfiguredBoardError:
                time.sleep(0.1)
                continue
            now_ms = int(time.time() * 1000)
            if (
                candidate_status.get("schema") == COORDINATOR_STATUS_SCHEMA
                and candidate_status.get("repository_commit") == head
                and candidate_status.get("repository_tree") == tree
                and candidate_status.get("configuration_revision")
                == board.configuration_revision
                and candidate_status.get("board_namespace")
                == board.board_namespace
                and candidate_status.get("launch_session_id")
                == launch_session_id
                and candidate_status.get("lifecycle_profile_id")
                == profile.profile_id
                and candidate_status.get("coordinator_pid") == process.pid
                and candidate_status.get("coordinator_process_start_ticks")
                == process_identity.start_time_ticks
                and candidate_status.get("coordinator_argv_cid")
                == content_identity({"argv": list(profile.argv)})
                and candidate_status.get("phase") == "launch_attested"
                and candidate_status.get("lane_status_paths")
                == [str(path) for path in expected_lane_paths]
                and type(candidate_status.get("started_at_ms")) is int
                and type(candidate_status.get("attested_at_ms")) is int
                and candidate_status["started_at_ms"]
                <= candidate_status["attested_at_ms"]
                <= now_ms + 5_000
                and now_ms - candidate_status["attested_at_ms"]
                <= launch_attestation_max_age_ms
                and _lane_statuses_ready(
                    board,
                    expected_lane_paths,
                    started_at_ms=candidate_status["started_at_ms"],
                    now_ms=now_ms,
                    coordinator_pid=process.pid,
                    coordinator_start_ticks=process_identity.start_time_ticks,
                    repository_commit=head,
                    repository_tree=tree,
                )
            ):
                status = candidate_status
                break
            time.sleep(0.1)
        if status is None:
            raise ConfiguredBoardError(
                "coordinator launch attestation or lane heartbeat readiness "
                "timed out"
            )
        if _git_identity(accepted_tree_root) != (head, tree):
            raise ConfiguredBoardError(
                "repository identity changed during coordinator launch"
            )
        try:
            final_process_identity = adapter._identity(  # noqa: SLF001
                process.pid,
                profile,
            )
        except (
            FileNotFoundError,
            ProcessLookupError,
            ProcessIdentityMismatch,
            OSError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise ConfiguredBoardError(
                "coordinator identity disappeared after lane readiness"
            ) from exc
        if process.poll() is not None or final_process_identity != process_identity:
            raise ConfiguredBoardError(
                "coordinator identity changed after lane readiness"
            )
        argv_cid = content_identity({"argv": list(profile.argv)})
        unsigned_receipt = {
            "schema": COORDINATOR_LAUNCH_RECEIPT_SCHEMA,
            "repository_commit": head,
            "repository_tree": tree,
            "configuration_revision": board.configuration_revision,
            "board_namespace": board.board_namespace,
            "launch_session_id": launch_session_id,
            "coordinator_pid": process.pid,
            "coordinator_pid_path": str(pid_path),
            "coordinator_log": str(log_path),
            "coordinator_status_path": str(status_path),
            "coordinator_status_cid": status["receipt_cid"],
            "coordinator_profile": profile.to_dict(),
            "coordinator_process_identity": process_identity.to_dict(),
            "coordinator_argv_cid": argv_cid,
        }
        return {
            **unsigned_receipt,
            "receipt_cid": content_identity(unsigned_receipt),
        }
    except BaseException as exc:
        fenced = True
        if process is not None:
            fenced = _fence_exact_coordinator_group(
                process,
                observed_start_ticks=observed_start_ticks,
            )
        if not fenced:
            fence_error = ConfiguredBoardError(
                "receipt coordinator failure could not be exactly fenced; "
                "preserving PID projection and control-plane capsule"
            )
            assert process is not None
            try:
                _repair_unreaped_coordinator_pid_projection(
                    pid_path,
                    descriptor,
                    reserved_identity,
                    process.pid,
                )
            except ConfiguredBoardError as projection_error:
                fence_error.add_note(str(projection_error))
            raise fence_error from exc
        _remove_reserved_coordinator_pid(pid_path, reserved_identity)
        if capsule_parent is not None:
            try:
                shutil.rmtree(capsule_parent)
            except OSError:
                pass
        raise
    finally:
        os.close(descriptor)
        if sealed is not None:
            os.close(sealed.descriptor)


def _run_plan_bound_coordinator(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None,
    accepted_control_plane_descriptor: int = -1,
) -> int:
    """Publish and execute fresh exact waves until drain or the run bound."""

    _require_plan_bound_process_launch_policy(board, implement=implement)
    from .multi_supervisor_runner import PLAN_BOUND_REPLAN_RETURN_CODE
    from .multi_supervisor_runner import main as multi_supervisor_main

    started = time.monotonic()
    base_stamp = utc_run_stamp()
    wave_index = 0
    while True:
        elapsed = time.monotonic() - started
        if math.isfinite(duration_seconds) and elapsed >= duration_seconds:
            return 0
        if (
            wave_index >= MAX_COORDINATOR_WAVES
            and not _eaaef_plan_bound_profile(board)
        ):
            print(
                json.dumps(
                    {
                        "valid": False,
                        "errors": [
                            "adaptive coordinator exceeded its wave bound"
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        try:
            current_board = load_configured_board(
                board.config_path,
                repo_root=board.repo_root,
            )
            _require_plan_bound_process_launch_policy(
                current_board,
                implement=implement,
            )
            if current_board.board_namespace != board.board_namespace:
                raise ConfiguredBoardError(
                    "coordinator configuration changed board namespace"
                )
            receipt = materialize_configured_board_execution_plan(current_board)
        except (ConfiguredBoardError, OSError, RuntimeError, ValueError) as exec_error:
            detail = str(exec_error)
            retryable = any(
                marker in detail
                for marker in (
                    "provider_infeasible",
                    "resource_infeasible",
                    "stale_capacity",
                )
            )
            print(
                json.dumps(
                    {
                        "valid": not retryable,
                        "errors": [f"adaptive_plan: {exec_error}"],
                        "retryable": retryable,
                        "wave_index": wave_index,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            if retryable:
                retry_seconds = float(
                    board.payload.get("poll_interval_seconds") or 5
                )
                time.sleep(max(1.0, min(retry_seconds, 30.0)))
                wave_index += 1
                continue
            return 2
        if receipt is None:
            print(
                json.dumps(
                    {
                        "schema": (
                            "ipfs_accelerate_py/agent-supervisor/"
                            "configured-board-coordinator-result@1"
                        ),
                        "board_namespace": board.board_namespace,
                        "waves_completed": wave_index,
                        "reason": "no_dependency_ready_retry_admissible_tasks",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                _eaaef_plan_bound_profile(board)
                and not math.isfinite(duration_seconds)
            ):
                retry_seconds = float(
                    board.payload.get("poll_interval_seconds") or 5
                )
                time.sleep(max(1.0, min(retry_seconds, 30.0)))
                wave_index += 1
                continue
            return 0
        remaining = (
            max(0.0, duration_seconds - elapsed)
            if math.isfinite(duration_seconds)
            else float("inf")
        )
        plan = configured_board_launch_plan(
            current_board,
            implement=implement,
            detach=False,
            duration_seconds=remaining,
            stamp=f"{base_stamp}-wave-{wave_index}",
            parallelism_receipt=receipt,
            accepted_control_plane_pin=accepted_control_plane_pin,
            accepted_control_plane_descriptor=(
                accepted_control_plane_descriptor
            ),
        )
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        _apply_configured_board_environment(plan)
        try:
            result = int(multi_supervisor_main(plan["argv"]))
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "valid": False,
                        "errors": [f"wave_dispatch: {type(exc).__name__}: {exc}"],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            raise
        if result == PLAN_BOUND_REPLAN_RETURN_CODE:
            wave_index += 1
            continue
        if result != 0:
            return result
        wave_index += 1


def _remove_owned_coordinator_pid(board: ConfiguredBoard) -> bool:
    """Remove only this coordinator's detached-launch PID projection."""

    pid_path = (
        board.path(board.runtime_paths["state"])
        / "configured-board-master.pid"
    )
    try:
        _lexical_repo_artifact(board.repo_root, pid_path)
        with serialized_lock_update(pid_path):
            payload, evidence = _read_stable_regular_bytes(
                pid_path,
                max_bytes=32,
            )
            if payload is None or not re.fullmatch(rb"[1-9][0-9]*\n", payload):
                return False
            recorded_pid = int(payload[:-1].decode("ascii"))
            if recorded_pid != os.getpid():
                return False
            observed = os.lstat(pid_path)
            if (
                evidence.get("state") != "present"
                or int(evidence.get("device", -1)) != int(observed.st_dev)
                or int(evidence.get("inode", -1)) != int(observed.st_ino)
                or stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or int(observed.st_nlink) != 1
                or int(observed.st_uid) != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o600
            ):
                return False
            pid_path.unlink()
            return True
    except (
        ConfiguredBoardError,
        _StableArtifactReadError,
        OSError,
        UnicodeError,
        ValueError,
    ):
        return False


def _repair_authoritative_board_projection_before_launch(
    *,
    config_path: Path,
    repo_root: Path,
    command: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Restore an opted-in immutable board projection before real launch.

    Preflight and dry-run commands remain read-only.  A real launch may repair
    only the closed, receipt-sealed drift class implemented by the projection
    repairer; every inconclusive case fails before scheduler validation or
    process creation.
    """

    if command != "launch" or dry_run:
        return {
            "enabled": False,
            "repaired": False,
            "reason_code": "read_only_command",
        }
    from .authoritative_board_projection import (
        BoardProjectionRepairError,
        repair_authoritative_board_projection,
    )

    try:
        return repair_authoritative_board_projection(
            config_path,
            repo_root=repo_root,
        )
    except BoardProjectionRepairError as exc:
        raise ConfiguredBoardError(
            f"authoritative board projection repair: {exc}"
        ) from exc


@contextlib.contextmanager
def _isolated_launch_receipt_stream() -> Any:
    """Keep the machine receipt on stdout and route every other writer away."""

    try:
        stdout_descriptor = 1
        stderr_descriptor = 2
        os.fstat(stdout_descriptor)
        os.fstat(stderr_descriptor)
    except OSError as exc:
        raise ConfiguredBoardError(
            "launch receipt descriptors are unavailable"
        ) from exc

    sys.stdout.flush()
    sys.stderr.flush()
    restore_descriptor = os.dup(stdout_descriptor)
    try:
        receipt_descriptor = os.dup(stdout_descriptor)
    except BaseException:
        os.close(restore_descriptor)
        raise
    receipt_stream = os.fdopen(
        receipt_descriptor,
        "w",
        encoding=getattr(sys.stdout, "encoding", None) or "utf-8",
        errors="strict",
        newline="\n",
        closefd=True,
    )
    redirected = False
    try:
        os.dup2(stderr_descriptor, stdout_descriptor)
        redirected = True
        with contextlib.redirect_stdout(sys.stderr):
            yield receipt_stream
    finally:
        try:
            if redirected:
                try:
                    sys.stdout.flush()
                finally:
                    os.dup2(restore_descriptor, stdout_descriptor)
        finally:
            os.close(restore_descriptor)
            receipt_stream.close()


def _run_parsed_command(
    args: argparse.Namespace,
    *,
    launch_receipt_stream: Any | None = None,
) -> int:
    control_plane_pin: AgentImplementationControlPlanePin | None = None
    control_plane_descriptor = -1
    control_plane_parent: Path | None = None
    live_context = None
    try:
        live_values = (
            bool(args.configured_board_live_capsule_pin_json),
            args.configured_board_live_capsule_fd >= 3,
            bool(args.configured_board_live_admission_json),
            bool(args.configured_board_live_native_launch_json),
            args.configured_board_live_native_fd >= 3,
        )
        bootstrap_values = (
            args.state_owner_bootstrap_fd >= 3,
            bool(str(args.state_owner_bootstrap_store_id or "").strip()),
        )
        root = Path(args.repo_root).resolve()
        requested_config = Path(args.config)
        if not requested_config.is_absolute():
            requested_config = root / requested_config
        exact_live_config = (
            Path(os.path.abspath(requested_config))
            == root / LGCVF_LIVE_CONFIG_PATH
        )
        if any(live_values) and not all(live_values):
            raise ConfiguredBoardError(
                "LGCVF configured-board live launch fields are incomplete"
            )
        if any(bootstrap_values) and not all(bootstrap_values):
            raise ConfiguredBoardError(
                "LGCVF state-owner bootstrap fields are incomplete"
            )
        if all(live_values) != all(bootstrap_values):
            raise ConfiguredBoardError(
                "LGCVF live capsule and state-owner bootstrap are bidirectional"
            )
        if exact_live_config and not all(live_values):
            raise ConfiguredBoardError(
                "the LGCVF Quack candidate requires its complete live capsule"
            )
        admitted_config_bytes: bytes | None = None
        if all(live_values):
            if not exact_live_config:
                raise ConfiguredBoardError(
                    "LGCVF live capsule cannot authorize a different config"
                )
            try:
                live_context = verify_lgcvf_configured_board_live_context(
                    capsule_pin_json=(
                        args.configured_board_live_capsule_pin_json
                    ),
                    capsule_descriptor=(
                        args.configured_board_live_capsule_fd
                    ),
                    admission_json=args.configured_board_live_admission_json,
                    native_launch_json=(
                        args.configured_board_live_native_launch_json
                    ),
                    native_descriptor=args.configured_board_live_native_fd,
                )
                from ...agent_implementation_route import (
                    read_lgcvf_configured_board_live_capsule_member,
                )

                admitted_config_bytes = (
                    read_lgcvf_configured_board_live_capsule_member(
                        live_context.capsule_pin,
                        live_context.capsule_descriptor,
                        live_context.capsule_pin.candidate_config_path,
                    )
                )
                disk_config_bytes, _disk_evidence = (
                    _read_stable_regular_bytes(
                        requested_config,
                        max_bytes=4_194_304,
                    )
                )
                if disk_config_bytes != admitted_config_bytes:
                    raise ValueError(
                        "repository config differs from its sealed capsule"
                    )
            except (OSError, ValueError) as exc:
                raise ConfiguredBoardError(
                    "LGCVF configured-board live launch binding is invalid"
                ) from exc
            projection_repair = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "authoritative-board-projection-repair@1"
                ),
                "enabled": False,
                "repaired": False,
                "reason_code": "policy_absent_in_sealed_config",
            }
        else:
            projection_repair = (
                _repair_authoritative_board_projection_before_launch(
                    config_path=args.config,
                    repo_root=args.repo_root,
                    command=str(args.command or ""),
                    dry_run=bool(getattr(args, "dry_run", False)),
                )
            )
        board = (
            load_configured_board(
                args.config,
                repo_root=args.repo_root,
                config_bytes=admitted_config_bytes,
            )
            if admitted_config_bytes is not None
            else load_configured_board(
                args.config,
                repo_root=args.repo_root,
            )
        )
        if live_context is not None and (
            "authoritative_board_projection_repair" in board.payload
            or board.board_namespace != LGCVF_LIVE_BOARD_NAMESPACE
        ):
            raise ConfiguredBoardError(
                "sealed LGCVF board identity differs from the live profile"
            )
        preflight = (
            preflight_configured_board(
                board,
                admitted_live_validator_sha256=str(
                    live_context.admission.validator_sha256
                ),
            )
            if live_context is not None
            else preflight_configured_board(board)
        )
        has_control_plane = bool(args.accepted_control_plane_pin_json)
        has_descriptor = args.accepted_control_plane_fd >= 3
        has_parent = args.accepted_control_plane_capsule_parent is not None
        if len({has_control_plane, has_descriptor, has_parent}) != 1:
            raise ConfiguredBoardError(
                "accepted control-plane launch fields are incomplete"
            )
        if has_control_plane:
            try:
                control_plane_pin = parse_accepted_control_plane_pin(
                    args.accepted_control_plane_pin_json
                )
                control_plane_descriptor = int(
                    args.accepted_control_plane_fd
                )
                verify_agent_implementation_sealed_control_plane(
                    control_plane_pin,
                    control_plane_descriptor,
                )
            except (OSError, ValueError) as exc:
                raise ConfiguredBoardError(
                    "accepted control-plane launch binding is invalid"
                ) from exc
            control_plane_parent = Path(
                args.accepted_control_plane_capsule_parent
            )
            if (
                control_plane_parent.parent != Path(tempfile.gettempdir())
                or not control_plane_parent.name.startswith(
                    "asref-configured-control-plane-"
                )
                or Path(control_plane_pin.capsule_root).parent
                != control_plane_parent
                or (
                    control_plane_pin.source_head,
                    control_plane_pin.source_tree,
                )
                != _git_identity(board.repo_root)
            ):
                raise ConfiguredBoardError(
                    "accepted control-plane launch provenance is foreign"
                )
        if args.accepted_tree_root is not None:
            accepted_tree_root = _canonical_no_symlink_root(
                args.accepted_tree_root
            )
            module_tree_root = (
                board.repo_root
                if control_plane_pin is not None
                else Path(__file__).resolve().parents[3]
            )
            if (
                accepted_tree_root != module_tree_root
                or accepted_tree_root != board.repo_root.resolve()
            ):
                raise ConfiguredBoardError(
                    "configured scheduler accepted-tree root is foreign"
                )
        has_coordinator_session = bool(args.coordinator_launch_session)
        has_coordinator_status = args.coordinator_status_path is not None
        if has_coordinator_session != has_coordinator_status:
            raise ConfiguredBoardError(
                "coordinator launch session binding is incomplete"
            )
        if has_coordinator_session:
            expected_status_path = _expected_coordinator_status_path(
                board,
                args.coordinator_launch_session,
            )
            if (
                args.command != "launch"
                or not bool(args.foreground)
                or bool(args.dry_run)
                or control_plane_pin is None
                or args.accepted_tree_root is None
                or args.coordinator_status_path != expected_status_path
                or _plan_bound_profile(board)
            ):
                raise ConfiguredBoardError(
                    "coordinator launch session is not an admitted foreground child"
                )
        if launch_receipt_stream is not None and _plan_bound_profile(board):
            raise ConfiguredBoardError(
                "launch-receipt-only does not admit adaptive plan-bound profiles"
            )
    except ConfiguredBoardError as exc:
        print(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "configured-board-error@1"
                    ),
                    "valid": False,
                    "errors": [str(exc)],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if args.command == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if preflight["valid"] else 2
    if not preflight["valid"]:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 2

    detach = not bool(args.foreground)
    if launch_receipt_stream is not None:
        try:
            launch_receipt = _launch_detached_receipt_coordinator(
                board,
                implement=bool(args.implement),
                duration_seconds=float(args.duration_seconds),
            )
            receipt_cid = launch_receipt.get("receipt_cid")
            unsigned_receipt = dict(launch_receipt)
            unsigned_receipt.pop("receipt_cid", None)
            if (
                set(launch_receipt) != COORDINATOR_LAUNCH_RECEIPT_FIELDS
                or receipt_cid != content_identity(unsigned_receipt)
            ):
                raise ConfiguredBoardError(
                    "coordinator launch returned a non-closed receipt"
                )
        except (ConfiguredBoardError, OSError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "valid": False,
                        "errors": [f"coordinator_launch: {exc}"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        launch_receipt_stream.write(
            json.dumps(
                launch_receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        launch_receipt_stream.flush()
        return 0

    if _plan_bound_profile(board):
        try:
            _require_plan_bound_process_launch_policy(
                board,
                implement=bool(args.implement),
            )
        except ConfiguredBoardError as exc:
            print(
                json.dumps(
                    {
                        "schema": (
                            "ipfs_accelerate_py/agent-supervisor/"
                            "configured-board-launch-no-go@1"
                        ),
                        "valid": False,
                        "process_started": False,
                        "errors": [str(exc)],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if args.dry_run:
            plan = configured_board_launch_plan(
                board,
                implement=bool(args.implement),
                detach=detach,
                duration_seconds=float(args.duration_seconds),
            )
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if detach:
            plan = configured_board_launch_plan(
                board,
                implement=bool(args.implement),
                detach=True,
                duration_seconds=float(args.duration_seconds),
            )
            try:
                launch_receipt = _launch_detached_plan_bound_coordinator(
                    board,
                    implement=bool(args.implement),
                    duration_seconds=float(args.duration_seconds),
                )
                plan.update(launch_receipt)
            except (ConfiguredBoardError, OSError) as exc:
                notes = [
                    str(note)
                    for note in getattr(exc, "__notes__", ())
                    if str(note).strip()
                ]
                print(
                    json.dumps(
                        {
                            "valid": False,
                            "errors": [f"coordinator_launch: {exc}", *notes],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 2
            if launch_receipt_stream is None:
                print(json.dumps(plan, indent=2, sort_keys=True))
            else:
                launch_receipt_stream.write(
                    json.dumps(launch_receipt, indent=2, sort_keys=True) + "\n"
                )
                launch_receipt_stream.flush()
            return 0
        if control_plane_pin is None:
            try:
                return _launch_foreground_plan_bound_coordinator(
                    board,
                    implement=bool(args.implement),
                    duration_seconds=float(args.duration_seconds),
                )
            except (ConfiguredBoardError, OSError, ValueError) as exc:
                notes = [
                    str(note)
                    for note in getattr(exc, "__notes__", ())
                    if str(note).strip()
                ]
                print(
                    json.dumps(
                        {
                            "valid": False,
                            "errors": [f"coordinator_launch: {exc}", *notes],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 2
        try:
            return _run_plan_bound_coordinator(
                board,
                implement=bool(args.implement),
                duration_seconds=float(args.duration_seconds),
                accepted_control_plane_pin=control_plane_pin,
                accepted_control_plane_descriptor=control_plane_descriptor,
            )
        finally:
            _remove_owned_coordinator_pid(board)
            if control_plane_parent is not None:
                _cleanup_plan_bound_control_plane(
                    control_plane_pin,
                    control_plane_parent,
                )

    plan = configured_board_launch_plan(
        board,
        implement=bool(args.implement),
        detach=detach,
        duration_seconds=float(args.duration_seconds),
        configured_board_live_capsule_pin_json=(
            live_context.capsule_pin_json if live_context is not None else ""
        ),
        configured_board_live_capsule_descriptor=(
            live_context.capsule_descriptor if live_context is not None else -1
        ),
        configured_board_live_admission_json=(
            live_context.admission_json if live_context is not None else ""
        ),
        configured_board_live_native_launch_json=(
            live_context.native_launch_json if live_context is not None else ""
        ),
        configured_board_live_native_descriptor=(
            live_context.native_descriptor if live_context is not None else -1
        ),
        state_owner_bootstrap_fd=(
            args.state_owner_bootstrap_fd
            if live_context is not None
            else -1
        ),
        state_owner_bootstrap_store_id=(
            args.state_owner_bootstrap_store_id
            if live_context is not None
            else ""
        ),
    )
    plan["authoritative_board_projection_repair"] = projection_repair
    if has_coordinator_session:
        _bind_foreground_wave_pid(plan, board)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    _apply_configured_board_environment(plan)
    from .multi_supervisor_runner import main as multi_supervisor_main

    previous_umask: int | None = None
    private_lane_runtime = has_coordinator_session or live_context is not None
    try:
        if private_lane_runtime:
            previous_umask = os.umask(0o077)
            _prepare_coordinator_lane_status_permissions(board)
        if has_coordinator_session:
            assert args.coordinator_status_path is not None
            _publish_coordinator_launch_attestation(
                board,
                launch_session_id=args.coordinator_launch_session,
                status_path=args.coordinator_status_path,
            )
        return int(multi_supervisor_main(plan["argv"]))
    finally:
        if has_coordinator_session:
            _remove_owned_coordinator_pid(board)
            if control_plane_parent is not None:
                assert control_plane_pin is not None
                _cleanup_plan_bound_control_plane(
                    control_plane_pin,
                    control_plane_parent,
                )
        if previous_umask is not None:
            os.umask(previous_umask)


def main(argv: Sequence[str] | None = None) -> int:
    from .process_security import (
        capture_state_authority_credentials,
        harden_state_authority_process,
    )

    harden_state_authority_process()
    capture_state_authority_credentials()
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    receipt_only = bool(getattr(args, "launch_receipt_only", False))
    if receipt_only and (bool(args.dry_run) or bool(args.foreground)):
        parser.error(
            "--launch-receipt-only requires a detached, non-dry launch"
        )
    if not receipt_only:
        return _run_parsed_command(args)

    # Preserve a dedicated receipt descriptor before redirecting stdout at the
    # descriptor boundary. This also fences native code and inherited child
    # stdout, not only Python ``print`` calls.
    with _isolated_launch_receipt_stream() as receipt_stream:
        return _run_parsed_command(
            args,
            launch_receipt_stream=receipt_stream,
        )


__all__ = (
    "ConfiguredBoard",
    "ConfiguredBoardError",
    "configured_board_capacity_observation",
    "configured_board_common_args",
    "configured_board_launch_plan",
    "load_configured_board",
    "materialize_configured_board_execution_plan",
    "main",
    "preflight_configured_board",
)
