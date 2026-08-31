#!/usr/bin/env python3
"""Deterministic, fail-closed validator for the SAWM R2 control program.

The Markdown documents are immutable operator inputs, never task-completion
authority.  This validator checks their closed structure and the scheduler
binding, including the append-only M25 native-DuckDB preload successor.
Accepted task state remains in the datasets-authoritative DuckDB store
reached through the current Quack owner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BOARD_NAMESPACE = "semantic-addressed-world-model-v1"
PLAN_REVISION = "SAWM-PLAN-R2"
ROOT_GOAL = "SAWM-G000"
TASK_IDS = tuple(f"SAWM-{index:03d}" for index in range(45))
GOAL_IDS = (
    "SAWM-G000",
    "SAWM-G010", "SAWM-G011", "SAWM-G012", "SAWM-G013",
    "SAWM-G020", "SAWM-G021", "SAWM-G022", "SAWM-G023",
    "SAWM-G030", "SAWM-G031", "SAWM-G032", "SAWM-G033",
    "SAWM-G040", "SAWM-G041", "SAWM-G042", "SAWM-G043",
    "SAWM-G050", "SAWM-G051", "SAWM-G052", "SAWM-G053", "SAWM-G054",
    "SAWM-G060", "SAWM-G061", "SAWM-G062", "SAWM-G063",
    "SAWM-G070", "SAWM-G071", "SAWM-G072",
)

PLAN_PATH = REPO_ROOT / "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md"
OBJECTIVES_PATH = REPO_ROOT / "docs/architecture/semantic_addressed_world_model.objectives.md"
TODO_PATH = REPO_ROOT / "docs/architecture/semantic_addressed_world_model.todo.md"
INVENTORY_ROOT = REPO_ROOT / "docs/architecture/semantic_addressed_world_model_inventory"
SEAL_PATH = REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
SCHEDULER_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
BENCHMARK_PATH = REPO_ROOT / "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json"
NATIVE_AUTHORIZATION_PATH = (
    "config/semantic_addressed_world_model_native_dependency.authorization.json"
)

INVENTORY_PATHS = tuple(
    INVENTORY_ROOT / name
    for name in (
        "repository_baseline.json",
        "authority_matrix.json",
        "overlap_gap_matrix.json",
        "identity_inventory.json",
        "interface_inventory.json",
        "dependency_graph.json",
        "capability_matrix.json",
        "rollout_baseline.json",
        "prior_materialization_migration.json",
    )
)

CONTROL_RELATIVE_PATHS = (
    ".gitignore",
    "requirements.txt",
    "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
    "docs/architecture/semantic_addressed_world_model.objectives.md",
    "docs/architecture/semantic_addressed_world_model.todo.md",
    *(path.relative_to(REPO_ROOT).as_posix() for path in INVENTORY_PATHS),
    "config/semantic_addressed_world_model_dependencies.seal.json",
    "config/semantic_addressed_world_model_native_dependency.authorization.json",
    "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
    "scripts/validate_semantic_addressed_world_model_dependencies.py",
    "scripts/validate_semantic_addressed_world_model_board.py",
    "scripts/materialize_semantic_addressed_world_model_program.py",
    "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    "ipfs_accelerate_py/agent_implementation_route.py",
    "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
    "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_extension_projection.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_live_capsule.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_attestation.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
    "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    "test/api/semantic_world/test_semantic_addressed_world_model_quack_protocol.py",
    "test/api/test_agent_supervisor_configured_board_extension_projection.py",
    "test/api/test_agent_supervisor_configured_board_live_capsule.py",
    "test/api/test_agent_supervisor_configured_board_scheduler.py",
    "test/api/test_agent_supervisor_database_coordination.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
    "test/api/test_agent_supervisor_database_portal_bridge.py",
    "test/api/test_agent_supervisor_grok_quota_terra_gate.py",
    "test/api/test_agent_supervisor_native_dependency_pin.py",
    "test/api/test_agent_supervisor_project_dependency_preflight.py",
    "test/api/test_agent_supervisor_provider_command_binding.py",
    "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
)

GOAL_PARENT = {
    "SAWM-G000": "",
    "SAWM-G010": "SAWM-G000",
    "SAWM-G011": "SAWM-G010", "SAWM-G012": "SAWM-G010", "SAWM-G013": "SAWM-G010",
    "SAWM-G020": "SAWM-G000",
    "SAWM-G021": "SAWM-G020", "SAWM-G022": "SAWM-G020", "SAWM-G023": "SAWM-G020",
    "SAWM-G030": "SAWM-G000",
    "SAWM-G031": "SAWM-G030", "SAWM-G032": "SAWM-G030", "SAWM-G033": "SAWM-G030",
    "SAWM-G040": "SAWM-G000",
    "SAWM-G041": "SAWM-G040", "SAWM-G042": "SAWM-G040", "SAWM-G043": "SAWM-G040",
    "SAWM-G050": "SAWM-G000",
    "SAWM-G051": "SAWM-G050", "SAWM-G052": "SAWM-G050", "SAWM-G053": "SAWM-G050", "SAWM-G054": "SAWM-G050",
    "SAWM-G060": "SAWM-G000",
    "SAWM-G061": "SAWM-G060", "SAWM-G062": "SAWM-G060", "SAWM-G063": "SAWM-G060",
    "SAWM-G070": "SAWM-G000",
    "SAWM-G071": "SAWM-G070", "SAWM-G072": "SAWM-G070",
}

TASK_FIELDS = (
    "stable task id", "status", "completion", "completion mode",
    "is schedulable", "review only", "priority", "track", "depends on",
    "dependencies json", "goal id", "parent goal id", "subgoal id",
    "owning repository", "exact inputs", "outputs", "outputs json",
    "predicted files", "predicted files json", "predicted symbols",
    "public interfaces", "interfaces", "preconditions", "declared effects", "validation",
    "validation commands json", "evidence requirements", "acceptance",
    "conflict policy", "context budget tokens", "no-model route",
    "model fallback", "rollout mode", "protected paths", "limitations",
    "bundle", "parallel lane", "resource class", "implementation stage",
    "implementation timeout seconds", "provider role", "network policy",
    "risk class", "write scope", "external effect scope", "prohibited effects",
    "rollback or compensation procedure", "board namespace", "plan revision",
)

GOAL_FIELDS = (
    "stable goal id", "status", "parent", "parent goal ids json", "depends on",
    "dependencies json", "child goal ids json", "fib priority", "priority",
    "track", "bundle", "parallel lane", "resource class", "goal",
    "refinement", "producing tasks", "evidence", "evidence requirements json",
    "evidence criteria", "outputs", "predicted files", "predicted files json",
    "interfaces", "validation", "acceptance", "gap tasks", "rollout constraint",
    "authority constraint", "conflict policy",
)

ALLOWED_TASK_STATUSES = frozenset({"todo", "completed"})
ALLOWED_GOAL_STATUSES = frozenset(
    {"open", "active", "reopened", "provisionally_complete", "analysis_inconclusive"}
)
ROLLOUT_ORDER = ("bootstrap", "shadow_write", "shadow_read", "guarded", "required")
HISTORICAL_VALIDATION_PREFIX = (
    "PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. "
    "/home/barberb/.local/bin/python "
)
OPERATIONAL_VALIDATION_PREFIX = (
    "PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. python "
)
# Kept as a compatibility alias for tests and callers that inspect the
# immutable R2 Markdown definitions directly.
VALIDATION_PREFIX = HISTORICAL_VALIDATION_PREFIX
LEARNED_TASKS = frozenset(f"SAWM-{index:03d}" for index in range(25, 32))
REQUIRED_MODE_TASKS = frozenset(f"SAWM-{index:03d}" for index in range(38, 45))


@dataclass(frozen=True)
class Card:
    identifier: str
    title: str
    metadata: Mapping[str, str]


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(REPO_ROOT)} must contain an object")
    return value


def _dependency_validator_module(root: Path):
    """Load the sibling seal validator without making ``scripts`` a package."""

    path = root / "scripts/validate_semantic_addressed_world_model_dependencies.py"
    name = "_sawm_dependency_validator_for_board"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("unable to load the SAWM dependency validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _m26_validation_modules(root: Path) -> tuple[Any, Any]:
    """Load successor validators from the exact supplied repository root."""

    dependency = _dependency_validator_module(root)
    path = root / "scripts/materialize_semantic_addressed_world_model_program.py"
    spec = importlib.util.spec_from_file_location(
        "sawm_board_m26_materializer", path
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load the M26 materializer")
    materializer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(materializer)
    return dependency, materializer


def _m6_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse the one exact M6 migration contract across both static gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(module._m6_source_migration_errors(scheduler, seal, migration))
    except Exception as exc:
        return [f"M6 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m7_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse the exact M7 source-only repair contract across static gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(module._m7_source_repair_errors(scheduler, seal, migration))
    except Exception as exc:
        return [f"M7 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m8_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse the exact M8 source-only successor contract across static gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(module._m8_source_repair_errors(scheduler, seal, migration))
    except Exception as exc:
        return [f"M8 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m9_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse the exact M9 live-recovery contract across both static gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(module._m9_live_recovery_errors(scheduler, seal, migration))
    except Exception as exc:
        return [f"M9 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m10_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse the exact M10 live-projection contract across static gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m10_live_projection_errors(scheduler, seal, migration)
        )
    except Exception as exc:
        return [f"M10 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m11_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse the exact M11 provider-launch repair contract across static gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m11_provider_retry_errors(scheduler, seal, migration)
        )
    except Exception as exc:
        return [f"M11 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m12_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse the exact M12 declared-output repair contract across static gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m12_declared_output_retry_errors(
                scheduler,
                seal,
                migration,
            )
        )
    except Exception as exc:
        return [f"M12 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m13_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse the exact M13 initial-refresh repair contract across static gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m13_quack_refresh_errors(
                scheduler,
                seal,
                migration,
            )
        )
    except Exception as exc:
        return [f"M13 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m16_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M16 accepted-source repair contract across gates."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m16_accepted_source_retry_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M16 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m34_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M34 JSON-emission normalization contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m34_json_emission_normalization_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M34 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m33_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M33 normalized live-preflight contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m33_live_preflight_contract_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M33 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m32_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M32 live-preflight plan-anchor repair contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m32_live_preflight_plan_anchor_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M32 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m31_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M31 detached-coordinator PID recovery contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m31_detached_coordinator_pid_recovery_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M31 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m30_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M30 stopped-owner restart/source-seal contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m30_stopped_owner_restart_source_seal_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M30 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m29_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M29 committed-evidence verification contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m29_committed_evidence_verification_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M29 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m28_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M28 live-claim/admission-recovery contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m28_live_claim_admission_recovery_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M28 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m27_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M27 dead-owner parallel-resume contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m27_dead_owner_parallel_resume_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M27 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m26_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M26 stopped-runtime recovery successor contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m26_automatic_stall_recovery_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M26 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m25_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M25 native-DuckDB preload successor contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m25_native_duckdb_preload_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M25 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m24_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M24 sidecar-reopen successor contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m24_sidecar_reopen_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M24 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m23_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M23 sealed multi-lane successor contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m23_multi_lane_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M23 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m22_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M22 live-preflight receipt compatibility contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m22_live_preflight_receipt_compatibility_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M22 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m21_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M21 generation-realization successor contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m21_generation_realization_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M21 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m20_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M20 post-materialization isolation contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m20_test_isolation_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M20 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m19_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M19 live catalog-inventory contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m19_live_catalog_inventory_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M19 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m17_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M17 post-commit source-binding contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m17_source_binding_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M17 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m18_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M18 portal-completion persistence contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m18_portal_completion_persistence_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M18 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _active_successor_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Select the newest declared successor without truthiness fallback.

    Key presence selects M34 before every historical successor.  Consequently
    an empty, null,
    or otherwise malformed newest declaration is validated at that revision
    and cannot silently reactivate historical authority.  Every predecessor
    remains independently checked as immutable history.
    """

    m34_key = "json_emission_normalization_successor_materialization"
    m34_seal_key = f"{m34_key}_cid"
    m34_presence = (
        m34_key in scheduler,
        m34_key in migration,
        m34_seal_key in seal,
    )
    if any(m34_presence):
        errors = _m34_migration_errors(scheduler, seal, migration)
        if not all(m34_presence):
            errors.append(
                "M34 JSON-emission normalization authority is only partially declared"
            )
        for validator in (
            _m33_migration_errors,
            _m32_migration_errors,
            _m31_migration_errors,
            _m30_migration_errors,
            _m29_migration_errors,
            _m28_migration_errors,
            _m27_migration_errors,
            _m26_migration_errors,
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m33_key = "live_preflight_contract_successor_materialization"
    m33_seal_key = f"{m33_key}_cid"
    m33_presence = (
        m33_key in scheduler,
        m33_key in migration,
        m33_seal_key in seal,
    )
    if any(m33_presence):
        errors = _m33_migration_errors(scheduler, seal, migration)
        if not all(m33_presence):
            errors.append("M33 live-preflight contract authority is only partially declared")
        for validator in (
            _m32_migration_errors,
            _m31_migration_errors,
            _m30_migration_errors,
            _m29_migration_errors,
            _m28_migration_errors,
            _m27_migration_errors,
            _m26_migration_errors,
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m32_key = "live_preflight_plan_anchor_successor_materialization"
    m32_seal_key = f"{m32_key}_cid"
    m32_presence = (
        m32_key in scheduler,
        m32_key in migration,
        m32_seal_key in seal,
    )
    if any(m32_presence):
        errors = _m32_migration_errors(scheduler, seal, migration)
        if not all(m32_presence):
            errors.append("M32 live-preflight authority is only partially declared")
        for validator in (
            _m31_migration_errors,
            _m30_migration_errors,
            _m29_migration_errors,
            _m28_migration_errors,
            _m27_migration_errors,
            _m26_migration_errors,
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m31_key = "detached_coordinator_pid_recovery_successor_materialization"
    m31_seal_key = f"{m31_key}_cid"
    m31_presence = (
        m31_key in scheduler,
        m31_key in migration,
        m31_seal_key in seal,
    )
    if any(m31_presence):
        errors = _m31_migration_errors(scheduler, seal, migration)
        if not all(m31_presence):
            errors.append(
                "M31 detached-coordinator authority is only partially declared"
            )
        for validator in (
            _m30_migration_errors,
            _m29_migration_errors,
            _m28_migration_errors,
            _m27_migration_errors,
            _m26_migration_errors,
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m30_key = "stopped_owner_restart_source_seal_successor_materialization"
    m30_seal_key = f"{m30_key}_cid"
    m30_presence = (
        m30_key in scheduler,
        m30_key in migration,
        m30_seal_key in seal,
    )
    if any(m30_presence):
        errors = _m30_migration_errors(scheduler, seal, migration)
        if not all(m30_presence):
            errors.append("M30 stopped-owner restart authority is only partially declared")
        for validator in (
            _m29_migration_errors,
            _m28_migration_errors,
            _m27_migration_errors,
            _m26_migration_errors,
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m29_key = "committed_evidence_verification_successor_materialization"
    m29_seal_key = f"{m29_key}_cid"
    m29_presence = (
        m29_key in scheduler,
        m29_key in migration,
        m29_seal_key in seal,
    )
    if any(m29_presence):
        errors = _m29_migration_errors(scheduler, seal, migration)
        if not all(m29_presence):
            errors.append(
                "M29 committed-evidence-verification authority is only "
                "partially declared"
            )
        for validator in (
            _m28_migration_errors,
            _m27_migration_errors,
            _m26_migration_errors,
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m28_key = "live_claim_admission_recovery_successor_materialization"
    m28_seal_key = f"{m28_key}_cid"
    m28_presence = (
        m28_key in scheduler,
        m28_key in migration,
        m28_seal_key in seal,
    )
    if any(m28_presence):
        errors = _m28_migration_errors(scheduler, seal, migration)
        if not all(m28_presence):
            errors.append(
                "M28 live-claim/admission-recovery authority is only partially "
                "declared"
            )
        for validator in (
            _m27_migration_errors,
            _m26_migration_errors,
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m27_key = "dead_owner_parallel_resume_successor_materialization"
    m27_seal_key = f"{m27_key}_cid"
    m27_presence = (
        m27_key in scheduler,
        m27_key in migration,
        m27_seal_key in seal,
    )
    if any(m27_presence):
        errors = _m27_migration_errors(scheduler, seal, migration)
        if not all(m27_presence):
            errors.append(
                "M27 dead-owner parallel-resume authority is only partially declared"
            )
        for validator in (
            _m26_migration_errors,
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m26_key = "automatic_stall_recovery_successor_materialization"
    m26_seal_key = f"{m26_key}_cid"
    m26_presence = (
        m26_key in scheduler,
        m26_key in migration,
        m26_seal_key in seal,
    )
    if any(m26_presence):
        errors = _m26_migration_errors(scheduler, seal, migration)
        if not all(m26_presence):
            errors.append(
                "M26 automatic-stall-recovery successor authority is only "
                "partially declared"
            )
        for validator in (
            _m25_migration_errors,
            _m24_migration_errors,
            _m23_migration_errors,
            _m22_migration_errors,
            _m21_migration_errors,
            _m20_migration_errors,
            _m19_migration_errors,
            _m18_migration_errors,
            _m17_migration_errors,
            _m16_migration_errors,
        ):
            errors.extend(
                validator(
                    scheduler,
                    seal,
                    migration,
                    require_active_runtime=False,
                )
            )
        return errors

    m25_key = "native_duckdb_preload_successor_materialization"
    m25_seal_key = f"{m25_key}_cid"
    m25_presence = (
        m25_key in scheduler,
        m25_key in migration,
        m25_seal_key in seal,
    )
    if any(m25_presence):
        errors = _m25_migration_errors(scheduler, seal, migration)
        if not all(m25_presence):
            errors.append(
                "M25 native-DuckDB preload successor authority is only "
                "partially declared"
            )
        errors.extend(
            _m24_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m23_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m22_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m21_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m20_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m19_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m18_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m17_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m24_key = "multi_lane_sidecar_reopen_successor_materialization"
    m24_seal_key = f"{m24_key}_cid"
    m24_presence = (
        m24_key in scheduler,
        m24_key in migration,
        m24_seal_key in seal,
    )
    if any(m24_presence):
        errors = _m24_migration_errors(scheduler, seal, migration)
        if not all(m24_presence):
            errors.append(
                "M24 sidecar-reopen successor authority is only partially declared"
            )
        errors.extend(
            _m23_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m22_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m21_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m20_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m19_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m18_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m17_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m23_key = "multi_lane_successor_materialization"
    m23_seal_key = "multi_lane_successor_materialization_cid"
    m23_presence = (
        m23_key in scheduler,
        m23_key in migration,
        m23_seal_key in seal,
    )
    if any(m23_presence):
        errors = _m23_migration_errors(scheduler, seal, migration)
        if not all(m23_presence):
            errors.append(
                "M23 multi-lane successor authority is only partially declared"
            )
        errors.extend(
            _m22_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m21_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m20_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m19_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m18_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m17_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m22_key = "live_preflight_receipt_compatibility_successor_materialization"
    m22_seal_key = (
        "live_preflight_receipt_compatibility_successor_materialization_cid"
    )
    m22_presence = (
        m22_key in scheduler,
        m22_key in migration,
        m22_seal_key in seal,
    )
    if any(m22_presence):
        errors = _m22_migration_errors(scheduler, seal, migration)
        if not all(m22_presence):
            errors.append(
                "M22 live-preflight receipt compatibility successor authority "
                "is only partially declared"
            )
        errors.extend(
            _m21_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m20_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m19_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m18_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m17_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m21_key = "generation_realization_successor_materialization"
    m21_seal_key = "generation_realization_successor_materialization_cid"
    m21_presence = (
        m21_key in scheduler,
        m21_key in migration,
        m21_seal_key in seal,
    )
    if any(m21_presence):
        errors = _m21_migration_errors(scheduler, seal, migration)
        if not all(m21_presence):
            errors.append(
                "M21 generation-realization successor authority is only "
                "partially declared"
            )
        errors.extend(
            _m20_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m19_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m18_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m17_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m20_key = "test_isolation_successor_materialization"
    m20_seal_key = "test_isolation_successor_materialization_cid"
    m20_presence = (
        m20_key in scheduler,
        m20_key in migration,
        m20_seal_key in seal,
    )
    if any(m20_presence):
        errors = _m20_migration_errors(scheduler, seal, migration)
        if not all(m20_presence):
            errors.append(
                "M20 test-isolation successor authority is only partially declared"
            )
        errors.extend(
            _m19_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m18_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m17_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m19_key = "live_catalog_inventory_successor_materialization"
    m19_seal_key = "live_catalog_inventory_successor_materialization_cid"
    m19_presence = (
        m19_key in scheduler,
        m19_key in migration,
        m19_seal_key in seal,
    )
    if any(m19_presence):
        errors = _m19_migration_errors(scheduler, seal, migration)
        if not all(m19_presence):
            errors.append(
                "M19 live-catalog-inventory successor authority is only partially declared"
            )
        errors.extend(
            _m18_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m17_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m18_key = "portal_completion_persistence_successor_materialization"
    m18_seal_key = "portal_completion_persistence_successor_materialization_cid"
    m18_presence = (
        m18_key in scheduler,
        m18_key in migration,
        m18_seal_key in seal,
    )
    if any(m18_presence):
        errors = _m18_migration_errors(scheduler, seal, migration)
        if not all(m18_presence):
            errors.append(
                "M18 portal-completion successor authority is only partially declared"
            )
        errors.extend(
            _m17_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m17_key = "source_binding_successor_materialization"
    m17_seal_key = "source_binding_successor_materialization_cid"
    m17_presence = (
        m17_key in scheduler,
        m17_key in migration,
        m17_seal_key in seal,
    )
    if any(m17_presence):
        errors = _m17_migration_errors(scheduler, seal, migration)
        if not all(m17_presence):
            errors.append(
                "M17 source-binding successor authority is only partially declared"
            )
        # M16 remains immutable historical authority.  A malformed M17 never
        # falls back to it, but its exact accepted-source repair is still
        # checked independently.
        errors.extend(
            _m16_migration_errors(
                scheduler,
                seal,
                migration,
                require_active_runtime=False,
            )
        )
        return errors

    m16_key = "accepted_source_retry_successor_materialization"
    m16_seal_key = "accepted_source_retry_successor_materialization_cid"
    m16_presence = (
        m16_key in scheduler,
        m16_key in migration,
        m16_seal_key in seal,
    )
    if any(m16_presence):
        errors = _m16_migration_errors(scheduler, seal, migration)
        if not all(m16_presence):
            errors.append(
                "M16 accepted-source retry authority is only partially declared"
            )
        # M15 remains immutable historical authority.  A malformed M16 must
        # never reactivate it, but the exact predecessor control is still
        # independently checked.
        try:
            module = _dependency_validator_module(REPO_ROOT)
            errors.extend(
                module._m15_historical_authority_errors(
                    scheduler,
                    seal,
                    migration,
                )
            )
        except Exception as exc:
            errors.append(
                f"M15 historical validator unavailable: {type(exc).__name__}: {exc}"
            )
        return errors

    m15_key = "runtime_root_rebind_successor_materialization"
    m15_seal_key = "runtime_root_rebind_successor_materialization_cid"
    m15_presence = (m15_key in scheduler, m15_key in migration, m15_seal_key in seal)
    if any(m15_presence):
        try:
            module = _dependency_validator_module(REPO_ROOT)
            errors = list(
                module._m15_runtime_root_rebind_errors(scheduler, seal, migration)
            )
        except Exception as exc:
            errors = [
                f"M15 migration validator unavailable: {type(exc).__name__}: {exc}"
            ]
        if not all(m15_presence):
            errors.append("M15 runtime-root authority is only partially declared")
        return errors

    m14_key = "stale_owner_restart_successor_materialization"
    m14_seal_key = "stale_owner_restart_successor_materialization_cid"
    m14_presence = (m14_key in scheduler, m14_key in migration, m14_seal_key in seal)
    if any(m14_presence):
        try:
            module = _dependency_validator_module(REPO_ROOT)
            errors = list(module._m14_stale_owner_restart_errors(scheduler, seal, migration))
        except Exception as exc:
            errors = [f"M14 migration validator unavailable: {type(exc).__name__}: {exc}"]
        if not all(m14_presence):
            errors.append("M14 stale-owner authority is only partially declared")
        return errors

    m13_key = "quack_refresh_successor_materialization"
    m13_seal_key = "quack_refresh_successor_materialization_cid"
    m13_presence = (
        m13_key in scheduler,
        m13_key in migration,
        m13_seal_key in seal,
    )
    if any(m13_presence):
        errors = _m13_migration_errors(scheduler, seal, migration)
        if not all(m13_presence):
            errors.append("M13 Quack-refresh authority is only partially declared")
        errors.extend(_m12_migration_errors(scheduler, seal, migration))
        errors.extend(_m11_migration_errors(scheduler, seal, migration))
        errors.extend(_m10_migration_errors(scheduler, seal, migration))
        errors.extend(_m9_migration_errors(scheduler, seal, migration))
        errors.extend(_m8_migration_errors(scheduler, seal, migration))
        return errors

    m12_key = "declared_output_retry_successor_materialization"
    m12_seal_key = "declared_output_retry_successor_materialization_cid"
    m12_presence = (
        m12_key in scheduler,
        m12_key in migration,
        m12_seal_key in seal,
    )
    if any(m12_presence):
        errors = _m12_migration_errors(scheduler, seal, migration)
        if not all(m12_presence):
            errors.append(
                "M12 declared-output retry authority is only partially declared"
            )
        errors.extend(_m11_migration_errors(scheduler, seal, migration))
        errors.extend(_m10_migration_errors(scheduler, seal, migration))
        errors.extend(_m9_migration_errors(scheduler, seal, migration))
        errors.extend(_m8_migration_errors(scheduler, seal, migration))
        return errors

    m11_key = "live_provider_retry_successor_materialization"
    m11_seal_key = "live_provider_retry_successor_materialization_cid"
    m11_presence = (
        m11_key in scheduler,
        m11_key in migration,
        m11_seal_key in seal,
    )
    if any(m11_presence):
        errors = _m11_migration_errors(scheduler, seal, migration)
        if not all(m11_presence):
            errors.append("M11 provider-retry authority is only partially declared")
        errors.extend(_m10_migration_errors(scheduler, seal, migration))
        errors.extend(_m9_migration_errors(scheduler, seal, migration))
        errors.extend(_m8_migration_errors(scheduler, seal, migration))
        return errors

    m10_key = "live_projection_successor_materialization"
    m10_seal_key = "live_projection_successor_materialization_cid"
    m10_presence = (
        m10_key in scheduler,
        m10_key in migration,
        m10_seal_key in seal,
    )
    if any(m10_presence):
        errors = _m10_migration_errors(scheduler, seal, migration)
        if not all(m10_presence):
            errors.append("M10 live-projection authority is only partially declared")
        errors.extend(_m9_migration_errors(scheduler, seal, migration))
        errors.extend(_m8_migration_errors(scheduler, seal, migration))
        return errors

    m9_key = "live_recovery_successor_materialization"
    m9_seal_key = "live_recovery_successor_materialization_cid"
    m9_presence = (
        m9_key in scheduler,
        m9_key in migration,
        m9_seal_key in seal,
    )
    if any(m9_presence):
        errors = _m9_migration_errors(scheduler, seal, migration)
        if not all(m9_presence):
            errors.append("M9 live-recovery authority is only partially declared")
        # The predecessor remains immutable historical authority; M9 changes
        # only the adjacent active execution generation.
        errors.extend(_m8_migration_errors(scheduler, seal, migration))
        return errors

    m8_key = "source_repair_successor_materialization"
    m8_seal_key = "source_repair_successor_materialization_cid"
    m8_presence = (
        m8_key in scheduler,
        m8_key in migration,
        m8_seal_key in seal,
    )
    if any(m8_presence):
        errors = _m8_migration_errors(scheduler, seal, migration)
        if not all(m8_presence):
            errors.append("M8 source-repair authority is only partially declared")
        return errors
    return [
        "active M8/M9/M10/M11/M12/M13/M14/M15/M16/M17/M18/M19/M20/M21/M22/M23/M24/M25 "
        "successor authority is absent"
    ]


def _normalize_field(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("_", " "))


def _parse_cards(path: Path, *, goal: bool) -> tuple[Card, ...]:
    text = path.read_text(encoding="utf-8")
    identity = r"SAWM-G\d{3}" if goal else r"SAWM-\d{3}"
    matches = list(re.finditer(rf"^## ({identity})\b\s*(?:[-—:]\s*)?(.*)$", text, re.MULTILINE))
    cards: list[Card] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        metadata: dict[str, str] = {}
        current = ""
        for raw in block.splitlines():
            item = re.match(r"^\s*-\s+([^:]+):\s*(.*)$", raw)
            if item:
                current = _normalize_field(item.group(1))
                if current in metadata:
                    raise ValueError(f"{match.group(1)} has duplicate field {current!r}")
                metadata[current] = item.group(2).strip()
            elif current and raw.startswith(("  ", "\t")) and raw.strip():
                metadata[current] = (metadata[current] + " " + raw.strip()).strip()
            elif raw.strip():
                current = ""
        cards.append(Card(match.group(1), match.group(2).strip(), metadata))
    return tuple(cards)


def _json_list(card: Card, field: str, errors: list[str]) -> list[Any]:
    raw = card.metadata.get(field, "")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{card.identifier}: {field} is invalid JSON: {exc}")
        return []
    if not isinstance(value, list):
        errors.append(f"{card.identifier}: {field} must be a JSON array")
        return []
    return value


def _bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _csv(value: str) -> tuple[str, ...]:
    if value.strip().lower() in {"", "none", "[]", "n/a"}:
        return ()
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _safe_relative(value: str) -> bool:
    text = value.strip().replace("\\", "/")
    path = PurePosixPath(text)
    return bool(
        text
        and not path.is_absolute()
        and ".." not in path.parts
        and "\x00" not in text
        and not any(character in text for character in "*?[]{}")
    )


def _paths(items: Iterable[Any], *, card: Card, field: str, errors: list[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for item in items:
        if isinstance(item, str):
            path = item.strip()
        elif isinstance(item, Mapping):
            path = str(item.get("path") or "").strip()
        else:
            errors.append(f"{card.identifier}: {field} entry must be string or object")
            continue
        if not _safe_relative(path):
            errors.append(f"{card.identifier}: {field} path is not exact and contained: {path!r}")
            continue
        selected.append(path)
    if len(selected) != len(set(selected)):
        errors.append(f"{card.identifier}: {field} contains duplicate exact paths")
    return tuple(selected)


def _acyclic(adjacency: Mapping[str, Iterable[str]]) -> tuple[bool, tuple[str, ...]]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: list[str] = []

    def visit(node: str, trail: tuple[str, ...]) -> bool:
        if node in visiting:
            cycle.extend((*trail, node))
            return False
        if node in visited:
            return True
        visiting.add(node)
        for dependency in adjacency.get(node, ()):
            if dependency in adjacency and not visit(dependency, (*trail, node)):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    valid = all(visit(node, ()) for node in adjacency if node not in visited)
    return valid, tuple(cycle)


def _depends_transitively(adjacency: Mapping[str, tuple[str, ...]], task: str, dependency: str) -> bool:
    pending = list(adjacency.get(task, ()))
    seen: set[str] = set()
    while pending:
        item = pending.pop()
        if item == dependency:
            return True
        if item in seen:
            continue
        seen.add(item)
        pending.extend(adjacency.get(item, ()))
    return False


def _append(checks: list[dict[str, Any]], errors: list[str], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def _repository_for_path(path: str) -> str:
    if path.startswith("ipfs_datasets_py/"):
        return "ipfs_datasets_py"
    if path.startswith("ipfs_kit_py/"):
        return "ipfs_kit_py"
    return "ipfs_accelerate_py"


def _canonical_identity(value: Mapping[str, Any], *, identity_field: str) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _m5_migration_errors(
    config: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify the frozen M4 authority and the one bounded M5 recovery CAS."""

    errors: list[str] = []
    prior_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m4/control.duckdb"
    )
    target_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m5/control.duckdb"
    )
    m4_control_sha256 = (
        "d0c531d2ea30c512beb3587152527c4f605b5a5bf89e6311bf7aaa1974bff670"
    )
    m4_event_prefix = (
        "77b1a6b834658038c0f0cc44870a28f9fb480750e34dda408ab5718e83ee8911"
    )
    m4_projection = (
        "baguqeera65d24eqqbusuk6vznpmhm4nr65fas3i75dytebq5bev72bs6jaha"
    )
    m4_source_binding = (
        "sha256:72ae538afc063f98a4e7b0a799a619d6a8c499ceb51c9ceb4bffe95de5f9d323"
    )
    sawm_001_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    target_projection = (
        "baguqeerafx22x24mx7qrjjkfmyfdamtjkjd3ikrmhfqrp2gesqhe33l5467q"
    )
    expected_requeue = {
        "schema": "sawm/nonterminal-task-requeue-authorization@1",
        "authorized": True,
        "authority": "operator_source_migration",
        "task_alias": "SAWM-001",
        "task_cid": sawm_001_cid,
        "from_status": "in_progress",
        "from_revision": 2,
        "to_status": "todo",
        "to_revision": 3,
        "reason": (
            "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery"
        ),
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "expected_event_watermark": 119,
        "worker_self_approval": False,
    }

    expected_inventory = {
        "schema": "sawm/prior-materialization-migration-inventory@3",
        "migration_revision": "SAWM-R2-M5",
        "migration_kind": (
            "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery"
        ),
        "supersession_reason": (
            "source_authority_revision_and_preprovider_task_requeue"
        ),
        "prior_store_id": prior_store,
        "target_store_id": target_store,
        "prior_plan_revision": 5,
        "target_plan_revision": 6,
        "target_generation": 7,
        "prior_event_watermark": 116,
        "prior_event_prefix_sha256": m4_event_prefix,
        "prior_projection_cid": m4_projection,
        "prior_control_store_sha256": m4_control_sha256,
        "prior_source_binding_cid": m4_source_binding,
        "prior_source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "prior_source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "prior_materialization_receipt_cid": (
            "sha256:b24b3a2a0d325540761aaee03ea79bc88836b58577dab8583fec43f0de4c2a21"
        ),
        "prior_authority_preserved": True,
    }
    if any(migration.get(key) != value for key, value in expected_inventory.items()):
        errors.append("M5 inventory does not bind the exact frozen M4 authority")

    program = config.get("database_program")
    prior = config.get("prior_materialization")
    if not isinstance(program, Mapping) or (
        program.get("store_id") != target_store
        or program.get("store_generation") != "7"
    ):
        errors.append("M5 scheduler does not bind run-r2-m5 generation 7")
    if not isinstance(prior, Mapping):
        errors.append("M5 scheduler prior-materialization binding is absent")
        prior = {}
    expected_prior = {
        "migration_revision": "SAWM-R2-M5",
        "reason": "source_authority_revision_and_preprovider_task_requeue",
        "store_id": prior_store,
        "source_binding_cid": m4_source_binding,
        "source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "projection_cid": m4_projection,
        "control_store_sha256": m4_control_sha256,
        "event_watermark": 116,
        "event_prefix_sha256": m4_event_prefix,
        "migration_history_count": 4,
        "prior_plan_revision": 5,
        "target_plan_revision": 6,
        "migration_event_watermark": 118,
        "target_event_watermark": 119,
        "target_projection_cid": target_projection,
        "preserve_append_only": True,
    }
    if any(prior.get(key) != value for key, value in expected_prior.items()):
        errors.append("M5 scheduler prior-materialization fields are not exact")
    if dict(prior.get("nonterminal_task_requeue") or {}) != expected_requeue:
        errors.append("M5 scheduler does not authorize exactly one SAWM-001 recovery CAS")

    sealed = seal.get("source_migration")
    if not isinstance(sealed, Mapping):
        errors.append("M5 dependency-seal source migration is absent")
        sealed = {}
    expected_sealed = {
        "migration_revision": "SAWM-R2-M5",
        "migration_kind": (
            "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery"
        ),
        "supersession_reason": (
            "source_authority_revision_and_preprovider_task_requeue"
        ),
        "mode": "append_only_source_authority_revision",
        "prior_store_id": prior_store,
        "target_store_id": target_store,
        "prior_plan_revision": 5,
        "target_plan_revision": 6,
        "prior_event_watermark": 116,
        "prior_event_prefix_sha256": m4_event_prefix,
        "prior_projection_cid": m4_projection,
        "prior_control_store_sha256": m4_control_sha256,
        "prior_source_binding_cid": m4_source_binding,
        "prior_source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "prior_source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "prior_migration_count": 4,
        "prior_migration_receipt_cid": (
            "sha256:b24b3a2a0d325540761aaee03ea79bc88836b58577dab8583fec43f0de4c2a21"
        ),
        "migration_event_watermark": 118,
        "target_event_watermark": 119,
        "target_projection_cid": target_projection,
        "accepted_definition_rewrite_allowed": False,
        "accepted_completion_replay_allowed": False,
        "prior_authority_preserved": True,
    }
    if any(sealed.get(key) != value for key, value in expected_sealed.items()):
        errors.append("M5 dependency seal does not preserve the exact M4 authority")
    if dict(sealed.get("nonterminal_task_requeue") or {}) != expected_requeue:
        errors.append("M5 dependency seal recovery authorization is not exact")

    history = migration.get("migration_history")
    if (
        not isinstance(history, list)
        or len(history) != 4
        or any(not isinstance(entry, Mapping) for entry in history)
    ):
        errors.append("M1-through-M4 migration history is not a four-entry sequence")
        history = []
    if history:
        if [entry.get("migration_revision") for entry in history] != [
            "SAWM-R2-M1",
            "SAWM-R2-M2",
            "SAWM-R2-M3",
            "SAWM-R2-M4",
        ]:
            errors.append("M1-through-M4 migration revisions are not contiguous")
        for index, entry in enumerate(history[:3]):
            if (
                entry.get("schema") != "sawm/source-migration-history-entry@1"
                or not isinstance(entry.get("prior_event_watermark"), int)
                or entry.get("target_event_watermark")
                != entry.get("prior_event_watermark") + 2
            ):
                errors.append(
                    f"M{index + 1} history must use @1 and add exactly two events"
                )
        for previous, current in zip(history, history[1:], strict=False):
            if any(
                current.get(prior_key) != previous.get(target_key)
                for prior_key, target_key in (
                    ("prior_store_id", "target_store_id"),
                    ("prior_control_store_sha256", "target_control_store_sha256"),
                    ("prior_event_watermark", "target_event_watermark"),
                    ("prior_event_prefix_sha256", "target_event_prefix_sha256"),
                    ("prior_source_binding_cid", "current_source_binding_cid"),
                    ("prior_source_head", "current_source_head"),
                    ("prior_source_tree", "current_source_tree"),
                )
            ):
                errors.append("migration history predecessor/target continuity is broken")
                break
        m4 = history[3]
        expected_m4 = {
            "schema": "sawm/source-migration-history-entry@2",
            "migration_revision": "SAWM-R2-M4",
            "prior_event_watermark": 113,
            "migration_event_watermark": 115,
            "target_event_watermark": 116,
            "migration_event_prefix_sha256": (
                "724c47eaf1c70d6ddcf61059f133c752f2787a619aba040e4f2334ecbd287731"
            ),
            "target_event_prefix_sha256": m4_event_prefix,
            "migration_projection_cid": (
                "baguqeeramvvb5ij2hs4eniw735kyvfv3t253vp3tn7atk6vmw2qrcfuqf3sq"
            ),
            "projection_cid": m4_projection,
            "target_control_store_sha256": m4_control_sha256,
            "target_store_id": prior_store,
            "current_source_binding_cid": m4_source_binding,
            "post_migration_event_id": (
                "baguqeera3qvtzkbfquvcwhk3weocoesvqlajklm736surlkl25jn2ljkllha"
            ),
            "post_migration_event_type": "intent.task_status_changed",
            "post_migration_task_cid": sawm_001_cid,
            "post_migration_task_revision": 2,
            "post_migration_task_status": "in_progress",
        }
        if any(m4.get(key) != value for key, value in expected_m4.items()):
            errors.append(
                "M4 @2 history does not distinguish watermark 115 migration "
                "from watermark 116 frozen target"
            )
        if (
            m4.get("migration_receipt_cid")
            != migration.get("prior_materialization_receipt_cid")
            or m4.get("migration_receipt_path")
            != migration.get("prior_materialization_receipt_path")
        ):
            errors.append("M4 history receipt does not bind the frozen predecessor")

    failure = migration.get("preprovider_task_failure")
    if not isinstance(failure, Mapping):
        errors.append("frozen M4 pre-provider task failure is absent")
        failure = {}
    expected_failure = {
        "schema": "sawm/pre-provider-task-failure@1",
        "authority_class": "operator_frozen_predecessor_observation",
        "authoritative_completion_evidence": False,
        "store_id": prior_store,
        "control_store_sha256": m4_control_sha256,
        "database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
        "plan_revision": 5,
        "owner_generation": 6,
        "source_binding_cid": m4_source_binding,
        "source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "canonical_event_watermark": 116,
        "canonical_event_prefix_sha256": m4_event_prefix,
        "canonical_projection_cid": m4_projection,
        "canonical_claim_event_id": (
            "baguqeera3qvtzkbfquvcwhk3weocoesvqlajklm736surlkl25jn2ljkllha"
        ),
        "canonical_claim_event_type": "intent.task_status_changed",
        "task_alias": "SAWM-001",
        "task_cid": sawm_001_cid,
        "task_previous_status": "todo",
        "task_status": "in_progress",
        "task_revision": 2,
        "execution_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m4/control.execution.duckdb"
        ),
        "execution_store_sha256": (
            "84e010b517a791039a6db16c2860a3292f6fb123d9884521ea52dc31f4223989"
        ),
        "coordination_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m4/control.coordination.duckdb"
        ),
        "coordination_store_sha256": (
            "6e4211ed41f02e84e95877a20f94d56660f94d00e2bb80769d4fa4550032a3a2"
        ),
        "portal_attempt_path": (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m4/"
            "state/sawm_database_portal_attempts/72474c065ac95c1e878159f3"
        ),
        "portal_attempt_binding_path": (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m4/"
            "state/sawm_database_portal_attempts/72474c065ac95c1e878159f3/"
            "database-attempt-binding.json"
        ),
        "portal_attempt_binding_id": (
            "sha256:7c9ef3b030749b39ee01b772a3b54b43a77035dbb16f7ea1ec440f4646009f83"
        ),
        "portal_attempt_binding_sha256": (
            "db00bde3459f4eb4973532110647baabe2b2ca335a6704b3dbd22f706ef0e92f"
        ),
        "portal_task_projection_sha256": (
            "5710f2c084a967201a9c0ae1fd19d4aaaf7cb5257799462fba2eb768ac31dd01"
        ),
        "portal_event_log_sha256": (
            "aca7c522a6868965405696bfcadcc88ca47236eebfca6ca50171e20c9c9a035e"
        ),
        "portal_event_manifest_sha256": (
            "48fd39f4e4fe4a4c0d744111589d21eb84cc524fbc8a8c3ae67e458bec0bc67f"
        ),
        "portal_event_snapshot_id": (
            "event-log-snapshot:sha256:"
            "806ba18bf8230033aadf344c166a193b85c8cf77fd4a21792e4e61cd8ad02171"
        ),
        "portal_event_tail_id": (
            "sha256:46e15c941d784c8e7e986b2668bd2a457a13777c3c08297ec8f936c31732590d"
        ),
        "portal_event_count": 13,
        "portal_event_first_sequence": 1,
        "portal_event_last_sequence": 13,
        "database_attempt_terminal_reason": (
            "external_protected_checkout_recovery_required"
        ),
        "attempt_consumed": False,
        "retry_deferred": True,
        "successor_migration_revision": "SAWM-R2-M5",
        "successor_plan_revision": 6,
        "successor_generation": 7,
        "successor_store_id": target_store,
    }
    if any(failure.get(key) != value for key, value in expected_failure.items()):
        errors.append("frozen M4 control/companion-store/portal evidence is not exact")
    if failure.get("canonical_authority_counts") != {
        "completion_receipts": 1,
        "effect_claims": 0,
        "merge_attempts": 0,
        "provider_calls": 0,
        "provider_invocations": 0,
        "provider_responses": 0,
        "task_assignments": 0,
        "task_attempts": 0,
        "task_claims": 0,
    }:
        errors.append("frozen M4 canonical authority counts are not exact")
    if failure.get("execution_authority_counts") != {
        "attempt_phases": 3,
        "database_task_attempts": 1,
        "effect_claims": 0,
        "provider_invocations": 0,
    } or failure.get("coordination_authority_counts") != {
        "fenced_leases": 1,
        "resource_claims": 0,
        "task_attempts": 1,
        "task_claims": 1,
        "task_completions": 0,
    }:
        errors.append("frozen M4 companion authority counts are not exact")
    for field in (
        "provider_call_allowed",
        "provider_dispatch_attempted",
        "provider_dispatched",
        "provider_invocation_recorded",
        "implementation_provider_invoked",
        "effect_claim_recorded",
        "implementation_commit_created",
        "merge_attempted",
        "task_completed",
    ):
        if failure.get(field) is not False:
            errors.append(f"frozen M4 evidence must retain {field}=false")
    if failure.get("merge_queue_request_count") != 0:
        errors.append("frozen M4 evidence must retain zero merge requests")

    historical_failure = migration.get("preworker_launch_failure")
    if not isinstance(historical_failure, Mapping) or (
        len(history) == 4
        and (
            historical_failure.get("store_id") != history[2].get("target_store_id")
            or historical_failure.get("source_head")
            != history[2].get("current_source_head")
            or historical_failure.get("source_tree")
            != history[2].get("current_source_tree")
        )
    ):
        errors.append("historical M3 pre-worker failure binding is not preserved")

    repair_paths = tuple(migration.get("bounded_control_plane_repair_paths") or ())
    required_repair_paths = {
        "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
        "test/api/test_agent_supervisor_database_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "test/api/test_agent_supervisor_project_dependency_preflight.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
        "test/api/test_agent_supervisor_provider_command_binding.py",
    }
    if (
        not repair_paths
        or len(repair_paths) != len(set(repair_paths))
        or any(path not in CONTROL_RELATIVE_PATHS for path in repair_paths)
        or not required_repair_paths.issubset(repair_paths)
    ):
        errors.append("M5 bounded repair sources/tests are not exact protected controls")
    return errors


def _stable_regular_bytes(path: Path, *, maximum: int) -> bytes:
    if not path.is_absolute():
        raise ValueError("path is not absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("source is unavailable") from exc
    if resolved != path:
        raise ValueError("source path is noncanonical or contains a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("source is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ValueError("source is not stable owner-held regular-file evidence")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("source changed after read") from exc

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(after) != identity(current)
    ):
        raise ValueError("source changed while read")
    return raw


def _extension_pin_errors(
    name: str,
    configured: object,
    sealed: object,
) -> list[str]:
    errors: list[str] = []
    extra = {"service_external_access_limitation"} if name == "quack" else set()
    fields = {
        "path",
        "info_path",
        "version",
        "sha256",
        "size",
        "info_sha256",
        "info_size",
        "network_install_allowed",
        "unsigned_extension_allowed",
        *extra,
    }
    if type(configured) is not dict or type(sealed) is not dict:
        return [f"{name} extension pin must be an exact object"]
    if configured != sealed:
        errors.append(f"scheduler {name} pin differs from the dependency seal")
    pin = sealed
    if set(pin) != fields:
        errors.append(f"sealed {name} extension pin fields are noncanonical")
        return errors
    path_value = pin.get("path")
    info_path_value = pin.get("info_path")
    version = pin.get("version")
    payload_digest = pin.get("sha256")
    info_digest = pin.get("info_sha256")
    payload_size = pin.get("size")
    info_size = pin.get("info_size")
    if (
        type(path_value) is not str
        or type(info_path_value) is not str
        or type(version) is not str
        or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}", version) is None
        or type(payload_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", payload_digest) is None
        or type(info_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", info_digest) is None
        or type(payload_size) is not int
        or not 0 < payload_size <= 64 * 1024 * 1024
        or type(info_size) is not int
        or not 0 < info_size <= 64 * 1024
        or pin.get("network_install_allowed") is not False
        or pin.get("unsigned_extension_allowed") is not False
        or (
            name == "quack"
            and pin.get("service_external_access_limitation")
            != "canonical_writer_sealed; "
            "pinned_extension_preloaded_only_in_locked_read_only_loopback_replica"
        )
    ):
        errors.append(f"sealed {name} extension pin values are noncanonical")
        return errors
    payload_path = Path(path_value)
    info_path = Path(info_path_value)
    if (
        payload_path.name != f"{name}.duckdb_extension"
        or info_path != payload_path.with_name(f"{payload_path.name}.info")
    ):
        errors.append(f"sealed {name} extension paths are not exact")
        return errors
    try:
        payload = _stable_regular_bytes(payload_path, maximum=64 * 1024 * 1024)
        info = _stable_regular_bytes(info_path, maximum=64 * 1024)
    except ValueError as exc:
        errors.append(f"sealed {name} extension source is invalid: {exc}")
        return errors
    if len(payload) != payload_size or hashlib.sha256(payload).hexdigest() != payload_digest:
        errors.append(f"sealed {name} extension payload differs from its pin")
    if len(info) != info_size or hashlib.sha256(info).hexdigest() != info_digest:
        errors.append(f"sealed {name} extension metadata differs from its pin")
    return errors


def _native_extension_identity(
    seal: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    """Derive the only extension engine/platform admitted by native DuckDB."""

    native = seal.get("configured_board_native_dependency")
    native_pin = native.get("pin") if type(native) is dict else None
    toolchain = seal.get("toolchain")
    if type(native_pin) is not dict or type(toolchain) is not dict:
        return "", "", ["native DuckDB/toolchain extension identity is absent"]

    native_platforms = {
        ("linux", "aarch64"): "linux_arm64",
        ("linux", "x86_64"): "linux_amd64",
        ("darwin", "arm64"): "osx_arm64",
        ("darwin", "x86_64"): "osx_amd64",
        ("win32", "AMD64"): "windows_amd64",
    }
    toolchain_platforms = {
        ("Linux", "aarch64"): "linux_arm64",
        ("Linux", "x86_64"): "linux_amd64",
        ("Darwin", "arm64"): "osx_arm64",
        ("Darwin", "x86_64"): "osx_amd64",
        ("Windows", "AMD64"): "windows_amd64",
    }
    native_platform = native_platforms.get(
        (
            str(native_pin.get("platform_name") or ""),
            str(native_pin.get("platform_machine") or ""),
        )
    )
    toolchain_platform = toolchain_platforms.get(
        (
            str(toolchain.get("operating_system") or ""),
            str(toolchain.get("machine") or ""),
        )
    )
    engine = str(native_pin.get("engine_version") or "")
    distribution = str(native_pin.get("distribution_version") or "")
    toolchain_version = str(toolchain.get("duckdb_distribution_version") or "")
    errors: list[str] = []
    if (
        not engine
        or engine != f"v{distribution}"
        or distribution != toolchain_version
    ):
        errors.append("native DuckDB engine identity differs from the sealed toolchain")
    if (
        native_platform is None
        or toolchain_platform is None
        or native_platform != toolchain_platform
    ):
        errors.append("native DuckDB platform identity differs from the sealed toolchain")
    return engine, native_platform or "", errors


def _configured_board_dependency_errors(
    root: Path,
    config: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> list[str]:
    """Validate exact configured-board dependency and local extension custody."""

    errors: list[str] = []
    expected_paths = {
        "validator_path": "scripts/validate_semantic_addressed_world_model_board.py",
        "dependency_validator_path": (
            "scripts/validate_semantic_addressed_world_model_dependencies.py"
        ),
        "dependency_seal_path": (
            "config/semantic_addressed_world_model_dependencies.seal.json"
        ),
    }
    for field, expected in expected_paths.items():
        if config.get(field) != expected:
            errors.append(f"scheduler {field} does not bind {expected}")
    try:
        dependency_validator = _dependency_validator_module(root)
        _runtime, runtime_errors = (
            dependency_validator._validation_runtime_contract_errors(config, seal)
        )
        errors.extend(runtime_errors)
    except Exception as exc:
        errors.append(
            "validation runtime contract validator unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    environment_policy = seal.get("environment_policy")
    fixed_environment = (
        environment_policy.get("fixed")
        if isinstance(environment_policy, Mapping)
        and isinstance(environment_policy.get("fixed"), Mapping)
        else {}
    )
    if (
        not isinstance(environment_policy, Mapping)
        or environment_policy.get("duckdb_extension_home")
        != "exact_private_read_only_projection"
        or fixed_environment.get(
            "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY"
        )
        != "disabled"
    ):
        errors.append(
            "configured-board extension projection/install environment is not sealed"
        )
    validation = seal.get("validation")
    if (
        type(validation) is not dict
        or set(validation)
        != {
            "validator",
            "expected_result",
            "current_tree_revalidation_required_before_launch",
            "network_required",
            "database_open_required_for_static_seal_validation",
            "seal_mismatch_disposition",
        }
        or validation.get("validator") != expected_paths["dependency_validator_path"]
        or validation.get("current_tree_revalidation_required_before_launch") is not True
        or validation.get("network_required") is not False
        or validation.get("database_open_required_for_static_seal_validation") is not False
        or validation.get("seal_mismatch_disposition") != "fail_closed"
    ):
        errors.append("dependency seal validation policy is not exact and fail-closed")
    if any(
        expected not in tuple(config.get("protected_paths") or ())
        for expected in expected_paths.values()
    ):
        errors.append("scheduler validator/seal bindings are not protected paths")

    quack_owner = config.get("quack_owner")
    if not isinstance(quack_owner, Mapping):
        errors.append("scheduler Quack owner is absent")
        quack_owner = {}
    expected_engine, expected_platform, identity_errors = (
        _native_extension_identity(seal)
    )
    errors.extend(identity_errors)
    quack_pin = seal.get("quack_extension_pin")
    httpfs_pin = seal.get("httpfs_extension_pin")
    errors.extend(
        _extension_pin_errors("quack", quack_owner.get("pinned_extension"), quack_pin)
    )
    errors.extend(
        _extension_pin_errors(
            "httpfs",
            quack_owner.get("pinned_httpfs_extension"),
            httpfs_pin,
        )
    )
    if type(quack_pin) is dict and type(httpfs_pin) is dict:
        quack_path = Path(str(quack_pin.get("path") or ""))
        httpfs_path = Path(str(httpfs_pin.get("path") or ""))
        if quack_path.parent != httpfs_path.parent:
            errors.append("Quack and httpfs pins do not share one exact engine/platform root")
        for name, extension_path in (
            ("quack", quack_path),
            ("httpfs", httpfs_path),
        ):
            if (
                extension_path.parent.name != expected_platform
                or extension_path.parent.parent.name != expected_engine
            ):
                errors.append(
                    f"{name} extension path engine/platform differs from "
                    "native DuckDB/toolchain"
                )

    projection = seal.get("configured_board_quack_projection")
    expected_projection_fields = {
        "schema",
        "source_path",
        "info_path",
        "pin",
        "load_policy",
        "network_install_allowed",
        "unsigned_extension_allowed",
    }
    if type(projection) is not dict or set(projection) != expected_projection_fields:
        errors.append("configured-board Quack projection fields are noncanonical")
    else:
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
                inspect_configured_board_extension_sources,
                parse_configured_board_extension_pin,
            )

            projection_pin = parse_configured_board_extension_pin(projection.get("pin"))
            observed_projection = inspect_configured_board_extension_sources(
                projection.get("source_path"),
                projection.get("info_path"),
                name=projection_pin.name,
                engine_version=projection_pin.engine_version,
                platform=projection_pin.platform,
            )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            errors.append(
                "configured-board Quack projection is invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if (
                observed_projection != projection_pin
                or projection.get("schema")
                != "semantic-addressed-world-model/configured-board-quack-projection@1"
                or projection.get("load_policy") != "local_load_only"
                or projection.get("network_install_allowed") is not False
                or projection.get("unsigned_extension_allowed") is not False
                or type(quack_pin) is not dict
                or projection.get("source_path") != quack_pin.get("path")
                or projection.get("info_path") != quack_pin.get("info_path")
                or projection_pin.payload_sha256
                != "sha256:" + str(quack_pin.get("sha256") or "")
                or projection_pin.payload_size != quack_pin.get("size")
                or projection_pin.info_sha256
                != "sha256:" + str(quack_pin.get("info_sha256") or "")
                or projection_pin.info_size != quack_pin.get("info_size")
            ):
                errors.append("configured-board Quack projection differs from its sealed source")
            if (
                projection_pin.engine_version != expected_engine
                or projection_pin.platform != expected_platform
            ):
                errors.append(
                    "configured-board extension engine/platform differs from "
                    "native DuckDB/toolchain"
                )
            projection_source = Path(str(projection.get("source_path") or ""))
            if (
                projection_source.parent.name != projection_pin.platform
                or projection_source.parent.parent.name
                != projection_pin.engine_version
            ):
                errors.append("configured-board Quack projection engine/platform path mismatches")

    native = seal.get("configured_board_native_dependency")
    native_fields = {
        "schema",
        "source_path",
        "acceptance",
        "pin",
        "sealed_memfd_required",
        "ambient_site_import_allowed",
        "ambient_loader_environment_allowed",
    }
    if type(native) is not dict or set(native) != native_fields:
        errors.append("configured-board native dependency fields are noncanonical")
        return errors
    acceptance = native.get("acceptance")
    if type(acceptance) is not dict or set(acceptance) != {
        "schema",
        "path",
        "sha256",
        "size",
        "authorization_id",
    }:
        errors.append("native dependency authorization reference is noncanonical")
        return errors
    if (
        native.get("schema")
        != "semantic-addressed-world-model/configured-board-native-dependency@1"
        or type(native.get("source_path")) is not str
        or not Path(native["source_path"]).is_absolute()
        or native.get("sealed_memfd_required") is not True
        or native.get("ambient_site_import_allowed") is not False
        or native.get("ambient_loader_environment_allowed") is not False
        or acceptance.get("schema")
        != "semantic-addressed-world-model/native-dependency-authorization-reference@1"
        or acceptance.get("path") != NATIVE_AUTHORIZATION_PATH
    ):
        errors.append("configured-board native dependency policy is not exact")
        return errors
    try:
        authorization_path = (root / NATIVE_AUTHORIZATION_PATH).resolve(strict=True)
        if authorization_path != root / NATIVE_AUTHORIZATION_PATH:
            raise ValueError("authorization path is noncanonical")
        authorization_raw = _stable_regular_bytes(
            authorization_path,
            maximum=32 * 1024,
        )
        authorization = json.loads(
            authorization_raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
        )
        from ipfs_accelerate_py.agent_implementation_route import (
            inspect_agent_supervisor_native_dependency_source,
            parse_agent_supervisor_native_dependency_pin,
        )

        native_pin = parse_agent_supervisor_native_dependency_pin(native.get("pin"))
        observed_native_pin = inspect_agent_supervisor_native_dependency_source(
            native.get("source_path"),
            distribution_version=native_pin.distribution_version,
            engine_version=native_pin.engine_version,
        )
    except (
        ImportError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        errors.append(
            "configured-board native dependency is invalid: "
            f"{type(exc).__name__}: {exc}"
        )
        return errors
    authorization_fields = {
        "schema",
        "board_namespace",
        "plan_revision",
        "status",
        "scope",
        "dependency_id",
        "payload_sha256",
        "python_executable_sha256",
        "authority_basis",
        "inspection_is_authority",
        "authorization_may_claim_task_completion",
        "authorization_id",
    }
    if type(authorization) is not dict or set(authorization) != authorization_fields:
        errors.append("native dependency authorization fields are noncanonical")
        return errors
    authorization_id = _canonical_identity(
        authorization,
        identity_field="authorization_id",
    )
    if (
        observed_native_pin != native_pin
        or native.get("source_path") != str(Path(native.get("source_path")))
        or Path(native.get("source_path")).name != native_pin.extension_filename
        or acceptance.get("size") != len(authorization_raw)
        or acceptance.get("sha256")
        != "sha256:" + hashlib.sha256(authorization_raw).hexdigest()
        or acceptance.get("authorization_id") != authorization_id
        or authorization.get("authorization_id") != authorization_id
        or authorization.get("schema")
        != "semantic-addressed-world-model/native-dependency-launch-authorization@1"
        or authorization.get("board_namespace") != BOARD_NAMESPACE
        or authorization.get("plan_revision") != PLAN_REVISION
        or authorization.get("status") != "accepted"
        or authorization.get("scope") != "configured-board-live-control-plane"
        or authorization.get("authority_basis")
        != "operator-owned protected control inside the accepted immutable source capsule"
        or authorization.get("dependency_id") != native_pin.dependency_id
        or authorization.get("payload_sha256") != native_pin.payload_sha256
        or authorization.get("python_executable_sha256")
        != native_pin.python_executable_sha256
        or authorization.get("inspection_is_authority") is not False
        or authorization.get("authorization_may_claim_task_completion") is not False
    ):
        errors.append("native dependency authorization does not bind the exact sealed pin")
    toolchain = seal.get("toolchain") if isinstance(seal.get("toolchain"), Mapping) else {}
    if (
        native_pin.distribution_version != toolchain.get("duckdb_distribution_version")
        or native_pin.engine_version
        != f"v{toolchain.get('duckdb_distribution_version', '')}"
    ):
        errors.append("native dependency version differs from the sealed DuckDB toolchain")
    return errors


def validate_program(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    required = (
        root / PLAN_PATH.relative_to(REPO_ROOT),
        root / OBJECTIVES_PATH.relative_to(REPO_ROOT),
        root / TODO_PATH.relative_to(REPO_ROOT),
        root / SEAL_PATH.relative_to(REPO_ROOT),
        root / SCHEDULER_PATH.relative_to(REPO_ROOT),
        root / BENCHMARK_PATH.relative_to(REPO_ROOT),
        *(root / path.relative_to(REPO_ROOT) for path in INVENTORY_PATHS),
        *(root / path for path in CONTROL_RELATIVE_PATHS if path.startswith(("scripts/", "test/", "benchmarks/"))),
    )
    missing = sorted(path.relative_to(root).as_posix() for path in required if not path.is_file())
    _append(checks, errors, "control_files_present", not missing, missing)
    if missing:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-board-validation@1",
            "valid": False,
            "board_namespace": BOARD_NAMESPACE,
            "plan_revision": PLAN_REVISION,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    try:
        tasks = _parse_cards(root / TODO_PATH.relative_to(REPO_ROOT), goal=False)
        goals = _parse_cards(root / OBJECTIVES_PATH.relative_to(REPO_ROOT), goal=True)
        config = _load_json(root / SCHEDULER_PATH.relative_to(REPO_ROOT))
        seal = _load_json(root / SEAL_PATH.relative_to(REPO_ROOT))
        benchmark = _load_json(root / BENCHMARK_PATH.relative_to(REPO_ROOT))
        migration = _load_json(
            root
            / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json"
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _append(checks, errors, "control_documents_parse", False, f"{type(exc).__name__}: {exc}")
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-board-validation@1",
            "valid": False,
            "board_namespace": BOARD_NAMESPACE,
            "plan_revision": PLAN_REVISION,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    task_ids = tuple(card.identifier for card in tasks)
    goal_ids = tuple(card.identifier for card in goals)
    _append(checks, errors, "task_population", task_ids == TASK_IDS, {"expected": TASK_IDS, "observed": task_ids})
    _append(checks, errors, "task_ids_unique", len(set(task_ids)) == len(task_ids), sorted(k for k, v in Counter(task_ids).items() if v != 1))
    _append(checks, errors, "goal_population", goal_ids == GOAL_IDS, {"expected": GOAL_IDS, "observed": goal_ids})
    _append(checks, errors, "goal_ids_unique", len(set(goal_ids)) == len(goal_ids), sorted(k for k, v in Counter(goal_ids).items() if v != 1))

    missing_task_fields = {
        card.identifier: [field for field in TASK_FIELDS if field not in card.metadata]
        for card in tasks
        if any(field not in card.metadata for field in TASK_FIELDS)
    }
    missing_goal_fields = {
        card.identifier: [field for field in GOAL_FIELDS if field not in card.metadata]
        for card in goals
        if any(field not in card.metadata for field in GOAL_FIELDS)
    }
    _append(checks, errors, "task_fields_closed", not missing_task_fields, missing_task_fields)
    _append(checks, errors, "goal_fields_closed", not missing_goal_fields, missing_goal_fields)

    goal_dependencies: dict[str, tuple[str, ...]] = {}
    goal_structure_errors: list[str] = []
    for card in goals:
        meta = card.metadata
        if meta.get("stable goal id") != card.identifier:
            goal_structure_errors.append(f"{card.identifier}: stable goal id mismatch")
        if meta.get("status", "").lower() not in ALLOWED_GOAL_STATUSES:
            goal_structure_errors.append(f"{card.identifier}: closed goal status violation")
        parent = meta.get("parent", "").strip()
        if parent.lower() in {"none", "null", "root", ""}:
            parent = ""
        if parent != GOAL_PARENT.get(card.identifier, "unknown"):
            goal_structure_errors.append(
                f"{card.identifier}: parent {parent!r} != {GOAL_PARENT.get(card.identifier)!r}"
            )
        parent_json = [str(item) for item in _json_list(card, "parent goal ids json", goal_structure_errors)]
        expected_parent_json = [] if not parent else [parent]
        if parent_json != expected_parent_json:
            goal_structure_errors.append(f"{card.identifier}: parent goal IDs JSON mismatch")
        deps = tuple(str(item) for item in _json_list(card, "dependencies json", goal_structure_errors))
        if set(deps) != set(_csv(meta.get("depends on", ""))):
            goal_structure_errors.append(f"{card.identifier}: dependency text/JSON mismatch")
        if any(dep not in GOAL_IDS or dep == card.identifier for dep in deps):
            goal_structure_errors.append(f"{card.identifier}: invalid goal dependency")
        goal_dependencies[card.identifier] = deps
    goal_acyclic, goal_cycle = _acyclic(goal_dependencies)
    if not goal_acyclic:
        goal_structure_errors.append("goal dependency cycle: " + " -> ".join(goal_cycle))
    _append(checks, errors, "goal_hierarchy_and_dependencies", not goal_structure_errors, goal_structure_errors)

    task_dependencies: dict[str, tuple[str, ...]] = {}
    task_structure_errors: list[str] = []
    task_outputs: dict[str, tuple[str, ...]] = {}
    task_predicted: dict[str, tuple[str, ...]] = {}
    for card in tasks:
        meta = card.metadata
        if meta.get("stable task id") != card.identifier:
            task_structure_errors.append(f"{card.identifier}: stable task id mismatch")
        status = meta.get("status", "").lower()
        if status not in ALLOWED_TASK_STATUSES:
            task_structure_errors.append(f"{card.identifier}: status {status!r} is outside the closed initial set")
        expected_status = "completed" if card.identifier == "SAWM-000" else "todo"
        if status != expected_status:
            task_structure_errors.append(f"{card.identifier}: initial status must be {expected_status}")
        deps = tuple(str(item) for item in _json_list(card, "dependencies json", task_structure_errors))
        if set(deps) != set(_csv(meta.get("depends on", ""))):
            task_structure_errors.append(f"{card.identifier}: dependency text/JSON mismatch")
        if any(dep not in TASK_IDS or dep == card.identifier for dep in deps):
            task_structure_errors.append(f"{card.identifier}: invalid task dependency")
        task_dependencies[card.identifier] = deps
        for field in ("goal id", "parent goal id", "subgoal id"):
            if meta.get(field) not in GOAL_IDS:
                task_structure_errors.append(f"{card.identifier}: {field} is not a current goal")
        if meta.get("board namespace") != BOARD_NAMESPACE or meta.get("plan revision") != PLAN_REVISION:
            task_structure_errors.append(f"{card.identifier}: namespace or plan revision mismatch")
        task_outputs[card.identifier] = _paths(
            _json_list(card, "outputs json", task_structure_errors),
            card=card, field="outputs json", errors=task_structure_errors,
        )
        task_predicted[card.identifier] = _paths(
            _json_list(card, "predicted files json", task_structure_errors),
            card=card, field="predicted files json", errors=task_structure_errors,
        )
    task_acyclic, task_cycle = _acyclic(task_dependencies)
    if not task_acyclic:
        task_structure_errors.append("task dependency cycle: " + " -> ".join(task_cycle))
    if task_dependencies.get("SAWM-000"):
        task_structure_errors.append("SAWM-000 must not depend on an implementation task")
    if task_dependencies.get("SAWM-001") != ("SAWM-000",):
        task_structure_errors.append("SAWM-001 must be the sole initial implementation frontier")
    _append(checks, errors, "task_dependencies_and_goal_bindings", not task_structure_errors, task_structure_errors)

    rollout_errors: list[str] = []
    rollout_chain = (("SAWM-035", "SAWM-034"), ("SAWM-036", "SAWM-035"), ("SAWM-037", "SAWM-036"))
    for task, dependency in rollout_chain:
        if not _depends_transitively(task_dependencies, task, dependency):
            rollout_errors.append(f"{task} must depend transitively on {dependency}")
    for task in REQUIRED_MODE_TASKS:
        card = tasks[TASK_IDS.index(task)] if task in task_ids else None
        if card is not None and card.metadata.get("rollout mode", "").lower() != "required":
            rollout_errors.append(f"{task}: rollout mode must be required")
        if task != "SAWM-038" and not _depends_transitively(task_dependencies, task, "SAWM-037"):
            rollout_errors.append(f"{task} must transitively follow required-mode activation")
    if not _depends_transitively(task_dependencies, "SAWM-044", "SAWM-043"):
        rollout_errors.append("SAWM-044 must transitively follow the self-hosted capstone")
    _append(checks, errors, "rollout_ordering", not rollout_errors, rollout_errors)

    completion_errors: list[str] = []
    for card in tasks:
        meta = card.metadata
        if card.identifier == "SAWM-000":
            if meta.get("completion mode", "").lower() != "trusted_manual":
                completion_errors.append("SAWM-000 completion mode must be trusted_manual")
            if _bool(meta.get("is schedulable", "")) is not False or _bool(meta.get("review only", "")) is not True:
                completion_errors.append("SAWM-000 must be non-schedulable operator review")
            operator_text = " ".join(
                meta.get(field, "") for field in ("provider role", "evidence requirements", "acceptance")
            ).lower()
            if "operator" not in operator_text or "current" not in operator_text or "receipt" not in operator_text:
                completion_errors.append("SAWM-000 must require operator current-tree receipt evidence")
        else:
            if meta.get("completion mode", "").lower() != "automatic":
                completion_errors.append(f"{card.identifier}: completion mode must be automatic")
            if _bool(meta.get("is schedulable", "")) is not True or _bool(meta.get("review only", "")) is not False:
                completion_errors.append(f"{card.identifier}: implementation task schedulability mismatch")
        if not meta.get("evidence requirements", "").strip() or not meta.get("acceptance", "").strip():
            completion_errors.append(f"{card.identifier}: evidence/acceptance is empty")
    if config.get("authority_policy", {}).get("markdown_is_task_completion_authority") is not False:
        completion_errors.append("scheduler must make Markdown non-authoritative for completion")
    _append(checks, errors, "completion_authority", not completion_errors, completion_errors)

    validation_errors: list[str] = []
    forbidden_validation = re.compile(r"(?:^|\s)(?:pip|pip3)\s+install\b|\bcurl\b|\bwget\b|git\s+clean|reset\s+--hard", re.I)
    historical_commands: dict[str, list[str]] = {}
    for card in tasks:
        commands = _json_list(card, "validation commands json", validation_errors)
        historical_commands[card.identifier] = [
            command for command in commands if isinstance(command, str)
        ]
        if not commands:
            validation_errors.append(f"{card.identifier}: validation command list is empty")
        for command in commands:
            if not isinstance(command, str) or not command.startswith(HISTORICAL_VALIDATION_PREFIX):
                validation_errors.append(
                    f"{card.identifier}: immutable Markdown validation definition changed"
                )
            elif forbidden_validation.search(command):
                validation_errors.append(f"{card.identifier}: validation contains forbidden installer/network/destructive command")
        if not card.metadata.get("validation", "").strip():
            validation_errors.append(f"{card.identifier}: human-readable validation is empty")
        network = card.metadata.get("network policy", "").lower()
        if not any(term in network for term in ("no network", "network disabled", "offline", "deny")):
            validation_errors.append(f"{card.identifier}: ordinary validation must be network-disabled")
    nonoperator_commands = [
        command
        for task_id in TASK_IDS[1:]
        for command in historical_commands.get(task_id, ())
    ]
    replacement_count = sum(
        command.count("/home/barberb/.local/bin/python")
        for command in nonoperator_commands
    )
    operational_commands = [
        command.replace("/home/barberb/.local/bin/python", "python")
        for command in nonoperator_commands
    ]
    if (
        len(operational_commands) != 46
        or replacement_count != 46
        or any(
            not command.startswith(OPERATIONAL_VALIDATION_PREFIX)
            or "/home/barberb/.local/bin/python" in command
            for command in operational_commands
        )
    ):
        validation_errors.append(
            "44-task M6 operational validation command revision is not exact"
        )
    else:
        try:
            from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
                validation_shell_command,
            )

            for command in operational_commands:
                validation_shell_command(command)
        except (ImportError, ValueError) as exc:
            validation_errors.append(
                "M6 bare-python operational validation is not launchable: "
                f"{type(exc).__name__}: {exc}"
            )
    _append(checks, errors, "current_validation_commands", not validation_errors, validation_errors)

    ownership_errors: list[str] = []
    owners_by_path: dict[str, list[str]] = defaultdict(list)
    protected = set(CONTROL_RELATIVE_PATHS)
    for card in tasks:
        owner = card.metadata.get("owning repository", "").strip().lower()
        normalized_owner = "operator" if "operator" in owner else next(
            (candidate for candidate in ("ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py") if candidate in owner),
            owner,
        )
        if normalized_owner not in {"operator", "ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py"}:
            ownership_errors.append(f"{card.identifier}: unknown repository owner {owner!r}")
        for path in task_outputs.get(card.identifier, ()):
            owners_by_path[path].append(card.identifier)
            if path in protected and card.identifier != "SAWM-000":
                ownership_errors.append(f"{card.identifier}: implementation task owns protected path {path}")
            actual_owner = _repository_for_path(path)
            if normalized_owner not in {"operator", actual_owner}:
                ownership_errors.append(
                    f"{card.identifier}: {path} belongs to {actual_owner}, not {normalized_owner}"
                )
        if card.identifier != "SAWM-000":
            for path in task_predicted.get(card.identifier, ()):
                if path in protected:
                    ownership_errors.append(f"{card.identifier}: predicts a protected-path mutation {path}")
    for path, task_list in sorted(owners_by_path.items()):
        if len(task_list) < 2:
            continue
        ordered = all(
            _depends_transitively(task_dependencies, later, earlier)
            or _depends_transitively(task_dependencies, earlier, later)
            for index, later in enumerate(task_list)
            for earlier in task_list[:index]
        )
        policies = " ".join(tasks[TASK_IDS.index(task)].metadata.get("conflict policy", "") for task in task_list).lower()
        if not ordered or not any(term in policies for term in ("serial", "successor", "integration owner")):
            ownership_errors.append(f"{path}: duplicate output owners are not explicitly serialized: {task_list}")
    _append(checks, errors, "exact_output_ownership_and_protection", not ownership_errors, ownership_errors)

    authority_errors: list[str] = []
    forbidden_positive = re.compile(
        r"\b(?:model|worker|prediction|retrieval result|ann result)\s+(?:may|can|shall|will)\s+"
        r"(?:self[- ]?)?(?:approve|accept|authorize|complete|prove)\b",
        re.I,
    )
    for card in tasks:
        rendered = " ".join(card.metadata.values())
        if forbidden_positive.search(rendered):
            authority_errors.append(f"{card.identifier}: declares model/worker self-authority")
        if not card.metadata.get("prohibited effects", "").strip():
            authority_errors.append(f"{card.identifier}: prohibited effects are empty")
        fallback = card.metadata.get("model fallback", "").lower()
        if fallback and not any(
            term in fallback
            for term in (
                "proposal", "untrusted", "residual", "none", "unavailable",
                "abstain", "forbidden", "bounded",
            )
        ):
            authority_errors.append(f"{card.identifier}: model fallback is not proposal/residual bounded")
    authority_policy = config.get("authority_policy")
    if not isinstance(authority_policy, Mapping):
        authority_errors.append("scheduler authority_policy is missing")
    else:
        required_false = (
            "markdown_is_task_completion_authority", "ann_or_neural_result_is_authoritative",
            "model_can_self_approve", "worker_can_self_approve",
            "model_can_create_proof_authority", "model_can_create_completion_authority",
            "automatic_file_fallback_from_quack", "raw_llm_sql_allowed",
            "implementation_provider_receives_state_credentials",
        )
        for field in required_false:
            if authority_policy.get(field) is not False:
                authority_errors.append(f"authority_policy.{field} must be false")
    _append(checks, errors, "no_self_approval_or_authority_weakening", not authority_errors, authority_errors)

    learned_errors: list[str] = []
    # Candidate/benchmark tasks are allowed to terminate typed-unavailable;
    # only the foundry/promotion task may admit a checkpoint.  Bind candidate
    # tasks to the corpus and frozen benchmark, then require the full promotion
    # vocabulary exactly where authority can actually be granted.
    for task_id in tuple(f"SAWM-{index:03d}" for index in range(25, 30)):
        if not _depends_transitively(task_dependencies, task_id, "SAWM-023"):
            learned_errors.append(f"{task_id}: does not transitively bind the admitted corpus")
        if not _depends_transitively(task_dependencies, task_id, "SAWM-024"):
            learned_errors.append(f"{task_id}: does not transitively bind the frozen held-out benchmark")
        card = tasks[TASK_IDS.index(task_id)]
        evidence = " ".join(card.metadata.get(field, "") for field in
                            ("preconditions", "evidence requirements", "acceptance", "limitations")).lower()
        if "calibration" not in evidence and "training_unavailable" not in evidence:
            learned_errors.append(f"{task_id}: lacks calibration or typed training unavailability")
        if not any(term in evidence for term in ("unavailable", "abstain", "ood")):
            learned_errors.append(f"{task_id}: lacks unavailable/OOD abstention")
    promotion = tasks[TASK_IDS.index("SAWM-030")]
    promotion_text = " ".join(promotion.metadata.get(field, "") for field in
                              ("preconditions", "evidence requirements", "acceptance", "limitations")).lower().replace("-", "_")
    for concept, alternatives in {
        "corpus": ("corpus", "training_unavailable"),
        "checkpoint": ("checkpoint", "training_unavailable"),
        "calibration": ("calibration", "training_unavailable"),
        "held_out": ("held_out", "held out", "training_unavailable"),
        "lineage": ("lineage", "training_unavailable"),
    }.items():
        if not any(term in promotion_text for term in alternatives):
            learned_errors.append(f"SAWM-030: missing promotion gate {concept}")
    if not _depends_transitively(task_dependencies, "SAWM-031", "SAWM-030"):
        learned_errors.append("SAWM-031 must consume the admitted foundry/promotion result")
    _append(checks, errors, "learned_capability_promotion_gates", not learned_errors, learned_errors)

    config_errors: list[str] = []
    if config.get("board_namespace") != BOARD_NAMESPACE or config.get("plan_revision") != PLAN_REVISION:
        config_errors.append("scheduler namespace/revision mismatch")
    projection = config.get("initial_projection") if isinstance(config.get("initial_projection"), Mapping) else {}
    if projection.get("task_count") != 45 or projection.get("goal_count") != 29:
        config_errors.append("initial projection population mismatch")
    if projection.get("completed_task_ids") != ["SAWM-000"] or projection.get("ready_task_ids") != ["SAWM-001"]:
        config_errors.append("initial projection frontier mismatch")
    m34_key = "json_emission_normalization_successor_materialization"
    m34_selected = any(
        (
            m34_key in config,
            m34_key in migration,
            f"{m34_key}_cid" in seal,
        )
    )
    m33_key = "live_preflight_contract_successor_materialization"
    m33_selected = any(
        (
            m33_key in config,
            m33_key in migration,
            f"{m33_key}_cid" in seal,
        )
    )
    m32_key = "live_preflight_plan_anchor_successor_materialization"
    m32_selected = any(
        (
            m32_key in config,
            m32_key in migration,
            f"{m32_key}_cid" in seal,
        )
    )
    m31_key = "detached_coordinator_pid_recovery_successor_materialization"
    m31_selected = any(
        (
            m31_key in config,
            m31_key in migration,
            f"{m31_key}_cid" in seal,
        )
    )
    m30_key = "stopped_owner_restart_source_seal_successor_materialization"
    m30_selected = any(
        (
            m30_key in config,
            m30_key in migration,
            f"{m30_key}_cid" in seal,
        )
    )
    m29_key = "committed_evidence_verification_successor_materialization"
    m29_selected = any(
        (
            m29_key in config,
            m29_key in migration,
            f"{m29_key}_cid" in seal,
        )
    )
    m28_key = "live_claim_admission_recovery_successor_materialization"
    m28_selected = any(
        (
            m28_key in config,
            m28_key in migration,
            f"{m28_key}_cid" in seal,
        )
    )
    m27_key = "dead_owner_parallel_resume_successor_materialization"
    m27_selected = any(
        (
            m27_key in config,
            m27_key in migration,
            f"{m27_key}_cid" in seal,
        )
    )
    m26_key = "automatic_stall_recovery_successor_materialization"
    m26_selected = any(
        (
            m26_key in config,
            m26_key in migration,
            f"{m26_key}_cid" in seal,
        )
    )
    m25_key = "native_duckdb_preload_successor_materialization"
    m25_selected = any(
        (
            m25_key in config,
            m25_key in migration,
            f"{m25_key}_cid" in seal,
        )
    )
    m24_key = "multi_lane_sidecar_reopen_successor_materialization"
    m24_selected = any(
        (
            m24_key in config,
            m24_key in migration,
            f"{m24_key}_cid" in seal,
        )
    )
    m23_key = "multi_lane_successor_materialization"
    m23_selected = not m25_selected and not m24_selected and any(
        (
            m23_key in config,
            m23_key in migration,
            f"{m23_key}_cid" in seal,
        )
    )
    multi_lane_selected = (
        m34_selected
        or m33_selected
        or m32_selected
        or m31_selected
        or m29_selected
        or m28_selected
        or m27_selected
        or m26_selected
        or m25_selected
        or m24_selected
        or m23_selected
    )
    expected_lane_count = 4 if multi_lane_selected else 1
    if (
        type(config.get("max_lanes")) is not int
        or config.get("max_lanes") != expected_lane_count
    ):
        config_errors.append(
            "four lanes are required for M34/M33/M32/M31/M30/M29/M28/M27/M26/M25/M24/M23"
            if multi_lane_selected
            else "one lane is required until sidecars are lane-scoped"
        )
    if (
        config.get("strict_task_sharding") is not True
        or config.get("idle_lane_work_stealing") != ""
    ):
        config_errors.append("strict no-stealing lane policy mismatch")
    lanes = config.get("lanes")
    if multi_lane_selected and (
        type(lanes) is not list
        or [
            (
                lane.get("index"),
                lane.get("name"),
                lane.get("strict_shard_remainder"),
                lane.get("initial_task_ids"),
            )
            for lane in lanes
            if type(lane) is dict
        ]
        != [
            (0, "sawm-lane-0", 0, ["SAWM-008"]),
            (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
            (2, "sawm-lane-2", 2, ["SAWM-015"]),
            (3, "sawm-lane-3", 3, ["SAWM-012"]),
        ]
        or any(type(lane) is not dict for lane in lanes)
        or any(
            type(value) is not int
            for lane in lanes
            if type(lane) is dict
            for value in (
                lane.get("index"),
                lane.get("strict_shard_remainder"),
            )
        )
    ):
        config_errors.append(
            "M31/M29/M28/M27/M26/M25/M24/M23 exact four-lane identity mismatch"
        )
    provider = config.get("provider") if isinstance(config.get("provider"), Mapping) else {}
    expected_provider = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.6",
        "primary_executable": "/home/barberb/.local/bin/grok",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "medium",
        "secrets_from_environment_only": True,
        "secrets_in_argv_prompts_logs_or_receipts": False,
        "probe_before_live_launch": True,
        "provider_results_are_completion_authority": False,
    }
    provider_without_cap = dict(provider)
    provider_cap = provider_without_cap.pop("max_concurrency", None)
    provider_cap_valid = (
        type(provider_cap) is int
        and (provider_cap >= 4 if multi_lane_selected else provider_cap == 1)
    )
    if (
        type(config.get("provider")) is not dict
        or provider_without_cap != expected_provider
        or not provider_cap_valid
    ):
        config_errors.append("ordered provider route mismatch")
    m22_key = "live_preflight_receipt_compatibility_successor_materialization"
    m22_selected = not m24_selected and not m23_selected and any(
        (
            m22_key in config,
            m22_key in migration,
            f"{m22_key}_cid" in seal,
        )
    )
    m21_key = "generation_realization_successor_materialization"
    m21_selected = not m24_selected and not m23_selected and not m22_selected and any(
        (
            m21_key in config,
            m21_key in migration,
            "generation_realization_successor_materialization_cid" in seal,
        )
    )
    m20_key = "test_isolation_successor_materialization"
    m20_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and any(
        (
            m20_key in config,
            m20_key in migration,
            "test_isolation_successor_materialization_cid" in seal,
        )
    )
    m19_key = "live_catalog_inventory_successor_materialization"
    m19_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and any(
        (
            m19_key in config,
            m19_key in migration,
            "live_catalog_inventory_successor_materialization_cid" in seal,
        )
    )
    m18_key = "portal_completion_persistence_successor_materialization"
    m18_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and any(
        (
            m18_key in config,
            m18_key in migration,
            "portal_completion_persistence_successor_materialization_cid" in seal,
        )
    )
    m17_key = "source_binding_successor_materialization"
    m17_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and any(
        (
            m17_key in config,
            m17_key in migration,
            "source_binding_successor_materialization_cid" in seal,
        )
    )
    m16_key = "accepted_source_retry_successor_materialization"
    m16_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and not m17_selected and any(
        (
            m16_key in config,
            m16_key in migration,
            "accepted_source_retry_successor_materialization_cid" in seal,
        )
    )
    m15_key = "runtime_root_rebind_successor_materialization"
    m15_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and not m17_selected and not m16_selected and any(
        (
            m15_key in config,
            m15_key in migration,
            "runtime_root_rebind_successor_materialization_cid" in seal,
        )
    )
    m14_key = "stale_owner_restart_successor_materialization"
    m14_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and not m17_selected and not m16_selected and not m15_selected and any(
        (
            m14_key in config,
            m14_key in migration,
            "stale_owner_restart_successor_materialization_cid" in seal,
        )
    )
    m13_key = "quack_refresh_successor_materialization"
    m13_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and not m17_selected and not m16_selected and not m15_selected and not m14_selected and any(
        (
            m13_key in config,
            m13_key in migration,
            "quack_refresh_successor_materialization_cid" in seal,
        )
    )
    m12_key = "declared_output_retry_successor_materialization"
    m12_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and not m17_selected and not m16_selected and not m15_selected and not m14_selected and not m13_selected and any(
        (
            m12_key in config,
            m12_key in migration,
            "declared_output_retry_successor_materialization_cid" in seal,
        )
    )
    m11_key = "live_provider_retry_successor_materialization"
    m11_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and not m17_selected and not m16_selected and not m15_selected and not m14_selected and not m13_selected and not m12_selected and any(
        (
            m11_key in config,
            m11_key in migration,
            "live_provider_retry_successor_materialization_cid" in seal,
        )
    )
    m10_key = "live_projection_successor_materialization"
    m10_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and not m17_selected and not m16_selected and not m15_selected and not m14_selected and not m13_selected and not m12_selected and not m11_selected and any(
        (
            m10_key in config,
            m10_key in migration,
            "live_projection_successor_materialization_cid" in seal,
        )
    )
    m9_key = "live_recovery_successor_materialization"
    m9_selected = not m24_selected and not m23_selected and not m22_selected and not m21_selected and not m20_selected and not m19_selected and not m18_selected and not m17_selected and not m16_selected and not m15_selected and not m14_selected and not m13_selected and not m12_selected and not m11_selected and not m10_selected and any(
        (
            m9_key in config,
            m9_key in migration,
            "live_recovery_successor_materialization_cid" in seal,
        )
    )
    active_run = (
        "run-r2-m27"
        if m34_selected
        else "run-r2-m27"
        if m33_selected
        else "run-r2-m27"
        if m32_selected
        else "run-r2-m27"
        if m31_selected
        else "run-r2-m27"
        if m30_selected
        else "run-r2-m27"
        if m29_selected
        else "run-r2-m27"
        if m28_selected
        else "run-r2-m27"
        if m27_selected
        else
        "run-r2-m26"
        if m26_selected
        else "run-r2-m25"
        if m25_selected
        else "run-r2-m24"
        if m24_selected
        else "run-r2-m23"
        if m23_selected
        else "run-r2-m22"
        if m22_selected
        else "run-r2-m21"
        if m21_selected
        else "run-r2-m20"
        if m20_selected
        else "run-r2-m19"
        if m19_selected
        else "run-r2-m18"
        if m18_selected
        else "run-r2-m17"
        if m17_selected
        else "run-r2-m16"
        if m16_selected
        else "run-r2-m15"
        if m15_selected
        else "run-r2-m14"
        if m14_selected
        else "run-r2-m13"
        if m13_selected
        else "run-r2-m12"
        if m12_selected
        else "run-r2-m11"
        if m11_selected
        else "run-r2-m10"
        if m10_selected
        else "run-r2-m9"
        if m9_selected
        else "run-r2-m8"
    )
    active_generation = (
        "29"
        if m34_selected
        else "29"
        if m33_selected
        else "29"
        if m32_selected
        else "29"
        if m31_selected
        else "28"
        if m30_selected
        else "27"
        if m29_selected
        else "27"
        if m28_selected
        else "26"
        if m27_selected
        else
        "25"
        if m26_selected
        else "24"
        if m25_selected
        else "24"
        if m24_selected
        else "23"
        if m23_selected
        else "22"
        if m22_selected
        else "21"
        if m21_selected
        else "21"
        if m20_selected
        else "20"
        if m19_selected
        else "19"
        if m18_selected
        else "18"
        if m17_selected
        else "17"
        if m16_selected
        else "16"
        if m15_selected
        else "15"
        if m14_selected
        else "14"
        if m13_selected
        else "14"
        if m12_selected
        else "13"
        if m11_selected
        else "12"
        if m10_selected
        else "11"
        if m9_selected
        else "10"
    )
    active_port = (
        24070
        if m34_selected
        else 24070
        if m33_selected
        else 24070
        if m32_selected
        else 24070
        if m31_selected
        else 24070
        if m30_selected
        else 24070
        if m29_selected
        else 24070
        if m28_selected
        else 24070
        if m27_selected
        else
        24069
        if m26_selected
        else 24068
        if m25_selected
        else 24067
        if m24_selected
        else 24066
        if m23_selected
        else 24065
        if m22_selected
        else 24064
        if m21_selected
        else 24063
        if m20_selected
        else 24062
        if m19_selected
        else 24061
        if m18_selected
        else 24060
        if m17_selected
        else 24059
        if m16_selected
        else 24058
        if m15_selected
        else 24057
        if m14_selected
        else 24056
        if m13_selected
        else 45256
        if m12_selected
        else 45255
        if m11_selected
        else 45253
        if m10_selected
        else 45251
        if m9_selected
        else 45250
    )
    active_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        f"{active_run}/control.duckdb"
    )
    program = config.get("database_program") if isinstance(config.get("database_program"), Mapping) else {}
    if (
        program.get("authority_mode") != "quack"
        or program.get("task_source_kind") != "duckdb"
        or program.get("quack_endpoint") != f"quack:127.0.0.1:{active_port}"
        or program.get("endpoint_secret_handle") != "env://SAWM_QUACK_TOKEN"
        or program.get("failover_policy") != "fail_closed"
        or program.get("store_generation") != active_generation
        or program.get("store_id") != active_store
    ):
        config_errors.append("DuckDB + Quack authority binding mismatch")
    if m34_selected:
        successor = config.get(m34_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m34_json_emission_normalization_authority()
            )
            reference = materializer._m34_authority_reference()
            source_chain = expected.get("source_chain", {})
            if (
                successor != reference
                or migration.get(m34_key) != reference
                or seal.get(f"{m34_key}_cid") != materializer._identity(expected)
                or source_chain.get("base_control_commit")
                != "a5aa77fe58cd706ed8a1a2ae9d3f1e652f28b7f9"
                or source_chain.get("base_control_tree")
                != "8fe38992e84c291a0cab6d2afc6cda5bb4f08ab8"
                or source_chain.get("initial_control_commit")
                != materializer._M34_INITIAL_CONTROL_COMMIT
                or source_chain.get("initial_control_commit")
                != "e342b63f3f143bb85ed4744e5391c4f8e7c961cd"
                or source_chain.get("initial_control_tree")
                != materializer._M34_INITIAL_CONTROL_TREE
                or source_chain.get("initial_control_tree")
                != "30f0f07fa41947647461ccbd44d6dc4008848259"
                or source_chain.get("final_reseal_parent")
                != materializer._M34_INITIAL_CONTROL_COMMIT
                or source_chain.get("final_reseal_parent")
                != "e342b63f3f143bb85ed4744e5391c4f8e7c961cd"
                or source_chain.get("initial_control_blobs")
                != dict(materializer._M34_INITIAL_CONTROL_BLOBS)
                or module._m34_json_emission_normalization_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M34 JSON-emission authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M34 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M34 active runtime paths are not exactly preserved")
    elif m33_selected:
        successor = config.get(m33_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = materializer._expected_m33_live_preflight_contract_authority()
            reference = materializer._m33_authority_reference()
            source_chain = expected.get("source_chain", {})
            if (
                successor != reference
                or migration.get(m33_key) != reference
                or seal.get(f"{m33_key}_cid") != materializer._identity(expected)
                or source_chain.get("initial_control_commit")
                != materializer._M33_INITIAL_CONTROL_COMMIT
                or source_chain.get("initial_control_commit")
                != "b0526d4085b231f1eca0cf638743e8d455debc08"
                or source_chain.get("initial_control_tree")
                != materializer._M33_INITIAL_CONTROL_TREE
                or module._m33_live_preflight_contract_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M33 live-preflight contract authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M33 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M33 active runtime paths are not exactly preserved")
    elif m32_selected:
        successor = config.get(m32_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = materializer._expected_m32_live_preflight_plan_anchor_authority()
            reference = materializer._m32_authority_reference()
            source_chain = expected.get("source_chain", {})
            if (
                successor != reference
                or migration.get(m32_key) != reference
                or seal.get(f"{m32_key}_cid") != materializer._identity(expected)
                or source_chain.get("initial_control_commit")
                != materializer._M32_INITIAL_CONTROL_COMMIT
                or source_chain.get("initial_control_commit")
                != "547485ffacd636c6046b00ed23dc0f4c53de4315"
                or source_chain.get("initial_control_tree")
                != materializer._M32_INITIAL_CONTROL_TREE
                or module._m32_live_preflight_plan_anchor_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append("M32 live-preflight authority/CID/source differs")
        except Exception as exc:
            config_errors.append(
                f"M32 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M32 active runtime paths are not exactly preserved")
    elif m31_selected:
        successor = config.get(m31_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m31_detached_coordinator_pid_recovery_authority()
            )
            source_chain = expected.get("source_chain", {})
            if (
                type(successor) is not dict
                or materializer._identity(successor) != materializer._identity(expected)
                or type(migration.get(m31_key)) is not dict
                or materializer._identity(migration.get(m31_key))
                != materializer._identity(expected)
                or seal.get(f"{m31_key}_cid") != materializer._identity(expected)
                or source_chain.get("initial_control_commit")
                != "07aed87e3ebc4ef5667541435fd04f2d62a39b25"
                or source_chain.get("initial_control_tree")
                != "c9fe9a657d2d0e1e05172688dbc44dea6f699265"
                or source_chain.get("final_reseal_parent")
                != "07aed87e3ebc4ef5667541435fd04f2d62a39b25"
                or module._m31_detached_coordinator_pid_recovery_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M31 detached-coordinator authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M31 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M31 active runtime paths are not exactly preserved")
    elif m30_selected:
        successor = config.get(m30_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m30_stopped_owner_restart_source_seal_authority()
            )
            if (
                type(successor) is not dict
                or materializer._identity(successor) != materializer._identity(expected)
                or type(migration.get(m30_key)) is not dict
                or materializer._identity(migration.get(m30_key))
                != materializer._identity(expected)
                or seal.get(f"{m30_key}_cid") != materializer._identity(expected)
                or module._m30_stopped_owner_restart_source_seal_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append("M30 stopped-owner authority/CID/source differs")
        except Exception as exc:
            config_errors.append(
                f"M30 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M30 active runtime paths are not exactly preserved")
    elif m29_selected:
        successor = config.get(m29_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m29_committed_evidence_verification_authority()
            )
            if (
                type(successor) is not dict
                or materializer._identity(successor)
                != materializer._identity(expected)
                or type(migration.get(m29_key)) is not dict
                or materializer._identity(migration.get(m29_key))
                != materializer._identity(expected)
                or seal.get(f"{m29_key}_cid")
                != materializer._identity(expected)
                or module
                ._m29_committed_evidence_verification_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M29 committed-evidence-verification authority/CID/source "
                    "differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M29 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M29 active runtime paths are not exactly preserved")
    elif m28_selected:
        successor = config.get(m28_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m28_live_claim_admission_recovery_authority()
            )
            if (
                type(successor) is not dict
                or materializer._identity(successor)
                != materializer._identity(expected)
                or type(migration.get(m28_key)) is not dict
                or materializer._identity(migration.get(m28_key))
                != materializer._identity(expected)
                or seal.get(f"{m28_key}_cid")
                != materializer._identity(expected)
                or module
                ._m28_live_claim_admission_recovery_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M28 live-claim/admission-recovery authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M28 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M28 active runtime paths are not exactly preserved")
    elif m27_selected:
        successor = config.get(m27_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m27_dead_owner_parallel_resume_authority()
            )
            if (
                type(successor) is not dict
                or materializer._identity(successor)
                != materializer._identity(expected)
                or type(migration.get(m27_key)) is not dict
                or materializer._identity(migration.get(m27_key))
                != materializer._identity(expected)
                or seal.get(f"{m27_key}_cid")
                != materializer._identity(expected)
                or module._m27_dead_owner_parallel_resume_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M27 dead-owner parallel-resume authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M27 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M27 active runtime paths are not exactly fresh")
    elif m26_selected:
        successor = config.get(m26_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m26"
        )
        required_m26 = {
            "schema": (
                "sawm/automatic-stall-recovery-successor-materialization-"
                "authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M26",
            "migration_kind": m26_key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m25/control.duckdb"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                f"{runtime_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": runtime_root,
            "target_generation": 25,
            "target_quack_port": 24_069,
            "target_plan_revision": 27,
            "target_event_watermark": 262,
            "task_revision_changes": 2,
            "task_status_changes": 2,
            "coordination_semantic_changes": 4,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if (
            not isinstance(successor, Mapping)
            or any(
                successor.get(key) != value
                for key, value in required_m26.items()
            )
        ):
            config_errors.append(
                "M26 automatic-stall-recovery successor authority is not exact"
            )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m26_automatic_stall_recovery_authority()
            )
            if (
                type(successor) is not dict
                or materializer._identity(successor)
                != materializer._identity(expected)
                or type(migration.get(m26_key)) is not dict
                or materializer._identity(migration.get(m26_key))
                != materializer._identity(expected)
                or seal.get(f"{m26_key}_cid")
                != materializer._identity(expected)
                or module._m26_automatic_stall_recovery_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M26 automatic-stall-recovery authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M26 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M26 active runtime paths are not exactly fresh")
    elif m25_selected:
        successor = config.get(m25_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m25"
        )
        required_m25 = {
            "schema": (
                "sawm/native-duckdb-preload-successor-materialization-"
                "authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M25",
            "migration_kind": m25_key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m24/control.duckdb"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                f"{runtime_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": runtime_root,
            "target_generation": 24,
            "target_quack_port": 24_068,
            "target_plan_revision": 26,
            "target_event_watermark": 251,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "coordination_semantic_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if (
            not isinstance(successor, Mapping)
            or any(
                successor.get(key) != value
                for key, value in required_m25.items()
            )
        ):
            config_errors.append(
                "M25 native-DuckDB preload successor authority is not exact"
            )
        try:
            module = _dependency_validator_module(REPO_ROOT)
            materializer_spec = importlib.util.spec_from_file_location(
                "sawm_board_m25_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if materializer_spec is None or materializer_spec.loader is None:
                raise RuntimeError("M25 materializer cannot be loaded")
            materializer = importlib.util.module_from_spec(materializer_spec)
            materializer_spec.loader.exec_module(materializer)
            expected = (
                materializer._expected_m25_native_duckdb_preload_authority()
            )
            if (
                type(successor) is not dict
                or materializer._identity(successor)
                != materializer._identity(expected)
                or type(migration.get(m25_key)) is not dict
                or materializer._identity(migration.get(m25_key))
                != materializer._identity(expected)
                or seal.get(f"{m25_key}_cid")
                != materializer._identity(expected)
                or module._m25_native_duckdb_preload_successor_errors(
                    config, seal, migration, root=REPO_ROOT
                )
            ):
                config_errors.append(
                    "M25 native-DuckDB preload authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M25 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M25 active runtime paths are not exactly fresh")
    elif m24_selected:
        successor = config.get(m24_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m24"
        )
        required_m24 = {
            "schema": (
                "sawm/multi-lane-sidecar-reopen-successor-"
                "materialization-authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M24",
            "migration_kind": m24_key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m23/control.duckdb"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                f"{runtime_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": runtime_root,
            "target_generation": 24,
            "target_quack_port": 24_067,
            "target_plan_revision": 25,
            "target_event_watermark": 249,
        }
        if (
            not isinstance(successor, Mapping)
            or any(
                successor.get(key) != value
                for key, value in required_m24.items()
            )
        ):
            config_errors.append(
                "M24 sidecar-reopen successor authority is not exact"
            )
        try:
            module = _dependency_validator_module(REPO_ROOT)
            materializer_spec = importlib.util.spec_from_file_location(
                "sawm_board_m24_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if materializer_spec is None or materializer_spec.loader is None:
                raise RuntimeError("M24 materializer cannot be loaded")
            materializer = importlib.util.module_from_spec(materializer_spec)
            materializer_spec.loader.exec_module(materializer)
            expected = materializer._expected_m24_sidecar_reopen_authority()
            if (
                type(successor) is not dict
                or materializer._identity(successor)
                != materializer._identity(expected)
                or type(migration.get(m24_key)) is not dict
                or materializer._identity(migration.get(m24_key))
                != materializer._identity(expected)
                or seal.get(f"{m24_key}_cid")
                != materializer._identity(expected)
                or module._m24_sidecar_reopen_successor_errors(
                    config, seal, migration, root=REPO_ROOT
                )
            ):
                config_errors.append(
                    "M24 sidecar-reopen authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M24 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M24 active runtime paths are not exactly fresh")
    elif m23_selected:
        multi_lane = config.get(m23_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m23"
        )
        required_m23 = {
            "schema": "sawm/multi-lane-successor-materialization-authorization@1",
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M23",
            "migration_kind": m23_key,
            "supersession_mode": m23_key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m22/control.duckdb"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                f"{runtime_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": runtime_root,
            "target_generation": 23,
            "target_quack_port": 24_066,
            "target_plan_revision": 24,
            "target_event_watermark": 244,
        }
        if (
            not isinstance(multi_lane, Mapping)
            or any(
                multi_lane.get(key) != value
                for key, value in required_m23.items()
            )
        ):
            config_errors.append("M23 multi-lane successor authority is not exact")
        try:
            module = _dependency_validator_module(REPO_ROOT)
            materializer_spec = importlib.util.spec_from_file_location(
                "sawm_board_m23_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if materializer_spec is None or materializer_spec.loader is None:
                raise RuntimeError("M23 materializer cannot be loaded")
            materializer = importlib.util.module_from_spec(materializer_spec)
            materializer_spec.loader.exec_module(materializer)
            expected = materializer._expected_m23_multi_lane_authority()
            if (
                type(multi_lane) is not dict
                or materializer._identity(multi_lane)
                != materializer._identity(expected)
                or type(migration.get(m23_key)) is not dict
                or materializer._identity(migration.get(m23_key))
                != materializer._identity(expected)
                or seal.get(f"{m23_key}_cid") != materializer._identity(expected)
                or module._m23_multi_lane_successor_errors(
                    config, seal, migration, root=REPO_ROOT
                )
            ):
                config_errors.append(
                    "M23 multi-lane successor authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M23 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M23 active runtime paths are not exactly fresh")
    elif m22_selected:
        compatibility = config.get(m22_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m22"
        )
        required_m22 = {
            "schema": (
                "sawm/live-preflight-receipt-compatibility-successor-"
                "materialization-authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M22",
            "migration_kind": m22_key,
            "supersession_mode": m22_key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m21/control.duckdb"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                f"{runtime_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": runtime_root,
            "target_generation": 22,
            "target_quack_port": 24_065,
            "target_plan_revision": 23,
            "target_event_watermark": 233,
            "event_suffix_length": 2,
            "ordinary_source_changes": 0,
            "coordination_semantic_changes": 0,
            "plan_revision_changes": 1,
            "evidence_node_changes": 1,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "goal_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "worker_self_approval": False,
        }
        if (
            not isinstance(compatibility, Mapping)
            or any(
                compatibility.get(key) != value
                for key, value in required_m22.items()
            )
        ):
            config_errors.append(
                "M22 live-preflight receipt compatibility authority is not exact"
            )
        try:
            module = _dependency_validator_module(REPO_ROOT)
            materializer_spec = importlib.util.spec_from_file_location(
                "sawm_board_m22_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if materializer_spec is None or materializer_spec.loader is None:
                raise RuntimeError("M22 materializer cannot be loaded")
            materializer = importlib.util.module_from_spec(materializer_spec)
            materializer_spec.loader.exec_module(materializer)
            expected = (
                materializer._expected_m22_live_preflight_receipt_compatibility_authority()
            )
            if (
                compatibility != expected
                or compatibility != migration.get(m22_key)
                or seal.get(f"{m22_key}_cid") != materializer._identity(expected)
                or module._m22_live_preflight_receipt_compatibility_successor_errors(
                    config, seal, migration, root=REPO_ROOT
                )
            ):
                config_errors.append(
                    "M22 live-preflight receipt compatibility "
                    "authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M22 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M22 active runtime paths are not exactly fresh")
    elif m21_selected:
        realization = config.get(m21_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m21"
        )
        required_m21 = {
            "schema": (
                "sawm/generation-realization-successor-materialization-"
                "authorization@1"
            ),
            "migration_revision": "SAWM-R2-M21",
            "migration_kind": "generation_realization_successor_materialization",
            "supersession_mode": (
                "generation_realization_successor_materialization"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                f"{runtime_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": runtime_root,
            "target_generation": 21,
            "target_quack_port": 24_064,
            "target_plan_revision": 22,
            "target_event_watermark": 231,
            "ordinary_source_changes": 0,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "coordination_semantic_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if (
            not isinstance(realization, Mapping)
            or any(
                realization.get(key) != value
                for key, value in required_m21.items()
            )
        ):
            config_errors.append(
                "M21 generation-realization authority is not exact"
            )
        try:
            module = _dependency_validator_module(REPO_ROOT)
            materializer_spec = importlib.util.spec_from_file_location(
                "sawm_board_m21_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if materializer_spec is None or materializer_spec.loader is None:
                raise RuntimeError("M21 materializer cannot be loaded")
            materializer = importlib.util.module_from_spec(materializer_spec)
            materializer_spec.loader.exec_module(materializer)
            expected = (
                materializer._expected_m21_generation_realization_authority()
            )
            if (
                realization != expected
                or realization != migration.get(m21_key)
                or seal.get("generation_realization_successor_materialization_cid")
                != materializer._identity(expected)
                or module._m21_generation_realization_successor_errors(
                    config, seal, migration, root=REPO_ROOT
                )
            ):
                config_errors.append(
                    "M21 generation-realization authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M21 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M21 active runtime paths are not exactly fresh")
    elif m20_selected:
        isolation = config.get(m20_key)
        required_m20 = {
            "schema": "sawm/post-materialization-test-isolation-repair-authorization@1",
            "migration_revision": "SAWM-R2-M20",
            "supersession_mode": (
                "source_only_post_materialization_test_isolation_repair"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m20/control.coordination.duckdb"
            ),
            "target_runtime_root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m20"
            ),
            "target_generation": 21,
            "target_quack_port": 24_063,
            "target_plan_revision": 21,
            "target_event_watermark": 229,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "coordination_semantic_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if (
            not isinstance(isolation, Mapping)
            or any(isolation.get(key) != value for key, value in required_m20.items())
        ):
            config_errors.append("M20 test-isolation authority is not exact")
        try:
            module = _dependency_validator_module(REPO_ROOT)
            materializer_spec = importlib.util.spec_from_file_location(
                "sawm_board_m20_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if materializer_spec is None or materializer_spec.loader is None:
                raise RuntimeError("M20 materializer cannot be loaded")
            materializer = importlib.util.module_from_spec(materializer_spec)
            materializer_spec.loader.exec_module(materializer)
            if (
                isolation != migration.get(m20_key)
                or seal.get("test_isolation_successor_materialization_cid")
                != materializer._identity(isolation)
                or module._m20_test_isolation_successor_errors(
                    config, seal, migration, root=REPO_ROOT
                )
            ):
                config_errors.append("M20 test-isolation authority/CID differs")
        except Exception as exc:
            config_errors.append(
                f"M20 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        runtime_root = required_m20["target_runtime_root"]
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M20 active runtime paths are not exactly fresh")
    elif m19_selected:
        catalog = config.get(m19_key)
        required_m19 = {
            "schema": "sawm/live-quack-catalog-inventory-repair-authorization@1",
            "migration_revision": "SAWM-R2-M19",
            "supersession_mode": "source_only_live_quack_catalog_inventory_repair",
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m19/control.coordination.duckdb"
            ),
            "target_runtime_root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m19"
            ),
            "target_generation": 20,
            "target_quack_port": 24_062,
            "target_plan_revision": 20,
            "target_event_watermark": 227,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "coordination_semantic_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if (
            not isinstance(catalog, Mapping)
            or any(catalog.get(key) != value for key, value in required_m19.items())
        ):
            config_errors.append("M19 live-catalog-inventory authority is not exact")
        try:
            module = _dependency_validator_module(REPO_ROOT)
            materializer_spec = importlib.util.spec_from_file_location(
                "sawm_board_m19_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if materializer_spec is None or materializer_spec.loader is None:
                raise RuntimeError("M19 materializer cannot be loaded")
            materializer = importlib.util.module_from_spec(materializer_spec)
            materializer_spec.loader.exec_module(materializer)
            if (
                catalog != migration.get(m19_key)
                or seal.get(
                    "live_catalog_inventory_successor_materialization_cid"
                )
                != materializer._identity(catalog)
                or module._m19_live_catalog_inventory_successor_errors(
                    config, seal, migration, root=REPO_ROOT
                )
            ):
                config_errors.append(
                    "M19 live-catalog-inventory authority/CID differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M19 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        runtime_root = required_m19["target_runtime_root"]
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M19 active runtime paths are not exactly fresh")
    elif m18_selected:
        portal = config.get(m18_key)
        required_m18 = {
            "schema": "sawm/portal-completion-persistence-repair-authorization@1",
            "migration_revision": "SAWM-R2-M18",
            "supersession_mode": (
                "append_only_portal_completion_persistence_repair_and_exact_rearm"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m18/control.coordination.duckdb"
            ),
            "target_runtime_root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m18"
            ),
            "target_generation": 19,
            "target_quack_port": 24_061,
            "target_plan_revision": 19,
            "target_event_watermark": 225,
            "target_projection_cid": (
                "baguqeeraqsmnfs6rzwc6bvjjdszmryjtdrtg5aosnk2wdsppaxymsrrpqyxa"
            ),
            "task_revision_changes": 1,
            "task_status_changes": 1,
            "coordination_semantic_changes": 1,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if (
            not isinstance(portal, Mapping)
            or any(portal.get(key) != value for key, value in required_m18.items())
            or set(portal.get("task_rearms", {})) != {"SAWM-007"}
            or set(portal.get("failure_receipts", {})) != {"SAWM-007"}
        ):
            config_errors.append("M18 portal-completion authority is not exact")
        try:
            module = _dependency_validator_module(REPO_ROOT)
            materializer_spec = importlib.util.spec_from_file_location(
                "sawm_board_m18_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if materializer_spec is None or materializer_spec.loader is None:
                raise RuntimeError("M18 materializer cannot be loaded")
            materializer = importlib.util.module_from_spec(materializer_spec)
            materializer_spec.loader.exec_module(materializer)
            if (
                portal != migration.get(m18_key)
                or seal.get(
                    "portal_completion_persistence_successor_materialization_cid"
                )
                != materializer._identity(portal)
                or module._m18_portal_completion_persistence_errors(
                    config, seal, migration, root=REPO_ROOT
                )
            ):
                config_errors.append("M18 portal-completion authority/CID differs")
        except Exception as exc:
            config_errors.append(
                f"M18 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        runtime_root = required_m18["target_runtime_root"]
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M18 active runtime paths are not exactly fresh")
    elif m17_selected:
        source_binding = config.get(m17_key)
        required_m17 = {
            "schema": (
                "sawm/post-commit-source-binding-successor-authorization@1"
            ),
            "migration_revision": "SAWM-R2-M17",
            "supersession_mode": "post_commit_exact_source_binding_repair",
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m17/control.coordination.duckdb"
            ),
            "target_runtime_root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m17"
            ),
            "target_generation": 18,
            "target_quack_port": 24060,
            "target_plan_revision": 18,
            "target_event_watermark": 211,
            "target_projection_cid": (
                "baguqeerat5ph3demwcyfmvtxjsq4dxdei5lf6xva2jhylwvgh2s4ewttscza"
            ),
            "target_coordination_projection_digest": (
                "sha256:3a1871c7bd682348897fd10da9a5515f671e1986beedff889c26d7dfe5a91773"
            ),
            "target_coordination_event_count": 674,
            "prior_event_watermark": 209,
            "prior_generation": 17,
            "prior_plan_revision": 17,
            "coordination_semantic_changes": 0,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "worker_self_approval": False,
        }
        if type(source_binding) is not dict or any(
            source_binding.get(key) != value
            for key, value in required_m17.items()
        ):
            config_errors.append("M17 source-binding authority is not exact")
        elif (
            source_binding != migration.get(m17_key)
            or seal.get("source_binding_successor_materialization_cid")
            != "sha256:"
            + hashlib.sha256(
                json.dumps(
                    source_binding,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        ):
            config_errors.append("M17 source-binding authority/CID differs")
        runtime_root = required_m17["target_runtime_root"]
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        } or any(
            program.get(key) != value
            for key, value in {
                "event_store_path": f"{runtime_root}/events",
                "runtime_registry_path": f"{runtime_root}/registry",
                "worktree_root": f"{runtime_root}/worktrees",
            }.items()
        ):
            config_errors.append("M17 active runtime paths are not exactly fresh")
    elif m16_selected:
        accepted_source_retry = config.get(m16_key)
        required_m16 = {
            "schema": "sawm/portal-accepted-source-repair-authorization@1",
            "migration_revision": "SAWM-R2-M16",
            "supersession_mode": (
                "append_only_accepted_source_repair_and_exact_rearm"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m16/control.coordination.duckdb"
            ),
            "target_runtime_root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m16"
            ),
            "target_generation": 17,
            "target_quack_port": 24059,
            "target_plan_revision": 17,
            "target_event_watermark": 209,
            "target_projection_cid": (
                "baguqeerakzd5xe55z5l6nifvwumea7unzobnhokoigbfkkdzg2chmrl4xa6q"
            ),
            "target_coordination_projection_digest": (
                "sha256:3a1871c7bd682348897fd10da9a5515f671e1986beedff889c26d7dfe5a91773"
            ),
            "target_coordination_event_count": 674,
            "prior_event_watermark": 205,
            "prior_generation": 16,
            "prior_plan_revision": 16,
            "coordination_semantic_changes": 2,
            "task_revision_changes": 2,
            "task_status_changes": 2,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "worker_self_approval": False,
        }
        if type(accepted_source_retry) is not dict or any(
            accepted_source_retry.get(key) != value
            for key, value in required_m16.items()
        ):
            config_errors.append(
                "M16 accepted-source retry authority is not exact"
            )
        elif (
            accepted_source_retry != migration.get(m16_key)
            or seal.get("accepted_source_retry_successor_materialization_cid")
            != "sha256:"
            + hashlib.sha256(
                json.dumps(
                    accepted_source_retry,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        ):
            config_errors.append(
                "M16 accepted-source retry authority/CID differs"
            )
        runtime_root = required_m16["target_runtime_root"]
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        } or any(
            program.get(key) != value
            for key, value in {
                "event_store_path": f"{runtime_root}/events",
                "runtime_registry_path": f"{runtime_root}/registry",
                "worktree_root": f"{runtime_root}/worktrees",
            }.items()
        ):
            config_errors.append("M16 active runtime paths are not exactly fresh")
    elif m15_selected:
        rebind = config.get(m15_key)
        required_m15 = {
            "schema": "sawm/runtime-root-rebind-repair-authorization@1",
            "migration_revision": "SAWM-R2-M15",
            "target_store_id": active_store,
            "target_runtime_root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m15"
            ),
            "target_generation": 16,
            "target_quack_port": 24058,
            "target_plan_revision": 16,
            "target_event_watermark": 201,
            "target_projection_cid": (
                "baguqeeraaiqn3rqfg7gr4ks25n5ffjt3k4wcf4du56534qzyk7z3j7hewxbq"
            ),
        }
        if type(rebind) is not dict or any(
            rebind.get(key) != value for key, value in required_m15.items()
        ):
            config_errors.append("M15 runtime-root rebind authority is not exact")
        elif (
            rebind != migration.get(m15_key)
            or seal.get("runtime_root_rebind_successor_materialization_cid")
            != "sha256:"
            + hashlib.sha256(
                json.dumps(
                    rebind,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        ):
            config_errors.append("M15 runtime-root rebind authority/CID differs")
        runtime_root = required_m15["target_runtime_root"]
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        } or any(
            program.get(key) != value
            for key, value in {
                "event_store_path": f"{runtime_root}/events",
                "runtime_registry_path": f"{runtime_root}/registry",
                "worktree_root": f"{runtime_root}/worktrees",
            }.items()
        ):
            config_errors.append("M15 active runtime paths are not exactly fresh")
    elif m14_selected:
        restart = config.get(m14_key)
        required_m14 = {
            "schema": "sawm/stale-owner-restart-repair-authorization@1",
            "migration_revision": "SAWM-R2-M14", "target_store_id": active_store,
            "target_generation": 15, "target_quack_port": 24057,
            "target_plan_revision": 15, "target_event_watermark": 199,
            "target_projection_cid": "baguqeerae7evi25zx65b6ezizdp7s6mjdd252yz4cjj2tcd5u4dfgikrxula",
        }
        if type(restart) is not dict or any(restart.get(key) != value for key, value in required_m14.items()):
            config_errors.append("M14 stale-owner restart authority is not exact")
        elif restart != migration.get(m14_key) or seal.get("stale_owner_restart_successor_materialization_cid") != "sha256:" + hashlib.sha256(json.dumps(restart, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest():
            config_errors.append("M14 stale-owner restart authority/CID differs")
    elif m13_selected:
        refresh_successor = config.get(m13_key)
        expected_m13_fields = {
            "schema": "sawm/quack-initial-refresh-repair-authorization@1",
            "migration_revision": "SAWM-R2-M13",
            "migration_kind": "quack_initial_refresh_rebind_repair",
            "supersession_mode": "source_only_quack_initial_refresh_lifecycle_repair",
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m13/control.coordination.duckdb"
            ),
            "target_generation": 14,
            "target_quack_port": 24056,
            "target_plan_revision": 14,
            "target_event_watermark": 191,
            "target_projection_cid": (
                "baguqeeraqpkofyd3pnjnaqiwwefz7pkubim7kaklbp65puqu47ckb2vdmtxa"
            ),
            "coordination_semantic_changes": 0,
            "plan_revision_changes": 1,
            "evidence_node_changes": 1,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "worker_self_approval": False,
        }
        expected_m13_cid = (
            "sha256:2a6be82b1d160f07ee192b6093eb8fae3518ab2df713155bce98f7fb280e9920"
        )
        if type(refresh_successor) is not dict or any(
            refresh_successor.get(field) != expected
            for field, expected in expected_m13_fields.items()
        ):
            config_errors.append("M13 active Quack-refresh authority is not exact")
        elif (
            refresh_successor != migration.get(m13_key)
            or "sha256:"
            + hashlib.sha256(
                json.dumps(
                    refresh_successor,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            != expected_m13_cid
            or seal.get("quack_refresh_successor_materialization_cid")
            != expected_m13_cid
        ):
            config_errors.append("M13 Quack-refresh authority/CID differs")
    elif m12_selected:
        declared_output_retry = config.get(m12_key)
        expected_m12_fields = {
            "schema": "sawm/portal-declared-output-repair-authorization@1",
            "migration_revision": "SAWM-R2-M12",
            "migration_kind": "database_portal_declared_path_projection_repair",
            "supersession_mode": (
                "source_only_database_portal_declared_path_adapter_and_retry_rearm"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m12/control.coordination.duckdb"
            ),
            "target_generation": 14,
            "target_quack_port": 45256,
            "target_plan_revision": 13,
            "target_event_watermark": 189,
            "implementation_provider_invocations_observed": 2,
            "implementation_provider_model_calls_observed": 92,
            "implementation_provider_tokens_observed": 11_202_083,
            "implementation_provider_cost_usd_observed": "1.23726850",
            "settlement_provider_invocation_count": 0,
            "provider_execution_accounting_mismatch": True,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "execution_sidecar_copied": False,
            "read_replica_sidecar_copied": False,
            "worker_self_approval": False,
        }
        expected_m12_cid = (
            "sha256:786dde1f1728b907c3e28e5a09c746842c0a4a5ac0333f3ccb25f5290e8227a8"
        )
        if type(declared_output_retry) is not dict or any(
            declared_output_retry.get(field) != expected
            for field, expected in expected_m12_fields.items()
        ):
            config_errors.append(
                "M12 active declared-output retry authority is not exact"
            )
        elif (
            declared_output_retry != migration.get(m12_key)
            or "sha256:"
            + hashlib.sha256(
                json.dumps(
                    declared_output_retry,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            != expected_m12_cid
            or seal.get("declared_output_retry_successor_materialization_cid")
            != expected_m12_cid
        ):
            config_errors.append("M12 declared-output retry authority/CID differs")
    elif m11_selected:
        provider_retry = config.get(m11_key)
        expected_m11_fields = {
            "schema": "sawm/provider-launch-repair-authorization@1",
            "migration_revision": "SAWM-R2-M11",
            "migration_kind": "authenticated_grok_container_lifecycle_repair",
            "supersession_mode": (
                "source_only_grok_container_lifecycle_and_retry_rearm"
            ),
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m11/control.coordination.duckdb"
            ),
            "target_generation": 13,
            "target_plan_revision": 12,
            "implementation_provider_invocations_observed": 1,
            "settlement_provider_invocation_count": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "execution_sidecar_copied": False,
            "worker_self_approval": False,
        }
        if type(provider_retry) is not dict or any(
            provider_retry.get(field) != expected
            for field, expected in expected_m11_fields.items()
        ):
            config_errors.append("M11 active provider-retry authority is not exact")
        elif (
            provider_retry != migration.get(m11_key)
            or "sha256:"
            + hashlib.sha256(
                json.dumps(
                    provider_retry,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            != "sha256:7f8404735adae0fb7ed890cda97dc674b78ae915efa193207e98abd95eb5193e"
            or seal.get("live_provider_retry_successor_materialization_cid")
            != "sha256:7f8404735adae0fb7ed890cda97dc674b78ae915efa193207e98abd95eb5193e"
        ):
            config_errors.append("M11 provider-retry authority/CID differs")
    elif m10_selected:
        live_projection = config.get(m10_key)
        expected_m10_cid = (
            "sha256:f27878def0ee9b406d0dbaac669728278e4f375cb267ee7f76330faaf9b14f10"
        )
        expected_m10_fields = {
            "schema": "sawm/live-control-projection-repair-authorization@1",
            "migration_revision": "SAWM-R2-M10",
            "migration_kind": (
                "authenticated_live_task_and_semantic_projection_comparator_repair"
            ),
            "supersession_mode": "source_only_live_projection_comparator_repair",
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m10/control.coordination.duckdb"
            ),
            "target_generation": 12,
            "target_plan_revision": 11,
            "target_event_watermark": 179,
            "target_projection_cid": (
                "baguqeerareq2bngq3hffyk5vidym2ukeleg5gehpaxhqvvdjayn7ucxplcaq"
            ),
            "target_semantic_authority_digest": (
                "sha256:a4903791c91cc2e9c3337f2abdfd6af78389f7036d54cb7f8e43246cd4f0c023"
            ),
            "target_coordination_projection_digest": (
                "sha256:7fb9bacb0f76fe832cc34dd5fb2ccdef13ddafeef4ec2a3d532cb902aa62011e"
            ),
            "target_coordination_event_count": 36,
            "event_suffix_length": 2,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "coordination_semantic_changes": 0,
        }
        if type(live_projection) is not dict or any(
            live_projection.get(field) != expected
            for field, expected in expected_m10_fields.items()
        ):
            config_errors.append("M10 active live-projection authority is not exact")
        elif (
            live_projection != migration.get(m10_key)
            or "sha256:"
            + hashlib.sha256(
                json.dumps(
                    live_projection,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            != expected_m10_cid
            or seal.get("live_projection_successor_materialization_cid")
            != expected_m10_cid
        ):
            config_errors.append("M10 projection/failure authority CID is not exact")
    elif m9_selected:
        live_recovery = config.get(m9_key)
        expected_m9_cid = (
            "sha256:e7834d12a6150a7d4dd1f90dcb5369d73a6d8620a38bf6baa0cbbb3da17bebb9"
        )
        expected_m9_fields = {
            "schema": "sawm/control-plane-runtime-recovery-authorization@1",
            "migration_revision": "SAWM-R2-M9",
            "migration_kind": "board_scoped_checkout_lock_and_portal_deferral_recovery",
            "target_store_id": active_store,
            "target_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m9/control.coordination.duckdb"
            ),
            "target_generation": 11,
            "target_plan_revision": 10,
            "target_event_watermark": 177,
            "target_projection_cid": (
                "baguqeeraebsdnrj7pvgd6yp26yr6ob7ocbrfzjwg57xfjnr6sn4xfkuvkq2q"
            ),
            "target_coordination_projection_digest": (
                "sha256:7fb9bacb0f76fe832cc34dd5fb2ccdef13ddafeef4ec2a3d532cb902aa62011e"
            ),
            "event_suffix_length": 3,
        }
        if type(live_recovery) is not dict or any(
            live_recovery.get(field) != expected
            for field, expected in expected_m9_fields.items()
        ):
            config_errors.append("M9 active dual-store recovery authority is not exact")
        elif (
            live_recovery != migration.get(m9_key)
            or "sha256:"
            + hashlib.sha256(
                json.dumps(
                    live_recovery,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            != expected_m9_cid
            or seal.get("live_recovery_successor_materialization_cid")
            != expected_m9_cid
        ):
            # The canonical authority binds the exact fourteen repair paths,
            # frozen live failure, sanctioned task rearm, and unchanged
            # provider strategy in addition to the active dual-store fields.
            config_errors.append("M9 recovery/failure/rearm authority CID is not exact")
    prior = config.get("prior_materialization") if isinstance(config.get("prior_materialization"), Mapping) else {}
    if (
        prior.get("program_definition_cid") != migration.get("prior_program_definition_cid")
        or prior.get("plan_root_cid") != migration.get("prior_plan_root_cid")
        or set(migration.get("prior_task_cids") or {}) != set(TASK_IDS)
        or set(migration.get("prior_goal_cids") or {}) != set(GOAL_IDS)
        or migration.get("prior_task_count") != len(TASK_IDS)
        or migration.get("prior_goal_count") != len(GOAL_IDS)
        or migration.get("definition_source_binding_cid")
        != "sha256:cf4d9fa1ba595286866f5406e61b2ac71e4ed3730af70a2b07f88d0c16905e5e"
    ):
        config_errors.append("append-only prior-SAWM migration binding mismatch")
    config_errors.extend(_m6_migration_errors(config, seal, migration))
    config_errors.extend(_m7_migration_errors(config, seal, migration))
    config_errors.extend(_active_successor_migration_errors(config, seal, migration))
    config_errors.extend(_configured_board_dependency_errors(root, config, seal))
    ducklake = config.get("ducklake_history_projection") if isinstance(config.get("ducklake_history_projection"), Mapping) else {}
    if not (
        ducklake.get("authority") is False
        and ducklake.get("scheduling_prerequisite") is False
        and ducklake.get("completion_prerequisite") is False
        and ducklake.get("load_local_only") is True
        and ducklake.get("install_or_network_forbidden") is True
    ):
        config_errors.append("DuckLake must be local-only and non-authoritative")
    if tuple(config.get("protected_paths") or ()) != CONTROL_RELATIVE_PATHS:
        config_errors.append("scheduler protected paths differ from the exact operator controls")
    live_policy = config.get("configured_board_live_capsule")
    if not isinstance(live_policy, Mapping):
        config_errors.append("configured-board live capsule policy is absent")
    else:
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_live_capsule import (
                parse_configured_board_live_capsule_policy,
            )

            live_paths = parse_configured_board_live_capsule_policy(live_policy)
        except (ImportError, ValueError) as exc:
            config_errors.append(
                "configured-board live capsule policy is invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if frozenset(live_paths) != frozenset(CONTROL_RELATIVE_PATHS):
                config_errors.append(
                    "configured-board live capsule paths differ from operator controls"
                )
    expected_direction = {
        "ipfs_datasets_py": [], "ipfs_kit_py": [],
        "ipfs_accelerate_py": ["ipfs_datasets_py", "ipfs_kit_py"],
    }
    if config.get("package_dependency_direction") != expected_direction:
        config_errors.append("package dependency direction mismatch")
    if config.get("rollout_order") != list(ROLLOUT_ORDER):
        config_errors.append("closed rollout order mismatch")
    _append(checks, errors, "scheduler_authority_binding", not config_errors, config_errors)

    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            load_configured_board,
        )

        loaded = load_configured_board(
            root / SCHEDULER_PATH.relative_to(REPO_ROOT), repo_root=root
        )
        scheduler_load_detail: Any = {
            "namespace": loaded.board_namespace,
            "lanes": loaded.max_lanes,
            "database_authority": loaded.resolved_database_program().authority_mode,
        }
        scheduler_load_valid = True
    except Exception as exc:  # typed into deterministic report
        scheduler_load_valid = False
        scheduler_load_detail = f"{type(exc).__name__}: {exc}"
    _append(checks, errors, "current_scheduler_schema_load", scheduler_load_valid, scheduler_load_detail)

    benchmark_errors: list[str] = []
    if benchmark.get("board_namespace") != BOARD_NAMESPACE or benchmark.get("plan_revision") != PLAN_REVISION:
        benchmark_errors.append("benchmark namespace/revision mismatch")
    if benchmark.get("status") != "design_frozen" or benchmark.get("results_available") is not False:
        benchmark_errors.append("benchmark must be a result-free frozen design")
    ablations = benchmark.get("ablation_ladder")
    if not isinstance(ablations, list) or [item.get("id") for item in ablations if isinstance(item, Mapping)] != list("ABCDEFGHIJKL"):
        benchmark_errors.append("benchmark ablation ladder must contain A through L")
    floors = benchmark.get("safety_floors") if isinstance(benchmark.get("safety_floors"), Mapping) else {}
    if not floors or any(value != 0 for value in floors.values()):
        benchmark_errors.append("all frozen safety floors must be zero")
    if any(config.get("release_safety_floors", {}).get(key) != value for key, value in floors.items()):
        benchmark_errors.append("benchmark safety floors differ from scheduler release floors")
    _append(checks, errors, "benchmark_freeze", not benchmark_errors, benchmark_errors)

    if seal.get("board_namespace") != BOARD_NAMESPACE or seal.get("plan_revision") != PLAN_REVISION:
        _append(checks, errors, "dependency_seal_binding", False, "seal namespace/revision mismatch")
    else:
        _append(checks, errors, "dependency_seal_binding", True, seal.get("schema"))

    plan_text = (root / PLAN_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8").lower()
    required_plan_terms = (
        "exact identity", "semantic projection", "typed semantic relation", "authority",
        "physical ipld/merkle", "logical program graph", "duckdb", "quack", "ducklake",
        "bootstrap", "shadow_write", "shadow_read", "guarded", "required",
        "tactician", "hammer", "cegis", "contextcompiler", "world root",
    )
    absent_terms = [term for term in required_plan_terms if term not in plan_text]
    _append(checks, errors, "architecture_control_terms", not absent_terms, absent_terms)

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-board-validation@1",
        "valid": not errors,
        "board_namespace": BOARD_NAMESPACE,
        "plan_revision": PLAN_REVISION,
        "task_count": len(tasks),
        "goal_count": len(goals),
        "markdown_completion_is_authority": False,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true", help="run the complete static board gate")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = validate_program(args.repo_root)
    except Exception as exc:  # fail closed while retaining machine-readable output
        report = {
            "schema": "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-board-validation@1",
            "valid": False,
            "board_namespace": BOARD_NAMESPACE,
            "plan_revision": PLAN_REVISION,
            "errors": [f"unhandled_validator_error: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "checks": [],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
