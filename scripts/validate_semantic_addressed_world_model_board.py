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
_M42_AUTHORITY_CID = (
    "sha256:1e1df3ea3d6b1805dc32f6dd43c61bb95d99e2d08da4c96872441dbc9fdbf587"
)
_M43_AUTHORITY_CID = (
    "sha256:6ddc11cb9e37da82023e5a89124298f532cfc943cc347fa5ea67ff13bcd1eb43"
)
_M43_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M43_FINAL_CONTROL_AUTHORITY_CID"
_M44_AUTHORITY_CID = (
    "sha256:36cba6a8006b50d36d8de2ed6cf76d4adf8d688669a8c6414621173409535845"
)
_M44_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M44_FINAL_CONTROL_AUTHORITY_CID"
_M44_SUCCESSOR_KEY = (
    "post_m43_hardened_procfs_user_manager_restart_successor_materialization"
)
_M45_AUTHORITY_CID = (
    "sha256:1fece222571aff1238d6e32465cae69f3f766d3e4da303b8775a5df892b95ef7"
)
_M45_AUTHORITY_SIZE = 14_585
_M45_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M45_FINAL_CONTROL_AUTHORITY_CID"
_M45_SUCCESSOR_KEY = (
    "failed_pre_authoritative_m44_validation_successor_materialization"
)
_M48_AUTHORITY_CID = (
    "sha256:a7b262b976b38eb94646a9d73c595ff49ab9b1f89fa528443745ed964be025f6"
)
_M48_AUTHORITY_SIZE = 17_315
_M48_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M48_FINAL_CONTROL_AUTHORITY_CID"
_M48_SUCCESSOR_KEY = "post_m47_clean_shutdown_restart_successor_materialization"
_M48_TARGET_PROJECTION_CID = (
    "baguqeerar773puawanjlnneg2heko27svsonv5dfg5pioayyqemfu73ljpwa"
)
_M48_M47_RECEIPT_CID = (
    "sha256:52d8954b5c31a07deafd6077cc9786a3de170017660bca0746dffae08c128919"
)
_M49_AUTHORITY_CID = (
    "sha256:712ace8de59282858a149b8a65457311bf79ccee8abd1907df862f958916e0cc"
)
_M49_AUTHORITY_SIZE = 17_098
_M49_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M49_FINAL_CONTROL_AUTHORITY_CID"
_M49_SUCCESSOR_KEY = "post_m48_successor_report_fix_materialization"
_M49_TARGET_PROJECTION_CID = (
    "baguqeeracfikkmsrysxh3fmhmhri3iduzcki2kqxgoyz2gmszjdkxexm25mq"
)
_M49_M48_RECEIPT_CID = (
    "sha256:72bbd190a41c137d579c6f9372cd85aa8e8b469dc910ebed78be2db28bd09698"
)
_M50_AUTHORITY_CID = (
    "sha256:57e54164eb55a8015c9324db6a414981446a2a455c0308cd65f6a8e219563ea9"
)
_M50_AUTHORITY_SIZE = 29_249
_M50_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M50_FINAL_CONTROL_AUTHORITY_CID"
_M50_SUCCESSOR_KEY = (
    "post_m49_fenced_worktree_quarantine_recovery_successor_materialization"
)
_M50_TARGET_PROJECTION_CID = (
    "baguqeeradyejswcdsx6tnmfgrvglwhvqkuewvpynydwrhlvtenacf3xg2pfa"
)
_M51_AUTHORITY_CID = (
    "sha256:64c4319d273cb9d561a8176c2a152c458e58df10c612c9a9b8235b21ccd4eceb"
)
_M51_AUTHORITY_SIZE = 28_704
_M51_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M51_FINAL_CONTROL_AUTHORITY_CID"
_M51_SUCCESSOR_KEY = "live_quack_catalog_compatibility_successor_materialization"
_M51_TARGET_PROJECTION_CID = _M50_TARGET_PROJECTION_CID
_M52_AUTHORITY_CID = (
    "sha256:97457278d806c1706fc43ad177bea8c66f18c02f2dbcc399121c26aba926e52d"
)
_M52_AUTHORITY_SIZE = 30_930
_M52_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M52_FINAL_CONTROL_AUTHORITY_CID"
_M52_SUCCESSOR_KEY = "test_compatibility_and_control_hash_successor_materialization"
_M52_TARGET_PROJECTION_CID = _M51_TARGET_PROJECTION_CID
_M53_AUTHORITY_CID = (
    "sha256:a0e0e768087e70a27aff88ae1959a638534ef6f9e5b94e8b6d4030b15b01edc8"
)
_M53_AUTHORITY_SIZE = 30_561
_M53_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M53_FINAL_CONTROL_AUTHORITY_CID"
_M53_SUCCESSOR_KEY = (
    "post_reboot_stale_ready_generation_37_restart_successor_materialization"
)
_M53_TARGET_PROJECTION_CID = (
    "baguqeerayh3goxbqiclzj3lzaxzgmqlqaopjmcsbuhastfmtti2xyfd5qxwa"
)
_M55_AUTHORITY_CID = (
    "sha256:fd5d70d8f5f82fe78b4880b2298c8efd1c7bdae2739d6a21933bbab6d24c0c3c"
)
_M55_AUTHORITY_SIZE = 30_271
_M55_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M55_FINAL_CONTROL_AUTHORITY_CID"
_M55_SUCCESSOR_KEY = (
    "live_ready_owner_missing_client_token_vault_restart_successor_materialization"
)
_M55_TARGET_PROJECTION_CID = (
    "baguqeera4y5u77bcjvlfmbe44mpnkzip7hnzs5eqnbcu5mkn7sehgck7mbwq"
)
_M56_AUTHORITY_CID = (
    "sha256:68d8ce25334ca14915eaad83a5eb63b5e03845f11a516423b17a5f144ca84e7a"
)
_M56_AUTHORITY_SIZE = 30_631
_M56_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M56_FINAL_CONTROL_AUTHORITY_CID"
_M56_SUCCESSOR_KEY = (
    "post_m55_live_ready_owner_missing_client_token_vault_restart_successor_materialization"
)
_M56_TARGET_PROJECTION_CID = (
    "baguqeerapoydtvtpy75iszsuwdulvft5zlt4frblsf3fwcd4jwyk2bbb4vva"
)
_M57_AUTHORITY_CID = (
    "sha256:ca402f78a63e84a937f1f62ea97ebbccc67264bb2c47c4aaaac3473ec9fb9590"
)
_M57_AUTHORITY_SIZE = 45_580
_M57_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M57_FINAL_CONTROL_AUTHORITY_CID"
_M57_SUCCESSOR_KEY = (
    "post_m56_live_ready_owner_missing_client_token_vault_restart_successor_materialization"
)
_M57_TARGET_PROJECTION_CID = (
    "baguqeerabphddilf44cxwqvyhfauvfxskeuczbgnwmzbb33dmtj6vpmgyexq"
)
_M50_M49_RECEIPT_CID = (
    "sha256:d5bfeb6dd987b05c2407d93f66d73c6a70bcd2b4f17e8381a93a2bb265acae47"
)
_M47_AUTHORITY_CID = (
    "sha256:73879d0dd4f622ea850a13ef1857dfdf06a79fab170904af62975d856d65e444"
)
_M47_AUTHORITY_SIZE = 18_795
_M47_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M47_FINAL_CONTROL_AUTHORITY_CID"
_M47_SUCCESSOR_KEY = (
    "ignored_python_cache_preservation_and_recovery_successor_materialization"
)
_M47_TARGET_PROJECTION_CID = (
    "baguqeeraadvcue2olr4ies6ctvwaf4rzgafn5myeyomldgeaazkq53cco5wq"
)
_M46_AUTHORITY_CID = (
    "sha256:47ed315018541ef5c4c759c1c486cae2a411821ed4e9902877b08439b79b4455"
)
_M46_AUTHORITY_SIZE = 19_365
_M46_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M46_FINAL_CONTROL_AUTHORITY_CID"
_M46_SUCCESSOR_KEY = "legacy_no_delta_rescue_recovery_successor_materialization"
_M46_M45_RECEIPT_CID = (
    "sha256:46508d2540469d9cbc3ab0cc5220db0bf0cf0cb3cfdce6eed2ff8c8f4da71a97"
)
_M46_TARGET_PROJECTION_CID = (
    "baguqeera3gwgex35vb2k2c2u2vq5qq6d65immvntj3nl2fqfzjpts6jvrfta"
)
_M47_M46_RECEIPT_CID = (
    "sha256:3f9d79c33306ada7b3074609c9fd1e43beee7e474a9fd8b4caacdd65966513c2"
)
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


def _m39_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M39 committed-event reconciliation contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m39_committed_m38_evidence_reconciliation_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M39 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m57_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Validate M57's token-vault generation-41 restart authority."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m57_post_m56_live_ready_owner_missing_client_token_vault_restart_errors(
                scheduler, seal, migration, root=REPO_ROOT
            )
        )
    except Exception as exc:
        return [f"M57 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m56_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Validate M56's token-vault generation-40 restart authority."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m56_post_m55_live_ready_owner_missing_client_token_vault_restart_errors(
                scheduler, seal, migration, root=REPO_ROOT
            )
        )
    except Exception as exc:
        return [f"M56 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m55_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Validate M55's token-vault generation-39 restart authority."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m55_live_ready_owner_missing_client_token_vault_restart_errors(
                scheduler, seal, migration, root=REPO_ROOT
            )
        )
    except Exception as exc:
        return [f"M55 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m53_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Validate M53's post-reboot generation-38 restart authority."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m53_post_reboot_stale_ready_restart_errors(
                scheduler, seal, migration, root=REPO_ROOT
            )
        )
    except Exception as exc:
        return [f"M53 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m52_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse M52's exact repair/hash/catalog source contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m52_test_compatibility_and_control_hash_successor_errors(
                scheduler, seal, migration, root=REPO_ROOT
            )
        )
    except Exception as exc:
        return [f"M52 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m51_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse M51's exact offline-catalog/live-query source contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(module._m51_live_quack_catalog_compatibility_errors(
            scheduler, seal, migration, root=REPO_ROOT
        ))
    except Exception as exc:
        return [f"M51 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m50_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse M50's exact generation-36 fenced recovery source contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m50_post_m49_fenced_recovery_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
            )
        )
    except Exception as exc:
        return [f"M50 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m49_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Reuse M49's exact same-owner event-306 source contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m49_post_m48_successor_report_fix_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
            )
        )
    except Exception as exc:
        return [f"M49 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m48_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse M48's exact generation-35 clean-shutdown restart contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m48_post_m47_clean_shutdown_restart_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M48 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m47_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse M47's exact generation-34 cache-preservation contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module
            ._m47_ignored_python_cache_preservation_and_recovery_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M47 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m46_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse M46's exact generation-33 legacy-rescue recovery contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m46_legacy_no_delta_rescue_recovery_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M46 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m45_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse M45's exact prepublication validation-successor contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module
            ._m45_failed_pre_authoritative_m44_validation_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M45 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m44_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse M44's exact post-M43 generation-32 restart contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module
            ._m44_post_m43_hardened_procfs_user_manager_restart_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M44 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m43_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse M43's exact stopped-generation lifecycle-recovery contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m43_dead_attempt_lifecycle_recovery_restart_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M43 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m42_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse M42's exact failed-pre-authoritative-M41 projection contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module
            ._m42_failed_pre_authoritative_m41_evidence_projection_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M42 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m41_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M41 failed-pre-authoritative-M40 validation contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m41_failed_pre_authoritative_m40_validation_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M41 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m40_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M40 failed-pre-authoritative-M39 contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m40_failed_pre_authoritative_m39_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M40 migration validator unavailable: {type(exc).__name__}: {exc}"]


# Live M38 event bodies are Quack envelopes; verification compares canonical JSON.
def _m38_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M38 custody-first restart contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m38_pre_authoritative_custody_restart_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M38 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m37_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M37 post-reboot generation restart contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m37_post_reboot_generation_restart_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M37 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m36_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M36 operator-task binding correction contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m36_operator_task_binding_correction_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M36 migration validator unavailable: {type(exc).__name__}: {exc}"]


def _m35_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    require_active_runtime: bool = True,
) -> list[str]:
    """Reuse the exact M35 immutable-authority identity contract."""

    try:
        module = _dependency_validator_module(REPO_ROOT)
        return list(
            module._m35_immutable_authority_identity_normalization_successor_errors(
                scheduler,
                seal,
                migration,
                root=REPO_ROOT,
                require_active_runtime=require_active_runtime,
            )
        )
    except Exception as exc:
        return [f"M35 migration validator unavailable: {type(exc).__name__}: {exc}"]


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

    Key presence selects M52 before every historical successor. Consequently
    an empty, null,
    or otherwise malformed newest declaration is validated at that revision
    and cannot silently reactivate historical authority.  Every predecessor
    remains independently checked as immutable history.
    """

    m57_key = _M57_SUCCESSOR_KEY
    m57_presence = (
        m57_key in scheduler,
        m57_key in migration,
        f"{m57_key}_cid" in seal,
    )
    if any(m57_presence):
        errors = _m57_migration_errors(scheduler, seal, migration)
        if not all(m57_presence):
            errors.append("M57 successor authority is only partially declared")
        if errors:
            return errors
        for validator in (
            _m36_migration_errors, _m35_migration_errors, _m34_migration_errors,
            _m33_migration_errors, _m32_migration_errors, _m31_migration_errors,
            _m30_migration_errors, _m29_migration_errors, _m28_migration_errors,
            _m27_migration_errors, _m26_migration_errors, _m25_migration_errors,
            _m24_migration_errors, _m23_migration_errors, _m22_migration_errors,
            _m21_migration_errors, _m20_migration_errors, _m19_migration_errors,
            _m18_migration_errors, _m17_migration_errors, _m16_migration_errors,
        ):
            errors.extend(
                validator(scheduler, seal, migration, require_active_runtime=False)
            )
        return errors

    m56_key = _M56_SUCCESSOR_KEY
    m56_presence = (
        m56_key in scheduler,
        m56_key in migration,
        f"{m56_key}_cid" in seal,
    )
    if any(m56_presence):
        errors = _m56_migration_errors(scheduler, seal, migration)
        if not all(m56_presence):
            errors.append("M56 successor authority is only partially declared")
        if errors:
            return errors
        for validator in (
            _m36_migration_errors, _m35_migration_errors, _m34_migration_errors,
            _m33_migration_errors, _m32_migration_errors, _m31_migration_errors,
            _m30_migration_errors, _m29_migration_errors, _m28_migration_errors,
            _m27_migration_errors, _m26_migration_errors, _m25_migration_errors,
            _m24_migration_errors, _m23_migration_errors, _m22_migration_errors,
            _m21_migration_errors, _m20_migration_errors, _m19_migration_errors,
            _m18_migration_errors, _m17_migration_errors, _m16_migration_errors,
        ):
            errors.extend(
                validator(scheduler, seal, migration, require_active_runtime=False)
            )
        return errors

    m55_key = _M55_SUCCESSOR_KEY
    m55_presence = (
        m55_key in scheduler,
        m55_key in migration,
        f"{m55_key}_cid" in seal,
    )
    if any(m55_presence):
        errors = _m55_migration_errors(scheduler, seal, migration)
        if not all(m55_presence):
            errors.append("M55 successor authority is only partially declared")
        if errors:
            return errors
        for validator in (
            _m36_migration_errors, _m35_migration_errors, _m34_migration_errors,
            _m33_migration_errors, _m32_migration_errors, _m31_migration_errors,
            _m30_migration_errors, _m29_migration_errors, _m28_migration_errors,
            _m27_migration_errors, _m26_migration_errors, _m25_migration_errors,
            _m24_migration_errors, _m23_migration_errors, _m22_migration_errors,
            _m21_migration_errors, _m20_migration_errors, _m19_migration_errors,
            _m18_migration_errors, _m17_migration_errors, _m16_migration_errors,
        ):
            errors.extend(
                validator(scheduler, seal, migration, require_active_runtime=False)
            )
        return errors

    m53_key = _M53_SUCCESSOR_KEY
    m53_presence = (
        m53_key in scheduler,
        m53_key in migration,
        f"{m53_key}_cid" in seal,
    )
    if any(m53_presence):
        errors = _m53_migration_errors(scheduler, seal, migration)
        if not all(m53_presence):
            errors.append("M53 successor authority is only partially declared")
        if errors:
            return errors
        for validator in (
            _m36_migration_errors, _m35_migration_errors, _m34_migration_errors,
            _m33_migration_errors, _m32_migration_errors, _m31_migration_errors,
            _m30_migration_errors, _m29_migration_errors, _m28_migration_errors,
            _m27_migration_errors, _m26_migration_errors, _m25_migration_errors,
            _m24_migration_errors, _m23_migration_errors, _m22_migration_errors,
            _m21_migration_errors, _m20_migration_errors, _m19_migration_errors,
            _m18_migration_errors, _m17_migration_errors, _m16_migration_errors,
        ):
            errors.extend(
                validator(scheduler, seal, migration, require_active_runtime=False)
            )
        return errors

    m52_key = _M52_SUCCESSOR_KEY
    m52_presence = (
        m52_key in scheduler,
        m52_key in migration,
        f"{m52_key}_cid" in seal,
    )
    if any(m52_presence):
        errors = _m52_migration_errors(scheduler, seal, migration)
        if not all(m52_presence):
            errors.append("M52 successor authority is only partially declared")
        if errors:
            return errors
        for validator in (
            _m36_migration_errors, _m35_migration_errors, _m34_migration_errors,
            _m33_migration_errors, _m32_migration_errors, _m31_migration_errors,
            _m30_migration_errors, _m29_migration_errors, _m28_migration_errors,
            _m27_migration_errors, _m26_migration_errors, _m25_migration_errors,
            _m24_migration_errors, _m23_migration_errors, _m22_migration_errors,
            _m21_migration_errors, _m20_migration_errors, _m19_migration_errors,
            _m18_migration_errors, _m17_migration_errors, _m16_migration_errors,
        ):
            errors.extend(
                validator(scheduler, seal, migration, require_active_runtime=False)
            )
        return errors

    m51_key = _M51_SUCCESSOR_KEY
    m51_presence = (
        m51_key in scheduler,
        m51_key in migration,
        f"{m51_key}_cid" in seal,
    )
    if any(m51_presence):
        errors = _m51_migration_errors(scheduler, seal, migration)
        if not all(m51_presence):
            errors.append("M51 successor authority is only partially declared")
        if errors:
            return errors
        for validator in (
            _m36_migration_errors, _m35_migration_errors, _m34_migration_errors,
            _m33_migration_errors, _m32_migration_errors, _m31_migration_errors,
            _m30_migration_errors, _m29_migration_errors, _m28_migration_errors,
            _m27_migration_errors, _m26_migration_errors, _m25_migration_errors,
            _m24_migration_errors, _m23_migration_errors, _m22_migration_errors,
            _m21_migration_errors, _m20_migration_errors, _m19_migration_errors,
            _m18_migration_errors, _m17_migration_errors, _m16_migration_errors,
        ):
            errors.extend(
                validator(scheduler, seal, migration, require_active_runtime=False)
            )
        return errors

    m50_key = _M50_SUCCESSOR_KEY
    m50_presence = (
        m50_key in scheduler,
        m50_key in migration,
        f"{m50_key}_cid" in seal,
    )
    if any(m50_presence):
        errors = _m50_migration_errors(scheduler, seal, migration)
        if not all(m50_presence):
            errors.append("M50 successor authority is only partially declared")
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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
                validator(scheduler, seal, migration, require_active_runtime=False)
            )
        return errors

    m49_key = _M49_SUCCESSOR_KEY
    m49_presence = (
        m49_key in scheduler,
        m49_key in migration,
        f"{m49_key}_cid" in seal,
    )
    if any(m49_presence):
        errors = _m49_migration_errors(scheduler, seal, migration)
        if not all(m49_presence):
            errors.append("M49 successor authority is only partially declared")
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m48_key = _M48_SUCCESSOR_KEY
    m48_presence = (
        m48_key in scheduler,
        m48_key in migration,
        f"{m48_key}_cid" in seal,
    )
    if any(m48_presence):
        errors = _m48_migration_errors(scheduler, seal, migration)
        if not all(m48_presence):
            errors.append("M48 successor authority is only partially declared")
        # M48 independently rehashes M47's final controls and preserved
        # receipt, so M47's now-historical current-HEAD gate is not re-entered.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m47_key = _M47_SUCCESSOR_KEY
    m47_presence = (
        m47_key in scheduler,
        m47_key in migration,
        f"{m47_key}_cid" in seal,
    )
    if any(m47_presence):
        errors = _m47_migration_errors(scheduler, seal, migration)
        if not all(m47_presence):
            errors.append("M47 successor authority is only partially declared")
        # M47 independently rehashes M46's final controls, preserved receipt,
        # and exact two-path repair without re-entering M46's obsolete
        # current-HEAD source gate.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m46_key = _M46_SUCCESSOR_KEY
    m46_presence = (
        m46_key in scheduler,
        m46_key in migration,
        f"{m46_key}_cid" in seal,
    )
    if any(m46_presence):
        errors = _m46_migration_errors(scheduler, seal, migration)
        if not all(m46_presence):
            errors.append("M46 successor authority is only partially declared")
        # M46 independently rehashes M45's final controls and preserved
        # receipt without re-entering M45's obsolete current-HEAD source gate.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m45_key = _M45_SUCCESSOR_KEY
    m45_presence = (
        m45_key in scheduler,
        m45_key in migration,
        f"{m45_key}_cid" in seal,
    )
    if any(m45_presence):
        errors = _m45_migration_errors(scheduler, seal, migration)
        if not all(m45_presence):
            errors.append("M45 successor authority is only partially declared")
        # M45's validator independently rehashes the exact M44 controls and
        # delegates unchanged runtime semantics without entering M44's old
        # current-HEAD source gate.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m44_key = _M44_SUCCESSOR_KEY
    m44_presence = (
        m44_key in scheduler,
        m44_key in migration,
        f"{m44_key}_cid" in seal,
    )
    if any(m44_presence):
        errors = _m44_migration_errors(scheduler, seal, migration)
        if not all(m44_presence):
            errors.append("M44 successor authority is only partially declared")
        # M44's validator checks the exact M43 triplet, receipt, event 297,
        # and events 298-301 as immutable history.  It intentionally does not
        # invoke M43's current-root source-delta gate after b2136e3.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m43_key = "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    m43_presence = (
        m43_key in scheduler,
        m43_key in migration,
        f"{m43_key}_cid" in seal,
    )
    if any(m43_presence):
        errors = _m43_migration_errors(scheduler, seal, migration)
        if not all(m43_presence):
            errors.append("M43 successor authority is only partially declared")
        # A malformed or unresealed M43 declaration must mask every older
        # authority. Once M43 is exact, M42 remains mandatory immutable
        # history but no longer owns the active runtime generation binding.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m42_key = (
        "failed_pre_authoritative_m41_evidence_projection_"
        "successor_materialization"
    )
    m42_presence = (
        m42_key in scheduler,
        m42_key in migration,
        f"{m42_key}_cid" in seal,
    )
    if any(m42_presence):
        errors = _m42_migration_errors(scheduler, seal, migration)
        if not all(m42_presence):
            errors.append("M42 successor authority is only partially declared")
        # M41 stopped during authenticated read-only projection verification.
        # Mask its current-HEAD validator only after M42 verifies the immutable
        # M41 triplet, the closed legacy overlay, and all three M42 surfaces.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m41_key = (
        "failed_pre_authoritative_m40_validation_successor_materialization"
    )
    m41_presence = (
        m41_key in scheduler,
        m41_key in migration,
        f"{m41_key}_cid" in seal,
    )
    if any(m41_presence):
        errors = _m41_migration_errors(scheduler, seal, migration)
        if not all(m41_presence):
            errors.append("M41 successor authority is only partially declared")
        # M40 failed its sealed suite before materialization. Mask its current-
        # HEAD validator only after M41 verifies that no-write failure, the
        # immutable M40 triplet, and the complete three-surface M41 seal.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m40_key = "failed_pre_authoritative_m39_successor_materialization"
    m40_presence = (
        m40_key in scheduler,
        m40_key in migration,
        f"{m40_key}_cid" in seal,
    )
    if any(m40_presence):
        errors = _m40_migration_errors(scheduler, seal, migration)
        if not all(m40_presence):
            errors.append("M40 successor authority is only partially declared")
        # M39 failed before submitting an authoritative mutation. Mask its
        # live/source-head validator only after M40 verifies the typed no-write
        # failure, immutable M39 controls, and complete three-surface seal.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m39_key = "committed_m38_evidence_reconciliation_successor_materialization"
    m39_presence = (
        m39_key in scheduler,
        m39_key in migration,
        f"{m39_key}_cid" in seal,
    )
    if any(m39_presence):
        errors = _m39_migration_errors(scheduler, seal, migration)
        if not all(m39_presence):
            errors.append("M39 reconciliation authority is only partially declared")
        # M38 ended after its event commit but before receipt publication. Mask
        # its live/source-head validator only after M39 verifies the immutable
        # event-291 authority and the complete three-surface successor seal.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m38_key = "pre_authoritative_custody_restart_successor_materialization"
    m38_presence = (
        m38_key in scheduler,
        m38_key in migration,
        f"{m38_key}_cid" in seal,
    )
    if any(m38_presence):
        errors = _m38_migration_errors(scheduler, seal, migration)
        if not all(m38_presence):
            errors.append("M38 custody restart authority is only partially declared")
        # M37 is historical but unmaterialized. Mask its live/source-head
        # validator only after the complete M38 triplet and chain validate.
        if errors:
            return errors
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m37_key = "post_reboot_generation_restart_successor_materialization"
    m37_seal_key = f"{m37_key}_cid"
    m37_presence = (
        m37_key in scheduler,
        m37_key in migration,
        m37_seal_key in seal,
    )
    if any(m37_presence):
        errors = _m37_migration_errors(scheduler, seal, migration)
        if not all(m37_presence):
            errors.append(
                "M37 post-reboot restart authority is only partially declared"
            )
        for validator in (
            _m36_migration_errors,
            _m35_migration_errors,
            _m34_migration_errors,
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

    m36_key = "operator_task_binding_correction_successor_materialization"
    m36_seal_key = f"{m36_key}_cid"
    m36_presence = (
        m36_key in scheduler,
        m36_key in migration,
        m36_seal_key in seal,
    )
    if any(m36_presence):
        errors = _m36_migration_errors(scheduler, seal, migration)
        if not all(m36_presence):
            errors.append(
                "M36 operator-task binding authority is only partially declared"
            )
        for validator in (
            _m35_migration_errors,
            _m34_migration_errors,
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

    m35_key = "immutable_authority_identity_normalization_successor_materialization"
    m35_seal_key = f"{m35_key}_cid"
    m35_presence = (
        m35_key in scheduler,
        m35_key in migration,
        m35_seal_key in seal,
    )
    if any(m35_presence):
        errors = _m35_migration_errors(scheduler, seal, migration)
        if not all(m35_presence):
            errors.append(
                "M35 immutable-authority identity authority is only partially declared"
            )
        for validator in (
            _m34_migration_errors,
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
    m57_key = _M57_SUCCESSOR_KEY
    m57_selected = any((
        m57_key in config,
        m57_key in migration,
        f"{m57_key}_cid" in seal,
    ))
    m56_key = _M56_SUCCESSOR_KEY
    m56_selected = any((
        m56_key in config,
        m56_key in migration,
        f"{m56_key}_cid" in seal,
    )) and not m57_selected
    m55_key = _M55_SUCCESSOR_KEY
    m55_selected = any((
        m55_key in config,
        m55_key in migration,
        f"{m55_key}_cid" in seal,
    )) and not m56_selected and not m57_selected
    m53_key = _M53_SUCCESSOR_KEY
    m53_selected = any((
        m53_key in config,
        m53_key in migration,
        f"{m53_key}_cid" in seal,
    )) and not m55_selected and not m56_selected and not m57_selected
    m52_key = _M52_SUCCESSOR_KEY
    m52_selected = any((
        m52_key in config,
        m52_key in migration,
        f"{m52_key}_cid" in seal,
    )) and not m53_selected and not m55_selected and not m56_selected and not m57_selected and not m57_selected
    m51_key = _M51_SUCCESSOR_KEY
    m51_selected = any((
        m51_key in config,
        m51_key in migration,
        f"{m51_key}_cid" in seal,
    ))
    m50_key = _M50_SUCCESSOR_KEY
    m50_selected = any(
        (
            m50_key in config,
            m50_key in migration,
            f"{m50_key}_cid" in seal,
        )
    )
    m49_key = _M49_SUCCESSOR_KEY
    m49_selected = any(
        (
            m49_key in config,
            m49_key in migration,
            f"{m49_key}_cid" in seal,
        )
    )
    m48_key = _M48_SUCCESSOR_KEY
    m48_selected = any(
        (
            m48_key in config,
            m48_key in migration,
            f"{m48_key}_cid" in seal,
        )
    )
    m47_key = _M47_SUCCESSOR_KEY
    m47_declared = any(
        (
            m47_key in config,
            m47_key in migration,
            f"{m47_key}_cid" in seal,
        )
    )
    m47_selected = m47_declared and not m48_selected
    m46_key = _M46_SUCCESSOR_KEY
    m46_declared = any(
        (
            m46_key in config,
            m46_key in migration,
            f"{m46_key}_cid" in seal,
        )
    )
    m46_selected = m46_declared and not m47_selected and not m48_selected
    m45_key = _M45_SUCCESSOR_KEY
    m45_declared = any(
        (
            m45_key in config,
            m45_key in migration,
            f"{m45_key}_cid" in seal,
        )
    )
    m45_selected = (
        m45_declared
        and not m46_selected
        and not m47_selected
        and not m48_selected
    )
    m44_key = _M44_SUCCESSOR_KEY
    m44_declared = any(
        (
            m44_key in config,
            m44_key in migration,
            f"{m44_key}_cid" in seal,
        )
    )
    m44_selected = (
        m44_declared
        and not m45_selected
        and not m46_selected
        and not m47_selected
        and not m48_selected
    )
    m43_key = "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    m43_declared = any(
        (
            m43_key in config,
            m43_key in migration,
            f"{m43_key}_cid" in seal,
        )
    )
    m43_selected = (
        m43_declared
        and not m44_selected
        and not m45_selected
        and not m46_selected
        and not m47_selected
        and not m48_selected
    )
    m42_key = (
        "failed_pre_authoritative_m41_evidence_projection_"
        "successor_materialization"
    )
    m42_selected = any(
        (
            m42_key in config,
            m42_key in migration,
            f"{m42_key}_cid" in seal,
        )
    )
    m41_key = (
        "failed_pre_authoritative_m40_validation_successor_materialization"
    )
    m41_selected = any(
        (
            m41_key in config,
            m41_key in migration,
            f"{m41_key}_cid" in seal,
        )
    )
    m40_key = "failed_pre_authoritative_m39_successor_materialization"
    m40_selected = any(
        (
            m40_key in config,
            m40_key in migration,
            f"{m40_key}_cid" in seal,
        )
    )
    m39_key = "committed_m38_evidence_reconciliation_successor_materialization"
    m39_selected = any(
        (
            m39_key in config,
            m39_key in migration,
            f"{m39_key}_cid" in seal,
        )
    )
    m38_key = "pre_authoritative_custody_restart_successor_materialization"
    m38_selected = any(
        (
            m38_key in config,
            m38_key in migration,
            f"{m38_key}_cid" in seal,
        )
    )
    m37_key = "post_reboot_generation_restart_successor_materialization"
    m37_selected = any(
        (
            m37_key in config,
            m37_key in migration,
            f"{m37_key}_cid" in seal,
        )
    )
    m36_key = "operator_task_binding_correction_successor_materialization"
    m36_selected = any(
        (
            m36_key in config,
            m36_key in migration,
            f"{m36_key}_cid" in seal,
        )
    )
    m35_key = "immutable_authority_identity_normalization_successor_materialization"
    m35_selected = any(
        (
            m35_key in config,
            m35_key in migration,
            f"{m35_key}_cid" in seal,
        )
    )
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
        m53_selected
        or m52_selected
        or m51_selected
        or m48_selected
        or m47_selected
        or m46_selected
        or m45_selected
        or m44_selected
        or m43_selected
        or m42_selected
        or m41_selected
        or m40_selected
        or m39_selected
        or m38_selected
        or m37_selected
        or m36_selected
        or m35_selected
        or m34_selected
        or m33_selected
        or m32_selected
        or m31_selected
        or m30_selected
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
            "four lanes are required for M48/M47/M46/M45/M44/M43/M42/M41/M40/M39/M38/M37/M36/M35/M34/M33/M32/M31/M30/M29/M28/M27/M26/M25/M24/M23"
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
        if m57_selected
        else "run-r2-m27"
        if m56_selected
        else "run-r2-m27"
        if m55_selected
        else "run-r2-m27"
        if m53_selected
        else "run-r2-m27"
        if m52_selected
        else "run-r2-m27"
        if m51_selected
        else "run-r2-m27"
        if m50_selected
        else "run-r2-m27"
        if m48_selected
        else "run-r2-m27"
        if m47_selected
        else "run-r2-m27"
        if m46_selected
        else "run-r2-m27"
        if m45_selected
        else "run-r2-m27"
        if m44_selected
        else "run-r2-m27"
        if m43_selected
        else "run-r2-m27"
        if m42_selected
        else "run-r2-m27"
        if m41_selected
        else "run-r2-m27"
        if m40_selected
        else "run-r2-m27"
        if m39_selected
        else "run-r2-m27"
        if m38_selected
        else "run-r2-m27"
        if m37_selected
        else "run-r2-m27"
        if m36_selected
        else "run-r2-m27"
        if m35_selected
        else "run-r2-m27"
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
        "42"
        if m57_selected
        else "41"
        if m56_selected
        else "40"
        if m55_selected
        else "38"
        if m53_selected
        else "37"
        if m52_selected
        else "37"
        if m51_selected
        else "36"
        if m50_selected
        else "35"
        if m48_selected
        else "34"
        if m47_selected
        else "33"
        if m46_selected
        else "32"
        if m45_selected
        else "32"
        if m44_selected
        else "31"
        if m43_selected
        else "30"
        if m42_selected
        else "30"
        if m41_selected
        else "30"
        if m40_selected
        else "30"
        if m39_selected
        else "30"
        if m38_selected
        else "30"
        if m37_selected
        else "29"
        if m36_selected
        else "29"
        if m35_selected
        else "29"
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
        if m57_selected
        else 24070
        if m56_selected
        else 24070
        if m55_selected
        else 24070
        if m53_selected
        else 24070
        if m52_selected
        else 24070
        if m51_selected
        else 24070
        if m50_selected
        else 24070
        if m48_selected
        else 24070
        if m47_selected
        else 24070
        if m46_selected
        else 24070
        if m45_selected
        else 24070
        if m44_selected
        else 24070
        if m43_selected
        else 24070
        if m42_selected
        else 24070
        if m41_selected
        else 24070
        if m40_selected
        else 24070
        if m39_selected
        else 24070
        if m38_selected
        else 24070
        if m37_selected
        else 24070
        if m36_selected
        else 24070
        if m35_selected
        else 24070
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
    if m57_selected:
        successor = config.get(m57_key)
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m57_post_m56_live_ready_owner_missing_client_token_vault_restart_authority()
            )
            reference = dict(materializer._m57_authority_reference())
            contract = materializer._validated_m57_live_preflight_contract(
                expected
            )
            materializer._assert_m57_source_delta(
                root, materializer.build_population(root), expected
            )
            m57_errors = module._m57_post_m56_live_ready_owner_missing_client_token_vault_restart_errors(
                config, seal, migration, root=root
            )
            if (
                successor != reference
                or migration.get(m57_key) != reference
                or seal.get(f"{m57_key}_cid") != _M57_AUTHORITY_CID
                or materializer._identity(expected) != _M57_AUTHORITY_CID
                or len(materializer._canonical(expected)) != _M57_AUTHORITY_SIZE
                or expected.get("migration_revision") != "SAWM-R2-M57"
                or expected.get("migration_kind") != m57_key
                or contract.get("target_generation") != 42
                or contract.get("target_event_watermark") != 327
                or contract.get("target_projection_cid")
                != _M57_TARGET_PROJECTION_CID
            ):
                m57_errors.append("M57 successor authority binding differs")
            config_errors.extend(m57_errors)
        except Exception as exc:
            config_errors.append(
                f"M57 successor authority unavailable: {type(exc).__name__}: {exc}"
            )
    if m56_selected:
        successor = config.get(m56_key)
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m56_post_m55_live_ready_owner_missing_client_token_vault_restart_authority()
            )
            reference = dict(materializer._m56_authority_reference())
            contract = materializer._validated_m56_live_preflight_contract(
                expected
            )
            m56_errors = module._m56_post_m55_live_ready_owner_missing_client_token_vault_restart_errors(
                config, seal, migration, root=root
            )
            if (
                successor != reference
                or migration.get(m56_key) != reference
                or seal.get(f"{m56_key}_cid") != _M56_AUTHORITY_CID
                or materializer._identity(expected) != _M56_AUTHORITY_CID
                or len(materializer._canonical(expected)) != _M56_AUTHORITY_SIZE
                or expected.get("migration_revision") != "SAWM-R2-M56"
                or expected.get("migration_kind") != m56_key
                or contract.get("target_generation") != 41
                or contract.get("target_event_watermark") != 326
                or contract.get("target_projection_cid")
                != _M56_TARGET_PROJECTION_CID
            ):
                m56_errors.append("M56 successor authority binding differs")
            config_errors.extend(m56_errors)
        except Exception as exc:
            config_errors.append(
                f"M56 successor authority unavailable: {type(exc).__name__}: {exc}"
            )
    if m55_selected:
        successor = config.get(m55_key)
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m55_live_ready_owner_missing_client_token_vault_restart_authority()
            )
            reference = dict(materializer._m55_authority_reference())
            contract = materializer._validated_m55_live_preflight_contract(
                expected
            )
            m55_errors = module._m55_live_ready_owner_missing_client_token_vault_restart_errors(
                config, seal, migration, root=root
            )
            if (
                successor != reference
                or migration.get(m55_key) != reference
                or seal.get(f"{m55_key}_cid") != _M55_AUTHORITY_CID
                or materializer._identity(expected) != _M55_AUTHORITY_CID
                or len(materializer._canonical(expected)) != _M55_AUTHORITY_SIZE
                or expected.get("migration_revision") != "SAWM-R2-M55"
                or expected.get("migration_kind") != m55_key
                or contract.get("target_generation") != 40
                or contract.get("target_event_watermark") != 321
                or contract.get("target_projection_cid")
                != _M55_TARGET_PROJECTION_CID
            ):
                m55_errors.append("M55 successor authority binding differs")
            config_errors.extend(m55_errors)
        except Exception as exc:
            config_errors.append(
                f"M55 successor authority unavailable: {type(exc).__name__}: {exc}"
            )
    if m53_selected:
        successor = config.get(m53_key)
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m53_post_reboot_stale_ready_restart_authority()
            )
            reference = dict(materializer._m53_authority_reference())
            contract = materializer._validated_m53_live_preflight_contract(
                expected
            )
            m53_errors = module._m53_post_reboot_stale_ready_restart_errors(
                config, seal, migration, root=root
            )
            if (
                successor != reference
                or migration.get(m53_key) != reference
                or seal.get(f"{m53_key}_cid") != _M53_AUTHORITY_CID
                or materializer._identity(expected) != _M53_AUTHORITY_CID
                or len(materializer._canonical(expected)) != _M53_AUTHORITY_SIZE
                or expected.get("migration_revision") != "SAWM-R2-M53"
                or expected.get("migration_kind") != m53_key
                or contract.get("target_generation") != 38
                or contract.get("target_event_watermark") != 316
                or contract.get("target_projection_cid")
                != _M53_TARGET_PROJECTION_CID
            ):
                m53_errors.append("M53 successor authority binding differs")
            config_errors.extend(m53_errors)
        except Exception as exc:
            config_errors.append(
                f"M53 successor authority unavailable: {type(exc).__name__}: {exc}"
            )
    if m52_selected:
        successor = config.get(m52_key)
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m52_live_quack_catalog_compatibility_authority()
            )
            reference = dict(materializer._m52_authority_reference())
            contract = materializer._validated_m52_live_preflight_contract(
                expected
            )
            protocol = expected.get("live_query_protocol", {})
            failed = expected.get("failed_m51_attempt", {})
            repair = expected.get("accepted_test_compatibility_repair", {})
            stopped = expected.get("stopped_owner", {})
            changes = expected.get("exact_changes", {})
            policy = expected.get("receipt_policy", {})
            m52_errors = (
                module
                ._m52_test_compatibility_and_control_hash_successor_errors(
                    config, seal, migration, root=root
                )
            )
            if (
                successor != reference
                or migration.get(m52_key) != reference
                or seal.get(f"{m52_key}_cid") != _M52_AUTHORITY_CID
                or materializer._identity(expected) != _M52_AUTHORITY_CID
                or len(materializer._canonical(expected)) != _M52_AUTHORITY_SIZE
                or m52_errors
                or expected.get("migration_revision") != "SAWM-R2-M52"
                or expected.get("migration_kind") != m52_key
                or expected.get("authorized") is not True
                or expected.get("target_generation") != 37
                or expected.get("target_event_watermark") != 311
                or expected.get("target_projection_cid")
                != _M52_TARGET_PROJECTION_CID
                or contract.get("prior_generation") != 36
                or contract.get("target_generation") != 37
                or contract.get("prior_event_watermark") != 310
                or contract.get("target_event_watermark") != 311
                or contract.get("corrected_control_store_sha256")
                != materializer._M52_PRIOR_CONTROL_SHA256
                or len(materializer._M52_PRIOR_CONTROL_SHA256) != 64
                or stopped.get("status_projection_stopped_at_absent") is not True
                or stopped.get("database_stopped_at")
                != "2026-09-01T19:25:29Z"
                or protocol.get("offline_current_database") != "control"
                or protocol.get(
                    "remote_information_schema_tables_queries_for_claimed_zero_forbidden"
                ) is not True
                or protocol.get("claimed_zero_live_query_kind")
                != "select_count_only"
                or protocol.get("claimed_zero_arbitrary_sql_forbidden") is not True
                or protocol.get("ddl_forbidden") is not True
                or failed.get("failure_location")
                != (
                    "scripts/materialize_semantic_addressed_world_model_program.py:"
                    "_check_m51_prestart_admission"
                )
                or failed.get("typed_underlying_error")
                != "MigrationRequired: M51 stopped control store bytes/mode differ"
                or failed.get("operator_exit_code") != 2
                or failed.get("owner_started") is not False
                or failed.get("m51_prestart_receipt_created") is not False
                or failed.get("m51_final_receipt_created") is not False
                or repair.get("repair_commit")
                != materializer._M52_REPAIR_COMMIT
                or repair.get("repair_parent")
                != materializer._M52_REPAIR_PARENT
                or repair.get("repair_tree") != materializer._M52_REPAIR_TREE
                or repair.get("changed_paths")
                != [materializer._M52_REPAIR_PATH]
                or repair.get("binary_diff_sha256")
                != materializer._M52_REPAIR_DIFF_SHA256
                or policy.get("completion_authority") is not False
                or policy.get("launch_authority") is not False
                or any(changes.get(field) != 0 for field in (
                    "task_revision_changes", "task_status_changes",
                    "goal_revision_changes", "goal_status_changes",
                    "implementation_provider_invocations", "provider_call_changes",
                    "provider_response_changes", "effect_claim_changes",
                    "merge_attempt_changes", "merge_base_changes",
                    "merge_queue_entry_changes", "accepted_completion_changes",
                ))
            ):
                config_errors.append(
                    "M52 test-compatibility/hash authority/source seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M52 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
    if m51_selected and not m52_selected and not m53_selected and not m55_selected and not m56_selected and not m57_selected and not m57_selected:
        successor = config.get(m51_key)
        try:
            module, materializer = _m26_validation_modules(root)
            expected = materializer._expected_m51_live_quack_catalog_compatibility_authority()
            reference = dict(materializer._m51_authority_reference())
            contract = materializer._validated_m51_live_preflight_contract(expected)
            protocol = expected.get("live_query_protocol", {})
            failed = expected.get("failed_m50_attempt", {})
            stopped = expected.get("stopped_owner", {})
            changes = expected.get("exact_changes", {})
            policy = expected.get("receipt_policy", {})
            m51_errors = module._m51_live_quack_catalog_compatibility_errors(
                config, seal, migration, root=root
            )
            if (
                successor != reference
                or migration.get(m51_key) != reference
                or seal.get(f"{m51_key}_cid") != _M51_AUTHORITY_CID
                or materializer._identity(expected) != _M51_AUTHORITY_CID
                or len(materializer._canonical(expected)) != _M51_AUTHORITY_SIZE
                or m51_errors
                or expected.get("migration_revision") != "SAWM-R2-M51"
                or expected.get("migration_kind") != m51_key
                or expected.get("authorized") is not True
                or expected.get("target_generation") != 37
                or expected.get("target_event_watermark") != 311
                or expected.get("target_projection_cid") != _M51_TARGET_PROJECTION_CID
                or contract.get("prior_generation") != 36
                or contract.get("target_generation") != 37
                or contract.get("prior_event_watermark") != 310
                or contract.get("target_event_watermark") != 311
                or stopped.get("status_projection_stopped_at_absent") is not True
                or stopped.get("database_stopped_at") != "2026-09-01T19:25:29Z"
                or protocol.get("offline_current_database") != "control"
                or protocol.get(
                    "remote_information_schema_tables_queries_for_claimed_zero_forbidden"
                ) is not True
                or protocol.get("claimed_zero_live_query_kind") != "select_count_only"
                or protocol.get("claimed_zero_arbitrary_sql_forbidden") is not True
                or protocol.get("ddl_forbidden") is not True
                or protocol.get("forbidden_ddl_prefixes") != [
                    "CREATE ", "ALTER ", "DROP ", "ATTACH ", "DETACH ",
                    "COPY ", "EXPORT ", "IMPORT ", "INSTALL ", "LOAD ",
                    "PRAGMA ", "SET ", "CALL ", "VACUUM ", "CHECKPOINT ",
                    "FORCE ", "TRUNCATE ",
                ]
                or failed.get("m50_receipt_created") is not False
                or failed.get("m50_event_311_created") is not False
                or policy.get("completion_authority") is not False
                or policy.get("launch_authority") is not False
                or any(changes.get(field) != 0 for field in (
                    "task_revision_changes", "task_status_changes",
                    "goal_revision_changes", "goal_status_changes",
                    "implementation_provider_invocations", "provider_call_changes",
                    "provider_response_changes", "effect_claim_changes",
                    "merge_attempt_changes", "merge_base_changes",
                    "merge_queue_entry_changes", "accepted_completion_changes",
                ))
            ):
                config_errors.append("M51 live-catalog authority/source seal differs")
        except Exception as exc:
            config_errors.append(
                f"M51 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
    if m50_selected and not m51_selected:
        successor = config.get(m50_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m50_post_m49_fenced_worktree_quarantine_recovery_authority()
            )
            contract = materializer._validated_m50_live_preflight_contract(expected)
            reference = dict(materializer._m50_authority_reference())
            configured_cid = str(materializer._M50_AUTHORITY_CID)
            receipt = expected.get("preserved_m49_receipt", {})
            stopped = expected.get("stopped_owner", {})
            repair = expected.get("accepted_control_plane_repair", {})
            rejected = expected.get("rejected_unsafe_historical_attempt", {})
            revert = expected.get("transparent_revert", {})
            target = expected.get("target_authority", {})
            chain = expected.get("source_chain", {})
            changes = expected.get("exact_changes", {})
            preservation = expected.get("preservation", {})
            zero_baseline = expected.get("claimed_zero_authority_baseline", {})
            receipt_policy = expected.get("receipt_policy", {})
            m50_errors = module._m50_post_m49_fenced_recovery_errors(
                config, seal, migration, root=root
            )
            if (
                successor != reference
                or migration.get(m50_key) != reference
                or seal.get(f"{m50_key}_cid") != configured_cid
                or configured_cid != _M50_AUTHORITY_CID
                or configured_cid == _M50_UNSEALED_AUTHORITY_CID
                or materializer._identity(expected) != configured_cid
                or len(materializer._canonical(expected)) != _M50_AUTHORITY_SIZE
                or m50_errors
                or expected.get("schema")
                != "sawm/post-m49-fenced-worktree-quarantine-recovery-authorization@1"
                or expected.get("migration_revision") != "SAWM-R2-M50"
                or expected.get("migration_kind") != m50_key
                or expected.get("authorized") is not True
                or expected.get("authority") != "operator_control_plane"
                or int(expected.get("target_generation") or 0) != 36
                or int(expected.get("target_event_watermark") or 0) != 311
                or int(expected.get("target_plan_revision") or 0) != 28
                or expected.get("target_projection_cid")
                != _M50_TARGET_PROJECTION_CID
                or receipt.get("receipt_cid") != _M50_M49_RECEIPT_CID
                or receipt.get("created_or_rewritten") is not False
                or stopped.get("generation") != 35
                or stopped.get("target_generation") != 36
                or stopped.get("prior_owner_process_must_be_dead") is not True
                or stopped.get("process_birth_identity_must_be_present") is not True
                or stopped.get("process_birth_owner_uid_must_match") is not True
                or stopped.get("prior_listener_must_be_absent") is not True
                or stopped.get("unknown_process_liveness_is_rejected") is not True
                or repair.get("repair_commit")
                != "cca184ec40dffad195f9df95f95540cdd3b9c02e"
                or repair.get("arbitrary_unreadable_procfs_exemption") is not False
                or repair.get("authority_weakened") is not False
                or rejected.get("accepted_authority") is not False
                or revert.get("restored_m49_exactly") is not True
                or target.get("event_watermark") != 311
                or target.get("projection_cid") != _M50_TARGET_PROJECTION_CID
                or chain.get("final_control_parent")
                != "cca184ec40dffad195f9df95f95540cdd3b9c02e"
                or chain.get("rejected_commit_count") != 1
                or chain.get("transparent_revert_commit_count") != 1
                or chain.get("accepted_repair_commit_count") != 1
                or chain.get("final_control_commit_count") != 1
                or chain.get("current_commit_identity_embedded_in_authority")
                is not False
                or expected.get("ordinary_source_changes") != 4
                or contract.get("prior_generation") != 35
                or contract.get("target_generation") != 36
                or contract.get("prior_event_watermark") != 310
                or contract.get("target_event_watermark") != 311
                or contract.get("same_live_owner_required") is not False
                or contract.get("generation_restart_authorized") is not True
                or changes.get("event_suffix_length") != 1
                or changes.get("evidence_node_changes") != 1
                or changes.get("evidence_event_changes") != 1
                or changes.get("validation_event_changes") != 0
                or changes.get("implementation_provider_invocations") != 0
                or changes.get("provider_call_changes") != 0
                or changes.get("provider_response_changes") != 0
                or changes.get("effect_claim_changes") != 0
                or changes.get("merge_attempt_changes") != 0
                or changes.get("merge_base_changes") != 0
                or changes.get("merge_queue_entry_changes") != 0
                or changes.get("task_revision_changes") != 0
                or changes.get("task_status_changes") != 0
                or changes.get("accepted_completion_changes") != 0
                or zero_baseline.get("counts")
                != dict(materializer._M50_ZERO_ROW_AUTHORITY_COUNTS)
                or zero_baseline.get(
                    "persistence_through_receipt_publication_claimed"
                )
                is not False
                or zero_baseline.get(
                    "fresh_live_revalidation_required_after_receipt_read"
                )
                is not True
                or receipt_policy.get("authoritative") is not False
                or receipt_policy.get("completion_authority") is not False
                or receipt_policy.get("launch_authority") is not False
                or receipt_policy.get("deny_only_without_fresh_live_revalidation")
                is not True
                or receipt_policy.get(
                    "fresh_live_revalidation_required_after_receipt_read"
                )
                is not True
                or preservation.get("m49_authority_preserved_exactly") is not True
                or preservation.get("m49_receipt_preserved_exactly") is not True
                or preservation.get("generation_bearing_owner_restart") is not True
                or preservation.get(
                    "claimed_zero_operational_authorities_observed_at_post_append_snapshot"
                )
                is not True
                or preservation.get("worker_self_approval") is not False
            ):
                config_errors.append(
                    "M50 fenced generation-36 recovery authority/source seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M50 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M50 active runtime paths are not exactly preserved")
    if m49_selected and not m50_selected and not m51_selected:
        successor = config.get(m49_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m49_post_m48_successor_report_fix_authority()
            )
            contract = materializer._validated_m49_live_preflight_contract(expected)
            reference = dict(materializer._m49_authority_reference())
            configured_cid = str(materializer._M49_AUTHORITY_CID)
            receipt = expected.get("preserved_m48_receipt", {})
            preserved = expected.get("preserved_m48_materialization", {})
            owner = expected.get("live_owner", {})
            repair = expected.get("accepted_control_plane_repair", {})
            target = expected.get("target_authority", {})
            chain = expected.get("source_chain", {})
            changes = expected.get("exact_changes", {})
            preservation = expected.get("preservation", {})
            m49_errors = module._m49_post_m48_successor_report_fix_errors(
                config, seal, migration, root=root
            )
            if (
                successor != reference
                or migration.get(m49_key) != reference
                or seal.get(f"{m49_key}_cid") != configured_cid
                or configured_cid != _M49_AUTHORITY_CID
                or configured_cid == _M49_UNSEALED_AUTHORITY_CID
                or materializer._identity(expected) != configured_cid
                or len(materializer._canonical(expected)) != _M49_AUTHORITY_SIZE
                or m49_errors
                or expected.get("schema")
                != "sawm/post-m48-successor-report-fix-authorization@1"
                or expected.get("migration_revision") != "SAWM-R2-M49"
                or expected.get("migration_kind") != m49_key
                or expected.get("authorized") is not True
                or expected.get("authority") != "operator_control_plane"
                or int(expected.get("target_generation") or 0) != 35
                or int(expected.get("target_event_watermark") or 0) != 306
                or int(expected.get("target_plan_revision") or 0) != 28
                or expected.get("target_projection_cid")
                != _M49_TARGET_PROJECTION_CID
                or receipt.get("receipt_cid") != _M49_M48_RECEIPT_CID
                or receipt.get("sha256")
                != "9a0f54ea8e42eb6ae8b87acbeed951aab1a4f643996958b9c09dcdcb52ae9e20"
                or receipt.get("size") != 6_495
                or receipt.get("mode") != "0600"
                or receipt.get("created_or_rewritten") is not False
                or preserved.get("authority_cid") != _M48_AUTHORITY_CID
                or preserved.get("receipt_cid") != _M49_M48_RECEIPT_CID
                or preserved.get("event_watermark") != 305
                or preserved.get("projection_cid") != _M48_TARGET_PROJECTION_CID
                or owner.get("server_id")
                != "server:b37f1c63-76ff-42e1-8404-10e15952f472"
                or owner.get("process_birth_id")
                != "birth:8067c25fb34a40bbfa92ff287a4c7441"
                or owner.get("same_owner_required") is not True
                or owner.get("generation_restart_authorized") is not False
                or repair.get("repair_commit")
                != "3025557ac782af4f55b59b5fb00c661ba6fcfd77"
                or repair.get("authority_weakened") is not False
                or target.get("event_watermark") != 306
                or target.get("projection_cid") != _M49_TARGET_PROJECTION_CID
                or chain.get("m48_final_control_commit")
                != "53f050c40ad78f61674f7383ebc2708e062a7677"
                or chain.get("repair_commit")
                != "3025557ac782af4f55b59b5fb00c661ba6fcfd77"
                or chain.get("final_control_parent")
                != "3025557ac782af4f55b59b5fb00c661ba6fcfd77"
                or chain.get("repair_commit_count") != 1
                or chain.get("final_control_commit_count") != 1
                or chain.get("current_commit_identity_embedded_in_authority")
                is not False
                or expected.get("ordinary_source_changes") != 2
                or contract.get("target_generation") != 35
                or contract.get("prior_event_watermark") != 305
                or contract.get("target_event_watermark") != 306
                or contract.get("same_live_owner_required") is not True
                or contract.get("generation_restart_authorized") is not False
                or contract.get("m48_receipt_must_be_preserved") is not True
                or contract.get("event_305_must_be_preserved") is not True
                or contract.get("event_306_must_be_absent_before_append") is not True
                or changes.get("event_suffix_length") != 1
                or changes.get("evidence_node_changes") != 1
                or changes.get("evidence_event_changes") != 1
                or changes.get("validation_event_changes") != 0
                or changes.get("owner_generation_changes") != 0
                or changes.get("task_revision_changes") != 0
                or changes.get("task_status_changes") != 0
                or changes.get("goal_revision_changes") != 0
                or changes.get("goal_status_changes") != 0
                or changes.get("plan_revision_changes") != 0
                or changes.get("accepted_completion_changes") != 0
                or changes.get("worker_self_approval") is not False
                or preservation.get("same_live_owner") is not True
                or preservation.get("same_store_generation") is not True
                or preservation.get("generation_restart") is not False
                or preservation.get("m48_authority_preserved_exactly") is not True
                or preservation.get("m48_receipt_preserved_exactly") is not True
                or preservation.get("event_305_preserved_exactly") is not True
                or preservation.get("worker_self_approval") is not False
            ):
                config_errors.append(
                    "M49 same-owner event-306 repair authority/source seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M49 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M49 active runtime paths are not exactly preserved")
    if m48_selected and not m49_selected:
        successor = config.get(m48_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m48_post_m47_clean_shutdown_restart_successor_authority()
            )
            contract = materializer._validated_m48_live_preflight_contract(expected)
            reference = dict(materializer._m48_authority_reference())
            configured_cid = str(materializer._M48_AUTHORITY_CID)
            stopped = expected.get("stopped_owner", {})
            artifacts = expected.get("stopped_prestart_artifacts", {})
            receipt = expected.get("preserved_m47_receipt", {})
            preserved = expected.get("preserved_m47_materialization", {})
            observation = expected.get(
                "failed_test_induced_shutdown_observation", {}
            )
            target = expected.get("target_authority", {})
            chain = expected.get("source_chain", {})
            changes = expected.get("exact_changes", {})
            preservation = expected.get("preservation", {})
            m48_errors = (
                module._m48_post_m47_clean_shutdown_restart_successor_errors(
                    config, seal, migration, root=root
                )
            )
            if (
                successor != reference
                or migration.get(m48_key) != reference
                or seal.get(f"{m48_key}_cid") != configured_cid
                or configured_cid != _M48_AUTHORITY_CID
                or configured_cid == _M48_UNSEALED_AUTHORITY_CID
                or materializer._identity(expected) != configured_cid
                or len(materializer._canonical(expected)) != _M48_AUTHORITY_SIZE
                or m48_errors
                or expected.get("schema")
                != "sawm/post-m47-clean-shutdown-restart-authorization@1"
                or expected.get("migration_revision") != "SAWM-R2-M48"
                or expected.get("migration_kind") != m48_key
                or expected.get("authorized") is not True
                or expected.get("authority") != "operator_control_plane"
                or int(expected.get("target_generation") or 0) != 35
                or int(expected.get("target_event_watermark") or 0) != 305
                or int(expected.get("target_plan_revision") or 0) != 28
                or expected.get("target_projection_cid")
                != _M48_TARGET_PROJECTION_CID
                or stopped.get("status") != "stopped"
                or stopped.get("generation") != 34
                or stopped.get("target_generation") != 35
                or artifacts.get("control_mode") != "0664"
                or artifacts.get("m48_receipt_absent") is not True
                or artifacts.get("m47_receipt_present") is not True
                or artifacts.get("m44_receipt_absent") is not True
                or receipt.get("receipt_cid") != _M48_M47_RECEIPT_CID
                or receipt.get("sha256")
                != "e8ef92a302f38d37d5704d67d1beb9575091c16c2ee61d6f680fd34aa4cfe731"
                or receipt.get("size") != 8_088
                or receipt.get("mode") != "0600"
                or receipt.get("created_or_rewritten") is not False
                or preserved.get("authority_cid") != _M47_AUTHORITY_CID
                or preserved.get("receipt_cid") != _M48_M47_RECEIPT_CID
                or preserved.get("event_watermark") != 304
                or preserved.get("event_id")
                != "baguqeeranmbjd63u4wungeeox327cskqjtyypsavnwjwdlh3pqj3zbmtkeua"
                or preserved.get("evidence_id")
                != "baguqeeraqy7idops72utponoumzonzd3hbo4f4tqmg74of223pcbvb2mouua"
                or preserved.get("migration_digest")
                != "sha256:bceaad94e642ea09ba9ef1d4567b809db1ef6e5f371b3ff5b86273a1b3efa1b8"
                or preserved.get("source_binding_cid")
                != "sha256:2f87cf1b9620144aca641b49acbbe39d12a19917318bd351e5210c81e206dae1"
                or preserved.get("final_control_commit")
                != "ffec7b3c57cd75843dbedfb260263a98c4104d36"
                or preserved.get("final_control_tree")
                != "100f37e1da44aa5aec079c922b962d1ac9fdc3db"
                or observation.get("test_suite_completed") is not False
                or observation.get("test_suite_exit_code") != 2
                or observation.get("passed_tests_before_interruption") != 21
                or observation.get("operational_history_only") is not True
                or observation.get("validation_authority") is not False
                or observation.get("task_completion_authority") is not False
                or observation.get("worker_self_approval") is not False
                or target.get("event_watermark") != 305
                or target.get("projection_cid") != _M48_TARGET_PROJECTION_CID
                or target.get("evidence_kind")
                != (
                    "operator_control_plane_post_m47_clean_shutdown_restart_"
                    "successor"
                )
                or chain.get("m47_final_control_commit")
                != "ffec7b3c57cd75843dbedfb260263a98c4104d36"
                or chain.get("m47_final_control_tree")
                != "100f37e1da44aa5aec079c922b962d1ac9fdc3db"
                or chain.get("final_control_parent")
                != "ffec7b3c57cd75843dbedfb260263a98c4104d36"
                or chain.get("ordinary_repair_commit_count") != 0
                or chain.get("final_control_commit_count") != 1
                or expected.get("ordinary_source_changes") != 0
                or contract.get("prior_generation") != 34
                or contract.get("target_generation") != 35
                or contract.get("prior_event_watermark") != 304
                or contract.get("target_event_watermark") != 305
                or contract.get("m47_receipt_must_be_preserved") is not True
                or contract.get("m44_receipt_must_remain_absent") is not True
                or contract.get("event_304_must_be_preserved") is not True
                or contract.get("event_305_must_be_absent_before_append")
                is not True
                or contract.get("generation_restart_authorized") is not True
                or changes.get("event_suffix_length") != 1
                or changes.get("evidence_node_changes") != 1
                or changes.get("evidence_event_changes") != 1
                or changes.get("validation_event_changes") != 0
                or changes.get("task_revision_changes") != 0
                or changes.get("task_status_changes") != 0
                or changes.get("goal_revision_changes") != 0
                or changes.get("goal_status_changes") != 0
                or changes.get("plan_revision_changes") != 0
                or changes.get("accepted_completion_changes") != 0
                or changes.get("worker_self_approval") is not False
                or preservation.get("generation_34_preserved_stopped") is not True
                or preservation.get("m47_authority_preserved_exactly") is not True
                or preservation.get("m47_receipt_preserved_exactly") is not True
                or preservation.get("event_304_preserved_exactly") is not True
                or preservation.get(
                    "failed_test_history_preserved_without_authority"
                )
                is not True
                or preservation.get("generation_bearing_owner_restart") is not True
                or preservation.get("worker_self_approval") is not False
            ):
                config_errors.append(
                    "M48 generation-35 clean-shutdown restart authority/source "
                    "seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M48 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M48 active runtime paths are not exactly preserved")
    if m47_selected:
        successor = config.get(m47_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m47_ignored_python_cache_preservation_and_recovery_successor_authority()
            )
            contract = materializer._validated_m47_live_preflight_contract(
                expected
            )
            reference = dict(materializer._m47_authority_reference())
            configured_cid = str(materializer._M47_AUTHORITY_CID)
            stopped = expected.get("stopped_owner", {})
            artifacts = expected.get("stopped_prestart_artifacts", {})
            receipt = expected.get("preserved_m46_receipt", {})
            preserved = expected.get("preserved_m46_materialization", {})
            repair = expected.get(
                "accepted_ignored_python_cache_preservation_repair", {}
            )
            chain = expected.get("source_chain", {})
            changes = expected.get("exact_changes", {})
            preservation = expected.get("preservation", {})
            m47_errors = (
                module
                ._m47_ignored_python_cache_preservation_and_recovery_successor_errors(
                    config, seal, migration, root=root
                )
            )
            if (
                successor != reference
                or migration.get(m47_key) != reference
                or seal.get(f"{m47_key}_cid") != configured_cid
                or configured_cid != _M47_AUTHORITY_CID
                or configured_cid == _M47_UNSEALED_AUTHORITY_CID
                or materializer._identity(expected) != configured_cid
                or len(materializer._canonical(expected)) != _M47_AUTHORITY_SIZE
                or m47_errors
                or expected.get("schema")
                != (
                    "sawm/ignored-python-cache-preservation-and-recovery-"
                    "authorization@1"
                )
                or expected.get("migration_revision") != "SAWM-R2-M47"
                or expected.get("migration_kind") != m47_key
                or expected.get("authorized") is not True
                or expected.get("authority") != "operator_control_plane"
                or int(expected.get("target_generation") or 0) != 34
                or int(expected.get("target_event_watermark") or 0) != 304
                or int(expected.get("target_plan_revision") or 0) != 28
                or expected.get("target_projection_cid")
                != _M47_TARGET_PROJECTION_CID
                or stopped.get("status") != "stopped"
                or stopped.get("server_id")
                != "server:33625550-a262-4c25-8d36-f83b4dea8c2d"
                or stopped.get("process_birth_id")
                != "birth:e66a259534ac16f234d37ef185a5a797"
                or stopped.get("generation") != 33
                or stopped.get("target_generation") != 34
                or stopped.get("started_at") != "2026-09-01T14:26:15Z"
                or stopped.get("stopped_at") != "2026-09-01T14:38:23Z"
                or stopped.get("startup_epoch") != 1_788_272_775
                or artifacts.get("owner_marker_absent") is not True
                or artifacts.get("stop_control_absent") is not True
                or artifacts.get("token_handoff_absent") is not True
                or artifacts.get("control_wal_absent") is not True
                or artifacts.get("coordination_wal_absent") is not True
                or artifacts.get("m47_receipt_absent") is not True
                or artifacts.get("m44_receipt_absent") is not True
                or receipt.get("receipt_cid") != _M47_M46_RECEIPT_CID
                or receipt.get("sha256")
                != "534c042365c36dcaaef99aaf8fe32af88852d9e376235cb715ade82da22b88cf"
                or receipt.get("size") != 8_673
                or receipt.get("mode") != "0600"
                or receipt.get("created_or_rewritten") is not False
                or preserved.get("authority_cid") != _M46_AUTHORITY_CID
                or preserved.get("receipt_cid") != _M47_M46_RECEIPT_CID
                or preserved.get("event_watermark") != 303
                or preserved.get("event_id")
                != (
                    "baguqeeraq2vliqq7mqgygm6a2wxfpw56zp5wv5bf5l7e6iwqr463vk3xjywq"
                )
                or preserved.get("evidence_id")
                != (
                    "baguqeeradmyg67icb3nurhnfhgk4fjck4b43hc5xgyndgaern3tkjxlda3sq"
                )
                or preserved.get("migration_digest")
                != (
                    "sha256:e557dd4917e6e350a8579d76c054ba5129f154ad25ac3a5eca03ecc0f69250f3"
                )
                or preserved.get("source_binding_cid")
                != (
                    "sha256:122537df3b3493bf093e59f4e16ac4b2f7e71a8317c4cc7ba658b9692ad73330"
                )
                or preserved.get("final_control_commit")
                != "c9474da7158066bffaa6cb3395bfa3801ba79111"
                or preserved.get("final_control_tree")
                != "e227a2708771a7edd6af7047bfba98fc6b9293a2"
                or preserved.get("m44_receipt_absent") is not True
                or repair.get("repair_parent")
                != "c9474da7158066bffaa6cb3395bfa3801ba79111"
                or repair.get("repair_commit")
                != "7fe0f09615c6d07ba72d3f6334b6bb4f6a512141"
                or repair.get("repair_tree")
                != "9d89c71fd4d9a89b731bf83f6024f411fbac2b53"
                or repair.get("binary_diff_sha256")
                != (
                    "e45a80dea318c2b3ec2851540b7058cd62d4713606379fa341132b28c8ead5c4"
                )
                or repair.get("changed_paths")
                != sorted(materializer._M47_REPAIR_BLOBS)
                or repair.get("blob_oids")
                != dict(materializer._M47_REPAIR_BLOBS)
                or repair.get("path_modes")
                != dict(materializer._M47_REPAIR_MODES)
                or repair.get("repair_scope")
                != "ignored_python_cache_preservation_and_recovery"
                or repair.get("exact_original_ref_and_tree_required") is not True
                or repair.get("current_tag_cpython_cache_required") is not True
                or repair.get("tracked_source_binding_required") is not True
                or repair.get("raw_bytes_preserved_in_worktree") is not True
                or repair.get("raw_bytes_copied_to_receipt") is not False
                or repair.get("recovery_classifier_executes_payloads") is not False
                or repair.get("admitted_as_declared_output") is not False
                or repair.get("cache_observation_advisory") is not True
                or repair.get("cache_observation_authoritative") is not False
                or repair.get("unmatched_ignored_artifacts_fail_closed") is not True
                or repair.get("unsafe_ignored_artifacts_fail_closed") is not True
                or repair.get("provider_dispatched") is not False
                or repair.get("mutation_authority") is not False
                or repair.get("merge_authority") is not False
                or repair.get("task_completion_authority") is not False
                or repair.get("validation_weakened") is not False
                or repair.get("authority_weakened") is not False
                or repair.get("worker_self_approval") is not False
                or chain.get("m46_final_control_commit")
                != "c9474da7158066bffaa6cb3395bfa3801ba79111"
                or chain.get("m46_final_control_tree")
                != "e227a2708771a7edd6af7047bfba98fc6b9293a2"
                or chain.get("m46_final_control_blobs")
                != dict(materializer._M47_M46_FINAL_CONTROL_BLOBS)
                or chain.get("m46_final_control_modes")
                != dict(materializer._M47_M46_FINAL_CONTROL_MODES)
                or chain.get("repair_parent")
                != "c9474da7158066bffaa6cb3395bfa3801ba79111"
                or chain.get("repair_commit")
                != "7fe0f09615c6d07ba72d3f6334b6bb4f6a512141"
                or chain.get("repair_tree")
                != "9d89c71fd4d9a89b731bf83f6024f411fbac2b53"
                or chain.get("repair_diff_sha256")
                != (
                    "e45a80dea318c2b3ec2851540b7058cd62d4713606379fa341132b28c8ead5c4"
                )
                or chain.get("repair_blobs")
                != dict(materializer._M47_REPAIR_BLOBS)
                or chain.get("repair_modes")
                != dict(materializer._M47_REPAIR_MODES)
                or chain.get("final_control_parent")
                != "7fe0f09615c6d07ba72d3f6334b6bb4f6a512141"
                or chain.get("repair_commit_count") != 1
                or chain.get("final_control_commit_count") != 1
                or chain.get("current_commit_identity_embedded_in_authority")
                is not False
                or contract.get("migration_revision") != "SAWM-R2-M47"
                or contract.get("prior_generation") != 33
                or contract.get("target_generation") != 34
                or contract.get("prior_event_watermark") != 303
                or contract.get("target_event_watermark") != 304
                or contract.get("m46_receipt_must_be_preserved") is not True
                or contract.get("m44_receipt_must_remain_absent") is not True
                or contract.get("event_303_must_be_preserved") is not True
                or contract.get("event_304_must_be_absent_before_append")
                is not True
                or contract.get("generation_restart_authorized") is not True
                or changes.get("event_suffix_length") != 1
                or changes.get("evidence_node_changes") != 1
                or changes.get("evidence_event_changes") != 1
                or changes.get("validation_event_changes") != 0
                or changes.get("store_generation_row_changes") != 1
                or changes.get("state_server_row_changes") != 1
                or changes.get("credential_row_changes") != 1
                or changes.get("server_epoch_row_changes") != 1
                or changes.get("capability_snapshot_row_changes") != 1
                or changes.get("task_revision_changes") != 0
                or changes.get("task_status_changes") != 0
                or changes.get("goal_revision_changes") != 0
                or changes.get("plan_revision_changes") != 0
                or changes.get("coordination_semantic_changes") != 0
                or changes.get("implementation_provider_invocations") != 0
                or changes.get("effect_claim_changes") != 0
                or changes.get("merge_attempt_changes") != 0
                or changes.get("accepted_completion_changes") != 0
                or changes.get("worker_self_approval") is not False
                or preservation.get("generation_33_preserved_stopped") is not True
                or preservation.get("m46_authority_preserved_exactly") is not True
                or preservation.get("m46_receipt_preserved_exactly") is not True
                or preservation.get("m46_receipt_created_or_rewritten") is not False
                or preservation.get("m44_receipt_remains_absent") is not True
                or preservation.get("event_303_preserved_exactly") is not True
                or preservation.get("task_goal_plan_heads_preserved") is not True
                or preservation.get("same_database_uuid") is not True
                or preservation.get("same_runtime_root") is not True
                or preservation.get("same_store_path") is not True
                or preservation.get("generation_bearing_owner_restart") is not True
                or preservation.get("coordination_store_bytes_preserved_exactly")
                is not True
                or preservation.get("coordination_semantic_changes") != 0
                or preservation.get("worker_self_approval") is not False
            ):
                config_errors.append(
                    "M47 generation-34 ignored-cache preservation/recovery "
                    "authority/source seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M47 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M47 active runtime paths are not exactly preserved")
    if m46_selected:
        successor = config.get(m46_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m46_legacy_no_delta_rescue_recovery_successor_authority()
            )
            contract = materializer._validated_m46_live_preflight_contract(expected)
            reference = dict(materializer._m46_authority_reference())
            configured_cid = str(materializer._M46_AUTHORITY_CID)
            stopped = expected.get("stopped_owner", {})
            artifacts = expected.get("stopped_prestart_artifacts", {})
            receipt = expected.get("preserved_m45_receipt", {})
            preserved = expected.get("preserved_m45_materialization", {})
            repair = expected.get("accepted_legacy_no_delta_rescue_repair", {})
            chain = expected.get("source_chain", {})
            changes = expected.get("exact_changes", {})
            preservation = expected.get("preservation", {})
            m46_errors = (
                module._m46_legacy_no_delta_rescue_recovery_successor_errors(
                    config, seal, migration, root=root
                )
            )
            if (
                successor != reference
                or migration.get(m46_key) != reference
                or seal.get(f"{m46_key}_cid") != configured_cid
                or configured_cid != _M46_AUTHORITY_CID
                or configured_cid == _M46_UNSEALED_AUTHORITY_CID
                or materializer._identity(expected) != configured_cid
                or len(materializer._canonical(expected)) != _M46_AUTHORITY_SIZE
                or m46_errors
                or expected.get("schema")
                != "sawm/legacy-no-delta-rescue-recovery-authorization@1"
                or expected.get("migration_revision") != "SAWM-R2-M46"
                or expected.get("migration_kind") != m46_key
                or expected.get("authorized") is not True
                or expected.get("authority") != "operator_control_plane"
                or int(expected.get("target_generation") or 0) != 33
                or int(expected.get("target_event_watermark") or 0) != 303
                or int(expected.get("target_plan_revision") or 0) != 28
                or expected.get("target_projection_cid")
                != _M46_TARGET_PROJECTION_CID
                or stopped.get("status") != "stopped"
                or stopped.get("generation") != 32
                or stopped.get("target_generation") != 33
                or artifacts.get("owner_marker_absent") is not True
                or artifacts.get("stop_control_absent") is not True
                or artifacts.get("token_handoff_absent") is not True
                or artifacts.get("control_wal_absent") is not True
                or artifacts.get("coordination_wal_absent") is not True
                or artifacts.get("m46_receipt_absent") is not True
                or artifacts.get("m44_receipt_absent") is not True
                or receipt.get("receipt_cid") != _M46_M45_RECEIPT_CID
                or receipt.get("created_or_rewritten") is not False
                or preserved.get("authority_cid") != _M45_AUTHORITY_CID
                or preserved.get("receipt_cid") != _M46_M45_RECEIPT_CID
                or preserved.get("m44_receipt_absent") is not True
                or repair.get("repair_parent")
                != "5408e29ba1067c1282589818d175d930aeade1de"
                or repair.get("repair_commit")
                != "e666d239ba37738da9830497be50c8b0737271fc"
                or repair.get("repair_tree")
                != "6bc26fda6ab6a0b1b27a17b561e6fdea04dce928"
                or repair.get("status_empty_or_nested_gitlink_only_required")
                is not True
                or repair.get("attestation_metadata_validated_exactly")
                is not True
                or repair.get("task_completion_authority") is not False
                or repair.get("authority_weakened") is not False
                or chain.get("m45_final_control_commit")
                != "5408e29ba1067c1282589818d175d930aeade1de"
                or chain.get("repair_commit")
                != "e666d239ba37738da9830497be50c8b0737271fc"
                or chain.get("final_control_parent")
                != "e666d239ba37738da9830497be50c8b0737271fc"
                or chain.get("repair_commit_count") != 1
                or chain.get("final_control_commit_count") != 1
                or contract.get("prior_generation") != 32
                or contract.get("target_generation") != 33
                or contract.get("prior_event_watermark") != 302
                or contract.get("target_event_watermark") != 303
                or contract.get("m45_receipt_must_be_preserved") is not True
                or contract.get("m44_receipt_must_remain_absent") is not True
                or contract.get("event_302_must_be_preserved") is not True
                or contract.get("event_303_must_be_absent_before_append")
                is not True
                or changes.get("event_suffix_length") != 1
                or changes.get("evidence_node_changes") != 1
                or changes.get("evidence_event_changes") != 1
                or changes.get("store_generation_row_changes") != 1
                or changes.get("state_server_row_changes") != 1
                or changes.get("task_revision_changes") != 0
                or changes.get("task_status_changes") != 0
                or changes.get("accepted_completion_changes") != 0
                or preservation.get("generation_32_preserved_stopped") is not True
                or preservation.get("m45_authority_preserved_exactly") is not True
                or preservation.get("m45_receipt_preserved_exactly") is not True
                or preservation.get("m44_receipt_remains_absent") is not True
                or preservation.get("event_302_preserved_exactly") is not True
                or preservation.get("coordination_store_bytes_preserved_exactly")
                is not True
                or preservation.get("worker_self_approval") is not False
            ):
                config_errors.append(
                    "M46 generation-33 legacy-rescue recovery authority/source "
                    "seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M46 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M46 active runtime paths are not exactly preserved")
    if m45_selected:
        successor = config.get(m45_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m45_failed_pre_authoritative_m44_validation_successor_authority()
            )
            contract = materializer._validated_m45_live_preflight_contract(
                expected
            )
            reference = dict(materializer._m45_authority_reference())
            configured_cid = str(materializer._M45_AUTHORITY_CID)
            preserved = expected.get("preserved_m44_authority", {})
            failed = expected.get("failed_m44_prepublication_validation", {})
            correction = expected.get("accepted_test_expectation_correction", {})
            delegated = expected.get("delegated_m44_materialization", {})
            changes = expected.get("exact_changes", {})
            preservation = expected.get("preservation", {})
            m45_errors = (
                module
                ._m45_failed_pre_authoritative_m44_validation_successor_errors(
                    config, seal, migration, root=root
                )
            )
            if (
                successor != reference
                or migration.get(m45_key) != reference
                or seal.get(f"{m45_key}_cid") != configured_cid
                or configured_cid != _M45_AUTHORITY_CID
                or configured_cid == _M45_UNSEALED_AUTHORITY_CID
                or materializer._identity(expected) != configured_cid
                or len(materializer._canonical(expected)) != _M45_AUTHORITY_SIZE
                or m45_errors
                or expected.get("schema")
                != (
                    "sawm/failed-pre-authoritative-m44-validation-successor-"
                    "authorization@1"
                )
                or expected.get("migration_revision") != "SAWM-R2-M45"
                or expected.get("migration_kind") != m45_key
                or expected.get("authorized") is not True
                or expected.get("authority") != "operator_control_plane"
                or int(expected.get("target_generation") or 0) != 32
                or int(expected.get("target_event_watermark") or 0) != 302
                or int(expected.get("target_plan_revision") or 0) != 28
                or preserved.get("authority_cid") != _M44_AUTHORITY_CID
                or preserved.get("canonical_body_size") != 20_272
                or preserved.get("authority_amended_or_rewritten") is not False
                or preserved.get("receipt_created") is not False
                or failed.get("failure_kind")
                != "obsolete_historical_operator_error_expectation"
                or failed.get("runtime_started") is not False
                or failed.get("m44_receipt_created") is not False
                or correction.get("exact_replacement_count") != 1
                or correction.get("operator_behavior_changed") is not False
                or correction.get("validation_weakened") is not False
                or delegated.get("authority_cid") != _M44_AUTHORITY_CID
                or delegated.get("m44_transition_semantics_preserved_exactly")
                is not True
                or delegated.get("m44_receipt_created") is not False
                or delegated.get("m45_receipt_published_last") is not True
                or contract.get("migration_revision") != "SAWM-R2-M45"
                or contract.get("prior_generation") != 31
                or contract.get("target_generation") != 32
                or contract.get("prior_event_watermark") != 301
                or contract.get("target_event_watermark") != 302
                or contract.get("m44_authority_must_be_preserved") is not True
                or contract.get("m44_receipt_must_be_absent_before_publication")
                is not True
                or changes.get("test_expectation_changes") != 1
                or changes.get("event_suffix_length") != 1
                or changes.get("task_revision_changes") != 0
                or changes.get("accepted_completion_changes") != 0
                or preservation.get("m44_authority_preserved_exactly") is not True
                or preservation.get("m44_receipt_absent_before_publication")
                is not True
                or preservation.get("coordination_store_bytes_preserved_exactly")
                is not True
                or preservation.get("worker_self_approval") is not False
            ):
                config_errors.append(
                    "M45 validation-successor authority/source seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M45 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M45 active runtime paths are not exactly preserved")
    if m44_selected:
        successor = config.get(m44_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m44_post_m43_hardened_procfs_user_manager_restart_successor_authority()
            )
            reference = dict(materializer._m44_authority_reference())
            configured_cid = str(materializer._M44_AUTHORITY_CID)
            prior = expected.get("prior_authority", {})
            stopped = expected.get("stopped_owner", {})
            receipt = expected.get("preserved_m43_receipt", {})
            preserved = expected.get("preserved_m43_materialization", {})
            suffix = expected.get("post_m43_operational_suffix", {})
            repair = expected.get("accepted_user_manager_procfs_repair", {})
            chain = expected.get("source_chain", {})
            changes = expected.get("exact_changes", {})
            preservation = expected.get("preservation", {})
            contract = expected.get("live_preflight_contract", {})
            m44_errors = (
                module
                ._m44_post_m43_hardened_procfs_user_manager_restart_successor_errors(
                    config, seal, migration, root=root
                )
            )
            if (
                successor != reference
                or migration.get(m44_key) != reference
                or seal.get(f"{m44_key}_cid") != configured_cid
                or configured_cid != _M44_AUTHORITY_CID
                or configured_cid == _M44_UNSEALED_AUTHORITY_CID
                or materializer._identity(expected) != configured_cid
                or m44_errors
                or expected.get("schema")
                != (
                    "sawm/post-m43-hardened-procfs-user-manager-restart-"
                    "authorization@1"
                )
                or expected.get("migration_revision") != "SAWM-R2-M44"
                or expected.get("migration_kind") != m44_key
                or expected.get("authorized") is not True
                or expected.get("authority") != "operator_control_plane"
                or int(expected.get("target_generation") or 0) != 32
                or int(expected.get("target_event_watermark") or 0) != 302
                or int(expected.get("target_plan_revision") or 0) != 28
                or expected.get("target_projection_cid")
                != "baguqeerageftwrpvliedl3nkrbqlchwo2tzogehnjeyu5iqn6p7jfes4tmea"
                or int(prior.get("event_watermark") or 0) != 301
                or prior.get("event_prefix_sha256")
                != "b521ef1ea54dd2a90e108e23ac63347e83a10e9d99298b143a49eb5119ab4c0e"
                or prior.get("projection_cid")
                != "baguqeeraelrmqrsph27tld2bk6ydvq336uff5hihbl42oelhkhuwsg3sqrpa"
                or prior.get("semantic_authority_digest")
                != "sha256:b0f775db0bab9821418d4d4054033480c638a7c0ae86160bd21cdd816456fd59"
                or stopped.get("status") != "stopped"
                or stopped.get("generation") != 31
                or stopped.get("target_generation") != 32
                or receipt.get("receipt_cid")
                != "sha256:eea44e0f2aae970c1580c59e7b3310904fad480f249b3b1f5df5cd887fc1f050"
                or receipt.get("created_or_rewritten") is not False
                or preserved.get("event_watermark") != 297
                or preserved.get("event_id")
                != "baguqeerazsdqgqrzx5xyny5mh4dwpu5olor6rf4oj2p5c65onpykcy35hrvq"
                or preserved.get("evidence_id")
                != "baguqeera2britcaqnxj7ulksuxeg3v5zvdf6uh6xmq2oklgmho3tepaw6ljq"
                or suffix.get("first_event_watermark") != 298
                or suffix.get("last_event_watermark") != 301
                or suffix.get("event_count") != 4
                or repair.get("repair_parent")
                != "de2864eac1f7e49ea44b1dd5f8ad689167374c18"
                or repair.get("repair_commit")
                != "b2136e3eb88600df829afa563b74ab0957ed4061"
                or repair.get("repair_tree")
                != "99acd200a8c7cd65087fe6df316c0428b731d492"
                or repair.get("binary_diff_sha256")
                != "66ce916093e2fc6b091ca1963c8eef45c7d4419b1b480c870241117dd92f73bb"
                or repair.get("validation_weakened") is not False
                or repair.get("authority_weakened") is not False
                or chain.get("m43_final_control_commit")
                != "de2864eac1f7e49ea44b1dd5f8ad689167374c18"
                or chain.get("repair_commit")
                != "b2136e3eb88600df829afa563b74ab0957ed4061"
                or chain.get("final_control_parent")
                != "b2136e3eb88600df829afa563b74ab0957ed4061"
                or chain.get("repair_commit_count") != 1
                or chain.get("final_control_commit_count") != 1
                or contract.get("prior_generation") != 31
                or contract.get("target_generation") != 32
                or contract.get("prior_event_watermark") != 301
                or contract.get("target_event_watermark") != 302
                or contract.get("events_297_through_301_must_be_preserved")
                is not True
                or contract.get("event_302_must_be_absent_before_append")
                is not True
                or changes.get("event_suffix_length") != 1
                or changes.get("evidence_node_changes") != 1
                or changes.get("evidence_event_changes") != 1
                or changes.get("task_revision_changes") != 0
                or changes.get("task_status_changes") != 0
                or changes.get("accepted_completion_changes") != 0
                or changes.get("worker_self_approval") is not False
                or preservation.get("m43_receipt_preserved_exactly") is not True
                or preservation.get("m43_event_297_preserved_exactly") is not True
                or preservation.get(
                    "post_m43_operational_suffix_preserved_exactly"
                )
                is not True
                or preservation.get("coordination_store_bytes_preserved_exactly")
                is not True
                or preservation.get("worker_self_approval") is not False
            ):
                config_errors.append(
                    "M44 generation-32 procfs restart authority/source seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M44 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M44 active runtime paths are not exactly preserved")
    if m43_selected:
        successor = config.get(m43_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m43_dead_attempt_lifecycle_recovery_restart_authority()
            )
            reference = dict(materializer._m43_authority_reference())
            configured_cid = materializer._M43_AUTHORITY_CID
            prior = expected.get("prior_authority", {})
            stopped = expected.get("stopped_owner", {})
            artifacts = expected.get("stopped_prestart_artifacts", {})
            receipt = expected.get("preserved_m42_receipt", {})
            repair = expected.get("accepted_lifecycle_repair", {})
            failed_validation = expected.get(
                "failed_pre_authoritative_control_validation", {}
            )
            fixture_repair = expected.get(
                "accepted_historical_fixture_repair", {}
            )
            failed_reseal = expected.get(
                "failed_pre_authoritative_reseal_validation", {}
            )
            verifier_repair = expected.get(
                "accepted_bounded_materializer_verifier_repair", {}
            )
            final_hardening = expected.get("final_control_hardening", {})
            prior_control = expected.get("prior_control_authorization", {})
            prior_revision3 = expected.get("prior_revision3_authority", {})
            revision4_validation = expected.get("revision4_validation_attempts", {})
            revision4_attempts = revision4_validation.get("attempts", ())
            revision3_suite = (
                revision4_attempts[0]
                if isinstance(revision4_attempts, Sequence)
                and len(revision4_attempts) == 2
                else {}
            )
            disposable_rehearsal = (
                revision4_attempts[1]
                if isinstance(revision4_attempts, Sequence)
                and len(revision4_attempts) == 2
                else {}
            )
            authoritative_anchors = revision4_validation.get(
                "authoritative_anchors_after_both_attempts", {}
            )
            successor_verifier_repair = expected.get(
                "accepted_successor_evidence_verifier_repair", {}
            )
            source_chain = expected.get("source_chain", {})
            changes = expected.get("exact_changes", {})
            contract = expected.get("live_preflight_contract", {})
            m42_expected = (
                materializer
                ._expected_m42_failed_pre_authoritative_m41_evidence_projection_successor_authority()
            )
            m42_reference = dict(materializer._m42_authority_reference())
            m43_errors = module._m43_dead_attempt_lifecycle_recovery_restart_successor_errors(
                config, seal, migration, root=root
            )
            if (
                successor != reference
                or migration.get(m43_key) != reference
                or seal.get(f"{m43_key}_cid") != configured_cid
                or configured_cid != _M43_AUTHORITY_CID
                or (
                    configured_cid == _M43_UNSEALED_AUTHORITY_CID
                    and not any("not resealed" in error for error in m43_errors)
                )
                or (
                    configured_cid != _M43_UNSEALED_AUTHORITY_CID
                    and (
                        materializer._identity(expected) != configured_cid
                        or m43_errors
                    )
                )
                or expected.get("schema")
                != "sawm/dead-attempt-lifecycle-recovery-restart-authorization@4"
                or expected.get("authorization_revision") != 4
                or expected.get("authorization_amended_at")
                != materializer._M43_AUTHORIZATION_AMENDED_AT
                or expected.get("prior_authorization_cid")
                != materializer._M43_REVISION3_AUTHORITY_CID
                or expected.get("control_recorded_at") != "2026-09-01T11:00:00Z"
                or expected.get("authorization_amendment_paths")
                != sorted(materializer._M43_OPERATOR_CONTROL_PATHS)
                or prior_control.get("authorization_cid")
                != materializer._M43_REVISION3_AUTHORITY_CID
                or prior_control.get("authorization_revision") != 3
                or prior_control.get("prior_authorization_cid")
                != materializer._M43_PRIOR_FINAL_AUTHORITY_CID
                or prior_control.get("control_recorded_at")
                != "2026-09-01T10:00:00Z"
                or prior_control.get("control_commit")
                != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
                or prior_control.get("control_tree")
                != materializer._M43_REVISION3_FINAL_CONTROL_TREE
                or prior_control.get(
                    "superseded_before_authoritative_event_297"
                )
                is not True
                or prior_control.get("event_297_appended") is not False
                or prior_control.get("receipt_published") is not False
                or prior_revision3
                != materializer._expected_m43_revision3_authority()
                or materializer._identity(prior_revision3)
                != materializer._M43_REVISION3_AUTHORITY_CID
                or prior_revision3.get("schema")
                != "sawm/dead-attempt-lifecycle-recovery-restart-authorization@3"
                or prior_revision3.get("authorization_revision") != 3
                or prior_revision3.get("control_recorded_at")
                != "2026-09-01T10:00:00Z"
                or prior_revision3.get("prior_authorization_cid")
                != materializer._M43_PRIOR_FINAL_AUTHORITY_CID
                or expected.get("migration_revision") != "SAWM-R2-M43"
                or expected.get("migration_kind") != m43_key
                or expected.get("target_generation") != 31
                or expected.get("target_event_watermark") != 297
                or expected.get("target_plan_revision") != 28
                or expected.get("target_projection_cid")
                != "baguqeerazspjonqzwhd5e2jmnpl4lacaasfmtkziur4awfrihibnh6mxkpoa"
                or expected.get("prior_m42_authority") != m42_expected
                or expected.get("prior_m42_reference") != m42_reference
                or materializer._identity(m42_expected) != _M42_AUTHORITY_CID
                or prior.get("event_watermark") != 296
                or prior.get("event_prefix_sha256")
                != "89c64a4f018c2a3cfdce675eb8fb27913674e76995d64d89cabec42dd2967b70"
                or prior.get("projection_cid")
                != "baguqeerasguaepwupk3d5vme3cqsbicujihwvxnemnoenvdtnu3uwrscbt6q"
                or prior.get("semantic_authority_digest")
                != "sha256:a9f7e45d543cd983b36d475f3145344c956c547524bf33adfdc630de2bda7ae0"
                or stopped.get("generation") != 30
                or stopped.get("target_generation") != 31
                or stopped.get("status") != "stopped"
                or stopped.get("revision") != 2
                or stopped.get("database_uuid")
                != "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4"
                or artifacts.get("control_sha256")
                != "6798563648545b3fc05f1b7638ad2d0448c743d3a788bf78208a5d28a76a95f7"
                or artifacts.get("control_size") != 43_528_192
                or artifacts.get("coordination_sha256")
                != "ddbdf352e6a41452c6584cfa06fc760b90a94f1ff6473ff2c5eeb93de7551785"
                or artifacts.get("coordination_size") != 16_789_504
                or artifacts.get("status_sha256")
                != "3f8c1227e7bc29c3057d238e550880cdfb6144a3687a73dea12e3ca063148a4c"
                or artifacts.get("status_size") != 2_408
                or receipt.get("sha256")
                != "ff9a24d339cf06eacb3573cd2825e0648a558efe5ec9539c0c4f489002ca609d"
                or receipt.get("size") != 14_112
                or receipt.get("receipt_cid")
                != "sha256:31565b6bfc8e071f4278acc88fd3500ca5c4d25eee63d0131b16ceca3e7a9169"
                or repair.get("changed_paths")
                != sorted(materializer._M43_LIFECYCLE_REPAIR_BLOBS)
                or repair.get("repair_parent")
                != "a8bce148b793dcd15ac742df3a29e5773a178f28"
                or repair.get("repair_commit")
                != "7e3fa1170edac23149e0d1f38f5ff6b5f5ddb571"
                or repair.get("repair_tree")
                != "df40d6f879753c8c2ca00f35fd28054a29fd600a"
                or repair.get("blob_oids")
                != {
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
                    "database_portal_bridge.py": (
                        "97f44d032063a8a98cfca277dd123c998244f076"
                    ),
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
                    "implementation_daemon_runner.py": (
                        "51e577170d975f056e3f97a9ded55d210bee8792"
                    ),
                    "test/api/test_agent_supervisor_database_portal_bridge.py": (
                        "07fb7dd738256126c96e1370dd5a60501a13647b"
                    ),
                    "test/api/test_agent_supervisor_configured_board_live_capsule.py": (
                        "4638112c6c2da24cb5912914192f332f2903bd6b"
                    ),
                }
                or repair.get("path_modes")
                != {path: "100644" for path in repair.get("changed_paths", ())}
                or repair.get("repair_path_set_finalized")
                is not materializer._M43_LIFECYCLE_REPAIR_PATHS_FINALIZED
                or repair.get("newest_first_attempt_selection") is not True
                or repair.get("prepared_receipt_precedes_lifecycle_cas") is not True
                or repair.get("marker_retirement_is_no_replace") is not True
                or failed_validation.get("authority_cid")
                != materializer._M43_PRE_AUTHORITATIVE_AUTHORITY_CID
                or failed_validation.get("control_commit")
                != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_COMMIT
                or failed_validation.get("control_tree")
                != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_TREE
                or failed_validation.get("control_blobs")
                != dict(materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_BLOBS)
                or failed_validation.get("control_modes")
                != dict(materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_MODES)
                or failed_validation.get("collected_tests") != 280
                or failed_validation.get("passed_tests") != 271
                or failed_validation.get("failed_tests") != 9
                or failed_validation.get("failed_test_ids")
                != list(materializer._M43_FAILED_TEST_IDS)
                or failed_validation.get("materializer_invoked") is not False
                or failed_validation.get("quack_started") is not False
                or failed_validation.get(
                    "authenticated_mutation_request_created"
                )
                is not False
                or failed_validation.get("event_297_rows_created") != 0
                or failed_validation.get("m43_receipt_created") is not False
                or any(
                    failed_validation.get(name) != 0
                    for name in (
                        "evidence_node_changes",
                        "evidence_event_changes",
                        "validation_event_changes",
                        "store_generation_row_changes",
                        "state_server_row_changes",
                        "credential_row_changes",
                        "server_epoch_row_changes",
                        "capability_snapshot_row_changes",
                        "task_revision_changes",
                        "task_status_changes",
                        "goal_revision_changes",
                        "plan_revision_changes",
                        "effect_claim_changes",
                        "merge_attempt_changes",
                        "implementation_provider_invocations",
                        "coordination_semantic_changes",
                        "accepted_completion_changes",
                    )
                )
                or fixture_repair.get("repair_parent")
                != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_COMMIT
                or fixture_repair.get("repair_commit")
                != materializer._M43_HISTORICAL_FIXTURE_REPAIR_COMMIT
                or fixture_repair.get("repair_tree")
                != materializer._M43_HISTORICAL_FIXTURE_REPAIR_TREE
                or fixture_repair.get("blob_oids")
                != dict(materializer._M43_HISTORICAL_FIXTURE_REPAIR_BLOBS)
                or fixture_repair.get("path_modes")
                != dict(materializer._M43_HISTORICAL_FIXTURE_REPAIR_MODES)
                or fixture_repair.get("focused_former_failures_replayed") != 9
                or fixture_repair.get("focused_former_failures_passed") != 9
                or fixture_repair.get("focused_replay_is_full_suite") is not False
                or fixture_repair.get("production_code_changes") != 0
                or fixture_repair.get(
                    "fixture_collection_or_selection_logic_changed"
                )
                is not False
                or fixture_repair.get("validation_weakened") is not False
                or fixture_repair.get("authority_weakened") is not False
                or failed_reseal.get("authority_cid")
                != materializer._M43_PRIOR_FINAL_AUTHORITY_CID
                or failed_reseal.get("control_commit")
                != materializer._M43_PRIOR_FINAL_CONTROL_COMMIT
                or failed_reseal.get("control_tree")
                != materializer._M43_PRIOR_FINAL_CONTROL_TREE
                or failed_reseal.get("observed_failure_count") != 2
                or failed_reseal.get("isolated_live_rehearsal", {}).get(
                    "disposable_generation_31_ready"
                )
                is not True
                or failed_reseal.get("full_suite_validation", {}).get("error")
                != "MigrationRequired: operator task-recovery event differs"
                or failed_reseal.get("authenticated_mutation_request_created")
                is not False
                or failed_reseal.get("event_297_rows_created") != 0
                or failed_reseal.get("accepted_completion_changes") != 0
                or verifier_repair.get("repair_parent")
                != materializer._M43_PRIOR_FINAL_CONTROL_COMMIT
                or verifier_repair.get("repair_commit")
                != materializer._M43_VERIFIER_REPAIR_COMMIT
                or verifier_repair.get("repair_tree")
                != materializer._M43_VERIFIER_REPAIR_TREE
                or verifier_repair.get("blob_oids")
                != dict(materializer._M43_VERIFIER_REPAIR_BLOBS)
                or verifier_repair.get("path_modes")
                != dict(materializer._M43_VERIFIER_REPAIR_MODES)
                or verifier_repair.get("repair_class_count") != 2
                or verifier_repair.get("repair_classes")
                != [
                    "duckdb_positional_row_normalization",
                    "task_revision_timestamp_source_binding",
                ]
                or verifier_repair.get("repair_site_count") != 5
                or verifier_repair.get("repair_sites")
                != [
                    "m43_operational_suffix_rows",
                    "m43_prestart_event_type_counts",
                    "m43_live_preappend_event_type_counts",
                    "m43_live_postappend_event_type_counts",
                    "m6_recovery_revision_recorded_at",
                ]
                or verifier_repair.get("validation_weakened") is not False
                or verifier_repair.get("authority_weakened") is not False
                or revision4_validation
                != materializer._m43_revision4_validation_attempts()
                or revision4_validation.get("schema")
                != "sawm/m43-revision-4-validation-attempts@1"
                or not isinstance(revision4_attempts, Sequence)
                or len(revision4_attempts) != 2
                or revision3_suite.get("kind")
                != "authoritative_current_tree_full_suite"
                or revision3_suite.get("source_commit")
                != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
                or revision3_suite.get("source_tree")
                != materializer._M43_REVISION3_FINAL_CONTROL_TREE
                or revision3_suite.get("result") != "passed"
                or revision3_suite.get("collected") != 280
                or revision3_suite.get("passed") != 280
                or revision3_suite.get("failed") != 0
                or revision3_suite.get("retained_log", {}).get("sha256")
                != materializer._M43_REVISION3_FULL_SUITE_LOG_SHA256
                or revision3_suite.get("retained_log", {}).get("size_bytes")
                != materializer._M43_REVISION3_FULL_SUITE_LOG_SIZE
                or disposable_rehearsal.get("kind")
                != "fresh_disposable_clone_partial_append_rehearsal"
                or disposable_rehearsal.get("disposable") is not True
                or disposable_rehearsal.get("authoritative") is not False
                or disposable_rehearsal.get("source_commit")
                != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
                or disposable_rehearsal.get("source_tree")
                != materializer._M43_REVISION3_FINAL_CONTROL_TREE
                or disposable_rehearsal.get("authorization_cid")
                != materializer._M43_REVISION3_AUTHORITY_CID
                or disposable_rehearsal.get("result")
                != "failed_after_authenticated_append_before_receipt"
                or disposable_rehearsal.get("typed_error")
                != (
                    "MigrationRequired: M42 exact evidence projection "
                    "membership differs"
                )
                or disposable_rehearsal.get("post_stop_generation") != 31
                or disposable_rehearsal.get("partial_append", {}).get(
                    "event_watermark_after"
                )
                != 297
                or disposable_rehearsal.get("partial_append", {}).get(
                    "evidence_count_after"
                )
                != 50
                or disposable_rehearsal.get("partial_append", {}).get(
                    "m43_receipt_present"
                )
                is not False
                or disposable_rehearsal.get("task_status_changes") != 0
                or disposable_rehearsal.get("accepted_completion_changes") != 0
                or authoritative_anchors.get("unchanged") is not True
                or authoritative_anchors.get("authoritative_event_watermark")
                != 296
                or authoritative_anchors.get("authoritative_evidence_count") != 49
                or authoritative_anchors.get("authoritative_generation") != 30
                or revision4_validation.get("worker_self_approval") is not False
                or successor_verifier_repair
                != materializer._m43_successor_evidence_verifier_repair()
                or successor_verifier_repair.get("schema")
                != "sawm/bounded-successor-evidence-verifier-repair@1"
                or successor_verifier_repair.get("repair_parent")
                != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
                or successor_verifier_repair.get("repair_commit")
                != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_COMMIT
                or successor_verifier_repair.get("repair_tree")
                != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_TREE
                or successor_verifier_repair.get("blob_oids")
                != dict(
                    materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_BLOBS
                )
                or successor_verifier_repair.get("path_modes")
                != dict(
                    materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_MODES
                )
                or successor_verifier_repair.get(
                    "permitted_successor_evidence_row_count"
                )
                != 1
                or successor_verifier_repair.get("successor_row_is_caller_bound")
                is not True
                or successor_verifier_repair.get(
                    "successor_row_content_identity_rehashed"
                )
                is not True
                or successor_verifier_repair.get(
                    "unlisted_extra_evidence_rejected"
                )
                is not True
                or successor_verifier_repair.get(
                    "wrong_successor_evidence_rejected"
                )
                is not True
                or successor_verifier_repair.get(
                    "preappend_default_behavior_changed"
                )
                is not False
                or successor_verifier_repair.get("m42_return_fields_changed")
                is not False
                or successor_verifier_repair.get("m42_return_counts_changed")
                is not False
                or successor_verifier_repair.get("authority_weakened") is not False
                or successor_verifier_repair.get("worker_self_approval") is not False
                or final_hardening.get(
                    "operational_suffix_malformed_rows_are_typed_conflicts"
                )
                is not True
                or final_hardening.get(
                    "pending_authority_rejected_by_operator_facade"
                )
                is not True
                or final_hardening.get(
                    "exact_successor_evidence_is_the_only_postappend_allowance"
                )
                is not True
                or final_hardening.get("m42_return_contract_preserved") is not True
                or final_hardening.get(
                    "unlisted_physical_evidence_still_fails_closed"
                )
                is not True
                or final_hardening.get("authority_weakened") is not False
                or source_chain.get("m42_final_control_commit")
                != materializer._M43_BASE_CONTROL_COMMIT
                or source_chain.get("lifecycle_repair_commit")
                != materializer._M43_LIFECYCLE_REPAIR_COMMIT
                or source_chain.get("lifecycle_repair_blobs")
                != dict(materializer._M43_LIFECYCLE_REPAIR_BLOBS)
                or source_chain.get("lifecycle_repair_modes")
                != dict(materializer._M43_LIFECYCLE_REPAIR_MODES)
                or source_chain.get("initial_control_commit")
                != materializer._M43_INITIAL_CONTROL_COMMIT
                or source_chain.get("initial_control_tree")
                != materializer._M43_INITIAL_CONTROL_TREE
                or source_chain.get("initial_control_blobs")
                != dict(materializer._M43_INITIAL_CONTROL_BLOBS)
                or source_chain.get("initial_control_modes")
                != dict(materializer._M43_INITIAL_CONTROL_MODES)
                or source_chain.get("failed_pre_authoritative_control_parent")
                != materializer._M43_INITIAL_CONTROL_COMMIT
                or source_chain.get("failed_pre_authoritative_control_commit")
                != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_COMMIT
                or source_chain.get("failed_pre_authoritative_control_tree")
                != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_TREE
                or source_chain.get("failed_pre_authoritative_control_blobs")
                != dict(materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_BLOBS)
                or source_chain.get("failed_pre_authoritative_control_modes")
                != dict(materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_MODES)
                or source_chain.get("historical_fixture_repair_parent")
                != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_COMMIT
                or source_chain.get("historical_fixture_repair_commit")
                != materializer._M43_HISTORICAL_FIXTURE_REPAIR_COMMIT
                or source_chain.get("historical_fixture_repair_tree")
                != materializer._M43_HISTORICAL_FIXTURE_REPAIR_TREE
                or source_chain.get("historical_fixture_repair_blobs")
                != dict(materializer._M43_HISTORICAL_FIXTURE_REPAIR_BLOBS)
                or source_chain.get("historical_fixture_repair_modes")
                != dict(materializer._M43_HISTORICAL_FIXTURE_REPAIR_MODES)
                or source_chain.get("prior_final_control_parent")
                != materializer._M43_HISTORICAL_FIXTURE_REPAIR_COMMIT
                or source_chain.get("prior_final_control_commit")
                != materializer._M43_PRIOR_FINAL_CONTROL_COMMIT
                or source_chain.get("prior_final_control_tree")
                != materializer._M43_PRIOR_FINAL_CONTROL_TREE
                or source_chain.get("prior_final_control_blobs")
                != dict(materializer._M43_PRIOR_FINAL_CONTROL_BLOBS)
                or source_chain.get("prior_final_control_modes")
                != dict(materializer._M43_PRIOR_FINAL_CONTROL_MODES)
                or source_chain.get(
                    "bounded_materializer_verifier_repair_parent"
                )
                != materializer._M43_PRIOR_FINAL_CONTROL_COMMIT
                or source_chain.get(
                    "bounded_materializer_verifier_repair_commit"
                )
                != materializer._M43_VERIFIER_REPAIR_COMMIT
                or source_chain.get("bounded_materializer_verifier_repair_tree")
                != materializer._M43_VERIFIER_REPAIR_TREE
                or source_chain.get("bounded_materializer_verifier_repair_blobs")
                != dict(materializer._M43_VERIFIER_REPAIR_BLOBS)
                or source_chain.get("bounded_materializer_verifier_repair_modes")
                != dict(materializer._M43_VERIFIER_REPAIR_MODES)
                or source_chain.get("revision3_final_control_parent")
                != materializer._M43_VERIFIER_REPAIR_COMMIT
                or source_chain.get("revision3_final_control_commit")
                != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
                or source_chain.get("revision3_final_control_tree")
                != materializer._M43_REVISION3_FINAL_CONTROL_TREE
                or source_chain.get("revision3_final_control_blobs")
                != dict(materializer._M43_REVISION3_FINAL_CONTROL_BLOBS)
                or source_chain.get("revision3_final_control_modes")
                != dict(materializer._M43_REVISION3_FINAL_CONTROL_MODES)
                or source_chain.get("successor_evidence_verifier_repair_parent")
                != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
                or source_chain.get("successor_evidence_verifier_repair_commit")
                != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_COMMIT
                or source_chain.get("successor_evidence_verifier_repair_tree")
                != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_TREE
                or source_chain.get("successor_evidence_verifier_repair_blobs")
                != dict(
                    materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_BLOBS
                )
                or source_chain.get("successor_evidence_verifier_repair_modes")
                != dict(
                    materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_MODES
                )
                or source_chain.get("final_reseal_parent")
                != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_COMMIT
                or source_chain.get(
                    "failed_pre_authoritative_control_commit_count"
                )
                != 2
                or source_chain.get("historical_fixture_repair_commit_count")
                != 1
                or source_chain.get(
                    "bounded_materializer_verifier_repair_commit_count"
                )
                != 1
                or source_chain.get("revision3_final_control_commit_count") != 1
                or source_chain.get(
                    "successor_evidence_verifier_repair_commit_count"
                )
                != 1
                or source_chain.get("failed_disposable_partial_append_count") != 1
                or source_chain.get("final_reseal_commit_count") != 3
                or source_chain.get("final_control_commit_count") != 4
                or expected.get("ordinary_source_changes")
                != len(materializer._M43_LIFECYCLE_REPAIR_BLOBS)
                or set(expected.get("operator_control_paths", ()))
                != set(materializer._M43_OPERATOR_CONTROL_PATHS)
                or contract.get("generation_restart_authorized") is not True
                or contract.get("m42_receipt_must_be_preserved") is not True
                or changes.get("event_suffix_length") != 1
                or changes.get("task_status_changes") != 0
                or changes.get("accepted_completion_changes") != 0
            ):
                config_errors.append(
                    "M43 stopped-generation recovery authority/source seal differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M43 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M43 active runtime paths are not exactly preserved")
    if m42_selected:
        successor = config.get(m42_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer
                ._expected_m42_failed_pre_authoritative_m41_evidence_projection_successor_authority()
            )
            reference = dict(materializer._m42_authority_reference())
            expected_seal_cid = materializer._identity(expected)
            source_chain = expected.get("source_chain", {})
            failed = expected.get(
                "failed_m41_pre_authoritative_materialization", {}
            )
            legacy = expected.get("exact_legacy_projection_authority", {})
            repair = expected.get("accepted_projection_repair", {})
            contract = expected.get("live_preflight_contract", {})
            prior = expected.get("prior_m41_authority", {})
            prior_reference = expected.get("prior_m41_reference", {})
            m41_expected = (
                materializer
                ._expected_m41_failed_pre_authoritative_m40_validation_successor_authority()
            )
            m41_reference = dict(materializer._m41_authority_reference())
            if (
                successor != reference
                or migration.get(m42_key) != reference
                or seal.get(f"{m42_key}_cid") != expected_seal_cid
                or expected_seal_cid != _M42_AUTHORITY_CID
                or expected_seal_cid != materializer._M42_AUTHORITY_CID
                or expected.get("schema")
                != (
                    "sawm/failed-pre-authoritative-m41-evidence-projection-"
                    "successor-authorization@1"
                )
                or expected.get("migration_revision") != "SAWM-R2-M42"
                or expected.get("migration_kind") != m42_key
                or expected.get("target_generation") != 30
                or expected.get("target_event_watermark") != 292
                or expected.get("target_projection_cid")
                != "baguqeera6t2s6prg5atpg4gkgrlmqclu34firbn4z2o3wsgv6btp7p6q66tq"
                or prior != m41_expected
                or prior_reference != m41_reference
                or source_chain.get("m41_final_control_commit")
                != materializer._M42_M41_FINAL_CONTROL_COMMIT
                or source_chain.get("m41_final_control_tree")
                != materializer._M42_M41_FINAL_CONTROL_TREE
                or source_chain.get("projection_repair_parent")
                != materializer._M42_M41_FINAL_CONTROL_COMMIT
                or source_chain.get("projection_repair_commit")
                != materializer._M42_PROJECTION_REPAIR_COMMIT
                or source_chain.get("projection_repair_tree")
                != materializer._M42_PROJECTION_REPAIR_TREE
                or source_chain.get("projection_repair_blobs")
                != dict(materializer._M42_PROJECTION_REPAIR_BLOBS)
                or source_chain.get("final_control_parent")
                != materializer._M42_PROJECTION_REPAIR_COMMIT
                or int(source_chain.get("bounded_repair_commit_count") or 0) != 1
                or source_chain.get("final_control_commit_is_current_head")
                is not True
                or failed.get("schema")
                != "sawm/pre-authoritative-m41-materialization-failure@1"
                or failed.get("attempt") != "SAWM-R2-M41-MATERIALIZE-A1"
                or failed.get("phase")
                != "exact_historical_evidence_projection_verification"
                or failed.get("failure_kind")
                != "landed_projection_differs_from_event_only_replay"
                or failed.get("materializer_invoked") is not True
                or failed.get("authenticated_live_quack_read_opened") is not True
                or failed.get("quack_mutation_request_created") is not False
                or failed.get("record_evidence_reached") is not False
                or failed.get("event_292_rows_created") != 0
                or failed.get("m41_receipt_created") is not False
                or legacy.get("manifest_cid")
                != materializer._M42_LEGACY_PROJECTION_MANIFEST_CID
                or legacy.get("evidence_projection_digest")
                != materializer._M42_LEGACY_EVIDENCE_PROJECTION_DIGEST
                or legacy.get("validation_runs_digest")
                != materializer._M42_LEGACY_VALIDATION_RUNS_DIGEST
                or legacy.get("validation_results_digest")
                != materializer._M42_LEGACY_VALIDATION_RESULTS_DIGEST
                or legacy.get("evidence_refresh_overlay_count") != 9
                or legacy.get("compact_validation_evidence_overlay_count") != 1
                or legacy.get("validation_attempt_overlay_count") != 1
                or legacy.get("overlay_is_closed") is not True
                or repair.get("repair_commit")
                != materializer._M42_PROJECTION_REPAIR_COMMIT
                or repair.get("m42_exact_overlay_verifier_added") is not True
                or repair.get("strict_default_m38_verifier_preserved") is not True
                or repair.get("validation_weakened") is not False
                or contract.get("generation_restart_authorized") is not False
                or contract.get("prior_event_watermark") != 291
                or contract.get("target_event_watermark") != 292
                or contract.get("m37_receipt_must_be_absent") is not True
                or contract.get("m38_receipt_must_be_absent") is not True
                or contract.get("m39_receipt_must_be_absent") is not True
                or contract.get("m40_receipt_must_be_absent") is not True
                or contract.get("m41_receipt_must_be_absent") is not True
                or (
                    not m43_declared
                    and module
                    ._m42_failed_pre_authoritative_m41_evidence_projection_successor_errors(
                        config,
                        seal,
                        migration,
                        root=root,
                    )
                )
            ):
                config_errors.append(
                    "M42 failed-pre-authoritative-M41-projection authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M42 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M42 active runtime paths are not exactly preserved")
    elif m41_selected:
        successor = config.get(m41_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m41_failed_pre_authoritative_m40_validation_successor_authority()
            )
            reference = dict(materializer._m41_authority_reference())
            expected_seal_cid = materializer._identity(expected)
            source_chain = expected.get("source_chain", {})
            failed = expected.get(
                "failed_m40_pre_authoritative_validation", {}
            )
            repair = expected.get(
                "accepted_historical_test_helper_repair", {}
            )
            contract = expected.get("live_preflight_contract", {})
            prior = expected.get("prior_m40_authority", {})
            if (
                successor != reference
                or migration.get(m41_key) != reference
                or seal.get(f"{m41_key}_cid") != expected_seal_cid
                or expected_seal_cid
                != "sha256:25ad5550b59024c8da9b4821fba2d7b1b49d2a17a781e5d4bd2cbe58cb7b0233"
                or expected.get("schema")
                != (
                    "sawm/failed-pre-authoritative-m40-validation-successor-"
                    "authorization@1"
                )
                or expected.get("migration_revision") != "SAWM-R2-M41"
                or expected.get("migration_kind") != m41_key
                or expected.get("target_generation") != 30
                or expected.get("target_event_watermark") != 292
                or expected.get("target_projection_cid")
                != "baguqeera6t2s6prg5atpg4gkgrlmqclu34firbn4z2o3wsgv6btp7p6q66tq"
                or source_chain.get("m40_final_control_commit")
                != "7564064e191c88679f3a8b540fd5db847b0b492c"
                or source_chain.get("m40_final_control_tree")
                != "aa7035cc4a3bd0f0c1073b2081bcbf9928db4e4e"
                or source_chain.get("historical_test_helper_repair_parent")
                != "7564064e191c88679f3a8b540fd5db847b0b492c"
                or source_chain.get("historical_test_helper_repair_commit")
                != "599baee49905108a4361d37dcc6bd01d2829ee79"
                or source_chain.get("historical_test_helper_repair_tree")
                != "a96d83f1928c7482cd1d49d673744e8f4084c732"
                or source_chain.get("final_control_parent")
                != "599baee49905108a4361d37dcc6bd01d2829ee79"
                or int(source_chain.get("bounded_repair_commit_count") or 0) != 1
                or source_chain.get("final_control_commit_is_current_head") is not True
                or failed.get("schema")
                != "sawm/pre-authoritative-m40-validation-failure@1"
                or failed.get("attempt") != "SAWM-R2-M40-SEALED-SUITE-A1"
                or failed.get("phase")
                != "sealed_semantic_world_test_suite"
                or failed.get("failure_kind")
                != "historical_fixture_omitted_m40_successor_control_key"
                or failed.get("tests_passed") != 285
                or failed.get("tests_failed") != 22
                or len(failed.get("failed_test_node_ids") or ()) != 22
                or failed.get("materializer_invoked") is not False
                or repair.get("repair_commit")
                != "599baee49905108a4361d37dcc6bd01d2829ee79"
                or repair.get(
                    "adds_m40_to_successor_control_keys_newest_first"
                )
                is not True
                or repair.get("m40_key_index") != 0
                or repair.get("last_failed_rerun_passed") != 22
                or repair.get("direct_regression_passed") is not True
                or contract.get("generation_restart_authorized") is not False
                or not isinstance(prior, Mapping)
                or prior.get("migration_revision") != "SAWM-R2-M40"
                or module._m41_failed_pre_authoritative_m40_validation_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M41 failed-pre-authoritative-M40-validation authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M41 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M41 active runtime paths are not exactly preserved")
    elif m40_selected:
        successor = config.get(m40_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m40_failed_pre_authoritative_m39_successor_authority()
            )
            reference = dict(materializer._m40_authority_reference())
            expected_seal_cid = materializer._identity(expected)
            source_chain = expected.get("source_chain", {})
            failed = expected.get("failed_m39_pre_authoritative_attempt", {})
            contract = expected.get("live_preflight_contract", {})
            prior = expected.get("prior_m39_authority", {})
            if (
                successor != reference
                or migration.get(m40_key) != reference
                or seal.get(f"{m40_key}_cid") != expected_seal_cid
                or expected_seal_cid
                != "sha256:377b6e7269a7025f236642f12aaf92264582f42a1efa4899eb4d6e64b0e41db2"
                or expected.get("schema")
                != "sawm/failed-pre-authoritative-m39-successor-authorization@1"
                or expected.get("migration_revision") != "SAWM-R2-M40"
                or expected.get("migration_kind") != m40_key
                or expected.get("target_generation") != 30
                or expected.get("target_event_watermark") != 292
                or expected.get("target_projection_cid")
                != "baguqeera6t2s6prg5atpg4gkgrlmqclu34firbn4z2o3wsgv6btp7p6q66tq"
                or source_chain.get("m39_final_control_commit")
                != "64651e11f9d390a98a9daecc70c672e329487a1b"
                or source_chain.get("m39_final_control_tree")
                != "75b32dda51c7475a0a54de28d6e6a3c693c2d7f3"
                or source_chain.get("restart_helper_repair_parent")
                != "64651e11f9d390a98a9daecc70c672e329487a1b"
                or source_chain.get("restart_helper_repair_commit")
                != "00d15b870f7fdaa8e7165a94719eb2c7d3df7eca"
                or source_chain.get("restart_helper_repair_tree")
                != "83c17ce2ed68d10ef54e53c0a5e312f2eae8060b"
                or source_chain.get("final_control_parent")
                != "00d15b870f7fdaa8e7165a94719eb2c7d3df7eca"
                or int(source_chain.get("bounded_repair_commit_count") or 0) != 1
                or source_chain.get("final_control_commit_is_current_head") is not True
                or failed.get("schema")
                != "sawm/pre-authoritative-m39-materialization-failure@1"
                or failed.get("attempt") != "SAWM-R2-M39-LIVE-A1"
                or failed.get("phase")
                != "generation_restart_row_verification"
                or failed.get("failure_kind")
                != "m39_top_level_authority_missing_stopped_owner"
                or failed.get("error") != "KeyError: 'stopped_owner'"
                or contract.get("generation_restart_authorized") is not False
                or not isinstance(prior, Mapping)
                or prior.get("migration_revision") != "SAWM-R2-M39"
                or module._m40_failed_pre_authoritative_m39_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M40 failed-pre-authoritative-M39 authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M40 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M40 active runtime paths are not exactly preserved")
    elif m39_selected:
        successor = config.get(m39_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m39_committed_m38_evidence_reconciliation_authority()
            )
            reference = dict(materializer._m39_authority_reference())
            expected_seal_cid = materializer._identity(expected)
            source_chain = expected.get("source_chain", {})
            prior = expected.get("prior_authority", {})
            contract = expected.get("live_preflight_contract", {})
            preservation = expected.get("preservation", {})
            if (
                successor != reference
                or migration.get(m39_key) != reference
                or seal.get(f"{m39_key}_cid") != expected_seal_cid
                or expected_seal_cid
                != "sha256:7b986174605b4f2739abf69473d0ce27dfe880d551c55144f076b8f981f633c2"
                or expected.get("schema")
                != "sawm/committed-m38-evidence-reconciliation-successor-authorization@1"
                or expected.get("migration_revision") != "SAWM-R2-M39"
                or expected.get("target_generation") != 30
                or expected.get("target_event_watermark") != 292
                or expected.get("target_projection_cid")
                != "baguqeera6t2s6prg5atpg4gkgrlmqclu34firbn4z2o3wsgv6btp7p6q66tq"
                or source_chain.get("base_control_commit")
                != "1ef0e89b5c4dd4418485adce4c9e6a0d66d18f94"
                or source_chain.get("json_comparison_repair_commit")
                != "146653af91fe3846cb98e49a54ae1173e3a3dc66"
                or source_chain.get("canonical_envelope_repair_commit")
                != "32d2966c4944157d664748c536cfa167f7ae38f5"
                or source_chain.get("materializer_repair_commit")
                != "b581305f42ad4eda6b3d749680e79107c1c150b3"
                or source_chain.get("materializer_repair_tree")
                != "c870812938d25731d11a694a2365c1650c03204c"
                or source_chain.get("final_control_parent")
                != "b581305f42ad4eda6b3d749680e79107c1c150b3"
                or int(source_chain.get("bounded_repair_commit_count") or 0) != 3
                or prior.get("event_watermark") != 291
                or prior.get("receipt_present") is not False
                or prior.get("authorization_cid")
                != "sha256:c6d6c0b6951301d6b8bda94efade51d3e6ceb25dac3a82cdbc100e189c9cac19"
                or contract.get("generation_restart_authorized") is not False
                or contract.get("m38_receipt_must_be_absent") is not True
                or preservation.get("committed_m38_event_291_preserved_exactly")
                is not True
                or preservation.get("m38_receipt_created_or_rewritten") is not False
                or module._m39_committed_m38_evidence_reconciliation_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M39 committed-M38 reconciliation authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M39 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M39 active runtime paths are not exactly preserved")
    elif m38_selected:
        successor = config.get(m38_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m38_pre_authoritative_custody_restart_authority()
            )
            reference = module._m38_authority_reference_for_source_state(
                materializer, expected
            )
            identity_state = module._m38_source_chain_identity_state(
                materializer, expected
            )
            expected_seal_cid = (
                "sha256:PENDING_M38_AUTHORITY_CID"
                if identity_state == "placeholder"
                else materializer._identity(expected)
            )
            source_chain = expected.get("source_chain", {})
            failed = expected.get("failed_m37_pre_authority_attempt", {})
            derivation = expected.get("target_projection_derivation", {})
            if (
                successor != reference
                or migration.get(m38_key) != reference
                or seal.get(f"{m38_key}_cid") != expected_seal_cid
                or source_chain.get("base_control_commit")
                != "02a16d6c76eb2f1f165b544c8c72d62c533d7d3b"
                or source_chain.get("runtime_repair_commit")
                != "ad30bfa90cd0309a77a1a9936815f739e072a8a7"
                or expected.get("control_recorded_at") != "2026-09-01T01:10:00Z"
                or (
                    identity_state == "sealed"
                    and expected_seal_cid
                    != "sha256:664af21f470ada7e4d4bf02313df473ed345c539b122f30036db8c8ee171a8eb"
                )
                or expected.get("target_event_watermark") != 291
                or expected.get("target_projection_cid")
                != "baguqeeravycbuo73fyu5mpad55qi5duk3la53lubqeu7nu6kjtnahehjtnsq"
                or failed.get("failure_phase")
                != "after_database_open_and_checkpoint_before_identity_publication"
                or derivation.get("recomputed_not_inherited") is not True
                or module._m38_pre_authoritative_custody_restart_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M38 custody restart authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M38 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M38 active runtime paths are not exactly preserved")
    elif m37_selected:
        successor = config.get(m37_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m37_post_reboot_generation_restart_authority()
            )
            reference = module._m37_authority_reference_for_source_state(
                materializer, expected
            )
            source_chain = expected.get("source_chain", {})
            prior = expected.get("prior_authority", {})
            m36_anchor = expected.get("m36_historical_anchor", {})
            stopped = expected.get("stopped_owner", {})
            identity_state = module._m37_source_chain_identity_state(
                materializer, expected
            )
            expected_seal_cid = (
                "sha256:PENDING_M37_AUTHORITY_CID"
                if identity_state == "placeholder"
                else materializer._identity(expected)
            )
            if (
                successor != reference
                or migration.get(m37_key) != reference
                or seal.get(f"{m37_key}_cid") != expected_seal_cid
                or source_chain.get("base_control_commit")
                != "a3db1cde328c5aeba86896d4f6813821251ceb7e"
                or source_chain.get("base_control_tree")
                != "fba8c205656afda738ac5f14c2841fb1da452ea0"
                or source_chain.get("initial_control_commit")
                != "fb6672403850bc4b473db8e5e759176e3028adc0"
                or source_chain.get("initial_control_tree")
                != "bb9132b3dd333ad11c60efac45186e1fc6cdb11a"
                or expected_seal_cid
                != "sha256:c776180b7e65de98d5de235765db60148f7693148512b335260ddb772563a795"
                or prior.get("schema") != "sawm/current-operational-head@1"
                or prior.get("event_watermark") != 290
                or m36_anchor.get("migration_revision") != "SAWM-R2-M36"
                or m36_anchor.get("event_watermark") != 286
                or stopped.get("generation") != 29
                or stopped.get("target_generation") != 30
                or expected.get("target_event_watermark") != 291
                or expected.get("target_projection_cid")
                != "baguqeeravycbuo73fyu5mpad55qi5duk3la53lubqeu7nu6kjtnahehjtnsq"
                or module._m37_post_reboot_generation_restart_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M37 post-reboot restart authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M37 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M37 active runtime paths are not exactly preserved")
    elif m36_selected:
        successor = config.get(m36_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m36_operator_task_binding_correction_authority()
            )
            reference = materializer._m36_authority_reference()
            source_chain = expected.get("source_chain", {})
            failed = expected.get("failed_m35_append", {})
            target = expected.get("target_authority", {})
            if (
                successor != reference
                or migration.get(m36_key) != reference
                or seal.get(f"{m36_key}_cid") != materializer._identity(expected)
                or source_chain.get("base_control_commit")
                != "5bbf2dec97458585ee95034c986057a1552b805d"
                or source_chain.get("base_control_tree")
                != "954736c103dd644d7896e032c9676ed38d989831"
                or source_chain.get("initial_control_commit")
                != "e208bc490b8d12f6d86b286d98d1ac63bb4e62be"
                or source_chain.get("initial_control_tree")
                != "53fb95b9fd8e4f6f776c0380b0f1f959ddd58326"
                or target.get("operator_task_alias") != "SAWM-000"
                or target.get("operator_task_cid")
                != "sha256:8b8f43dd51ea4d8467af0e5cae4100478f16666d36c6f4fad49c23fd8e43a3d6"
                or target.get("operator_task_status") != "completed"
                or target.get("operator_task_revision") != 2
                or failed.get("sealed_target_task_cid")
                != "sha256:308a38585461080c06bf51f36a5b9cff75c4bf5a6e88ddcbccbca73198a51d1d"
                or failed.get("m35_event_appended") is not False
                or failed.get("m35_receipt_created") is not False
                or module._m36_operator_task_binding_correction_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M36 operator-task binding authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M36 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M36 active runtime paths are not exactly preserved")
    elif m35_selected:
        successor = config.get(m35_key)
        runtime_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        try:
            module, materializer = _m26_validation_modules(root)
            expected = (
                materializer._expected_m35_immutable_authority_identity_normalization_authority()
            )
            reference = materializer._m35_authority_reference()
            source_chain = expected.get("source_chain", {})
            if (
                successor != reference
                or migration.get(m35_key) != reference
                or seal.get(f"{m35_key}_cid") != materializer._identity(expected)
                or source_chain.get("base_control_commit")
                != "4dfe1c4c81ffd65f6a2d5c5cdc38b1cd33f1f443"
                or source_chain.get("base_control_tree")
                != "894d9a4d206faf4e59328111023ec44fcf26f96e"
                or source_chain.get("initial_control_commit")
                != materializer._M35_INITIAL_CONTROL_COMMIT
                or source_chain.get("initial_control_commit")
                != "27c5e1e5925228757fc02378eb9d3b2a8addc60e"
                or source_chain.get("initial_control_tree")
                != materializer._M35_INITIAL_CONTROL_TREE
                or source_chain.get("initial_control_tree")
                != "75f9c888b22911d47e941fb03b771c8d8f92dd99"
                or source_chain.get("final_reseal_parent")
                != materializer._M35_INITIAL_CONTROL_COMMIT
                or source_chain.get("final_reseal_parent")
                != "27c5e1e5925228757fc02378eb9d3b2a8addc60e"
                or source_chain.get("initial_control_blobs")
                != dict(materializer._M35_INITIAL_CONTROL_BLOBS)
                or module._m35_immutable_authority_identity_normalization_successor_errors(
                    config, seal, migration, root=root
                )
            ):
                config_errors.append(
                    "M35 immutable-authority identity authority/CID/source differs"
                )
        except Exception as exc:
            config_errors.append(
                f"M35 authority validation unavailable: {type(exc).__name__}: {exc}"
            )
        if config.get("runtime_paths") != {
            "root": runtime_root,
            "state": f"{runtime_root}/state",
            "worktrees": f"{runtime_root}/worktrees",
            "merge_queue": f"{runtime_root}/merge-queue",
            "logs": f"{runtime_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }:
            config_errors.append("M35 active runtime paths are not exactly preserved")
    elif m34_selected:
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
