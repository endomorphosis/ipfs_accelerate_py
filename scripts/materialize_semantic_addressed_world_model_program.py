#!/usr/bin/env python3
"""Render, migrate, and verify the append-only SAWM R2 supervisor program.

The default action verifies the sealed prior authority, copies only its exact
canonical control-store bytes to a confined stage, appends the bounded source
migration, successor operational-validation revisions, and operator-authorized
settled-task recovery through the landed intent repository, and atomically
publishes the successor store. It never rewrites accepted task/goal definitions
or completion evidence, and it never copies predecessor execution or
coordination sidecars.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NAMESPACE = "semantic-addressed-world-model-v1"
REVISION = "SAWM-PLAN-R2"
ROOT_GOAL = "SAWM-G000"
SCHEMA = "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-materialization@1"
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"

_M3_PREWORKER_FAILURE_V2: dict[str, Any] = {
    "schema": "sawm/pre-worker-launch-failure@2",
    "command": "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
    "phase": "detached_coordinator_provider_entry_module_preflight",
    "exit_code": 2,
    "outer_launch_command": (
        "python scripts/ops/agent_supervisor/"
        "semantic_addressed_world_model.py launch"
    ),
    "outer_launch_exit_code": 0,
    "error_payload": {
        "schema": "sawm/coordinator-error@1",
        "valid": False,
        "error": (
            "ModuleNotFoundError: No module named "
            "'ipfs_accelerate_py.agent_supervisor.provider_command_environment'"
        ),
    },
    "error_payload_cid": (
        "sha256:7eccdc0993160ddf88afeaa8acd66f951c525d0758c21326a6075d850b996afb"
    ),
    "source_head": "29fb3cec38d682702c952dd20111b56d993345bb",
    "source_tree": "4e3e7532d4d850edabad1610d98bab82d4fe2410",
    "configuration_root": (
        "baguqeeraime2bkkng2cuttamu3qtrqxxlp3xim7hzpkpce7sunv7ptumfkeq"
    ),
    "control_plane_admission_cid": (
        "baguqeeraug4eqqturk3lnsf2xfmg2sqrrm3ij6v7u5pnlxlcfbivzhyi6enq"
    ),
    "control_plane_capsule_id": (
        "sha256:50b6acb20e822d5a841282fbafc7aa5d32dfce9af68bca99c2107fffeb50ef9b"
    ),
    "control_plane_archive_sha256": (
        "sha256:4ea5c26e631544abc5890e28f176685957a8fb367858c8653e9aee5289a00646"
    ),
    "coordinator_pid": 3669646,
    "coordinator_pid_projection_after_failure": "absent",
    "coordinator_log_path": (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m3/"
        "logs/configured-board-20260828T060454Z.log"
    ),
    "coordinator_log_sha256": (
        "sha256:d9932d7bb2f5d906a09a9fe9f1f28886913aec4a40dbdd803d1c17d049a8bc8b"
    ),
    "store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m3/control.duckdb"
    ),
    "owner_generation": 5,
    "owner_server_id": "server:67f0d1bb-0289-45ae-b2bc-33c87f040124",
    "owner_process_birth_id": "birth:eaf37ba44ad162fc5231eb6e4dbbf35e",
    "credential_handoff_retired": True,
    "accepted_control_plane_admitted": True,
    "provider_capability_probed": True,
    "worker_started": False,
    "task_claimed": False,
    "task_state_changed": False,
    "implementation_provider_invoked": False,
    "failure_time_authority": "unavailable",
}

_M5_FROZEN_PRIOR_BINDING: dict[str, Any] = {
    "prior_store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m5/control.duckdb"
    ),
    "prior_control_store_sha256": (
        "e2b8e8a15abe9c2a5b53dd540e9621c11a85f63fcc3799da331bb55e3bfbc408"
    ),
    "prior_event_watermark": 121,
    "prior_event_prefix_sha256": (
        "f07bb22f380f30901d0ca16af4bf6bbc3b544f039fb1f43728828d92a95a3f9e"
    ),
    "prior_projection_cid": (
        "baguqeeram7gdldpraa6twhuvfbqbeou4gefgmj5szitgb6ij2nmwobx727wq"
    ),
    "prior_database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
    "prior_generation": 7,
    "prior_plan_revision": 6,
}

_M6_EXPECTED_PROJECTION_CID = (
    "baguqeeraztioh4pdrleiio2hzt2dtxh7o7ae2jk237fvcflktepffj6dpuya"
)
_M6_OPERATIONAL_TASK_COUNT = 44
_M6_OPERATIONAL_VALIDATION_COUNT = 46
_M6_EVENT_SUFFIX_LENGTH = 47
_M6_MIGRATION_EVENT_OFFSET = 2
_M6_OPERATIONAL_FIRST_OFFSET = 3
_M6_OPERATIONAL_LAST_OFFSET = 46
_M6_RECOVERY_EVENT_OFFSET = 47
_M6_VALIDATION_TOKEN_FROM = "/home/barberb/.local/bin/python"
_M6_VALIDATION_TOKEN_TO = "python"
_M6_TASK_IDENTITY_MANIFEST_CID = (
    "sha256:902037a87e711082e57c8dcf002eab669f0dc83c35e9f372ba197a18f8acf1d3"
)
_M6_SUPERSESSION_REASON = (
    "source_authority_revision_and_settled_preprovider_task_requeue"
)
_M6_SUPERSESSION_MODE = (
    "source_authority_revision_and_operational_validation_recovery"
)


class MaterializationError(RuntimeError):
    """Fail-closed SAWM bootstrap error."""


class MigrationRequired(MaterializationError):
    """An existing append-only authority contains a different population."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def closed(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=closed)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _board_module(root: Path):
    import importlib.util
    name = "_sawm_board_validator_for_materializer"
    path = root / "scripts/validate_semantic_addressed_world_model_board.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MaterializationError("unable to load the sealed board parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode:
        raise MaterializationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _source_binding(root: Path) -> dict[str, Any]:
    controls = (
        ".gitignore",
        "requirements.txt",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        "docs/architecture/semantic_addressed_world_model.objectives.md",
        "docs/architecture/semantic_addressed_world_model.todo.md",
        "docs/architecture/semantic_addressed_world_model_inventory/repository_baseline.json",
        "docs/architecture/semantic_addressed_world_model_inventory/authority_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/overlap_gap_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/identity_inventory.json",
        "docs/architecture/semantic_addressed_world_model_inventory/interface_inventory.json",
        "docs/architecture/semantic_addressed_world_model_inventory/dependency_graph.json",
        "docs/architecture/semantic_addressed_world_model_inventory/capability_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/rollout_baseline.json",
        "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json",
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
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
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
        "test/api/test_agent_supervisor_native_dependency_pin.py",
        "test/api/test_agent_supervisor_project_dependency_preflight.py",
        "test/api/test_agent_supervisor_provider_command_binding.py",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
    )
    missing = [path for path in controls if not (root / path).is_file()]
    if missing:
        raise MaterializationError(
            "sealed source controls are missing: " + ", ".join(missing)
        )
    file_digests = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in controls
    }
    payload = {
        "schema": "sawm/current-source-binding@1",
        "head": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "branch": _git(root, "branch", "--show-current"),
        "datasets_gitlink": _git(root, "rev-parse", "HEAD:ipfs_datasets_py"),
        "kit_gitlink": _git(root, "rev-parse", "HEAD:ipfs_kit_py"),
        "control_sha256": file_digests,
    }
    return {**payload, "source_binding_cid": _identity(payload)}


def _metadata_payload(card: Any) -> dict[str, Any]:
    """Retain every closed board field without treating Markdown as state."""
    return {str(key).replace(" ", "_"): str(value) for key, value in sorted(card.metadata.items())}


def _operational_task_aliases() -> tuple[str, ...]:
    return tuple(f"SAWM-{ordinal:03d}" for ordinal in range(1, 45))


def _task_identity_manifest_cid(task_cids: Mapping[str, Any]) -> str:
    manifest = [
        {"task_alias": alias, "task_cid": str(task_cids[alias])}
        for alias in _operational_task_aliases()
    ]
    return _identity(manifest)


def _operational_validation_commands(
    commands: Sequence[Sequence[str]],
    *,
    task_alias: str,
) -> tuple[tuple[str, ...], int]:
    revised: list[tuple[str, ...]] = []
    replacements = 0
    for argv in commands:
        revised_argv: list[str] = []
        for part in argv:
            text = str(part)
            count = text.count(_M6_VALIDATION_TOKEN_FROM)
            replacements += count
            revised_argv.append(
                text.replace(_M6_VALIDATION_TOKEN_FROM, _M6_VALIDATION_TOKEN_TO)
            )
        revised.append(tuple(revised_argv))
    if not revised or replacements != len(revised):
        raise MigrationRequired(
            f"{task_alias} operational validation token cardinality differs"
        )
    return tuple(revised), replacements


def _operational_validation_receipt(
    population: Mapping[str, Any],
    *,
    task_alias: str,
    task_cid: str,
    expected_status: str,
    expected_revision: int,
    prior_validations: Sequence[Sequence[str]],
    operational_validations: Sequence[Sequence[str]],
    replacement_count: int,
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    receipt = {
        "schema": "sawm/operational-validation-revision@1",
        "authority_class": "operator_control_plane",
        "migration_revision": migration["migration_revision"],
        "task_cid": task_cid,
        "task_alias": task_alias,
        "historical_definition_cid": migration["prior_task_cids"][task_alias],
        "historical_definition_rewritten": False,
        "expected_status": expected_status,
        "expected_revision": int(expected_revision),
        "status": expected_status,
        "target_revision": int(expected_revision) + 1,
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "validation_count": len(prior_validations),
        "validation_token_from": _M6_VALIDATION_TOKEN_FROM,
        "validation_token_to": _M6_VALIDATION_TOKEN_TO,
        "validation_token_replacement_count": int(replacement_count),
        "prior_validation_digest": _identity(
            {
                "task_cid": task_cid,
                "validations": [list(argv) for argv in prior_validations],
            }
        ),
        "operational_validation_digest": _identity(
            {
                "task_cid": task_cid,
                "validations": [list(argv) for argv in operational_validations],
            }
        ),
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def build_population(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    board = _board_module(root)
    tasks = board._parse_cards(root / "docs/architecture/semantic_addressed_world_model.todo.md", goal=False)
    goals = board._parse_cards(root / "docs/architecture/semantic_addressed_world_model.objectives.md", goal=True)
    if tuple(item.identifier for item in tasks) != board.TASK_IDS or tuple(item.identifier for item in goals) != board.GOAL_IDS:
        raise MaterializationError("board population differs from the sealed 45-task/29-goal identity")
    source = _source_binding(root)
    migration = _load_json(
        root
        / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json"
    )
    migration_history = migration.get("migration_history")
    expected_migration_revision = (
        f"SAWM-R2-M{len(migration_history) + 1}"
        if isinstance(migration_history, list)
        else ""
    )
    if (
        migration.get("schema")
        != "sawm/prior-materialization-migration-inventory@3"
        or migration.get("migration_revision") != expected_migration_revision
        or migration.get("migration_kind")
        != "bounded_validation_runtime_and_operational_board_command_recovery"
        or migration.get("supersession_reason") != _M6_SUPERSESSION_REASON
    ):
        raise MaterializationError("the sealed append-only migration revision is missing")
    launch_failure = migration.get("preworker_launch_failure")
    if (
        not isinstance(launch_failure, dict)
        or launch_failure != _M3_PREWORKER_FAILURE_V2
        or _identity(launch_failure.get("error_payload"))
        != launch_failure.get("error_payload_cid")
    ):
        raise MaterializationError("the typed pre-worker launch failure is invalid")
    preprovider_failure = migration.get("preprovider_task_failure")
    if (
        not isinstance(preprovider_failure, dict)
        or preprovider_failure.get("schema") != "sawm/pre-provider-task-failure@2"
        or preprovider_failure.get("store_id") != migration.get("prior_store_id")
        or preprovider_failure.get("control_store_sha256")
        != migration.get("prior_control_store_sha256")
        or preprovider_failure.get("canonical_event_watermark")
        != migration.get("prior_event_watermark")
        or preprovider_failure.get("canonical_event_prefix_sha256")
        != migration.get("prior_event_prefix_sha256")
        or preprovider_failure.get("canonical_projection_cid")
        != migration.get("prior_projection_cid")
        or preprovider_failure.get("source_binding_cid")
        != migration.get("prior_source_binding_cid")
        or preprovider_failure.get("successor_migration_revision")
        != migration.get("migration_revision")
        or preprovider_failure.get("successor_plan_revision")
        != migration.get("target_plan_revision")
        or preprovider_failure.get("successor_generation")
        != migration.get("target_generation")
        or preprovider_failure.get("successor_store_id")
        != migration.get("target_store_id")
        or preprovider_failure.get("implementation_provider_invoked") is not False
        or preprovider_failure.get("provider_dispatched") is not False
        or preprovider_failure.get("effect_claim_recorded") is not False
        or preprovider_failure.get("implementation_commit_created") is not False
        or preprovider_failure.get("merge_attempted") is not False
        or preprovider_failure.get("task_completed") is not False
        or preprovider_failure.get("canonical_task_status") != "blocked"
        or int(preprovider_failure.get("canonical_task_revision") or 0) != 5
        or preprovider_failure.get("settlement_closed") is not True
    ):
        raise MaterializationError("the sealed pre-provider task failure is invalid")
    bounded_paths = migration.get("bounded_control_plane_repair_paths")
    if (
        not isinstance(bounded_paths, list)
        or len(bounded_paths) != len(set(bounded_paths))
        or any(path not in source["control_sha256"] for path in bounded_paths)
    ):
        raise MaterializationError("bounded control-plane repairs are not source-bound")
    prior_goal_cids = migration.get("prior_goal_cids")
    prior_task_cids = migration.get("prior_task_cids")
    if not isinstance(prior_goal_cids, dict) or not isinstance(prior_task_cids, dict):
        raise MaterializationError("prior SAWM identity maps are incomplete")
    if (
        _task_identity_manifest_cid(prior_task_cids)
        != _M6_TASK_IDENTITY_MANIFEST_CID
    ):
        raise MaterializationError("operational task identity manifest differs")

    goal_cids: dict[str, str] = {}
    goal_definitions: dict[str, dict[str, Any]] = {}
    for ordinal, card in enumerate(goals, 1):
        definition = {
            "schema": "sawm/goal-definition@1", "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "goal_id": card.identifier,
            "title": card.title, "ordinal": ordinal,
            "metadata": _metadata_payload(card),
            "source_binding_cid": migration["definition_source_binding_cid"],
        }
        goal_definitions[card.identifier] = definition
        goal_cids[card.identifier] = str(prior_goal_cids[card.identifier])
        if _identity(definition) != goal_cids[card.identifier]:
            raise MaterializationError(
                f"preserved goal definition does not rehash: {card.identifier}"
            )

    objective_id = str(migration["prior_objective_id"])
    objective_rows: list[dict[str, Any]] = []
    goal_edges: list[dict[str, Any]] = []
    for ordinal, card in enumerate(goals, 1):
        parent_alias = board.GOAL_PARENT[card.identifier]
        definition = goal_definitions[card.identifier]
        row = {
            "goal_cid": goal_cids[card.identifier], "goal_id": card.identifier,
            "goal_alias": card.identifier, "title": card.title, "ordinal": ordinal,
            "status": str(card.metadata.get("status") or "open"),
            "parent_goal_cid": goal_cids[parent_alias] if parent_alias else "",
            "definition_cid": goal_cids[card.identifier], "definition": definition,
            "board_namespace": NAMESPACE, "plan_revision": REVISION,
        }
        if card.identifier == ROOT_GOAL:
            row.update({"objective_id": objective_id, "objective_alias": ROOT_GOAL,
                        "priority": str(card.metadata.get("priority") or "P0")})
        objective_rows.append(row)
        if parent_alias:
            goal_edges.append({"parent_goal_cid": goal_cids[parent_alias],
                               "child_goal_cid": goal_cids[card.identifier],
                               "edge_kind": "goal_refinement"})

    plan_definition = {
        "schema": "sawm/plan-definition@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION, "root_goal_cid": goal_cids[ROOT_GOAL],
        "goal_definition_cids": [goal_cids[item.identifier] for item in goals],
        "source_binding_cid": migration["definition_source_binding_cid"],
    }
    plan_cid = str(migration["prior_plan_root_cid"])
    if _identity(plan_definition) != plan_cid:
        raise MaterializationError("preserved plan definition does not rehash")

    task_definitions: dict[str, dict[str, Any]] = {}
    task_cids: dict[str, str] = {}
    for ordinal, card in enumerate(tasks, 1):
        definition = {
            "schema": "sawm/task-definition@1", "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "task_id": card.identifier,
            "title": card.title, "ordinal": ordinal,
            "metadata": _metadata_payload(card),
            "source_binding_cid": migration["definition_source_binding_cid"],
        }
        task_definitions[card.identifier] = definition
        task_cids[card.identifier] = str(prior_task_cids[card.identifier])
        if _identity(definition) != task_cids[card.identifier]:
            raise MaterializationError(
                f"preserved task definition does not rehash: {card.identifier}"
            )

    taskboard: list[dict[str, Any]] = []
    for ordinal, card in enumerate(tasks, 1):
        meta = card.metadata
        deps = json.loads(meta["dependencies json"])
        outputs = json.loads(meta["outputs json"])
        validations = json.loads(meta["validation commands json"])
        goal_alias = str(meta["goal id"])
        definition = task_definitions[card.identifier]
        taskboard.append(
            {
                "task_cid": task_cids[card.identifier], "task_id": card.identifier,
                "task_alias": card.identifier, "title": card.title, "ordinal": ordinal,
                # The operator card is deliberately born ready.  Its Markdown
                # completed marker cannot bypass current evidence and CAS.
                "status": "ready" if card.identifier == "SAWM-000" else "todo",
                "priority": str(meta.get("priority") or "P2"),
                "goal_cid": goal_cids[goal_alias], "goal_id": goal_alias,
                "plan_cid": plan_cid,
                "depends_on": [task_cids[str(dep)] for dep in deps],
                # IntentRepository output keys use the closed safe-ID grammar;
                # retain the exact file path inside the canonical effect body.
                "outputs": [{"effect_id": _identity({"task": card.identifier, "path": str(path)}),
                             "declared_path": str(path), "effect": "declared_output"}
                            for path in outputs],
                "acceptance_criteria": [{
                    "ordinal": 1, "criterion": str(meta.get("acceptance") or ""),
                    # IntentRepository stores this complete mapping as the
                    # evidence policy, so the required kind belongs here.
                    **({"evidence_kind": "operator_control_validation"}
                       if card.identifier == "SAWM-000" else {}),
                }],
                "validation_commands": validations,
                "definition_cid": task_cids[card.identifier], "definition": definition,
                "board_namespace": NAMESPACE, "plan_revision": REVISION,
                "completion_mode": str(meta.get("completion mode") or "automatic"),
                "protected_paths": str(meta.get("protected paths") or ""),
                "repository_owner": str(meta.get("owning repository") or ""),
                "provider_role": str(meta.get("provider role") or ""),
                "rollout_mode": str(meta.get("rollout mode") or ""),
            }
        )

    program_definition = {
        "schema": "sawm/program-definition@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION,
        "source_binding_cid": migration["definition_source_binding_cid"],
        "plan_cid": plan_cid, "goal_cids": [goal_cids[item.identifier] for item in goals],
        "task_cids": [task_cids[item.identifier] for item in tasks],
    }
    program_definition_cid = str(migration["prior_program_definition_cid"])
    if _identity(program_definition) != program_definition_cid:
        raise MaterializationError("preserved program definition does not rehash")
    for row in (*objective_rows, *taskboard):
        row["program_definition_cid"] = program_definition_cid
    return {
        "schema": "sawm/program-population@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION, "program_definition_cid": program_definition_cid,
        "program_definition": program_definition, "repository_tree_id": source["source_binding_cid"],
        "source_binding": source, "plan_root_cid": plan_cid,
        # DatabaseTaskSource consumes `objectives` before `goals`; all 29 goal
        # records are intentionally supplied in this authoritative key.
        "objectives": objective_rows, "goal_edges": goal_edges,
        "plans": [{"plan_cid": plan_cid, "plan_alias": REVISION,
                   "goal_cid": goal_cids[ROOT_GOAL], "status": "active", **plan_definition}],
        "taskboard": taskboard, "migration_inventory": migration,
    }


def _store_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_offline(path: Path) -> None:
    # Direct DuckDB verification is an offline operation.  Once the Quack
    # state owner is live, every read and write must pass through that owner;
    # opening the file here would violate the single-owner authority boundary.
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        discover_live_quack_endpoint,
    )
    discovery = discover_live_quack_endpoint(path)
    owner_marker = path.with_name(f".{path.name}.state-owner.json")
    if discovery.uri or owner_marker.exists():
        detail = discovery.reason or "active_owner_marker"
        raise MaterializationError(
            f"offline store verification refused while Quack ownership may be active: {detail}"
        )


def _event_prefix_digest(connection: Any, watermark: int) -> tuple[str, int]:
    rows = connection.execute(
        """
        SELECT global_sequence, event_id, event_type, stream_id, sequence,
               task_cid, attempt_id, session_id, recorded_at, body_json
        FROM domain_events
        WHERE global_sequence <= ?
        ORDER BY global_sequence
        """,
        [int(watermark)],
    ).fetchall()
    normalized_rows = [
        [row[index] for index in range(10)]
        for row in rows
    ]
    payload = json.dumps(
        normalized_rows,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(rows)


def _verify_receipt_anchor(path: Path, expected_cid: str) -> None:
    if not path.is_file():
        raise MigrationRequired(f"sealed receipt is missing: {path}")
    receipt = _load_json(path)
    claimed = str(receipt.pop("receipt_cid", ""))
    if claimed != expected_cid or _identity(receipt) != expected_cid:
        raise MigrationRequired(f"sealed receipt identity differs: {path}")


def _verify_migration_history(
    root: Path,
    connection: Any,
    migration: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Verify the complete immutable predecessor chain before adding a suffix."""

    raw_history = migration.get("migration_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise MigrationRequired("sealed migration history is missing")
    history: list[Mapping[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for ordinal, raw in enumerate(raw_history, 1):
        if not isinstance(raw, Mapping):
            raise MigrationRequired("sealed migration history entry is not an object")
        entry = dict(raw)
        expected_revision = f"SAWM-R2-M{ordinal}"
        schema = entry.get("schema")
        prior_watermark = int(entry.get("prior_event_watermark") or 0)
        migration_watermark = int(
            entry.get("migration_event_watermark")
            or entry.get("target_event_watermark")
            or 0
        )
        materialization_watermark = int(
            entry.get("materialization_event_watermark")
            or entry.get("target_event_watermark")
            or 0
        )
        target_watermark = int(entry.get("target_event_watermark") or 0)
        if entry.get("migration_revision") != expected_revision:
            raise MigrationRequired("sealed migration history sequence differs")
        if schema == "sawm/source-migration-history-entry@1":
            if (
                ordinal > 3
                or migration_watermark != prior_watermark + 2
                or target_watermark != migration_watermark
            ):
                raise MigrationRequired("sealed @1 migration history sequence differs")
        elif schema == "sawm/source-migration-history-entry@2":
            if (
                ordinal != 4
                or migration_watermark != prior_watermark + 2
                or target_watermark != migration_watermark + 1
                or entry.get("post_migration_event_type")
                != "intent.task_status_changed"
                or entry.get("post_migration_task_status") != "in_progress"
                or int(entry.get("post_migration_task_revision") or 0) != 2
            ):
                raise MigrationRequired("sealed @2 migration history sequence differs")
        elif schema == "sawm/source-migration-history-entry@3":
            runtime_events = entry.get("post_materialization_events")
            if (
                ordinal != 5
                or migration_watermark != prior_watermark + 2
                or materialization_watermark != migration_watermark + 1
                or target_watermark != materialization_watermark + 2
                or not isinstance(runtime_events, list)
                or len(runtime_events) != 2
                or [int(item.get("global_sequence") or 0) for item in runtime_events]
                != [materialization_watermark + 1, materialization_watermark + 2]
                or [item.get("event_type") for item in runtime_events]
                != ["intent.task_status_changed", "intent.task_status_changed"]
                or [item.get("task_status") for item in runtime_events]
                != ["in_progress", "blocked"]
                or [int(item.get("task_revision") or 0) for item in runtime_events]
                != [4, 5]
            ):
                raise MigrationRequired("sealed @3 migration history sequence differs")
        else:
            raise MigrationRequired("sealed migration history schema differs")
        if previous is not None and (
            entry.get("prior_store_id") != previous.get("target_store_id")
            or entry.get("prior_control_store_sha256")
            != previous.get("target_control_store_sha256")
            or entry.get("prior_event_watermark")
            != previous.get("target_event_watermark")
            or entry.get("prior_event_prefix_sha256")
            != previous.get("target_event_prefix_sha256")
            or entry.get("prior_source_binding_cid")
            != previous.get("current_source_binding_cid")
        ):
            raise MigrationRequired("sealed migration history continuity differs")
        for path_key, digest_key in (
            ("prior_store_id", "prior_control_store_sha256"),
            ("target_store_id", "target_control_store_sha256"),
        ):
            store_path = (root / str(entry.get(path_key) or "")).resolve()
            if not store_path.is_relative_to(root) or not store_path.is_file():
                raise MigrationRequired("sealed migration-history store is missing")
            if _store_sha256(store_path) != entry.get(digest_key):
                raise MigrationRequired("sealed migration-history store bytes differ")
        receipt_path = (root / str(entry.get("migration_receipt_path") or "")).resolve()
        if not receipt_path.is_relative_to(root):
            raise MigrationRequired("sealed migration-history receipt escapes the source root")
        _verify_receipt_anchor(
            receipt_path,
            str(entry.get("migration_receipt_cid") or ""),
        )
        receipt_file_sha256 = str(entry.get("migration_receipt_file_sha256") or "")
        if (
            receipt_file_sha256
            and _store_sha256(receipt_path) != receipt_file_sha256
        ):
            raise MigrationRequired("sealed migration-history receipt bytes differ")
        evidence = connection.execute(
            "SELECT digest FROM evidence_nodes WHERE evidence_id = ? "
            "AND evidence_kind = 'operator_control_plane_source_migration'",
            [entry.get("migration_evidence_id")],
        ).fetchall()
        if len(evidence) != 1 or str(evidence[0][0]) != entry.get("migration_digest"):
            raise MigrationRequired("sealed migration-history evidence differs")
        event_sequences = [migration_watermark - 1, migration_watermark]
        expected_event_ids = [
            entry.get("plan_migration_event_id"),
            entry.get("migration_evidence_event_id"),
        ]
        if schema == "sawm/source-migration-history-entry@2":
            event_sequences.append(target_watermark)
            expected_event_ids.append(entry.get("post_migration_event_id"))
        elif schema == "sawm/source-migration-history-entry@3":
            event_sequences.append(materialization_watermark)
            expected_event_ids.append(entry.get("materialization_event_id"))
            for runtime_event in entry["post_materialization_events"]:
                event_sequences.append(int(runtime_event["global_sequence"]))
                expected_event_ids.append(runtime_event.get("event_id"))
        placeholders = ",".join("?" for _ in event_sequences)
        event_ids = connection.execute(
            "SELECT global_sequence, event_id, event_type, task_cid, body_json "
            f"FROM domain_events WHERE global_sequence IN ({placeholders}) "
            "ORDER BY global_sequence",
            event_sequences,
        ).fetchall()
        if [str(row[1]) for row in event_ids] != expected_event_ids:
            raise MigrationRequired("sealed migration-history event identities differ")
        if schema == "sawm/source-migration-history-entry@2":
            post_event = event_ids[-1]
            post_payload = json.loads(str(post_event[4]))
            post_body = post_payload.get("body") or {}
            if (
                int(post_event[0]) != target_watermark
                or str(post_event[2]) != entry.get("post_migration_event_type")
                or str(post_event[3]) != entry.get("post_migration_task_cid")
                or post_body.get("task_cid") != entry.get("post_migration_task_cid")
                or post_body.get("status") != entry.get("post_migration_task_status")
                or int(post_body.get("revision") or 0)
                != int(entry.get("post_migration_task_revision") or 0)
            ):
                raise MigrationRequired("sealed post-migration runtime event differs")
            migration_prefix, migration_count = _event_prefix_digest(
                connection, migration_watermark
            )
            if (
                migration_count != migration_watermark
                or migration_prefix != entry.get("migration_event_prefix_sha256")
            ):
                raise MigrationRequired("sealed M4 migration-only event prefix differs")
        elif schema == "sawm/source-migration-history-entry@3":
            materialization_event = event_ids[2]
            materialization_payload = json.loads(str(materialization_event[4]))
            materialization_body = materialization_payload.get("body") or {}
            if (
                int(materialization_event[0]) != materialization_watermark
                or str(materialization_event[2]) != "intent.task_status_changed"
                or str(materialization_event[3])
                != entry.get("materialization_task_cid")
                or materialization_body.get("task_cid")
                != entry.get("materialization_task_cid")
                or materialization_body.get("status") != "todo"
                or int(materialization_body.get("revision") or 0) != 3
            ):
                raise MigrationRequired("sealed M5 materialization event differs")
            for observed, expected in zip(
                event_ids[3:], entry["post_materialization_events"], strict=True
            ):
                payload = json.loads(str(observed[4]))
                body = payload.get("body") or {}
                if (
                    int(observed[0]) != int(expected["global_sequence"])
                    or str(observed[1]) != expected["event_id"]
                    or str(observed[2]) != expected["event_type"]
                    or str(observed[3]) != expected["task_cid"]
                    or body.get("task_cid") != expected["task_cid"]
                    or body.get("status") != expected["task_status"]
                    or int(body.get("revision") or 0)
                    != int(expected["task_revision"])
                ):
                    raise MigrationRequired("sealed M5 runtime event differs")
            for watermark, digest_key, noun in (
                (
                    migration_watermark,
                    "migration_event_prefix_sha256",
                    "migration-only",
                ),
                (
                    materialization_watermark,
                    "materialization_event_prefix_sha256",
                    "materialization",
                ),
                (target_watermark, "target_event_prefix_sha256", "runtime"),
            ):
                prefix, count = _event_prefix_digest(connection, watermark)
                if count != watermark or prefix != entry.get(digest_key):
                    raise MigrationRequired(
                        f"sealed M5 {noun} event prefix differs"
                    )
        history.append(entry)
        previous = entry

    latest = history[-1]
    if (
        latest.get("target_store_id") != migration.get("prior_store_id")
        or latest.get("target_control_store_sha256")
        != migration.get("prior_control_store_sha256")
        or latest.get("target_event_watermark")
        != migration.get("prior_event_watermark")
        or latest.get("target_event_prefix_sha256")
        != migration.get("prior_event_prefix_sha256")
        or latest.get("current_source_binding_cid")
        != migration.get("prior_source_binding_cid")
        or latest.get("projection_cid") != migration.get("prior_projection_cid")
        or latest.get("migration_receipt_cid")
        != migration.get("prior_materialization_receipt_cid")
        or latest.get("migration_receipt_path")
        != migration.get("prior_materialization_receipt_path")
    ):
        raise MigrationRequired("latest migration-history entry does not bind the predecessor")
    if latest.get("schema") in {
        "sawm/source-migration-history-entry@2",
        "sawm/source-migration-history-entry@3",
    }:
        latest_materialization_watermark = latest.get(
            "materialization_event_watermark",
            latest.get("migration_event_watermark"),
        )
        latest_materialization_projection = latest.get(
            "materialization_projection_cid",
            latest.get("migration_projection_cid"),
        )
        if (
            latest_materialization_watermark
            != migration.get("prior_materialization_event_watermark")
            or latest_materialization_projection
            != migration.get("prior_materialization_projection_cid")
            or latest.get("migration_receipt_file_sha256")
            != migration.get("prior_materialization_receipt_file_sha256")
        ):
            raise MigrationRequired("latest materialization receipt binding differs")
    return tuple(history)


def _assert_committed_clean_source(
    root: Path,
    population: Mapping[str, Any],
) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise MaterializationError(
            "materialization requires a clean committed source tree"
        )
    controls = tuple(population["source_binding"]["control_sha256"])
    untracked: list[str] = []
    mismatched: list[str] = []
    for relative in controls:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if result.returncode:
            untracked.append(relative)
            continue
        if _git(root, "hash-object", relative) != _git(
            root, "rev-parse", f"HEAD:{relative}"
        ):
            mismatched.append(relative)
    if untracked or mismatched:
        raise MaterializationError(
            "source controls are not exact committed blobs: "
            + json.dumps(
                {"untracked": untracked, "mismatched": mismatched},
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _verify_sealed_file(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    noun: str,
) -> Path:
    path = (root / str(relative_path or "")).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise MigrationRequired(f"sealed {noun} is missing or escapes the source root")
    expected = str(expected_sha256 or "")
    if expected.startswith("sha256:"):
        expected = expected.removeprefix("sha256:")
    if len(expected) != 64 or _store_sha256(path) != expected:
        raise MigrationRequired(f"sealed {noun} bytes differ")
    return path


def _table_count(connection: Any, table_name: str) -> int:
    row = connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
    return int(row[0]) if row is not None else -1


def _verify_preprovider_failure_artifacts(
    root: Path,
    migration: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the frozen M5 sidecars without making them successor authority."""

    failure = migration["preprovider_task_failure"]
    if (
        failure.get("authority_class")
        != "operator_frozen_predecessor_observation"
        or failure.get("owner_status") != "stopped"
        or failure.get("retry_deferred") is not True
        or failure.get("attempt_consumed") is not False
        or failure.get("provider_call_allowed") is not False
        or failure.get("provider_dispatch_attempted") is not False
        or failure.get("provider_invocation_recorded") is not False
        or failure.get("authoritative_completion_evidence") is not False
        or failure.get("settlement_closed") is not True
    ):
        raise MigrationRequired("sealed M5 pre-provider authority classification differs")

    execution_path = _verify_sealed_file(
        root,
        str(failure["execution_store_id"]),
        str(failure["execution_store_sha256"]),
        noun="M5 execution sidecar",
    )
    _verify_sealed_file(
        root,
        str(failure["execution_wal_path"]),
        str(failure["execution_wal_sha256"]),
        noun="M5 execution WAL",
    )
    coordination_path = _verify_sealed_file(
        root,
        str(failure["coordination_store_id"]),
        str(failure["coordination_store_sha256"]),
        noun="M5 coordination sidecar",
    )
    for path_key, digest_key, noun in (
        ("configured_board_launch_log_path", "configured_board_launch_log_sha256", "M5 launch log"),
        ("implementation_daemon_log_path", "implementation_daemon_log_sha256", "M5 implementation log"),
        ("supervisor_event_log_path", "supervisor_event_log_sha256", "M5 supervisor event log"),
        ("supervisor_status_path", "supervisor_status_sha256", "M5 supervisor status"),
        ("merge_queue_store_path", "merge_queue_store_sha256", "M5 merge queue"),
        ("quack_owner_status_path", "quack_owner_status_sha256", "M5 stopped owner status"),
        ("portal_attempt_binding_path", "portal_attempt_binding_sha256", "M5 portal attempt binding"),
    ):
        _verify_sealed_file(
            root,
            str(failure[path_key]),
            str(failure[digest_key]),
            noun=noun,
        )

    import duckdb

    execution = duckdb.connect(str(execution_path), read_only=True)
    try:
        execution_counts = {
            "database_task_attempts": _table_count(execution, "database_task_attempts"),
            "attempt_phases": _table_count(execution, "attempt_phases"),
            "provider_invocations": _table_count(execution, "provider_invocations"),
            "effect_claims": _table_count(execution, "effect_claims"),
        }
        if execution_counts != failure["execution_authority_counts"]:
            raise MigrationRequired("sealed M5 execution-sidecar counts differ")
        attempt = execution.execute(
            "SELECT claim_id, task_cid, task_alias, attempt_number, owner_session_id, "
            "fencing_token, fence_epoch, lease_id, committed_phase, status, revision "
            "FROM database_task_attempts WHERE attempt_id = ?",
            [failure["attempt_id"]],
        ).fetchall()
        expected_attempt = [
            (
                failure["claim_id"],
                failure["task_cid"],
                failure["task_alias"],
                1,
                failure["owner_session_id"],
                int(failure["fencing_token"]),
                int(failure["fence_epoch"]),
                failure["lease_id"],
                failure["database_attempt_committed_phase"],
                failure["database_attempt_status"],
                int(failure["database_attempt_revision"]),
            )
        ]
        if attempt != expected_attempt:
            raise MigrationRequired("sealed M5 execution attempt differs")
        phases = execution.execute(
            "SELECT phase, fencing_token, fence_epoch, revision, body_json "
            "FROM attempt_phases WHERE attempt_id = ? ORDER BY revision",
            [failure["attempt_id"]],
        ).fetchall()
        failed_body = json.loads(str(phases[-1][4])) if phases else {}
        if (
            [(str(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in phases]
            != [("claimed", 1, 1, 1), ("context", 1, 1, 2), ("failed", 1, 1, 3)]
            or failed_body.get("settlement_id") != failure["settlement_id"]
            or failed_body.get("failure_kind")
            != failure["settlement_failure_kind"]
            or failed_body.get("failure_payload_digest")
            != failure["failure_payload_digest"]
            or int(failed_body.get("provider_invocation_count") or 0) != 0
            or int(failed_body.get("effect_claim_count") or 0) != 0
        ):
            raise MigrationRequired("sealed M5 execution settlement differs")
    finally:
        execution.close()

    coordination = duckdb.connect(str(coordination_path), read_only=True)
    try:
        coordination_counts = {
            "task_attempts": _table_count(coordination, "task_attempts"),
            "task_claims": _table_count(coordination, "task_claims"),
            "fenced_leases": _table_count(coordination, "fenced_leases"),
            "resource_claims": _table_count(coordination, "resource_claims"),
            "task_completions": _table_count(coordination, "task_completions"),
        }
        if coordination_counts != failure["coordination_authority_counts"]:
            raise MigrationRequired("sealed M5 coordination-sidecar counts differ")
        attempt = coordination.execute(
            "SELECT task_cid, attempt_number, owner_session_id, fencing_token, "
            "fence_epoch, status, revision FROM task_attempts WHERE attempt_id = ?",
            [failure["attempt_id"]],
        ).fetchall()
        if attempt != [
            (
                failure["task_cid"],
                1,
                failure["owner_session_id"],
                int(failure["fencing_token"]),
                int(failure["fence_epoch"]),
                failure["coordination_attempt_status"],
                int(failure["coordination_attempt_revision"]),
            )
        ]:
            raise MigrationRequired("sealed M5 coordination attempt differs")
        claim = coordination.execute(
            "SELECT task_cid, owner_session_id, fencing_token, fence_epoch, "
            "expires_at_ms, state, revision, attempt_id, attempt_number, lease_id "
            "FROM task_claims WHERE claim_id = ?",
            [failure["claim_id"]],
        ).fetchall()
        if claim != [
            (
                failure["task_cid"],
                failure["owner_session_id"],
                int(failure["fencing_token"]),
                int(failure["fence_epoch"]),
                int(failure["coordination_claim_expires_at_epoch_ms"]),
                failure["coordination_claim_status"],
                int(failure["coordination_claim_revision"]),
                failure["attempt_id"],
                1,
                failure["lease_id"],
            )
        ]:
            raise MigrationRequired("sealed M5 coordination claim differs")
        lease = coordination.execute(
            "SELECT task_cid, owner_session_id, fencing_token, fence_epoch, "
            "expires_at_ms, state, revision, claim_id, attempt_id, attempt_number "
            "FROM fenced_leases WHERE lease_id = ?",
            [failure["lease_id"]],
        ).fetchall()
        if claim and lease != [
            (
                failure["task_cid"],
                failure["owner_session_id"],
                int(failure["fencing_token"]),
                int(failure["fence_epoch"]),
                int(failure["coordination_claim_expires_at_epoch_ms"]),
                failure["coordination_lease_status"],
                int(failure["coordination_lease_revision"]),
                failure["claim_id"],
                failure["attempt_id"],
                1,
            )
        ]:
            raise MigrationRequired("sealed M5 coordination lease differs")
        completion = coordination.execute(
            "SELECT status, body_json FROM task_completions WHERE task_cid = ?",
            [failure["task_cid"]],
        ).fetchall()
        completion_body = json.loads(str(completion[0][1])) if completion else {}
        if (
            len(completion) != 1
            or str(completion[0][0]) != "failed"
            or completion_body.get("settlement_id") != failure["settlement_id"]
        ):
            raise MigrationRequired("sealed M5 coordination settlement differs")
        ready = coordination.execute(
            "SELECT ready FROM coordination_tasks WHERE task_cid = ?",
            [failure["task_cid"]],
        ).fetchall()
        if ready != [(bool(failure["coordination_task_ready"]),)]:
            raise MigrationRequired("sealed M5 coordination readiness differs")
    finally:
        coordination.close()

    binding_path = _verify_sealed_file(
        root,
        str(failure["portal_attempt_binding_path"]),
        str(failure["portal_attempt_binding_sha256"]),
        noun="M5 portal attempt binding",
    )
    binding = _load_json(binding_path)
    claimed_binding_id = str(binding.pop("binding_id", ""))
    if (
        claimed_binding_id != failure["portal_attempt_binding_id"]
        or _identity(binding) != claimed_binding_id
        or binding.get("task_cid") != failure["task_cid"]
        or binding.get("task_alias") != failure["task_alias"]
        or int(binding.get("task_revision") or 0)
        != int(failure["canonical_claim_revision"])
        or binding.get("attempt_id") != failure["attempt_id"]
        or binding.get("claim_id") != failure["claim_id"]
        or binding.get("lease_id") != failure["lease_id"]
        or binding.get("projection_seed_digest")
        != "sha256:" + failure["portal_task_projection_sha256"]
        or binding.get("projection_authority") is not False
    ):
        raise MigrationRequired("sealed M5 portal attempt binding differs")

    portal_dir = binding_path.parent
    event_log = _verify_sealed_file(
        root,
        str((portal_dir / "portal-events.jsonl").relative_to(root)),
        str(failure["portal_event_log_sha256"]),
        noun="M5 portal event log",
    )
    manifest_path = _verify_sealed_file(
        root,
        str((portal_dir / "portal-events.jsonl.manifest.json").relative_to(root)),
        str(failure["portal_event_manifest_sha256"]),
        noun="M5 portal event manifest",
    )
    _verify_sealed_file(
        root,
        str((portal_dir / "task-projection.md").relative_to(root)),
        str(failure["portal_task_projection_sha256"]),
        noun="M5 portal task projection",
    )
    event_rows = [
        json.loads(line)
        for line in event_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(event_rows) != int(failure["portal_event_count"])
        or [int(row.get("sequence") or 0) for row in event_rows]
        != list(
            range(
                int(failure["portal_event_first_sequence"]),
                int(failure["portal_event_last_sequence"]) + 1,
            )
        )
        or event_rows[-1].get("event_id") != failure["portal_event_tail_id"]
        or any(
            row.get("previous_event_id")
            != ("" if index == 0 else event_rows[index - 1].get("event_id"))
            for index, row in enumerate(event_rows)
        )
    ):
        raise MigrationRequired("sealed M5 portal event chain differs")
    events_by_id = {str(row.get("event_id") or ""): row for row in event_rows}
    probe_event = events_by_id.get(str(failure["failure_event_id"])) or {}
    preflight = probe_event.get("dependency_preflight") or {}
    probe = preflight.get("probe") or {}
    exception_event = events_by_id.get(
        str(failure["implementation_exception_event_id"])
    ) or {}
    finished_event = events_by_id.get(
        str(failure["implementation_finished_event_id"])
    ) or {}
    if (
        probe_event.get("type") != failure["failure_event_type"]
        or probe_event.get("task_id") != failure["task_alias"]
        or preflight.get("receipt_id") != failure["preflight_receipt_id"]
        or preflight.get("retry_fingerprint")
        != failure["preflight_retry_fingerprint"]
        or preflight.get("reason") != failure["preflight_reason"]
        or probe.get("reason") != failure["probe_reason"]
        or preflight.get("validation_roots") != failure["validation_roots"]
        or exception_event.get("type") != "implementation_exception"
        or exception_event.get("failure_kind") != failure["failure_kind"]
        or exception_event.get("phase") != failure["failure_phase"]
        or exception_event.get("reason") != failure["failure_reason"]
        or exception_event.get("provider_call_allowed") is not False
        or finished_event.get("type") != "implementation_finished"
        or finished_event.get("provider_dispatched") is not False
        or finished_event.get("implementation_commit")
        or (finished_event.get("merge_result") or {}).get("merged") is not False
        or finished_event.get("attempt_consumed") is not False
    ):
        raise MigrationRequired("sealed M5 pre-provider portal failure differs")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("snapshot_id") != failure["portal_event_snapshot_id"]
        or manifest.get("last_event_id") != failure["portal_event_tail_id"]
        or int(manifest.get("earliest_sequence") or 0)
        != int(failure["portal_event_first_sequence"])
        or int(manifest.get("latest_sequence") or 0)
        != int(failure["portal_event_last_sequence"])
    ):
        raise MigrationRequired("sealed M5 portal manifest binding differs")
    for mutation in failure["owner_mutation_receipts"]:
        path = _verify_sealed_file(
            root,
            mutation["path"],
            mutation["sha256"],
            noun="M5 owner mutation receipt",
        )
        result = _load_json(path)
        observed = result.get("observed") or {}
        if (
            result.get("request_id") != mutation["request_id"]
            or result.get("result_cid") != mutation["result_cid"]
            or observed.get("event_id") != mutation["event_id"]
            or observed.get("old_status") != mutation["old_status"]
            or int(observed.get("old_revision") or 0)
            != int(mutation["old_revision"])
            or observed.get("new_status") != mutation["new_status"]
            or int(observed.get("new_revision") or 0)
            != int(mutation["new_revision"])
        ):
            raise MigrationRequired("sealed M5 owner mutation result differs")
    return {
        "execution_store_sha256": failure["execution_store_sha256"],
        "coordination_store_sha256": failure["coordination_store_sha256"],
        "portal_event_log_sha256": failure["portal_event_log_sha256"],
        "portal_event_manifest_sha256": failure["portal_event_manifest_sha256"],
        "settlement_id": failure["settlement_id"],
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
    }


def _verify_prior_store(
    root: Path,
    config: Mapping[str, Any],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    if any(
        migration.get(key) != expected
        for key, expected in _M5_FROZEN_PRIOR_BINDING.items()
    ):
        raise MigrationRequired("sealed M5 predecessor binding differs")
    if migration.get("preworker_launch_failure") != _M3_PREWORKER_FAILURE_V2:
        raise MigrationRequired("sealed M3 historical launch failure differs")
    preprovider_failure = migration["preprovider_task_failure"]
    configured = config.get("prior_materialization") or {}
    for inventory_key, config_key in (
        ("migration_revision", "migration_revision"),
        ("prior_store_id", "store_id"),
        ("prior_program_definition_cid", "program_definition_cid"),
        ("prior_plan_root_cid", "plan_root_cid"),
        ("prior_source_binding_cid", "source_binding_cid"),
        ("prior_projection_cid", "projection_cid"),
    ):
        if migration.get(inventory_key) != configured.get(config_key):
            raise MaterializationError(
                f"scheduler and migration inventory disagree on {inventory_key}"
            )
    if (
        configured.get("operator_task_cid")
        != migration["prior_task_cids"]["SAWM-000"]
        or configured.get("inventory_path")
        != "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json"
        or configured.get("reason") != migration["supersession_reason"]
        or configured.get("preserve_append_only") is not True
        or int(configured.get("migration_history_count") or 0)
        != len(migration["migration_history"])
        or int(configured.get("prior_plan_revision") or 0)
        != int(migration["prior_plan_revision"])
        or int(configured.get("target_plan_revision") or 0)
        != int(migration["target_plan_revision"])
        or int(configured.get("migration_event_watermark") or 0)
        != int(migration["prior_event_watermark"]) + _M6_MIGRATION_EVENT_OFFSET
        or int(configured.get("operational_validation_first_event_watermark") or 0)
        != int(migration["prior_event_watermark"]) + _M6_OPERATIONAL_FIRST_OFFSET
        or int(configured.get("operational_validation_last_event_watermark") or 0)
        != int(migration["prior_event_watermark"]) + _M6_OPERATIONAL_LAST_OFFSET
        or int(configured.get("target_event_watermark") or 0)
        != int(migration["prior_event_watermark"]) + _M6_EVENT_SUFFIX_LENGTH
        or configured.get("target_projection_cid") != _M6_EXPECTED_PROJECTION_CID
        or int(configured.get("prior_materialization_event_watermark") or 0)
        != int(migration["prior_materialization_event_watermark"])
        or configured.get("prior_materialization_projection_cid")
        != migration["prior_materialization_projection_cid"]
        or configured.get("prior_materialization_receipt_cid")
        != migration["prior_materialization_receipt_cid"]
        or configured.get("prior_materialization_receipt_path")
        != migration["prior_materialization_receipt_path"]
        or configured.get("prior_materialization_receipt_file_sha256")
        != migration["prior_materialization_receipt_file_sha256"]
    ):
        raise MaterializationError("scheduler prior-authority policy differs from its inventory")
    requeue = configured.get("operational_validation_requeue") or {}
    expected_requeue = {
        "schema": "sawm/operational-validation-requeue-authorization@1",
        "authorized": True,
        "authority": "operator_source_migration",
        "task_alias_first": "SAWM-001",
        "task_alias_last": "SAWM-044",
        "task_count": _M6_OPERATIONAL_TASK_COUNT,
        "task_identity_manifest_cid": _M6_TASK_IDENTITY_MANIFEST_CID,
        "validation_command_count": _M6_OPERATIONAL_VALIDATION_COUNT,
        "validation_token_from": _M6_VALIDATION_TOKEN_FROM,
        "validation_token_to": _M6_VALIDATION_TOKEN_TO,
        "recovered_task_alias": "SAWM-001",
        "recovered_task_cid": migration["prior_task_cids"]["SAWM-001"],
        "from_status": "blocked",
        "from_revision": 5,
        "operational_status": "blocked",
        "operational_revision": 6,
        "to_status": "todo",
        "to_revision": 7,
        "reason": migration["migration_kind"],
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "operational_validation_revision_changes": _M6_OPERATIONAL_TASK_COUNT,
        "expected_event_watermark": int(migration["prior_event_watermark"])
        + _M6_EVENT_SUFFIX_LENGTH,
        "worker_self_approval": False,
    }
    if requeue != expected_requeue:
        raise MaterializationError(
            "scheduler operational-validation requeue authority differs from its inventory"
        )
    if not isinstance(config.get("validation_runtime"), Mapping) or not config[
        "validation_runtime"
    ]:
        raise MaterializationError("scheduler validation runtime is not sealed")
    prior = (root / str(migration["prior_store_id"])).resolve()
    target = (root / str(migration["target_store_id"])).resolve()
    if not prior.is_relative_to(root) or not target.is_relative_to(root) or prior == target:
        raise MaterializationError("prior and target stores must be distinct confined paths")
    if not prior.is_file():
        raise MigrationRequired(f"sealed prior authority is missing: {prior}")
    _assert_offline(prior)
    initial_hash = _store_sha256(prior)
    if initial_hash != migration["prior_control_store_sha256"]:
        raise MigrationRequired("sealed prior control-store bytes differ")

    import duckdb
    connection = duckdb.connect(str(prior), read_only=True)
    try:
        task_rows = connection.execute(
            "SELECT task_alias, task_cid, status, revision, body_json "
            "FROM tasks ORDER BY ordinal"
        ).fetchall()
        goal_rows = connection.execute(
            "SELECT goal_alias, goal_cid, body_json FROM goals ORDER BY ordinal"
        ).fetchall()
        task_map = {str(row[0]): str(row[1]) for row in task_rows}
        goal_map = {str(row[0]): str(row[1]) for row in goal_rows}
        if len(task_rows) != int(migration["prior_task_count"]):
            raise MigrationRequired("sealed prior task count differs")
        if len(goal_rows) != int(migration["prior_goal_count"]):
            raise MigrationRequired("sealed prior goal count differs")
        if task_map != migration["prior_task_cids"]:
            raise MigrationRequired("sealed prior task identity map differs")
        if goal_map != migration["prior_goal_cids"]:
            raise MigrationRequired("sealed prior goal identity map differs")
        for row in task_rows:
            body = json.loads(str(row[4]))
            if body.get("program_definition_cid") != migration["prior_program_definition_cid"]:
                raise MigrationRequired(
                    f"sealed prior task program binding differs: {row[0]}"
                )
        for row in goal_rows:
            body = json.loads(str(row[2]))
            if body.get("program_definition_cid") != migration["prior_program_definition_cid"]:
                raise MigrationRequired(
                    f"sealed prior goal program binding differs: {row[0]}"
                )
        objective = connection.execute(
            "SELECT objective_id, objective_alias, body_json FROM objectives"
        ).fetchall()
        if len(objective) != 1 or (
            str(objective[0][0]) != migration["prior_objective_id"]
            or str(objective[0][1]) != ROOT_GOAL
            or json.loads(str(objective[0][2])).get("program_definition_cid")
            != migration["prior_program_definition_cid"]
        ):
            raise MigrationRequired("sealed prior objective binding differs")
        completed = [str(row[0]) for row in task_rows if str(row[2]) == "completed"]
        if completed != ["SAWM-000"]:
            raise MigrationRequired(f"sealed prior completion set differs: {completed}")
        statuses = {str(row[0]): str(row[2]) for row in task_rows}
        expected_statuses = {
            alias: (
                "completed"
                if alias == "SAWM-000"
                else "blocked"
                if alias == "SAWM-001"
                else "todo"
            )
            for alias in migration["prior_task_cids"]
        }
        if statuses != expected_statuses:
            raise MigrationRequired("sealed M5 canonical task states differ")
        sawm_001 = next(row for row in task_rows if row[0] == "SAWM-001")
        if (
            int(sawm_001[3])
            != int(preprovider_failure["canonical_task_revision"])
            or str(sawm_001[1]) != preprovider_failure["task_cid"]
        ):
            raise MigrationRequired("sealed M5 SAWM-001 revision binding differs")
        canonical_counts = {
            table_name: _table_count(connection, table_name)
            for table_name in preprovider_failure["canonical_authority_counts"]
        }
        if canonical_counts != preprovider_failure["canonical_authority_counts"]:
            raise MigrationRequired("sealed M5 canonical authority counts differ")
        owner = connection.execute(
            "SELECT server_id, process_birth_id, status "
            "FROM state_servers WHERE generation = ?",
            [int(preprovider_failure["owner_generation"])],
        ).fetchall()
        if owner != [
            (
                preprovider_failure["owner_server_id"],
                preprovider_failure["owner_process_birth_id"],
                "stopped",
            )
        ]:
            raise MigrationRequired("sealed M5 launch owner identity or stopped state differs")
        operator = next(row for row in task_rows if row[0] == "SAWM-000")
        if str(operator[2]) != migration["prior_operator_status"] or int(operator[3]) != 2:
            raise MigrationRequired("sealed prior operator status/revision differs")

        plan = connection.execute(
            "SELECT revision, body_json FROM plans WHERE plan_cid = ?",
            [migration["prior_plan_root_cid"]],
        ).fetchone()
        if plan is None or int(plan[0]) != int(migration["prior_plan_revision"]):
            raise MigrationRequired("sealed prior plan revision differs")
        plan_body = json.loads(str(plan[1]))
        history = migration["migration_history"]
        latest_history = history[-1]
        if (
            plan_body.get("source_binding_cid")
            != migration["definition_source_binding_cid"]
            or plan_body.get("current_source_binding_cid")
            != migration["prior_source_binding_cid"]
            or plan_body.get("source_migration_revision")
            != latest_history["migration_revision"]
            or plan_body.get("source_migration_digest")
            != latest_history["migration_digest"]
        ):
            raise MigrationRequired("sealed prior plan source binding differs")
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM control_plane_metadata"
            ).fetchall()
        )
        if metadata.get("database_uuid") != migration["prior_database_uuid"]:
            raise MigrationRequired("sealed prior database UUID differs")
        generation = connection.execute(
            "SELECT generation, database_uuid, birth_id FROM store_generations "
            "ORDER BY generation DESC LIMIT 1"
        ).fetchone()
        if generation is None or (
            int(generation[0]) != int(migration["prior_generation"])
            or str(generation[1]) != migration["prior_database_uuid"]
            or str(generation[2]) != migration["prior_process_birth_id"]
        ):
            raise MigrationRequired("sealed prior generation identity differs")
        evidence_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT evidence_id FROM evidence_nodes WHERE task_cid = ?",
                [migration["prior_task_cids"]["SAWM-000"]],
            ).fetchall()
        }
        if not {
            migration["prior_operator_evidence_id"],
            migration["prior_validation_evidence_id"],
        }.issubset(evidence_ids):
            raise MigrationRequired("sealed prior operator evidence chain differs")
        validation = connection.execute(
            "SELECT outcome FROM validation_results WHERE result_id = ?",
            [migration["prior_validation_result_id"]],
        ).fetchone()
        completion = connection.execute(
            "SELECT task_cid FROM completion_receipts WHERE receipt_cid = ?",
            [migration["prior_completion_receipt_cid"]],
        ).fetchone()
        if validation is None or str(validation[0]) != "passed":
            raise MigrationRequired("sealed prior validation result differs")
        if completion is None or str(completion[0]) != migration["prior_task_cids"]["SAWM-000"]:
            raise MigrationRequired("sealed prior completion receipt differs")
        digest, count = _event_prefix_digest(
            connection, int(migration["prior_event_watermark"])
        )
        if count != int(migration["prior_event_watermark"]):
            raise MigrationRequired("sealed prior event sequence is not contiguous")
        if digest != migration["prior_event_prefix_sha256"]:
            raise MigrationRequired("sealed prior event prefix differs")
        _verify_migration_history(root, connection, migration)
    finally:
        connection.close()

    _verify_receipt_anchor(
        root / str(migration["prior_materialization_receipt_path"]),
        str(migration["prior_materialization_receipt_cid"]),
    )
    preprovider_evidence = _verify_preprovider_failure_artifacts(root, migration)
    # Landed schema and replay verification may obtain a read/write adapter;
    # run those checks only on an isolated copy. The accepted prior authority
    # above is opened read-only and its bytes are rehashed below.
    with tempfile.TemporaryDirectory(prefix="sawm-r2-prior-", dir="/tmp") as temp_dir:
        prior_copy = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(prior, prior_copy)
        prior_verified = _verify_store(
            prior_copy,
            population,
            require_operator_complete=True,
            require_migration=False,
        )
    if prior_verified["projection_cid"] != migration["prior_projection_cid"]:
        raise MigrationRequired("sealed prior projection identity differs")
    if _store_sha256(prior) != initial_hash:
        raise MaterializationError("offline prior verification changed authority bytes")
    return {
        "valid": True,
        "database_path": str(prior),
        "database_sha256": initial_hash,
        "event_prefix_sha256": migration["prior_event_prefix_sha256"],
        "event_watermark": migration["prior_event_watermark"],
        "projection_cid": migration["prior_projection_cid"],
        "preprovider_evidence": preprovider_evidence,
    }


def _verify_store(
    path: Path,
    population: Mapping[str, Any],
    *,
    require_operator_complete: bool,
    require_migration: bool = True,
    migration_config: Mapping[str, Any] | None = None,
    expected_validation_digest: str = "",
) -> dict[str, Any]:
    _assert_offline(path)
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    schema = verify_datasets_authoritative_operational_schema(path)
    if schema.get("valid") is not True:
        raise MigrationRequired(
            "existing database does not verify as datasets-authoritative operational schema"
        )
    source = DatabaseTaskSource(
        path,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
    )
    try:
        migration_details: dict[str, Any] = {}
        snap = source.snapshot()
        expected_tasks = list(population["taskboard"])
        expected_goals = list(population["objectives"])
        migration = population["migration_inventory"]
        if (
            snap.task_count != 45
            or snap.goal_count != 29
            or snap.plan_root_cid != population["plan_root_cid"]
        ):
            raise MigrationRequired(
                "population counts/root conflict: "
                f"tasks={snap.task_count} goals={snap.goal_count} "
                f"root={snap.plan_root_cid}"
            )

        status_by_alias: dict[str, str] = {}
        revision_by_alias: dict[str, int] = {}
        operational_receipts: dict[str, dict[str, Any]] = {}
        for expected in expected_tasks:
            alias = str(expected["task_id"])
            raw = source.intent.get_task(str(expected["task_cid"]))
            observed = source.get_task(str(expected["task_cid"]))
            prior_validations = tuple(
                (str(command),) for command in expected["validation_commands"]
            )
            expected_validations: tuple[tuple[str, ...], ...] = prior_validations
            expected_body_receipt: Mapping[str, Any] | None = None
            if require_migration and alias in _operational_task_aliases():
                expected_validations, replacement_count = (
                    _operational_validation_commands(
                        prior_validations,
                        task_alias=alias,
                    )
                )
                prior_status = "blocked" if alias == "SAWM-001" else "todo"
                prior_revision = 5 if alias == "SAWM-001" else 1
                expected_body_receipt = _operational_validation_receipt(
                    population,
                    task_alias=alias,
                    task_cid=str(expected["task_cid"]),
                    expected_status=prior_status,
                    expected_revision=prior_revision,
                    prior_validations=prior_validations,
                    operational_validations=expected_validations,
                    replacement_count=replacement_count,
                )
                operational_receipts[alias] = dict(expected_body_receipt)
            expected_outputs = [dict(item) for item in expected["outputs"]]
            expected_acceptance = [
                dict(item) for item in expected["acceptance_criteria"]
            ]
            expected_identity = {
                "task_cid": str(expected["task_cid"]),
                "task_alias": alias,
                "repository_tree_id": migration["definition_source_binding_cid"],
            }
            if (
                observed is None
                or raw is None
                or observed.task_alias != alias
                or raw["identity"] != expected_identity
                or observed.body.get("definition_cid")
                != expected["definition_cid"]
                or observed.body.get("definition") != expected["definition"]
                or sorted(observed.dependencies) != sorted(expected["depends_on"])
                or [dict(item.get("effect") or {}) for item in observed.outputs]
                != expected_outputs
                or [dict(item.get("evidence_policy") or {}) for item in observed.acceptance]
                != expected_acceptance
                or tuple(
                    tuple(str(part) for part in item.get("argv") or ())
                    for item in observed.validations
                )
                != expected_validations
                or any(dict(item.get("policy") or {}) for item in observed.validations)
            ):
                raise MigrationRequired(f"task definition conflict: {alias}")
            if require_migration:
                if alias == "SAWM-000":
                    if "operational_validation_revision" in observed.body:
                        raise MigrationRequired("operator task acquired an operational revision")
                elif observed.body.get("operational_validation_revision") != expected_body_receipt:
                    raise MigrationRequired(
                        f"operational validation receipt differs: {alias}"
                    )
            status_by_alias[alias] = observed.status
            revision_by_alias[alias] = int(observed.revision)

        for expected in expected_goals:
            observed = source.get_goal(str(expected["goal_cid"]))
            if (
                observed is None
                or str(observed.get("goal_alias")) != expected["goal_id"]
                or (observed.get("body") or {}).get("definition_cid")
                != expected["definition_cid"]
                or (observed.get("body") or {}).get("definition")
                != expected["definition"]
            ):
                raise MigrationRequired(f"goal definition conflict: {expected['goal_id']}")
        if (
            require_operator_complete
            and status_by_alias.get("SAWM-000")
            not in {"completed", "complete", "done"}
        ):
            raise MigrationRequired("SAWM-000 lacks an admitted completion CAS")

        if require_migration:
            if migration_config is None or not expected_validation_digest:
                raise MaterializationError(
                    "migration verification requires exact config and validator binding"
                )
            expected_statuses = {
                alias: "completed" if alias == "SAWM-000" else "todo"
                for alias in migration["prior_task_cids"]
            }
            expected_revisions = {
                alias: (
                    2
                    if alias == "SAWM-000"
                    else 7
                    if alias == "SAWM-001"
                    else 2
                )
                for alias in migration["prior_task_cids"]
            }
            if (
                status_by_alias != expected_statuses
                or revision_by_alias != expected_revisions
            ):
                raise MigrationRequired("M6 task-state recovery projection differs")
            expected_target_watermark = (
                int(migration["prior_event_watermark"])
                + _M6_EVENT_SUFFIX_LENGTH
            )
            if int(snap.event_cursor) != expected_target_watermark:
                raise MigrationRequired("M6 materialization event watermark differs")
            plan = source.plans.get(str(population["plan_root_cid"]))
            if (
                plan is None
                or int(plan.get("revision") or 0)
                != int(migration["target_plan_revision"])
            ):
                raise MigrationRequired(
                    "append-only source migration plan revision is missing"
                )

            with source.intent._connection(write=False) as connection:
                migration_rows = connection.execute(
                    "SELECT evidence_id, digest, body_json, created_at FROM evidence_nodes "
                    "WHERE task_cid = ? AND evidence_kind = ? ORDER BY created_at",
                    [
                        migration["prior_task_cids"]["SAWM-000"],
                        "operator_control_plane_source_migration",
                    ],
                ).fetchall()
                history = tuple(migration["migration_history"])
                if len(migration_rows) != len(history) + 1:
                    raise MigrationRequired("source migration receipt chain length differs")
                parsed_rows = [
                    {
                        "evidence_id": str(row[0]),
                        "digest": str(row[1]),
                        "body": json.loads(str(row[2])),
                        "created_at": str(row[3]),
                    }
                    for row in migration_rows
                ]
                for entry in history:
                    matched = [
                        row
                        for row in parsed_rows
                        if row["evidence_id"] == entry["migration_evidence_id"]
                    ]
                    if (
                        len(matched) != 1
                        or matched[0]["digest"] != entry["migration_digest"]
                        or matched[0]["body"].get("migration_revision")
                        != entry["migration_revision"]
                    ):
                        raise MigrationRequired("prior source migration evidence differs")
                current_rows = [
                    row
                    for row in parsed_rows
                    if row["body"].get("migration_revision")
                    == migration["migration_revision"]
                ]
                if len(current_rows) != 1:
                    raise MigrationRequired(
                        "current source migration receipt is ambiguous"
                    )
                current_row = current_rows[0]
                migration_evidence_id = current_row["evidence_id"]
                migration_digest = current_row["digest"]
                migration_body = current_row["body"]
                migration_created_at = current_row["created_at"]
                expected_migration_body = _migration_body(
                    population,
                    migration_config,
                    expected_validation_digest,
                )
                if migration_body != expected_migration_body:
                    raise MigrationRequired("source migration receipt binding differs")
                if migration_digest != _identity(migration_body):
                    raise MigrationRequired("source migration evidence digest differs")
                expected_evidence_id = content_identity(
                    {
                        "task_cid": migration["prior_task_cids"]["SAWM-000"],
                        "evidence_kind": "operator_control_plane_source_migration",
                        "digest": migration_digest,
                        "body": migration_body,
                    }
                )
                if migration_evidence_id != expected_evidence_id:
                    raise MigrationRequired("source migration evidence identity differs")

                revision_rows = connection.execute(
                    "SELECT revision, body_json, recorded_at FROM plan_revisions "
                    "WHERE plan_cid = ? AND revision <= ? ORDER BY revision",
                    [population["plan_root_cid"], migration["target_plan_revision"]],
                ).fetchall()
                if [int(row[0]) for row in revision_rows] != list(
                    range(1, int(migration["target_plan_revision"]) + 1)
                ):
                    raise MigrationRequired(
                        "exact source migration plan revisions are missing"
                    )
                prior_plan_body = json.loads(
                    str(revision_rows[int(migration["prior_plan_revision"]) - 1][1])
                )
                plan_delta = _migration_plan_delta(population)
                expected_plan_body = {
                    **prior_plan_body,
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": migration["migration_revision"],
                    "source_migration_digest": migration_digest,
                    "supersession_mode": _M6_SUPERSESSION_MODE,
                    "last_delta": plan_delta,
                }
                observed_plan_body = json.loads(str(revision_rows[-1][1]))
                plan_recorded_at = str(revision_rows[-1][2])
                if observed_plan_body != expected_plan_body:
                    raise MigrationRequired("source migration plan revision body differs")

                prefix_digest, prefix_count = _event_prefix_digest(
                    connection,
                    int(migration["prior_event_watermark"]),
                )
                if (
                    prefix_count != int(migration["prior_event_watermark"])
                    or prefix_digest != migration["prior_event_prefix_sha256"]
                ):
                    raise MigrationRequired(
                        "accepted event prefix changed during migration"
                    )
                suffix_sequences = list(
                    range(
                        int(migration["prior_event_watermark"]) + 1,
                        expected_target_watermark + 1,
                    )
                )
                placeholders = ",".join("?" for _ in suffix_sequences)
                suffix_rows = connection.execute(
                    "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
                    "task_cid, attempt_id, session_id, recorded_at, body_json "
                    f"FROM domain_events WHERE global_sequence IN ({placeholders}) "
                    "ORDER BY global_sequence",
                    suffix_sequences,
                ).fetchall()
                if len(suffix_rows) != _M6_EVENT_SUFFIX_LENGTH:
                    raise MigrationRequired("migration event suffix differs")
                events: list[dict[str, Any]] = []
                for row in suffix_rows:
                    event = {
                        "event_id": str(row[0]),
                        "stream_id": str(row[1]),
                        "sequence": int(row[2]),
                        "global_sequence": int(row[3]),
                        "event_type": str(row[4]),
                        "task_cid": str(row[5]),
                        "attempt_id": str(row[6]),
                        "session_id": str(row[7]),
                        "recorded_at": str(row[8]),
                        "body": json.loads(str(row[9])),
                    }
                    payload = event["body"]
                    if (
                        content_identity(
                            {
                                "stream_id": event["stream_id"],
                                "sequence": event["sequence"],
                                "global_sequence": event["global_sequence"],
                                "event_type": event["event_type"],
                                "body": payload,
                            }
                        )
                        != event["event_id"]
                        or set(payload)
                        != {
                            "schema",
                            "event_type",
                            "subject_id",
                            "body",
                            "recorded_at",
                            "owner_id",
                        }
                        or payload.get("schema")
                        != "ipfs_accelerate_py/agent-supervisor/intent-event@1"
                        or payload.get("event_type") != event["event_type"]
                        or payload.get("recorded_at") != event["recorded_at"]
                        or payload.get("owner_id") != "sawm-r2-source-migrator"
                        or event["stream_id"] != "stream:intent"
                        or event["session_id"] != "session:intent"
                        or event["attempt_id"]
                        or event["sequence"] != event["global_sequence"]
                    ):
                        raise MigrationRequired("migration event identity differs")
                    events.append(event)

                plan_event = events[0]
                evidence_event = events[1]
                operational_events = events[2:-1]
                recovery_event = events[-1]
                expected_plan_event_body = {
                    "plan_cid": population["plan_root_cid"],
                    "goal_cid": migration["prior_goal_cids"][ROOT_GOAL],
                    "plan_alias": REVISION,
                    "status": "active",
                    "revision": int(migration["target_plan_revision"]),
                    "body": expected_plan_body,
                    "delta": plan_delta,
                    "recorded_at": plan_recorded_at,
                }
                if (
                    plan_event["event_type"]
                    != "intent.plan_revision_appended"
                    or plan_event["task_cid"]
                    or plan_event["body"].get("subject_id")
                    != population["plan_root_cid"]
                    or plan_event["body"].get("body")
                    != expected_plan_event_body
                ):
                    raise MigrationRequired("source migration plan event differs")
                expected_evidence_event_body = {
                    "evidence_id": migration_evidence_id,
                    "parent_evidence_id": "",
                    "task_cid": migration["prior_task_cids"]["SAWM-000"],
                    "evidence_kind": "operator_control_plane_source_migration",
                    "digest": migration_digest,
                    "body": migration_body,
                    "created_at": migration_created_at,
                    "revision": 0,
                }
                if (
                    evidence_event["event_type"] != "intent.evidence_recorded"
                    or evidence_event["task_cid"]
                    != migration["prior_task_cids"]["SAWM-000"]
                    or evidence_event["body"].get("subject_id")
                    != migration_evidence_id
                    or evidence_event["body"].get("body")
                    != expected_evidence_event_body
                ):
                    raise MigrationRequired("source migration evidence event differs")

                expected_by_alias = {
                    str(item["task_id"]): item for item in expected_tasks
                }
                operational_event_ids: dict[str, str] = {}
                if len(operational_events) != _M6_OPERATIONAL_TASK_COUNT:
                    raise MigrationRequired(
                        "operational validation event count differs"
                    )
                for event, alias in zip(
                    operational_events,
                    _operational_task_aliases(),
                    strict=True,
                ):
                    expected = expected_by_alias[alias]
                    task_cid = str(expected["task_cid"])
                    prior_status = "blocked" if alias == "SAWM-001" else "todo"
                    prior_revision = 5 if alias == "SAWM-001" else 1
                    target_revision = prior_revision + 1
                    prior_body_row = connection.execute(
                        "SELECT body_json FROM task_revisions "
                        "WHERE task_cid = ? AND revision = ?",
                        [task_cid, prior_revision],
                    ).fetchall()
                    if len(prior_body_row) == 1:
                        prior_body = json.loads(str(prior_body_row[0][0]))
                    elif not prior_body_row and prior_revision == 1:
                        # The original M1 projection predates per-task
                        # revision rows for untouched todo tasks. Reconstruct
                        # their immutable body from the sealed @1 population;
                        # the definition CID and all relation tables were
                        # independently checked above.
                        prior_body = {
                            key: value
                            for key, value in expected.items()
                            if key
                            not in {
                                "task_cid",
                                "task_id",
                                "task_alias",
                                "cid",
                                "goal_cid",
                                "goal_id",
                                "depends_on",
                                "dependencies",
                                "effects",
                                "outputs",
                                "acceptance_criteria",
                                "acceptance",
                                "validation_commands",
                                "validations",
                                "status",
                                "priority",
                                "ordinal",
                                "plan_cid",
                                "objective_id",
                            }
                        }
                    else:
                        raise MigrationRequired(
                            f"historical task revision is ambiguous: {alias}"
                        )
                    expected_body = {
                        **prior_body,
                        "operational_validation_revision": operational_receipts[
                            alias
                        ],
                    }
                    target_revision_rows = connection.execute(
                        "SELECT status, body_json, recorded_at FROM task_revisions "
                        "WHERE task_cid = ? AND revision = ?",
                        [task_cid, target_revision],
                    ).fetchall()
                    if len(target_revision_rows) != 1:
                        raise MigrationRequired(
                            f"operational task revision is ambiguous: {alias}"
                        )
                    target_revision_status = str(target_revision_rows[0][0])
                    target_revision_body = json.loads(
                        str(target_revision_rows[0][1])
                    )
                    target_recorded_at = str(target_revision_rows[0][2])
                    if (
                        target_revision_status != prior_status
                        or target_revision_body != expected_body
                        or not target_recorded_at
                    ):
                        raise MigrationRequired(
                            f"operational task revision differs: {alias}"
                        )
                    operational_validations, _ = _operational_validation_commands(
                        tuple(
                            (str(command),)
                            for command in expected["validation_commands"]
                        ),
                        task_alias=alias,
                    )
                    inner = event["body"].get("body") or {}
                    expected_inner = {
                        "task_cid": task_cid,
                        "task_alias": alias,
                        "goal_cid": str(expected["goal_cid"]),
                        "plan_cid": str(expected["plan_cid"]),
                        "objective_id": "",
                        "ordinal": int(expected["ordinal"]),
                        "status": prior_status,
                        "priority": str(expected["priority"]),
                        "revision": target_revision,
                        "identity": {
                            "task_cid": task_cid,
                            "task_alias": alias,
                            "repository_tree_id": migration[
                                "definition_source_binding_cid"
                            ],
                        },
                        "body": expected_body,
                        # Task revision and event envelope timestamps are made
                        # by two consecutive landed-authority calls.  Their
                        # values may straddle a clock tick; bind the task-body
                        # timestamp to the exact revision row rather than
                        # incorrectly requiring equality with the envelope.
                        "recorded_at": target_recorded_at,
                        "dependencies": sorted(expected["depends_on"]),
                        "outputs": [dict(item) for item in expected["outputs"]],
                        "acceptance": [
                            dict(item) for item in expected["acceptance_criteria"]
                        ],
                        "validations": [
                            {"argv": list(argv)}
                            for argv in operational_validations
                        ],
                    }
                    if (
                        event["event_type"] != "intent.task_upserted"
                        or event["task_cid"] != task_cid
                        or event["body"].get("subject_id") != task_cid
                        or inner != expected_inner
                    ):
                        differing_keys = sorted(
                            key
                            for key in set(inner) | set(expected_inner)
                            if inner.get(key) != expected_inner.get(key)
                        )
                        raise MigrationRequired(
                            "operational validation event differs: "
                            f"{alias}; fields={differing_keys}"
                        )
                    operational_event_ids[alias] = event["event_id"]

                recovery_receipt = _task_recovery_receipt(population)
                expected_recovery_body = {
                    "task_cid": migration["prior_task_cids"]["SAWM-001"],
                    "task_alias": "SAWM-001",
                    "goal_cid": migration["prior_goal_cids"]["SAWM-G011"],
                    "previous_status": "blocked",
                    "status": "todo",
                    "revision": 7,
                    "receipt": recovery_receipt,
                    "recorded_at": recovery_event["recorded_at"],
                }
                if (
                    recovery_event["event_type"]
                    != "intent.task_status_changed"
                    or recovery_event["task_cid"]
                    != migration["prior_task_cids"]["SAWM-001"]
                    or recovery_event["body"].get("subject_id")
                    != migration["prior_task_cids"]["SAWM-001"]
                    or (recovery_event["body"].get("body") or {})
                    != expected_recovery_body
                ):
                    raise MigrationRequired("operator task-recovery event differs")
                completion_rows = [
                    (str(row[0]), str(row[1]))
                    for row in connection.execute(
                        "SELECT receipt_cid, task_cid FROM completion_receipts "
                        "ORDER BY receipt_cid"
                    ).fetchall()
                ]
                if completion_rows != [
                    (
                        migration["prior_completion_receipt_cid"],
                        migration["prior_task_cids"]["SAWM-000"],
                    )
                ]:
                    raise MigrationRequired(
                        "operational recovery changed completion evidence"
                    )
                migration_details = {
                    "migration_digest": migration_digest,
                    "migration_evidence_id": migration_evidence_id,
                    "plan_migration_event_id": plan_event["event_id"],
                    "migration_evidence_event_id": evidence_event["event_id"],
                    "migration_event_watermark": int(
                        migration["prior_event_watermark"]
                    )
                    + _M6_MIGRATION_EVENT_OFFSET,
                    "operational_validation_receipts": operational_receipts,
                    "operational_validation_event_ids": operational_event_ids,
                    "operational_validation_first_event_watermark": int(
                        migration["prior_event_watermark"]
                    )
                    + _M6_OPERATIONAL_FIRST_OFFSET,
                    "operational_validation_last_event_watermark": int(
                        migration["prior_event_watermark"]
                    )
                    + _M6_OPERATIONAL_LAST_OFFSET,
                    "task_recovery_receipt": recovery_receipt,
                    "task_recovery_event_id": recovery_event["event_id"],
                    "target_event_watermark": expected_target_watermark,
                    "accepted_definition_changes": 0,
                    "accepted_completion_changes": 0,
                    "operational_validation_revision_changes": (
                        _M6_OPERATIONAL_TASK_COUNT
                    ),
                    "nonterminal_task_status_recovery_changes": 1,
                }

        # Replay in a short private directory, never beside either accepted
        # authority. Rebuild itself is the landed deterministic verifier.
        with tempfile.TemporaryDirectory(
            prefix="sawm-r2-replay-", dir="/tmp"
        ) as temp_dir:
            replay_copy = Path(temp_dir) / "control.duckdb"
            shutil.copyfile(path, replay_copy)
            replay = DatabaseTaskSource(
                replay_copy,
                install_schema=False,
                repository_tree_id=str(population["repository_tree_id"]),
                plan_root_cid=str(population["plan_root_cid"]),
            )
            try:
                projection_matches = replay.projection_matches_events()
            finally:
                replay.close()
        if not projection_matches:
            raise MigrationRequired(
                "event replay projection differs from the accepted projection"
            )
        if (
            require_migration
            and str(snap.projection_cid) != _M6_EXPECTED_PROJECTION_CID
        ):
            raise MigrationRequired("M6 deterministic projection identity differs")
        return {
            "valid": True,
            "task_count": snap.task_count,
            "goal_count": snap.goal_count,
            "projection_cid": snap.projection_cid,
            "event_watermark": snap.event_cursor,
            "projection_matches_events": True,
            "statuses": status_by_alias,
            "revisions": revision_by_alias,
            **migration_details,
        }
    finally:
        source.close()


def _validator_report(root: Path, script: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(root / script), "--check-all", "--repo-root", str(root)],
        cwd=root, check=False, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "ipfs_datasets_py"), str(root / "ipfs_kit_py"), str(root)))},
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"{script} did not emit deterministic JSON: {exc}; stderr={result.stderr[-1000:]}") from exc
    if result.returncode or report.get("valid") is not True:
        raise MaterializationError(f"{script} failed current-tree validation: {report.get('errors')}")
    return report


def _task_recovery_receipt(population: Mapping[str, Any]) -> dict[str, Any]:
    migration = population["migration_inventory"]
    failure = migration["preprovider_task_failure"]
    task = next(
        item for item in population["taskboard"] if item["task_id"] == "SAWM-001"
    )
    prior_validations = tuple(
        (str(command),) for command in task["validation_commands"]
    )
    operational_validations, replacement_count = _operational_validation_commands(
        prior_validations,
        task_alias="SAWM-001",
    )
    operational_receipt = _operational_validation_receipt(
        population,
        task_alias="SAWM-001",
        task_cid=str(failure["task_cid"]),
        expected_status="blocked",
        expected_revision=5,
        prior_validations=prior_validations,
        operational_validations=operational_validations,
        replacement_count=replacement_count,
    )
    return {
        "schema": "sawm/operator-settled-preprovider-task-recovery@1",
        "operation": "operator_settled_preprovider_task_requeue",
        "authority_class": "operator_control_plane",
        "migration_revision": migration["migration_revision"],
        "task_cid": failure["task_cid"],
        "task_alias": failure["task_alias"],
        "task_definition_cid": migration["prior_task_cids"]["SAWM-001"],
        "expected_status": "blocked",
        "expected_revision": 6,
        "status": "todo",
        "target_revision": 7,
        "frozen_status": failure["canonical_task_status"],
        "frozen_revision": int(failure["canonical_task_revision"]),
        "prior_event_id": failure["canonical_settlement_event_id"],
        "prior_event_watermark": int(failure["canonical_event_watermark"]),
        "prior_projection_cid": failure["canonical_projection_cid"],
        "preprovider_failure_cid": _identity(failure),
        "failure_kind": failure["failure_kind"],
        "failure_reason": failure["failure_reason"],
        "settlement_id": failure["settlement_id"],
        "settlement_failure_kind": failure["settlement_failure_kind"],
        "failure_payload_digest": failure["failure_payload_digest"],
        "attempt_id": failure["attempt_id"],
        "claim_id": failure["claim_id"],
        "lease_id": failure["lease_id"],
        "owner_session_id": failure["owner_session_id"],
        "fencing_token": int(failure["fencing_token"]),
        "fence_epoch": int(failure["fence_epoch"]),
        "operational_validation_revision_receipt_cid": operational_receipt[
            "receipt_cid"
        ],
        "settlement_verified": True,
        "historical_execution_sidecar_copied": False,
        "historical_coordination_sidecar_copied": False,
        "automatic_retry_admitted": False,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "nonterminal_task_status_recovery_changes": 1,
        "implementation_provider_invoked": False,
        "effect_claim_recorded": False,
        "implementation_commit_created": False,
        "merge_attempted": False,
        "task_completed": False,
        "worker_self_approval": False,
    }


def _migration_body(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    operational_receipts: dict[str, str] = {}
    for task in population["taskboard"]:
        alias = str(task["task_id"])
        if alias not in _operational_task_aliases():
            continue
        prior_status = "blocked" if alias == "SAWM-001" else "todo"
        prior_revision = 5 if alias == "SAWM-001" else 1
        prior_validations = tuple(
            (str(command),) for command in task["validation_commands"]
        )
        operational_validations, replacement_count = (
            _operational_validation_commands(
                prior_validations,
                task_alias=alias,
            )
        )
        operational_receipts[alias] = _operational_validation_receipt(
            population,
            task_alias=alias,
            task_cid=str(task["task_cid"]),
            expected_status=prior_status,
            expected_revision=prior_revision,
            prior_validations=prior_validations,
            operational_validations=operational_validations,
            replacement_count=replacement_count,
        )["receipt_cid"]
    return {
        "schema": "sawm/operator-control-plane-source-migration@3",
        "migration_revision": migration["migration_revision"],
        "board_namespace": NAMESPACE,
        "plan_revision": REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "operator_task_cid": migration["prior_task_cids"]["SAWM-000"],
        "prior_store_id": migration["prior_store_id"],
        "target_store_id": migration["target_store_id"],
        "prior_control_store_sha256": migration["prior_control_store_sha256"],
        "prior_event_prefix_sha256": migration["prior_event_prefix_sha256"],
        "prior_event_watermark": migration["prior_event_watermark"],
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "prior_plan_revision": migration["prior_plan_revision"],
        "target_plan_revision": migration["target_plan_revision"],
        "migration_history_receipt_cids": [
            entry["migration_receipt_cid"]
            for entry in migration["migration_history"]
        ],
        "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
        "prior_head": migration["prior_source_head"],
        "prior_tree": migration["prior_source_tree"],
        "current_head": population["source_binding"]["head"],
        "current_tree": population["source_binding"]["tree"],
        "bounded_control_plane_repair_paths": migration["bounded_control_plane_repair_paths"],
        "bounded_control_plane_repair_sha256": {
            path: population["source_binding"]["control_sha256"][path]
            for path in migration["bounded_control_plane_repair_paths"]
        },
        "quack_extension_pin": dict(config["quack_owner"]["pinned_extension"]),
        "preworker_launch_failure": dict(migration["preworker_launch_failure"]),
        "preprovider_task_failure": dict(migration["preprovider_task_failure"]),
        "validation_runtime": dict(config["validation_runtime"]),
        "operational_validation_requeue": dict(
            config["prior_materialization"]["operational_validation_requeue"]
        ),
        "operational_validation_revision_receipt_cids": operational_receipts,
        "nonterminal_task_recovery_receipt": _task_recovery_receipt(population),
        "validator_digest": validation_digest,
        "supersession_reason": migration["supersession_reason"],
        "supersession_mode": _M6_SUPERSESSION_MODE,
        "prior_authority_preserved": True,
        "accepted_task_definitions_rewritten": False,
        "accepted_goal_definitions_rewritten": False,
        "accepted_completion_changes": 0,
        "operational_validation_revision_changes": _M6_OPERATIONAL_TASK_COUNT,
        "nonterminal_task_status_recovery_changes": 1,
        "operator_completion_replayed": False,
        "worker_self_approval": False,
    }


def _migration_plan_delta(population: Mapping[str, Any]) -> dict[str, Any]:
    migration = population["migration_inventory"]
    return {
        "kind": str(migration["migration_kind"]),
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "prior_migration_receipt_cids": [
            entry["migration_receipt_cid"]
            for entry in migration["migration_history"]
        ],
        "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "operational_validation_revision_changes": _M6_OPERATIONAL_TASK_COUNT,
        "operational_validation_task_identity_manifest_cid": (
            _M6_TASK_IDENTITY_MANIFEST_CID
        ),
        "operational_validation_token_from": _M6_VALIDATION_TOKEN_FROM,
        "operational_validation_token_to": _M6_VALIDATION_TOKEN_TO,
        "nonterminal_task_status_recovery_changes": 1,
        "recovered_task_cid": migration["prior_task_cids"]["SAWM-001"],
        "recovered_task_frozen_status": "blocked",
        "recovered_task_frozen_revision": 5,
        "recovered_task_expected_status": "blocked",
        "recovered_task_expected_revision": 6,
        "recovered_task_status": "todo",
        "recovered_task_revision": 7,
    }


def _projection_at_watermark(
    path: Path,
    population: Mapping[str, Any],
    watermark: int,
) -> str:
    """Rebuild a bounded historical projection on a disposable copy."""

    with tempfile.TemporaryDirectory(prefix="sawm-r2-prefix-", dir="/tmp") as temp_dir:
        copy = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(path, copy)
        import duckdb
        connection = duckdb.connect(str(copy))
        try:
            connection.execute(
                "DELETE FROM domain_events WHERE global_sequence > ?",
                [int(watermark)],
            )
        finally:
            connection.close()
        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )
        source = DatabaseTaskSource(
            copy,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        try:
            snapshot = source.rebuild_from_events()
            if int(snapshot.event_cursor) != int(watermark):
                raise MigrationRequired("historical migration projection watermark differs")
            return str(snapshot.projection_cid)
        finally:
            source.close()


def _expected_migration_receipt(
    root: Path,
    target: Path,
    population: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    migration_watermark = (
        int(migration["prior_event_watermark"]) + _M6_MIGRATION_EVENT_OFFSET
    )
    operational_first_watermark = (
        int(migration["prior_event_watermark"]) + _M6_OPERATIONAL_FIRST_OFFSET
    )
    operational_last_watermark = (
        int(migration["prior_event_watermark"]) + _M6_OPERATIONAL_LAST_OFFSET
    )
    target_watermark = (
        int(migration["prior_event_watermark"]) + _M6_EVENT_SUFFIX_LENGTH
    )
    recovery_receipt = _task_recovery_receipt(population)
    receipt = {
        "schema": "sawm/non-authoritative-migration-receipt@3",
        "authoritative": False,
        "database_is_authority": True,
        "migration_revision": migration["migration_revision"],
        "program_definition_cid": population["program_definition_cid"],
        "migration_projection_cid": _projection_at_watermark(
            target, population, migration_watermark
        ),
        "migration_event_watermark": migration_watermark,
        "operational_validation_projection_cid": _projection_at_watermark(
            target, population, operational_last_watermark
        ),
        "operational_validation_first_event_watermark": (
            operational_first_watermark
        ),
        "operational_validation_last_event_watermark": operational_last_watermark,
        "operational_validation_revision_changes": _M6_OPERATIONAL_TASK_COUNT,
        "operational_validation_receipt_cids": {
            alias: receipt["receipt_cid"]
            for alias, receipt in sorted(
                verified["operational_validation_receipts"].items()
            )
        },
        "operational_validation_event_ids": dict(
            sorted(verified["operational_validation_event_ids"].items())
        ),
        "projection_cid": verified["projection_cid"],
        "target_event_watermark": target_watermark,
        "validation_digest": validation_digest,
        "migration_digest": verified["migration_digest"],
        "migration_evidence_id": verified["migration_evidence_id"],
        "plan_migration_event_id": verified["plan_migration_event_id"],
        "migration_evidence_event_id": verified["migration_evidence_event_id"],
        "task_recovery_event_id": verified["task_recovery_event_id"],
        "task_recovery_receipt_cid": _identity(recovery_receipt),
        "task_recovery_receipt": recovery_receipt,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "nonterminal_task_status_recovery_changes": 1,
        "prior_control_store_sha256": migration["prior_control_store_sha256"],
        "prior_event_prefix_sha256": migration["prior_event_prefix_sha256"],
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "prior_migration_receipt_cids": [
            entry["migration_receipt_cid"]
            for entry in migration["migration_history"]
        ],
        "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
        "preprovider_failure_cid": _identity(migration["preprovider_task_failure"]),
        "prior_database_path": migration["prior_store_id"],
        "database_path": str(target.relative_to(root)),
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def _ensure_migration_receipt(
    root: Path,
    target: Path,
    population: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    expected = _expected_migration_receipt(
        root, target, population, verified, validation_digest
    )
    receipt_path = target.parent / "migration-receipt.json"
    if receipt_path.exists():
        observed = _load_json(receipt_path)
        if observed != expected:
            raise MigrationRequired("external migration receipt differs from authority")
        claimed = str(observed.get("receipt_cid") or "")
        unhashed = dict(observed)
        unhashed.pop("receipt_cid", None)
        if claimed != _identity(unhashed):
            raise MigrationRequired("external migration receipt CID does not rehash")
        return observed

    # A persistent advisory lock survives as an inert inode, while its kernel
    # ownership never survives a crashed process.  This lets a later verifier
    # reconstruct a missing non-authoritative receipt without deleting a lock
    # that might belong to a concurrent live materializer.
    lock = receipt_path.with_name(f".{receipt_path.name}.publish.lock")
    fd = os.open(
        lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError("migration receipt lock is not a regular file")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the migration receipt publication lock"
                    ) from exc
                time.sleep(0.02)
        if receipt_path.exists():
            observed = _load_json(receipt_path)
            if observed != expected:
                raise MigrationRequired("concurrent migration receipt differs")
            return observed
        temporary = receipt_path.with_name(
            f".{receipt_path.name}.{os.getpid()}.tmp"
        )
        out = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            payload = _canonical(expected) + b"\n"
            view = memoryview(payload)
            while view:
                written = os.write(out, view)
                view = view[written:]
            os.fsync(out)
        finally:
            os.close(out)
        os.replace(temporary, receipt_path)
        directory = os.open(
            receipt_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return expected
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _ducklake_projection(root: Path, config: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    policy = config.get("ducklake_history_projection") or {}
    receipt_path = root / str(policy.get("receipt_path"))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": "sawm/ducklake-history-projection-receipt@1", "authority": False,
        "scheduling_prerequisite": False, "completion_prerequisite": False,
        "install_attempted": False, "network_used": False,
    }
    connection = None
    try:
        import duckdb
        from ipfs_accelerate_py.agent_supervisor.integrations.ducklake_history_projection import (
            project_history,
        )
        connection = duckdb.connect(":memory:")
        connection.execute("SET autoinstall_known_extensions = false")
        connection.execute("SET autoload_known_extensions = false")
        row = connection.execute("SELECT installed, install_path FROM duckdb_extensions() WHERE extension_name = 'ducklake'").fetchone()
        if not row or not bool(row[0]) or not str(row[1] or ""):
            raise RuntimeError("ducklake_extension_not_locally_installed")
        connection.execute("LOAD ducklake")  # local LOAD only; INSTALL is forbidden
        catalog = (root / str(policy["catalog_path"])).resolve()
        data = (root / str(policy["data_path"])).resolve()
        catalog.parent.mkdir(parents=True, exist_ok=True)
        data.mkdir(parents=True, exist_ok=True)
        def literal(value: Path) -> str:
            return "'" + str(value).replace("'", "''") + "'"
        connection.execute(
            "ATTACH " + literal(Path("ducklake:" + str(catalog)))
            + " AS sawm_history (DATA_PATH " + literal(data) + ")"
        )
        connection.execute("CREATE TABLE IF NOT EXISTS sawm_history.control_history (program_definition_cid VARCHAR, projection_cid VARCHAR, authoritative BOOLEAN)")
        connection.execute("INSERT INTO sawm_history.control_history VALUES (?, ?, false)", [record["program_definition_cid"], record["projection_cid"]])
        projection = dict(project_history({"receipt": record}))
        receipt.update({"status": "available", "typed_unavailability": None,
                        "projection": projection, "catalog_path": str(catalog), "data_path": str(data)})
    except Exception as exc:
        receipt.update({"status": "typed_unavailability", "typed_unavailability": type(exc).__name__ + ": " + str(exc), "projection": None})
    finally:
        if connection is not None:
            connection.close()
    receipt["receipt_cid"] = _identity(receipt)
    temporary = receipt_path.with_name(receipt_path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(_canonical(receipt) + b"\n")
    os.replace(temporary, receipt_path)
    return receipt


def _assert_fresh_successor_operational_state(target: Path) -> None:
    """Refuse to adopt execution/coordination state from an earlier launch."""

    prohibited = (
        target.with_name("control.execution.duckdb"),
        target.with_name("control.coordination.duckdb"),
        target.with_name(f".{target.name}.state-owner.json"),
        target.parent / "state",
        target.parent / "events",
        target.parent / "registry",
        target.parent / "worktrees",
        target.parent / "merge-queue",
        target.parent / "quack-owner",
    )
    present = [str(path) for path in prohibited if path.exists()]
    if present:
        raise MaterializationError(
            "successor execution/coordination authority is not fresh: "
            + json.dumps(present, sort_keys=True, separators=(",", ":"))
        )


def materialize(repo_root: Path | str = REPO_ROOT, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_json(config_file)
    population = build_population(root)
    _assert_committed_clean_source(root, population)
    migration = population["migration_inventory"]
    target = (root / str(config["database_program"]["store_id"])).resolve()
    if str(config["database_program"]["store_id"]) != str(migration["target_store_id"]):
        raise MaterializationError("scheduler target differs from the sealed migration target")
    if (
        int(config["database_program"].get("store_generation") or 0)
        != int(migration["target_generation"])
        or str(config["quack_owner"].get("database_path") or "")
        != str(migration["target_store_id"])
        or str(config["quack_owner"].get("store_id") or "")
        != str(migration["target_store_id"])
    ):
        raise MaterializationError("scheduler successor generation/owner binding differs")
    if not target.is_relative_to(root):
        raise MaterializationError("migration target escapes the repository root")
    dependency = _validator_report(root, "scripts/validate_semantic_addressed_world_model_dependencies.py")
    board = _validator_report(root, "scripts/validate_semantic_addressed_world_model_board.py")
    validation_digest = _identity({"dependency": dependency, "board": board,
                                   "program_definition_cid": population["program_definition_cid"]})
    prior = _verify_prior_store(root, config, population)
    prior_path = Path(prior["database_path"])
    if target.exists():
        verified = _verify_store(
            target,
            population,
            require_operator_complete=True,
            require_migration=True,
            migration_config=config,
            expected_validation_digest=validation_digest,
        )
        receipt = _ensure_migration_receipt(
            root, target, population, verified, validation_digest
        )
        return {"schema": SCHEMA, "valid": True, "action": "verified_existing_noop",
                "migration_required": False, "database_path": str(target),
                "program_definition_cid": population["program_definition_cid"],
                "receipt": receipt, **verified}

    _assert_fresh_successor_operational_state(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    preserved_stages = sorted(target.parent.glob(target.name + ".installing.*"))
    if preserved_stages:
        raise MaterializationError(
            "preserved prior staging attempt requires inspection: "
            + ", ".join(str(item) for item in preserved_stages)
        )
    stage = target.with_name(target.name + f".installing.{os.getpid()}.{validation_digest[-12:]}")
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    # Preserve only the accepted canonical control store. M5 execution and
    # coordination sidecars remain immutable historical evidence and are
    # deliberately never copied into the successor authority.
    shutil.copy2(prior_path, stage)
    if _store_sha256(stage) != migration["prior_control_store_sha256"]:
        raise MaterializationError("staged prior authority copy differs before migration")
    source = DatabaseTaskSource(stage, install_schema=False,
                                repository_tree_id=str(population["repository_tree_id"]),
                                plan_root_cid=str(population["plan_root_cid"]),
                                owner_id="sawm-r2-source-migrator")
    try:
        operator = source.get_task("SAWM-000")
        if operator is None or operator.status != "completed" or operator.revision != 2:
            raise MaterializationError("preserved SAWM-000 completion authority differs")
        migration_body = _migration_body(population, config, validation_digest)
        migration_digest = _identity(migration_body)
        plan_delta = _migration_plan_delta(population)
        plan_receipt = source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=int(migration["prior_plan_revision"]),
            body={
                "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
                "source_migration_revision": migration["migration_revision"],
                "source_migration_digest": migration_digest,
                "supersession_mode": _M6_SUPERSESSION_MODE,
            },
            delta=plan_delta,
        )
        evidence = source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_source_migration",
            digest=migration_digest,
            body=migration_body,
        )
        unchanged = source.get_task(operator.task_cid)
        if unchanged is None or unchanged.status != "completed" or unchanged.revision != 2:
            raise MaterializationError("source migration changed accepted SAWM-000 state")
        operational_event_ids: dict[str, str] = {}
        operational_receipts: dict[str, dict[str, Any]] = {}
        for alias in _operational_task_aliases():
            prior_task = source.intent.get_task(alias)
            if prior_task is None:
                raise MaterializationError(
                    f"preserved operational task is missing: {alias}"
                )
            expected_status = "blocked" if alias == "SAWM-001" else "todo"
            expected_revision = 5 if alias == "SAWM-001" else 1
            if (
                prior_task["task_cid"] != migration["prior_task_cids"][alias]
                or prior_task["status"] != expected_status
                or int(prior_task["revision"]) != expected_revision
            ):
                raise MaterializationError(
                    f"preserved operational task precondition differs: {alias}"
                )
            prior_validations = tuple(
                tuple(str(part) for part in item.get("argv") or ())
                for item in prior_task["validations"]
            )
            operational_validations, replacement_count = (
                _operational_validation_commands(
                    prior_validations,
                    task_alias=alias,
                )
            )
            operational_receipt = _operational_validation_receipt(
                population,
                task_alias=alias,
                task_cid=str(prior_task["task_cid"]),
                expected_status=expected_status,
                expected_revision=expected_revision,
                prior_validations=prior_validations,
                operational_validations=operational_validations,
                replacement_count=replacement_count,
            )
            upsert = source.intent.upsert_task(
                task_cid=str(prior_task["task_cid"]),
                task_alias=str(prior_task["task_alias"]),
                goal_cid=str(prior_task["goal_cid"]),
                ordinal=int(prior_task["ordinal"]),
                status=str(prior_task["status"]),
                priority=str(prior_task["priority"]),
                plan_cid=str(prior_task["plan_cid"]),
                objective_id=str(prior_task["objective_id"]),
                body={
                    **dict(prior_task["body"]),
                    "operational_validation_revision": operational_receipt,
                },
                identity=dict(prior_task["identity"]),
                expected_revision=expected_revision,
                dependencies=list(prior_task["dependencies"]),
                outputs=[dict(item["effect"]) for item in prior_task["outputs"]],
                acceptance=[
                    dict(item["evidence_policy"])
                    for item in prior_task["acceptance"]
                ],
                validations=[
                    {
                        **dict(prior["policy"]),
                        "argv": list(argv),
                    }
                    for prior, argv in zip(
                        prior_task["validations"],
                        operational_validations,
                        strict=True,
                    )
                ],
            )
            if (
                not upsert.changed
                or upsert.event_type != "intent.task_upserted"
                or int(upsert.revision) != expected_revision + 1
            ):
                raise MaterializationError(
                    f"operational validation revision was not admitted: {alias}"
                )
            operational_event_ids[alias] = upsert.event_id
            operational_receipts[alias] = operational_receipt
        candidate = source.get_task("SAWM-001")
        if (
            candidate is None
            or candidate.status != "blocked"
            or candidate.revision != 6
            or candidate.task_cid != migration["prior_task_cids"]["SAWM-001"]
        ):
            raise MaterializationError(
                "operationally revised SAWM-001 recovery precondition differs"
            )
        recovery_receipt = _task_recovery_receipt(population)
        recovery = source.compare_and_set_status(
            candidate.task_cid,
            expected_revision=6,
            status="todo",
            receipt=recovery_receipt,
        )
        if (
            not recovery.changed
            or recovery.previous_status != "blocked"
            or recovery.task.status != "todo"
            or recovery.task.revision != 7
        ):
            raise MaterializationError("operator SAWM-001 recovery CAS was not admitted")
    finally:
        source.close()
    verified = _verify_store(
        stage,
        population,
        require_operator_complete=True,
        require_migration=True,
        migration_config=config,
        expected_validation_digest=validation_digest,
    )
    expected_watermark = (
        int(migration["prior_event_watermark"]) + _M6_EVENT_SUFFIX_LENGTH
    )
    if int(verified["event_watermark"]) != expected_watermark:
        raise MaterializationError(
            f"source migration emitted an unexpected event suffix: {verified['event_watermark']}"
        )
    if _store_sha256(prior_path) != migration["prior_control_store_sha256"]:
        raise MaterializationError("source migration changed the preserved prior authority")
    lock = target.with_name(target.name + ".publish.lock")
    fd = None
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        if target.exists():
            raise MigrationRequired("another writer published a control store; inspect it append-only")
        os.link(stage, target)
        os.unlink(stage)
    finally:
        if fd is not None:
            os.close(fd)
            lock.unlink(missing_ok=True)
    receipt = _ensure_migration_receipt(
        root, target, population, verified, validation_digest
    )
    history = _ducklake_projection(root, config, {"program_definition_cid": population["program_definition_cid"],
                                                   "projection_cid": verified["projection_cid"]})
    report = {"schema": SCHEMA, "valid": True, "action": "migrated_append_only",
            "migration_required": False, "database_path": str(target),
            "program_definition_cid": population["program_definition_cid"],
            "validation_digest": validation_digest, "prior_authority": prior,
            "plan_migration_event_id": plan_receipt.event_id,
            "migration_evidence_event_id": evidence.event_id,
            "operational_validation_event_ids": operational_event_ids,
            "operational_validation_receipts": operational_receipts,
            "task_recovery_event_id": recovery.receipt_cid,
            "task_recovery_receipt": recovery_receipt,
            "migration_digest": migration_digest,
            "ducklake_history": history, **verified}
    return {**report, "receipt": receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("materialize", "render", "check"), default="materialize")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        population = build_population(args.repo_root)
        if args.action == "render":
            report: dict[str, Any] = {"valid": True, **population}
        elif args.action == "check":
            root = Path(args.repo_root).resolve()
            config = _load_json(args.config if args.config.is_absolute() else root / args.config)
            _assert_committed_clean_source(root, population)
            dependency = _validator_report(
                root, "scripts/validate_semantic_addressed_world_model_dependencies.py"
            )
            board = _validator_report(
                root, "scripts/validate_semantic_addressed_world_model_board.py"
            )
            validation_digest = _identity(
                {"dependency": dependency, "board": board,
                 "program_definition_cid": population["program_definition_cid"]}
            )
            prior = _verify_prior_store(root, config, population)
            path = root / str(config["database_program"]["store_id"])
            verified = _verify_store(
                path,
                population,
                require_operator_complete=True,
                require_migration=True,
                migration_config=config,
                expected_validation_digest=validation_digest,
            )
            receipt = _ensure_migration_receipt(
                root, path, population, verified, validation_digest
            )
            report = {"schema": SCHEMA, "valid": True, "action": "checked",
                      "database_path": str(path), "program_definition_cid": population["program_definition_cid"],
                      "prior_authority": prior, "receipt": receipt, **verified}
        else:
            report = materialize(args.repo_root, args.config)
    except MigrationRequired as exc:
        report = {"schema": SCHEMA, "valid": False, "action": "migration_required",
                  "migration_required": True, "error": str(exc)}
    except Exception as exc:
        report = {"schema": SCHEMA, "valid": False, "action": "failed",
                  "migration_required": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
