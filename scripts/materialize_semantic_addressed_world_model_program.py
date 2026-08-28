#!/usr/bin/env python3
"""Render, migrate, and verify the append-only SAWM R2 supervisor program.

The default action verifies the sealed prior authority, copies its exact bytes
to a confined stage, appends the bounded source-migration suffix through the
landed intent repository, and atomically publishes the successor store.  It
never rematerializes accepted task/goal definitions or completion evidence.
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
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_quack_protocol.py",
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
    if migration.get("migration_revision") != "SAWM-R2-M1":
        raise MaterializationError("the sealed append-only migration revision is missing")
    prior_goal_cids = migration.get("prior_goal_cids")
    prior_task_cids = migration.get("prior_task_cids")
    if not isinstance(prior_goal_cids, dict) or not isinstance(prior_task_cids, dict):
        raise MaterializationError("prior SAWM identity maps are incomplete")

    goal_cids: dict[str, str] = {}
    goal_definitions: dict[str, dict[str, Any]] = {}
    for ordinal, card in enumerate(goals, 1):
        definition = {
            "schema": "sawm/goal-definition@1", "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "goal_id": card.identifier,
            "title": card.title, "ordinal": ordinal,
            "metadata": _metadata_payload(card),
            "source_binding_cid": migration["prior_source_binding_cid"],
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
        "source_binding_cid": migration["prior_source_binding_cid"],
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
            "source_binding_cid": migration["prior_source_binding_cid"],
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
        "source_binding_cid": migration["prior_source_binding_cid"],
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


def _verify_prior_store(
    root: Path,
    config: Mapping[str, Any],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    migration = population["migration_inventory"]
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
    ):
        raise MaterializationError("scheduler prior-authority policy differs from its inventory")
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
        operator = next(row for row in task_rows if row[0] == "SAWM-000")
        if str(operator[2]) != migration["prior_operator_status"] or int(operator[3]) != 2:
            raise MigrationRequired("sealed prior operator status/revision differs")

        plan = connection.execute(
            "SELECT revision, body_json FROM plans WHERE plan_cid = ?",
            [migration["prior_plan_root_cid"]],
        ).fetchone()
        if plan is None or int(plan[0]) != 1:
            raise MigrationRequired("sealed prior plan revision differs")
        plan_body = json.loads(str(plan[1]))
        if plan_body.get("source_binding_cid") != migration["prior_source_binding_cid"]:
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
    finally:
        connection.close()

    _verify_receipt_anchor(
        prior.parent / "materialization-receipt.json",
        str(migration["prior_materialization_receipt_cid"]),
    )
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
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    schema = verify_datasets_authoritative_operational_schema(path)
    if schema.get("valid") is not True:
        raise MigrationRequired("existing database does not verify as datasets-authoritative operational schema")
    source = DatabaseTaskSource(path, install_schema=False,
                                repository_tree_id=str(population["repository_tree_id"]),
                                plan_root_cid=str(population["plan_root_cid"]))
    try:
        migration_details: dict[str, Any] = {}
        snap = source.snapshot()
        expected_tasks = list(population["taskboard"])
        expected_goals = list(population["objectives"])
        if snap.task_count != 45 or snap.goal_count != 29 or snap.plan_root_cid != population["plan_root_cid"]:
            raise MigrationRequired(f"population counts/root conflict: tasks={snap.task_count} goals={snap.goal_count} root={snap.plan_root_cid}")
        status_by_alias: dict[str, str] = {}
        for expected in expected_tasks:
            observed = source.get_task(str(expected["task_cid"]))
            expected_outputs = [str(item["declared_path"]) for item in expected["outputs"]]
            expected_acceptance = [str(item["criterion"]) for item in expected["acceptance_criteria"]]
            expected_validations = [[str(command)] for command in expected["validation_commands"]]
            if (
                observed is None
                or observed.task_alias != expected["task_id"]
                or observed.body.get("definition_cid") != expected["definition_cid"]
                or sorted(observed.dependencies) != sorted(expected["depends_on"])
                or [str((item.get("effect") or {}).get("declared_path")) for item in observed.outputs] != expected_outputs
                or [str(item.get("criterion")) for item in observed.acceptance] != expected_acceptance
                or [list(item.get("argv") or ()) for item in observed.validations] != expected_validations
            ):
                raise MigrationRequired(f"task definition conflict: {expected['task_id']}")
            status_by_alias[observed.task_alias] = observed.status
        for expected in expected_goals:
            observed = source.get_goal(str(expected["goal_cid"]))
            if observed is None or str(observed.get("goal_alias")) != expected["goal_id"] or (observed.get("body") or {}).get("definition_cid") != expected["definition_cid"]:
                raise MigrationRequired(f"goal definition conflict: {expected['goal_id']}")
        if require_operator_complete and status_by_alias.get("SAWM-000") not in {"completed", "complete", "done"}:
            raise MigrationRequired("SAWM-000 lacks an admitted completion CAS")
        migration = population["migration_inventory"]
        if require_migration:
            if migration_config is None or not expected_validation_digest:
                raise MaterializationError(
                    "migration verification requires exact config and validator binding"
                )
            plan = source.plans.get(str(population["plan_root_cid"]))
            if plan is None or int(plan.get("revision") or 0) < 2:
                raise MigrationRequired("append-only source migration plan revision is missing")
            evidence = source.intent._connection(write=False)
            with evidence as connection:
                migration_rows = connection.execute(
                    "SELECT evidence_id, digest, body_json, created_at FROM evidence_nodes "
                    "WHERE task_cid = ? AND evidence_kind = ? ORDER BY created_at",
                    [migration["prior_task_cids"]["SAWM-000"],
                     "operator_control_plane_source_migration"],
                ).fetchall()
                if len(migration_rows) != 1:
                    raise MigrationRequired("exactly one source migration receipt is required")
                migration_evidence_id = str(migration_rows[0][0])
                migration_digest = str(migration_rows[0][1])
                migration_body = json.loads(str(migration_rows[0][2]))
                migration_created_at = str(migration_rows[0][3])
                expected_migration_body = _migration_body(
                    population, migration_config, expected_validation_digest
                )
                if migration_body != expected_migration_body:
                    raise MigrationRequired("source migration receipt binding differs")
                if migration_digest != _identity(migration_body):
                    raise MigrationRequired("source migration evidence digest differs")
                from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
                    content_identity,
                )
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
                    "WHERE plan_cid = ? AND revision IN (1, 2) ORDER BY revision",
                    [population["plan_root_cid"]],
                ).fetchall()
                if len(revision_rows) != 2 or [int(row[0]) for row in revision_rows] != [1, 2]:
                    raise MigrationRequired("exact source migration plan revisions are missing")
                prior_plan_body = json.loads(str(revision_rows[0][1]))
                plan_delta = _migration_plan_delta(population)
                expected_plan_body = {
                    **prior_plan_body,
                    "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
                    "source_migration_revision": migration["migration_revision"],
                    "source_migration_digest": migration_digest,
                    "supersession_mode": "source_authority_revision_only",
                    "last_delta": plan_delta,
                }
                observed_plan_body = json.loads(str(revision_rows[1][1]))
                plan_recorded_at = str(revision_rows[1][2])
                if observed_plan_body != expected_plan_body:
                    raise MigrationRequired("source migration plan revision body differs")

                prefix_digest, prefix_count = _event_prefix_digest(
                    connection, int(migration["prior_event_watermark"])
                )
                if (
                    prefix_count != int(migration["prior_event_watermark"])
                    or prefix_digest != migration["prior_event_prefix_sha256"]
                ):
                    raise MigrationRequired("accepted event prefix changed during migration")
                suffix_rows = connection.execute(
                    "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
                    "task_cid, attempt_id, session_id, recorded_at, body_json "
                    "FROM domain_events WHERE global_sequence IN (?, ?) "
                    "ORDER BY global_sequence",
                    [int(migration["prior_event_watermark"]) + 1,
                     int(migration["prior_event_watermark"]) + 2],
                ).fetchall()
                if len(suffix_rows) != 2:
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
                    computed = content_identity(
                        {
                            "stream_id": event["stream_id"],
                            "sequence": event["sequence"],
                            "global_sequence": event["global_sequence"],
                            "event_type": event["event_type"],
                            "body": payload,
                        }
                    )
                    if (
                        computed != event["event_id"]
                        or set(payload)
                        != {"schema", "event_type", "subject_id", "body", "recorded_at", "owner_id"}
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
                plan_event, evidence_event = events
                expected_plan_event_body = {
                    "plan_cid": population["plan_root_cid"],
                    "goal_cid": migration["prior_goal_cids"][ROOT_GOAL],
                    "plan_alias": REVISION,
                    "status": "active",
                    "revision": 2,
                    "body": expected_plan_body,
                    "delta": plan_delta,
                    "recorded_at": plan_recorded_at,
                }
                if (
                    plan_event["event_type"] != "intent.plan_revision_appended"
                    or plan_event["task_cid"]
                    or plan_event["body"].get("subject_id") != population["plan_root_cid"]
                    or plan_event["body"].get("body") != expected_plan_event_body
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
                    or evidence_event["task_cid"] != migration["prior_task_cids"]["SAWM-000"]
                    or evidence_event["body"].get("subject_id") != migration_evidence_id
                    or evidence_event["body"].get("body") != expected_evidence_event_body
                ):
                    raise MigrationRequired("source migration evidence event differs")
                migration_details = {
                    "migration_digest": migration_digest,
                    "migration_evidence_id": migration_evidence_id,
                    "plan_migration_event_id": plan_event["event_id"],
                    "migration_evidence_event_id": evidence_event["event_id"],
                    "migration_event_watermark": int(migration["prior_event_watermark"]) + 2,
                }

        # Replay in a short private directory, never beside either accepted
        # authority. Rebuild itself is the landed deterministic verifier.
        with tempfile.TemporaryDirectory(prefix="sawm-r2-replay-", dir="/tmp") as temp_dir:
            replay_copy = Path(temp_dir) / "control.duckdb"
            shutil.copyfile(path, replay_copy)
            replay = DatabaseTaskSource(replay_copy, install_schema=False,
                                        repository_tree_id=str(population["repository_tree_id"]),
                                        plan_root_cid=str(population["plan_root_cid"]))
            try:
                projection_matches = replay.projection_matches_events()
            finally:
                replay.close()
        if not projection_matches:
            raise MigrationRequired("event replay projection differs from the accepted projection")
        return {"valid": True, "task_count": snap.task_count, "goal_count": snap.goal_count,
                "projection_cid": snap.projection_cid, "event_watermark": snap.event_cursor,
                "projection_matches_events": True, "statuses": status_by_alias,
                **migration_details}
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


def _migration_body(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    return {
        "schema": "sawm/operator-control-plane-source-migration@1",
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
        "validator_digest": validation_digest,
        "supersession_reason": migration["supersession_reason"],
        "supersession_mode": "source_authority_revision_only",
        "prior_authority_preserved": True,
        "accepted_task_definitions_rewritten": False,
        "accepted_goal_definitions_rewritten": False,
        "operator_completion_replayed": False,
        "worker_self_approval": False,
    }


def _migration_plan_delta(population: Mapping[str, Any]) -> dict[str, Any]:
    migration = population["migration_inventory"]
    return {
        "kind": "bounded_control_plane_launch_repair",
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
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
    migration_watermark = int(migration["prior_event_watermark"]) + 2
    receipt = {
        "schema": "sawm/non-authoritative-migration-receipt@1",
        "authoritative": False,
        "database_is_authority": True,
        "migration_revision": migration["migration_revision"],
        "program_definition_cid": population["program_definition_cid"],
        "projection_cid": _projection_at_watermark(
            target, population, migration_watermark
        ),
        "migration_event_watermark": migration_watermark,
        "validation_digest": validation_digest,
        "migration_digest": verified["migration_digest"],
        "migration_evidence_id": verified["migration_evidence_id"],
        "plan_migration_event_id": verified["plan_migration_event_id"],
        "migration_evidence_event_id": verified["migration_evidence_event_id"],
        "prior_control_store_sha256": migration["prior_control_store_sha256"],
        "prior_event_prefix_sha256": migration["prior_event_prefix_sha256"],
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
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
    # Preserve the accepted database, task/goal definitions, evidence, and
    # event prefix byte-for-byte before appending the source-only migration.
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
            expected_revision=1,
            body={
                "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
                "source_migration_revision": migration["migration_revision"],
                "source_migration_digest": migration_digest,
                "supersession_mode": "source_authority_revision_only",
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
    expected_watermark = int(migration["prior_event_watermark"]) + 2
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
