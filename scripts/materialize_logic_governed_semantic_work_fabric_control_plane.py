#!/usr/bin/env python3
"""Materialize and verify the LGSWF board in the existing DuckDB control plane.

The Markdown documents are a sealed bootstrap input only.  Once this command
creates the store, ``DatabaseTaskSource@1`` is task authority and the files are
never used for task-status mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "config/logic_governed_semantic_work_fabric_scheduler.json"
TASK_RE = re.compile(r"^## (LGSWF-(\d{3})) (.+)$", re.MULTILINE)
GOAL_RE = re.compile(r"^## (LGSWF-G\d{3}) (.+)$", re.MULTILINE)
META_RE = re.compile(r"^- ([^:\n]+):(?: (.*))?$", re.MULTILINE)
SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgswf-duckdb-materialization@1"
RECEIPT_RESULT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/content-addressed-receipt-result@1"
QUALIFICATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgswf-bootstrap-qualification@1"
_MANUAL_SEAL_STAGE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/manual-seal-stage-receipt@1"
)
_MANUAL_SEAL_PARTIAL_LINK_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/manual-seal-partial-evidence-link@1"
)
_MANUAL_SEAL_STAGE_GUARD_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/manual-seal-stage-guard-policy@1"
)


class MaterializationError(RuntimeError):
    """Fail-closed bootstrap materialization error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError("scheduler config root must be an object")
    program = value.get("database_program")
    if not isinstance(program, dict):
        raise MaterializationError("database_program is required")
    if program.get("task_source_kind") != "duckdb":
        raise MaterializationError("task_source_kind must be duckdb")
    if program.get("authority_mode") != "embedded":
        raise MaterializationError(
            "bootstrap materializer only admits the bounded embedded single writer"
        )
    if program.get("schema_revision") != "datasets-authoritative-operational-v1":
        raise MaterializationError(
            "bootstrap requires the datasets-authoritative operational schema"
        )
    if program.get("schema_profile") != "datasets-authoritative-operational":
        raise MaterializationError(
            "bootstrap schema profile must preserve datasets semantic authority"
        )
    if program.get("semantic_relations_permitted") is not False:
        raise MaterializationError("accelerator semantic-truth relations must be prohibited")
    writer = value.get("bootstrap_writer_policy")
    if not isinstance(writer, dict) or writer.get("maximum_processes") != 1:
        raise MaterializationError("bootstrap writer policy must cap processes at one")
    seal = value.get("bootstrap_seal")
    runtime = value.get("runtime_paths")
    if not isinstance(seal, dict) or not isinstance(runtime, dict):
        raise MaterializationError("bootstrap seal and runtime paths are required")
    evidence = str(runtime.get("evidence") or "").rstrip("/")
    expected_receipts = {
        "qualification_receipt_path": f"{evidence}/bootstrap/qualification.json",
        "receipt_path": f"{evidence}/bootstrap/duckdb-seal.json",
    }
    if any(seal.get(key) != path for key, path in expected_receipts.items()):
        raise MaterializationError(
            "bootstrap receipt paths must exactly match the runtime evidence root"
        )
    for key, path in expected_receipts.items():
        _relative_path(path, field=f"bootstrap_seal.{key}")
    return value


def _relative_path(value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise MaterializationError(f"{field} must be a safe repository-relative path")
    resolved = (ROOT / path).resolve(strict=False)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise MaterializationError(f"{field} escapes repository") from exc
    return resolved


def _metadata(body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in META_RE.finditer(body):
        key = match.group(1).strip()
        if key in result:
            raise MaterializationError(f"duplicate metadata field: {key}")
        result[key] = (match.group(2) or "").strip()
    return result


def _records(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str, str]]:
    matches = list(pattern.finditer(text))
    records: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        records.append((match.group(1), match.group(match.lastindex or 1), body))
    return records


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalized_body(
    *, task_id: str, title: str, metadata: Mapping[str, str], block: str
) -> dict[str, Any]:
    body = {
        re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_"): value
        for key, value in metadata.items()
    }
    body.update(
        {
            "task_id": task_id,
            "title": title,
            "objective": metadata.get("Objective", title),
            "completion": metadata.get("Completion", "auto"),
            "completion_contract": metadata.get("Completion contract", ""),
            "validation": metadata.get("Validation", ""),
            "validation_requirements": metadata.get("Validation requirements", ""),
            "proof_requirements": metadata.get("Proof requirements", ""),
            "acceptance": metadata.get("Acceptance", ""),
            "acceptance_criteria": metadata.get("Acceptance", ""),
            "outputs": _csv(metadata.get("Outputs", "")),
            "predicted_files": _csv(metadata.get("Predicted files", "")),
            "depends_on": _csv(metadata.get("Depends on", "")),
            "source_block_sha256": _identity(f"## {task_id} {title}{block}".encode("utf-8")),
        }
    )
    return body


def build_population(config: Mapping[str, Any]) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise MaterializationError(
            "refusing to bind a control-plane population to a dirty execution "
            "worktree; commit the exact board, policy, and source first"
        )
    taskboard_path = _relative_path(config.get("taskboard_path"), field="taskboard_path")
    objectives_path = _relative_path(config.get("objectives_path"), field="objectives_path")
    plan_path = _relative_path(config.get("plan_path"), field="plan_path")
    board_bytes = taskboard_path.read_bytes()
    objective_bytes = objectives_path.read_bytes()
    plan_bytes = plan_path.read_bytes()
    config_bytes = CONFIG_PATH.read_bytes()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    datasets_path = _relative_path(
        (config.get("source_binding") or {}).get("ipfs_datasets_submodule_path"),
        field="source_binding.ipfs_datasets_submodule_path",
    )
    datasets_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=datasets_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    datasets_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=datasets_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if datasets_head != str(
        (config.get("source_binding") or {}).get("ipfs_datasets_planning_revision") or ""
    ):
        raise MaterializationError("datasets execution base differs from its sealed head")
    base = str((config.get("source_binding") or {}).get("accelerator_required_ancestor") or "")
    if (
        subprocess.run(["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=ROOT).returncode
        != 0
    ):
        raise MaterializationError("configured accelerator base is not an ancestor")

    plan_root = _identity(
        {
            "schema": "lgswf-plan-root@1",
            "board": _identity(board_bytes),
            "objectives": _identity(objective_bytes),
            "plan": _identity(plan_bytes),
            "config": _identity(config_bytes),
            "source_head": head,
            "source_tree": tree,
        }
    )
    objective_text = objective_bytes.decode("utf-8")
    goals: list[dict[str, Any]] = []
    goal_cids: dict[str, str] = {}
    for ordinal, (goal_id, title, body_text) in enumerate(
        _records(objective_text, GOAL_RE), start=1
    ):
        metadata = _metadata(body_text)
        goal_cid = _identity(
            {
                "goal_id": goal_id,
                "title": title,
                "body_sha256": _identity(body_text.encode("utf-8")),
                "plan_root_cid": plan_root,
            }
        )
        goal_cids[goal_id] = goal_cid
        goals.append(
            {
                "goal_cid": goal_cid,
                "goal_id": goal_id,
                "goal_alias": goal_id,
                "title": title,
                "ordinal": ordinal,
                "status": "open",
                "objective_id": "objective:lgswf-root" if goal_id == "LGSWF-G000" else "",
                "objective_alias": "LGSWF-G000",
                "priority": metadata.get("Priority", "P0"),
                "body": metadata,
            }
        )
    if len(goals) != int((config.get("initial_projection") or {}).get("goal_count", -1)):
        raise MaterializationError("goal count differs from sealed projection")
    goal_edges: list[dict[str, Any]] = []
    for goal in goals:
        metadata = goal.get("body") if isinstance(goal.get("body"), Mapping) else {}
        child_alias = str(goal["goal_alias"])
        parent_alias = str(metadata.get("Parent") or "").strip()
        if parent_alias:
            if parent_alias not in goal_cids:
                raise MaterializationError(f"{child_alias} has unknown parent goal {parent_alias}")
            goal["parent_goal_cid"] = goal_cids[parent_alias]
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[parent_alias],
                    "child_goal_cid": goal_cids[child_alias],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency_alias in _csv(str(metadata.get("Depends on") or "")):
            if dependency_alias not in goal_cids:
                raise MaterializationError(
                    f"{child_alias} has unknown goal dependency {dependency_alias}"
                )
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency_alias],
                    "child_goal_cid": goal_cids[child_alias],
                    "edge_kind": "goal_dependency",
                }
            )

    task_text = board_bytes.decode("utf-8")
    tasks: list[dict[str, Any]] = []
    task_cids: dict[str, str] = {}
    parsed = _records(task_text, TASK_RE)
    for task_id, title, body_text in parsed:
        metadata = _metadata(body_text)
        task_cids[task_id] = _identity(
            {
                "task_id": task_id,
                "block_sha256": _identity(f"## {task_id} {title}{body_text}".encode("utf-8")),
                "plan_root_cid": plan_root,
                "repository_tree_id": tree,
            }
        )
    for ordinal, (task_id, title, body_text) in enumerate(parsed, start=1):
        metadata = _metadata(body_text)
        dependency_aliases = _csv(metadata.get("Depends on", ""))
        try:
            dependencies = [task_cids[item] for item in dependency_aliases]
        except KeyError as exc:
            raise MaterializationError(f"{task_id} has unknown dependency {exc}") from exc
        goal_alias = metadata.get("Subgoal ID") or metadata.get("Goal id") or "LGSWF-G000"
        if goal_alias not in goal_cids:
            raise MaterializationError(f"{task_id} has unknown goal {goal_alias}")
        output_paths = _csv(metadata.get("Outputs", ""))
        normalized = _normalized_body(
            task_id=task_id,
            title=title,
            metadata=metadata,
            block=body_text,
        )
        normalized["planning_lineage_revision"] = normalized.get("base_revision", "")
        owner = str(metadata.get("Owning repository") or "")
        if owner == "ipfs_datasets_py":
            normalized["base_revision"] = datasets_head
            normalized["base_repository_tree_id"] = datasets_tree
        elif owner == "ipfs_accelerate_py":
            normalized["base_revision"] = head
            normalized["base_repository_tree_id"] = tree
        else:
            raise MaterializationError(f"{task_id} has unsupported owning repository {owner!r}")
        normalized["owning_repository"] = owner
        normalized["accepted_plan_root_cid"] = plan_root
        tasks.append(
            {
                **normalized,
                "task_cid": task_cids[task_id],
                "task_id": task_id,
                "task_alias": task_id,
                "goal_cid": goal_cids[goal_alias],
                "plan_cid": plan_root,
                "objective_id": "objective:lgswf-root",
                "ordinal": ordinal,
                "status": metadata.get("Status", "todo"),
                "priority": metadata.get("Priority", "P0"),
                "title": title,
                "dependencies": dependencies,
                "outputs": [
                    {"path": path, "effect_id": _identity({"task": task_id, "path": path})}
                    for path in output_paths
                ],
                "acceptance": [str(metadata.get("Acceptance") or "")],
                "validations": [str(metadata.get("Validation") or "")],
            }
        )
    if len(tasks) != int((config.get("initial_projection") or {}).get("task_count", -1)):
        raise MaterializationError("task count differs from sealed projection")
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/lgswf-population@1",
        "repository_tree_id": tree,
        "source_head": head,
        "plan_root_cid": plan_root,
        "objectives": goals,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "LGSWF-PLAN-ACTUAL-R1-S1",
                "goal_cid": goal_cids["LGSWF-G000"],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
                "supersedes_plan_cid": config["supersedes_quarantined_plan_root_cid"],
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
    }


def _git_output(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise MaterializationError(
            f"git {' '.join(args)} failed for {repository}: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout.strip()


def _assert_population_source_current(
    config: Mapping[str, Any], population: Mapping[str, Any]
) -> dict[str, Any]:
    """Recheck the exact clean repository forest immediately before mutation."""

    status = _git_output(ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise MaterializationError("execution worktree changed after population construction")
    head = _git_output(ROOT, "rev-parse", "HEAD")
    tree = _git_output(ROOT, "rev-parse", "HEAD^{tree}")
    if head != population.get("source_head") or tree != population.get("repository_tree_id"):
        raise MaterializationError("execution HEAD/tree changed after population construction")

    binding = config.get("source_binding") or {}
    if not isinstance(binding, Mapping):
        raise MaterializationError("source_binding is not a mapping")
    nested: list[dict[str, Any]] = []
    for prefix in ("ipfs_datasets", "ipfs_kit", "mcp_plus_plus"):
        path_value = binding.get(f"{prefix}_submodule_path")
        revision = str(binding.get(f"{prefix}_planning_revision") or "")
        if not path_value and not revision:
            continue
        nested_path = _relative_path(path_value, field=f"source_binding.{prefix}_submodule_path")
        if _git_output(nested_path, "status", "--porcelain=v1", "--untracked-files=all"):
            raise MaterializationError(f"{prefix} nested worktree is dirty")
        nested_head = _git_output(nested_path, "rev-parse", "HEAD")
        nested_tree = _git_output(nested_path, "rev-parse", "HEAD^{tree}")
        if nested_head != revision:
            raise MaterializationError(f"{prefix} nested HEAD changed")
        relative = nested_path.relative_to(ROOT).as_posix()
        gitlink = _git_output(ROOT, "ls-tree", head, "--", relative).split()
        if (
            len(gitlink) < 3
            or gitlink[0] != "160000"
            or gitlink[1] != "commit"
            or gitlink[2] != nested_head
        ):
            raise MaterializationError(f"{prefix} gitlink does not match its exact nested HEAD")
        nested.append(
            {
                "repository": prefix,
                "path": relative,
                "head": nested_head,
                "tree": nested_tree,
            }
        )
    report = {
        "source_head": head,
        "repository_tree_id": tree,
        "worktree_clean": True,
        "nested_repositories": nested,
    }
    report["source_forest_root"] = _identity(report)
    return report


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    program = config["database_program"]
    control = _relative_path(program.get("store_id"), field="database_program.store_id")
    return {
        "control": control,
        "coordination": control.with_name(f"{control.stem}.coordination.duckdb"),
        "execution": control.with_name(f"{control.stem}.execution.duckdb"),
    }


def _verify_execution_store(
    config: Mapping[str, Any], schema_profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the daemon side store is bound and has no pre-seal effects."""

    path = _paths(config)["execution"]
    if not path.is_file():
        raise MaterializationError("daemon execution store is absent")
    try:
        import duckdb  # type: ignore

        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        raise MaterializationError("daemon execution store cannot be read safely") from exc
    required_tables = {
        "daemon_execution_metadata",
        "database_task_attempts",
        "attempt_phases",
        "provider_invocations",
        "effect_claims",
        "daemon_execution_events",
    }
    effect_tables = sorted(required_tables - {"daemon_execution_metadata"})
    try:
        installed_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        if installed_tables != required_tables:
            raise MaterializationError(
                "daemon execution schema differs from the closed bootstrap profile: "
                + json.dumps(
                    {
                        "missing": sorted(required_tables - installed_tables),
                        "extra": sorted(installed_tables - required_tables),
                    },
                    sort_keys=True,
                )
            )
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM daemon_execution_metadata ORDER BY key"
            ).fetchall()
        }
        expected_metadata = {
            "interface": "DatabaseImplementationDaemon@1",
            "schema": "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1",
            "authority_mode": "embedded",
            "logical_owner_session_id": "lgswf-materializer:single-writer",
            "state_schema_revision": str(config["database_program"]["schema_revision"]),
            "control_schema_profile_id": str(schema_profile.get("profile_id") or ""),
            "control_schema_fingerprint": str(schema_profile.get("schema_fingerprint") or ""),
        }
        mismatches = [key for key, value in expected_metadata.items() if metadata.get(key) != value]
        expected_metadata_keys = {*expected_metadata, "process_instance_id"}
        if (
            mismatches
            or not metadata.get("process_instance_id")
            or set(metadata) != expected_metadata_keys
        ):
            raise MaterializationError(
                "daemon execution metadata is stale: "
                + ", ".join(
                    mismatches
                    or [
                        "metadata_keys"
                        if set(metadata) != expected_metadata_keys
                        else "process_instance_id"
                    ]
                )
            )
        row_counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in effect_tables
        }
        nonempty = {table: count for table, count in row_counts.items() if count != 0}
        if nonempty:
            raise MaterializationError(
                "pre-seal execution effects are present: " + json.dumps(nonempty, sort_keys=True)
            )
        report = {
            "required_tables": sorted(required_tables),
            "metadata": metadata,
            "row_counts": row_counts,
        }
        report["execution_store_root"] = _identity(report)
        return report
    finally:
        connection.close()


def _decode_projection_body(value: Any, *, label: str) -> dict[str, Any]:
    """Decode an exact control-plane body without accepting scalar JSON."""

    try:
        decoded = json.loads(str(value or "{}"))
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise MaterializationError(f"{label} is not a mapping")
    return dict(decoded)


def _expected_control_population(population: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact objective/goal/plan projection written by the adapter."""

    objectives: list[dict[str, Any]] = []
    goals: list[dict[str, Any]] = []
    for index, item in enumerate(population["objectives"], start=1):
        objective_id = str(item.get("objective_id") or "")
        if objective_id:
            objectives.append(
                {
                    "objective_id": objective_id,
                    "objective_alias": str(item.get("objective_alias") or objective_id),
                    "parent_objective_id": "",
                    "title": str(item.get("title") or objective_id),
                    "status": str(item.get("status") or "open"),
                    "priority": str(item.get("priority") or "P2"),
                    "revision": 1,
                    "body": {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "objective_id",
                            "objective_alias",
                            "title",
                            "status",
                            "priority",
                        }
                    },
                }
            )
        goal_cid = str(item.get("goal_cid") or item.get("goal_id") or f"goal:cid:{index}")
        goal_alias = str(
            item.get("goal_alias") or item.get("goal_id") or item.get("alias") or goal_cid
        )
        goals.append(
            {
                "goal_cid": goal_cid,
                "goal_alias": goal_alias,
                "objective_id": objective_id,
                "parent_goal_cid": str(item.get("parent_goal_cid") or ""),
                "ordinal": int(item.get("ordinal") or index),
                "title": str(item.get("title") or goal_alias),
                "status": str(item.get("status") or "open"),
                "revision": 1,
                "body": {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "goal_cid",
                        "goal_id",
                        "goal_alias",
                        "title",
                        "status",
                        "ordinal",
                        "objective_id",
                    }
                },
            }
        )

    plans = [
        {
            "plan_cid": str(item["plan_cid"]),
            "goal_cid": str(item.get("goal_cid") or goals[0]["goal_cid"]),
            "plan_alias": str(item.get("plan_alias") or item.get("alias") or item["plan_cid"]),
            "status": str(item.get("status") or "active"),
            "revision": 1,
            "body": dict(item),
        }
        for item in population["plans"]
    ]
    result = {
        "objectives": sorted(objectives, key=lambda item: item["objective_id"]),
        "objective_revisions": sorted(
            (
                {
                    "objective_id": item["objective_id"],
                    "revision": item["revision"],
                    "status": item["status"],
                    "body": item["body"],
                }
                for item in objectives
            ),
            key=lambda item: (item["objective_id"], item["revision"]),
        ),
        "goals": sorted(goals, key=lambda item: item["goal_cid"]),
        "goal_edges": sorted(
            (
                {
                    "parent_goal_cid": str(item["parent_goal_cid"]),
                    "child_goal_cid": str(item["child_goal_cid"]),
                    "edge_kind": str(item["edge_kind"]),
                }
                for item in population.get("goal_edges") or ()
            ),
            key=lambda item: (
                item["parent_goal_cid"],
                item["child_goal_cid"],
                item["edge_kind"],
            ),
        ),
        "plans": sorted(plans, key=lambda item: item["plan_cid"]),
        "plan_revisions": sorted(
            (
                {
                    "plan_cid": item["plan_cid"],
                    "revision": item["revision"],
                    "body": item["body"],
                }
                for item in plans
            ),
            key=lambda item: (item["plan_cid"], item["revision"]),
        ),
    }
    return result


def _verify_control_population(
    config: Mapping[str, Any], population: Mapping[str, Any]
) -> dict[str, Any]:
    """Read and verify the complete immutable objective/goal/plan projection."""

    path = _paths(config)["control"]
    try:
        import duckdb  # type: ignore

        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        raise MaterializationError("control population cannot be opened read-only") from exc
    try:
        objectives = [
            {
                "objective_id": str(row[0]),
                "objective_alias": str(row[1]),
                "parent_objective_id": str(row[2] or ""),
                "title": str(row[3]),
                "status": str(row[4]),
                "priority": str(row[5]),
                "revision": int(row[6]),
                "body": _decode_projection_body(row[7], label=f"objective {row[0]} body"),
            }
            for row in connection.execute(
                """
                SELECT objective_id, objective_alias, parent_objective_id,
                       title, status, priority, revision, body_json
                FROM objectives ORDER BY objective_id
                """
            ).fetchall()
        ]
        objective_revisions = [
            {
                "objective_id": str(row[0]),
                "revision": int(row[1]),
                "status": str(row[2]),
                "body": _decode_projection_body(
                    row[3], label=f"objective revision {row[0]}:{row[1]} body"
                ),
            }
            for row in connection.execute(
                """
                SELECT objective_id, revision, status, body_json
                FROM objective_revisions ORDER BY objective_id, revision
                """
            ).fetchall()
        ]
        goals = [
            {
                "goal_cid": str(row[0]),
                "goal_alias": str(row[1]),
                "objective_id": str(row[2] or ""),
                "parent_goal_cid": str(row[3] or ""),
                "ordinal": int(row[4]),
                "title": str(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "body": _decode_projection_body(row[8], label=f"goal {row[0]} body"),
            }
            for row in connection.execute(
                """
                SELECT goal_cid, goal_alias, objective_id, parent_goal_cid,
                       ordinal, title, status, revision, body_json
                FROM goals ORDER BY goal_cid
                """
            ).fetchall()
        ]
        goal_edges = [
            {
                "parent_goal_cid": str(row[0]),
                "child_goal_cid": str(row[1]),
                "edge_kind": str(row[2]),
            }
            for row in connection.execute(
                """
                SELECT parent_goal_cid, child_goal_cid, edge_kind
                FROM goal_edges
                ORDER BY parent_goal_cid, child_goal_cid, edge_kind
                """
            ).fetchall()
        ]
        plans = [
            {
                "plan_cid": str(row[0]),
                "goal_cid": str(row[1]),
                "plan_alias": str(row[2]),
                "status": str(row[3]),
                "revision": int(row[4]),
                "body": _decode_projection_body(row[5], label=f"plan {row[0]} body"),
            }
            for row in connection.execute(
                """
                SELECT plan_cid, goal_cid, plan_alias, status, revision, body_json
                FROM plans ORDER BY plan_cid
                """
            ).fetchall()
        ]
        plan_revisions = [
            {
                "plan_cid": str(row[0]),
                "revision": int(row[1]),
                "body": _decode_projection_body(
                    row[2], label=f"plan revision {row[0]}:{row[1]} body"
                ),
            }
            for row in connection.execute(
                """
                SELECT plan_cid, revision, body_json
                FROM plan_revisions ORDER BY plan_cid, revision
                """
            ).fetchall()
        ]
    except MaterializationError:
        raise
    except Exception as exc:
        raise MaterializationError("control objective/goal/plan projection is unavailable") from exc
    finally:
        connection.close()

    observed = {
        "objectives": objectives,
        "objective_revisions": objective_revisions,
        "goals": goals,
        "goal_edges": goal_edges,
        "plans": plans,
        "plan_revisions": plan_revisions,
    }
    expected = _expected_control_population(population)
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise MaterializationError(
            "control objective/goal/plan population changed: " + ", ".join(mismatches)
        )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/control-population-projection@1",
        "population_root": _identity(observed),
        "counts": {key: len(value) for key, value in observed.items()},
    }


_TASK_BODY_TOP_LEVEL_FIELDS = frozenset(
    {
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
)


def _verify_task_records(
    records: list[Any],
    population: Mapping[str, Any],
    *,
    expected_stage: str,
) -> str:
    expected_tasks = list(population["tasks"])
    if len(records) != len(expected_tasks):
        raise MaterializationError("database task population is incomplete")
    projection: list[dict[str, Any]] = []
    for observed, expected in zip(records, expected_tasks, strict=True):
        expected_status = (
            "completed"
            if expected_stage == "sealed" and expected["task_alias"] == "LGSWF-006"
            else str(expected["status"])
        )
        expected_outputs = [
            {
                "ordinal": index,
                "path": str(item.get("path") or ""),
                "effect": dict(item),
            }
            for index, item in enumerate(expected["outputs"])
        ]
        expected_acceptance = [
            {
                "ordinal": index,
                "criterion": str(
                    criterion.get("criterion") if isinstance(criterion, Mapping) else criterion
                ),
                "evidence_policy": (
                    dict(criterion.get("evidence_policy") or {})
                    if isinstance(criterion, Mapping) and criterion.get("evidence_policy")
                    else {
                        "criterion": str(
                            criterion.get("criterion")
                            if isinstance(criterion, Mapping)
                            else criterion
                        )
                    }
                ),
            }
            for index, criterion in enumerate(expected["acceptance"])
        ]
        expected_validations = [
            {
                "ordinal": index,
                "argv": (
                    [str(item) for item in command.get("argv") or ()]
                    if isinstance(command, Mapping) and command.get("argv")
                    else [str(command.get("command") if isinstance(command, Mapping) else command)]
                ),
                "policy": (
                    dict(command.get("policy") or {}) if isinstance(command, Mapping) else {}
                ),
            }
            for index, command in enumerate(expected["validations"])
        ]
        exact_fields = {
            "task_cid": expected["task_cid"],
            "task_alias": expected["task_alias"],
            "goal_cid": expected["goal_cid"],
            "plan_cid": expected["plan_cid"],
            "objective_id": expected["objective_id"],
            "ordinal": int(expected["ordinal"]),
            "priority": expected["priority"],
            "status": expected_status,
            "dependencies": list(expected["dependencies"]),
            "outputs": expected_outputs,
            "acceptance": expected_acceptance,
            "validations": expected_validations,
        }
        observed_dict = observed.to_dict()
        observed_dependencies = observed_dict.get("dependencies") or []
        expected_dependencies = list(exact_fields["dependencies"])
        if isinstance(observed_dependencies, (list, tuple)):
            deps_match = sorted(str(item) for item in observed_dependencies) == sorted(
                str(item) for item in expected_dependencies
            )
            observed_dict = {
                **observed_dict,
                "dependencies": expected_dependencies if deps_match else list(observed_dependencies),
            }
        mismatches = [key for key, value in exact_fields.items() if observed_dict.get(key) != value]
        if mismatches:
            raise MaterializationError(
                f"{expected['task_alias']} task projection changed: " + ", ".join(mismatches)
            )
        expected_body = {
            key: value for key, value in expected.items() if key not in _TASK_BODY_TOP_LEVEL_FIELDS
        }
        observed_body = dict(observed.body)
        body_mismatches = [
            key for key, value in expected_body.items() if observed_body.get(key) != value
        ]
        allowed_extra = (
            {"completion_receipt"}
            if expected_stage == "sealed" and expected["task_alias"] == "LGSWF-006"
            else set()
        )
        unexpected = set(observed_body) - set(expected_body) - allowed_extra
        if body_mismatches or unexpected:
            raise MaterializationError(
                f"{expected['task_alias']} task specification changed: "
                + ", ".join(body_mismatches + sorted(unexpected))
            )
        projection.append(
            {
                **exact_fields,
                "body": expected_body,
            }
        )
    return _identity(projection)


def _verify_coordination_registry(
    projection: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    expected_stage: str,
    promoted_completion: Mapping[str, Any] | None = None,
    permitted_writer_lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the exact daemon registry without mutating coordination state."""
    expected_tasks = sorted(
        (
            {
                "task_cid": str(task["task_cid"]),
                "task_id": str(task["task_alias"]),
                "worktree_id": "",
                "ready": not (
                    expected_stage == "sealed" and str(task["task_alias"]) == "LGSWF-006"
                ),
                "body": {
                    "task_alias": str(task["task_alias"]),
                    "status": str(task["status"]),
                    "priority": str(task["priority"]),
                },
            }
            for task in population["tasks"]
        ),
        key=lambda item: item["task_cid"],
    )
    expected_dependencies = sorted(
        (
            {
                "task_cid": str(task["task_cid"]),
                "dependency_task_cid": str(dependency),
            }
            for task in population["tasks"]
            for dependency in task["dependencies"]
        ),
        key=lambda item: (item["task_cid"], item["dependency_task_cid"]),
    )
    if projection.get("tasks") != expected_tasks:
        raise MaterializationError(
            "coordination task registry differs from the materialized population"
        )
    if projection.get("dependency_edges") != expected_dependencies:
        raise MaterializationError(
            "coordination dependency registry differs from the accepted task graph"
        )

    expected_completions: list[dict[str, Any]] = []
    if expected_stage == "sealed":
        if not isinstance(promoted_completion, Mapping):
            raise MaterializationError(
                "sealed coordination registry has no promoted completion authority"
            )
        promoted_body = {
            key: value for key, value in promoted_completion.items() if key != "status"
        }
        expected_completions = [
            {
                "task_cid": str(population["task_cids_by_alias"]["LGSWF-006"]),
                "status": "succeeded",
                "body": promoted_body,
            }
        ]
    if projection.get("logical_completions") != expected_completions:
        raise MaterializationError(
            "coordination logical completion registry differs from control authority"
        )

    counts = projection.get("counts")
    if not isinstance(counts, Mapping):
        raise MaterializationError("coordination registry omitted exact state counts")
    writer_active = permitted_writer_lease is not None
    expected_common_counts = {
        "registered_tasks": len(expected_tasks),
        "dependency_edges": len(expected_dependencies),
        "logical_completions": len(expected_completions),
        "active_task_claims": 0,
        "active_resource_claims": 1 if writer_active else 0,
        "active_task_attempts": 0,
        "active_fenced_leases": 1 if writer_active else 0,
        "maintenance_leases": 0,
        "active_maintenance_leases": 0,
    }
    count_mismatches = [
        key for key, value in expected_common_counts.items() if counts.get(key) != value
    ]
    if count_mismatches:
        raise MaterializationError(
            "coordination registry is not quiescent: " + ", ".join(count_mismatches)
        )
    task_claim_rows = [dict(item) for item in projection.get("task_claims") or ()]
    task_attempt_rows = [dict(item) for item in projection.get("task_attempts") or ()]
    fenced_lease_rows = [dict(item) for item in projection.get("fenced_leases") or ()]
    resource_claim_rows = [dict(item) for item in projection.get("resource_claims") or ()]
    maintenance_rows = [dict(item) for item in projection.get("maintenance_leases") or ()]
    task_lease_rows = [item for item in fenced_lease_rows if item.get("lease_kind") == "task"]
    resource_lease_rows = [
        item for item in fenced_lease_rows if item.get("lease_kind") == "resource"
    ]
    if maintenance_rows or len(task_lease_rows) + len(resource_lease_rows) != len(
        fenced_lease_rows
    ):
        raise MaterializationError("coordination history contains a foreign lease authority")
    exact_totals = {
        "task_claims": len(task_claim_rows),
        "task_attempts": len(task_attempt_rows),
        "resource_claims": len(resource_claim_rows),
        "fenced_leases": len(fenced_lease_rows),
        "maintenance_leases": len(maintenance_rows),
    }
    if any(counts.get(key) != value for key, value in exact_totals.items()):
        raise MaterializationError("coordination claim/attempt/lease projections are incomplete")
    if not (
        len(task_claim_rows) == len(task_attempt_rows) == len(task_lease_rows)
        and len(resource_claim_rows) == len(resource_lease_rows)
    ):
        raise MaterializationError("coordination claim/attempt/lease projections are incomplete")

    if expected_stage == "unsealed" and not writer_active:
        if task_claim_rows or resource_claim_rows or fenced_lease_rows:
            raise MaterializationError(
                "fresh unsealed coordination registry contains execution history"
            )
    task_cid = str(population["task_cids_by_alias"]["LGSWF-006"])
    accepted_result_cid = ""
    expected_owner = ""
    if writer_active:
        writer_body = permitted_writer_lease.get("body")
        accepted_result_cid = str(
            writer_body.get("accepted_result_cid") if isinstance(writer_body, Mapping) else ""
        )
        expected_owner = str(permitted_writer_lease.get("owner_session_id") or "")
    elif isinstance(promoted_completion, Mapping):
        accepted_result_cid = str(promoted_completion.get("evidence_digest") or "")
        expected_owner = str(promoted_completion.get("owner_session_id") or "")

    if accepted_result_cid:
        expected_resource_id = (
            "lgswf-control-store:"
            + _identity(
                {
                    "plan_root_cid": population["plan_root_cid"],
                    "repository_tree_id": population["repository_tree_id"],
                }
            ).split(":", 1)[1]
        )
        if not resource_claim_rows or len(resource_claim_rows) != len(resource_lease_rows):
            raise MaterializationError("bootstrap writer history is incomplete")
        writer_body = {
            "kind": "trusted_manual_bootstrap_writer",
            "accepted_result_cid": accepted_result_cid,
            "plan_root_cid": population["plan_root_cid"],
        }
        writer_identity_fields = {
            "resource_kind": "database_writer",
            "resource_id": expected_resource_id,
            "owner_session_id": expected_owner,
            "task_cid": task_cid,
            "repository_id": str(population["source_head"]),
            "mode": "exclusive",
            "body": writer_body,
        }
        writer_leases_by_id = {
            str(item.get("lease_id") or ""): item for item in resource_lease_rows
        }
        active_writer_leases: list[dict[str, Any]] = []
        writer_order: list[tuple[int, int]] = []
        for resource_claim in resource_claim_rows:
            resource_lease = writer_leases_by_id.get(str(resource_claim.get("lease_id") or ""))
            if (
                not isinstance(resource_lease, Mapping)
                or any(
                    resource_claim.get(key) != value
                    for key, value in writer_identity_fields.items()
                )
                or any(
                    resource_lease.get(key) != value
                    for key, value in {
                        **writer_identity_fields,
                        "lease_kind": "resource",
                        "claim_id": resource_claim.get("claim_id"),
                    }.items()
                )
                or any(
                    resource_claim.get(key) != resource_lease.get(key)
                    for key in (
                        "lease_id",
                        "owner_session_id",
                        "fencing_token",
                        "fence_epoch",
                        "resource_kind",
                        "resource_id",
                        "task_cid",
                        "repository_id",
                        "mode",
                        "body",
                        "state",
                    )
                )
                or resource_claim.get("state")
                not in {
                    "accepted",
                    "released",
                    "expired",
                }
            ):
                raise MaterializationError(
                    "coordination history contains a foreign writer authority"
                )
            if resource_claim.get("state") == "accepted":
                active_writer_leases.append(dict(resource_lease))
            writer_order.append(
                (
                    int(resource_claim.get("fence_epoch") or 0),
                    int(resource_claim.get("fencing_token") or 0),
                )
            )
        if len(active_writer_leases) != (1 if writer_active else 0):
            raise MaterializationError("writer resource claim state is inconsistent")
        ordered_writers = sorted(writer_order)
        if (
            len(set(ordered_writers)) != len(ordered_writers)
            or any(epoch <= 0 or token <= 0 for epoch, token in ordered_writers)
            or any(
                later_token <= earlier_token
                for (_earlier_epoch, earlier_token), (_later_epoch, later_token) in zip(
                    ordered_writers, ordered_writers[1:]
                )
            )
        ):
            raise MaterializationError("writer history is not monotonically fenced")
        if writer_active:
            exact_writer_fields = (
                "lease_id",
                "lease_kind",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "state",
                "task_cid",
                "resource_kind",
                "resource_id",
                "repository_id",
                "claim_id",
                "body",
            )
            if any(
                active_writer_leases[0].get(field) != permitted_writer_lease.get(field)
                for field in exact_writer_fields
            ):
                raise MaterializationError("active writer differs from its admitted reservation")
    elif resource_claim_rows or resource_lease_rows:
        raise MaterializationError("coordination history contains an unbound writer authority")

    attempts_by_id = {str(item.get("attempt_id") or ""): item for item in task_attempt_rows}
    leases_by_id = {str(item.get("lease_id") or ""): item for item in task_lease_rows}
    succeeded_claims: list[dict[str, Any]] = []
    attempt_order: list[tuple[int, int]] = []
    for claim in task_claim_rows:
        attempt = attempts_by_id.get(str(claim.get("attempt_id") or ""))
        lease = leases_by_id.get(str(claim.get("lease_id") or ""))
        expected_claim_body = {
            "kind": "trusted_manual_bootstrap_seal",
            "accepted_result_cid": accepted_result_cid,
        }
        expected_key = (
            "manual-seal:" + accepted_result_cid.split(":", 1)[-1] if accepted_result_cid else ""
        )
        claim_fields = {
            "task_cid": task_cid,
            "owner_session_id": expected_owner,
            "idempotency_key": expected_key,
            "body": expected_claim_body,
        }
        if (
            not isinstance(attempt, Mapping)
            or not isinstance(lease, Mapping)
            or any(claim.get(key) != value for key, value in claim_fields.items())
        ):
            raise MaterializationError("coordination history contains a foreign task claim")
        linked_fields = (
            "task_cid",
            "attempt_id",
            "attempt_number",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
        )
        if (
            any(claim.get(key) != attempt.get(key) for key in linked_fields)
            or any(
                claim.get(key) != lease.get(key)
                for key in (*linked_fields, "claim_id", "lease_id", "idempotency_key", "body")
            )
            or lease.get("lease_kind") != "task"
        ):
            raise MaterializationError(
                "task claim, attempt and lease histories do not share one fence"
            )
        state_triple = (
            str(claim.get("state") or ""),
            str(attempt.get("status") or ""),
            str(lease.get("state") or ""),
        )
        if state_triple in {
            ("released", "succeeded", "released"),
            ("completed", "succeeded", "completed"),
        }:
            succeeded_claims.append(claim)
        elif state_triple != ("expired", "expired", "expired"):
            raise MaterializationError(
                "task claim history contains a nonterminal or foreign transition"
            )
        attempt_order.append(
            (int(claim.get("attempt_number") or 0), int(claim.get("fencing_token") or 0))
        )
    ordered_attempts = sorted(attempt_order)
    if (
        len({number for number, _token in ordered_attempts}) != len(ordered_attempts)
        or any(number <= 0 or token <= 0 for number, token in ordered_attempts)
        or any(
            later_token <= earlier_token
            for (_earlier_number, earlier_token), (_later_number, later_token) in zip(
                ordered_attempts, ordered_attempts[1:]
            )
        )
    ):
        raise MaterializationError("task attempt history is not monotonically fenced")
    if expected_stage == "unsealed" and succeeded_claims:
        raise MaterializationError(
            "unsealed coordination history contains a successful task attempt"
        )
    if expected_stage == "sealed":
        if len(succeeded_claims) != 1 or not isinstance(promoted_completion, Mapping):
            raise MaterializationError("sealed coordination history has no unique accepted attempt")
        accepted_claim = succeeded_claims[0]
        if any(
            accepted_claim.get(field) != promoted_completion.get(field)
            for field in (
                "task_cid",
                "claim_id",
                "attempt_id",
                "attempt_number",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
            )
        ):
            raise MaterializationError("accepted task history differs from the promoted completion")

    structure = {
        "tasks": projection["tasks"],
        "dependency_edges": projection["dependency_edges"],
        "logical_completions": projection["logical_completions"],
    }
    registry_spec = {
        "tasks": [
            {key: value for key, value in item.items() if key != "ready"}
            for item in projection["tasks"]
        ],
        "dependency_edges": projection["dependency_edges"],
    }
    history = {
        "task_claims": task_claim_rows,
        "task_attempts": task_attempt_rows,
        "fenced_leases": fenced_lease_rows,
        "resource_claims": resource_claim_rows,
        "maintenance_leases": maintenance_rows,
    }
    return {
        "schema": projection["schema"],
        "projection_root": projection["projection_root"],
        "structure_root": _identity(structure),
        "registry_spec_root": _identity(registry_spec),
        "history_root": _identity(history),
        "counts": dict(counts),
        "task_claim_state_counts": list(projection.get("task_claim_state_counts") or ()),
        "resource_claim_state_counts": list(projection.get("resource_claim_state_counts") or ()),
        "task_attempt_status_counts": list(projection.get("task_attempt_status_counts") or ()),
        "fenced_lease_kind_state_counts": list(
            projection.get("fenced_lease_kind_state_counts") or ()
        ),
        "maintenance_lease_state_counts": list(
            projection.get("maintenance_lease_state_counts") or ()
        ),
    }


def _verify_store(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    expected_stage: str = "sealed",
    permitted_writer_lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    if expected_stage not in {"unsealed", "sealed"}:
        raise MaterializationError(f"unknown bootstrap stage: {expected_stage!r}")
    paths = _paths(config)
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise MaterializationError(f"control-plane files missing: {missing}")
    schema_verification = verify_datasets_authoritative_operational_schema(paths["control"])
    if not bool(schema_verification.get("valid")):
        raise MaterializationError(
            "datasets-authoritative operational schema verification failed: "
            + json.dumps(schema_verification, sort_keys=True)
        )
    execution_verification = _verify_execution_store(config, schema_verification)
    control_population = _verify_control_population(config, population)
    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id="lgswf-materializer:verify",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    try:
        coordination_projection = read_coordination_registry_projection(paths["coordination"])
    except Exception as exc:
        raise MaterializationError("coordination authority cannot be verified read-only") from exc
    try:
        snapshot = task_source.snapshot()
        page = task_source.list_tasks(limit=100)
        ready = task_source.ready_tasks(limit=100)
        aliases = [item.task_alias for item in page.tasks]
        ready_aliases = [item.task_alias for item in ready.tasks]
        records = {item.task_alias: item for item in page.tasks}
        task_spec_root = _verify_task_records(
            list(page.tasks), population, expected_stage=expected_stage
        )
        successor_alias = "LGSWF-006"
        successor_cid = population["task_cids_by_alias"][successor_alias]
        successor = records.get(successor_alias)
        if successor is None:
            raise MaterializationError("corrected bootstrap successor is absent")
        coordination_tasks = {
            str(item.get("task_cid") or ""): item
            for item in coordination_projection.get("tasks") or ()
        }
        successor_coordination = coordination_tasks.get(successor_cid)
        if not isinstance(successor_coordination, Mapping):
            raise MaterializationError("corrected bootstrap successor is absent from coordination")
        if expected_stage == "unsealed":
            expected_ready = list(
                (config.get("initial_projection") or {}).get("ready_task_ids") or []
            )
            if expected_ready != [successor_alias]:
                raise MaterializationError("config does not expose only the manual seal")
            if str(successor.status).lower() != "todo":
                raise MaterializationError("manual bootstrap seal is not pending")
            if successor_coordination.get("ready") is not True or coordination_projection.get(
                "logical_completions"
            ):
                raise MaterializationError("manual bootstrap seal is not claimable")
            completed_aliases: list[str] = []
        else:
            expected_ready = ["LGSWF-001", "LGSWF-002", "LGSWF-003"]
            if str(successor.status).lower() not in {"completed", "complete", "done"}:
                raise MaterializationError("manual bootstrap seal is not accepted")
            completion_receipt = successor.body.get("completion_receipt")
            if not isinstance(completion_receipt, Mapping):
                raise MaterializationError("manual bootstrap seal has no control receipt")
            accepted_result_cid = str(completion_receipt.get("accepted_result_cid") or "")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", accepted_result_cid):
                raise MaterializationError("manual bootstrap seal has no accepted result identity")
            logical_completions = list(coordination_projection.get("logical_completions") or ())
            promoted_row = logical_completions[0] if len(logical_completions) == 1 else None
            promoted = (
                {**dict(promoted_row.get("body") or {}), "status": promoted_row.get("status")}
                if isinstance(promoted_row, Mapping)
                else None
            )
            if (
                not isinstance(promoted, Mapping)
                or promoted_row.get("task_cid") != successor_cid
                or promoted.get("status") != "succeeded"
            ):
                raise MaterializationError("coordination has no promoted bootstrap completion")
            control_preparation = completion_receipt.get("coordination_preparation")
            if not isinstance(control_preparation, Mapping):
                raise MaterializationError("control completion has no coordination preparation")
            preparation_fields = (
                "schema",
                "task_cid",
                "attempt_id",
                "attempt_number",
                "claim_id",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "control_expected_revision",
                "control_expected_status",
                "evidence_digest",
                "prepared_at_ms",
                "body",
                "preparation_digest",
            )
            mismatched_preparation = [
                field
                for field in preparation_fields
                if control_preparation.get(field) != promoted.get(field)
            ]
            if mismatched_preparation:
                raise MaterializationError(
                    "control and coordination completion bindings disagree: "
                    + ", ".join(mismatched_preparation)
                )
            if (
                promoted.get("task_cid") != successor_cid
                or promoted.get("evidence_digest") != accepted_result_cid
            ):
                raise MaterializationError(
                    "accepted result is not the promoted coordination evidence"
                )
            preparation_body = promoted.get("body")
            seal_basis = (
                preparation_body.get("seal_basis")
                if isinstance(preparation_body, Mapping)
                else None
            )
            if not isinstance(seal_basis, Mapping) or _identity(seal_basis) != accepted_result_cid:
                raise MaterializationError(
                    "accepted result does not identify the persisted seal basis"
                )
            if seal_basis.get("schema_profile_fingerprint") != schema_verification.get(
                "schema_fingerprint"
            ):
                raise MaterializationError(
                    "persisted seal basis does not bind the current schema fingerprint"
                )
            materialization_receipt = _load_materialization_receipt(config, population)
            qualification_receipt = _load_qualification_receipt(config, population)
            expected_seal_basis = _build_seal_basis(
                config=config,
                population=population,
                materialization_receipt=materialization_receipt,
                qualification_receipt=qualification_receipt,
                launch_plan=_render_launch_plan_evidence(config),
            )
            if dict(seal_basis) != expected_seal_basis:
                raise MaterializationError(
                    "persisted seal basis is stale or differs from current authorities"
                )
            cross_store_guard = promoted.get("cross_store_guard")
            control_completion = promoted.get("control_completion")
            if (
                not isinstance(cross_store_guard, Mapping)
                or not isinstance(control_completion, Mapping)
                or preparation_body.get("requires_cross_store_fence_guard")
                is not True
                or cross_store_guard.get("schema")
                != "ipfs_accelerate_py/agent-supervisor/cross-store-fence-guard@1"
                or cross_store_guard.get("preparation_digest")
                != promoted.get("preparation_digest")
                or cross_store_guard.get("control_result_digest")
                != _identity(successor.to_dict())
                or control_completion.get("receipt_digest")
                != cross_store_guard.get("control_result_digest")
            ):
                raise MaterializationError(
                    "accepted bootstrap completion lacks its cross-store fence guard"
                )
            if successor_coordination.get("ready") is not False:
                raise MaterializationError(
                    "coordination does not contain the accepted bootstrap seal"
                )
            for alias in expected_ready:
                candidate = coordination_tasks.get(population["task_cids_by_alias"][alias])
                if not isinstance(candidate, Mapping) or candidate.get("ready") is not True:
                    raise MaterializationError(
                        f"{alias} is not claimable after bootstrap acceptance"
                    )
            completed_aliases = [successor_alias]
        if ready_aliases != expected_ready:
            raise MaterializationError(
                f"database ready frontier mismatch: {ready_aliases!r} != {expected_ready!r}"
            )
        coordination_registry = _verify_coordination_registry(
            coordination_projection,
            population,
            expected_stage=expected_stage,
            promoted_completion=promoted if expected_stage == "sealed" else None,
            permitted_writer_lease=permitted_writer_lease,
        )
        active_leases = [
            dict(lease)
            for lease in coordination_projection.get("fenced_leases") or ()
            if lease.get("state") == "accepted"
        ]
        if permitted_writer_lease is None and active_leases:
            raise MaterializationError(
                "bootstrap verification found active leases: "
                + json.dumps(active_leases, sort_keys=True)
            )
        if permitted_writer_lease is not None:
            exact_writer_fields = (
                "lease_id",
                "lease_kind",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
                "state",
                "task_cid",
                "resource_kind",
                "resource_id",
                "repository_id",
                "claim_id",
            )
            if len(active_leases) != 1 or any(
                active_leases[0].get(field) != permitted_writer_lease.get(field)
                for field in exact_writer_fields
            ):
                raise MaterializationError(
                    "active lease is not the exact admitted bootstrap writer"
                )
        prepared = [
            item
            for item in coordination_projection.get("logical_completions") or ()
            if item.get("status") == "prepared"
        ]
        if prepared:
            raise MaterializationError(
                "bootstrap verification found unresolved completion preparations"
            )
        report = {
            "bootstrap_stage": expected_stage,
            "schema_profile": schema_verification,
            "execution_store": execution_verification,
            "control_population": control_population,
            "coordination_registry": coordination_registry,
            "task_source_snapshot": snapshot.to_dict(),
            "task_aliases": aliases,
            "task_spec_root": task_spec_root,
            "ready_task_aliases": ready_aliases,
            "completed_task_aliases": completed_aliases,
            "active_lease_count": len(active_leases),
            "active_task_claim_count": int(
                coordination_registry["counts"].get("active_task_claims") or 0
            ),
            "unresolved_preparation_count": len(prepared),
        }
        if expected_stage == "sealed":
            report["accepted_result_cid"] = accepted_result_cid
            report["completion_binding"] = {
                "task_cid": successor_cid,
                "claim_id": str(promoted["claim_id"]),
                "attempt_id": str(promoted["attempt_id"]),
                "lease_id": str(promoted["lease_id"]),
                "fencing_token": int(promoted["fencing_token"]),
                "fence_epoch": int(promoted["fence_epoch"]),
                "preparation_digest": str(promoted["preparation_digest"]),
                "seal_basis_cid": accepted_result_cid,
            }
    finally:
        task_source.close()
    report["database_identities"] = {key: _sha256_file(path) for key, path in sorted(paths.items())}
    return report


_LIVE_TASK_STATUSES = frozenset(
    {
        "proposed",
        "admitted",
        "pending",
        "ready",
        "todo",
        "queued",
        "retrying",
        "completed",
        "skipped",
        "complete",
        "done",
        "cancelled",
        "failed",
        "quarantined",
        "rejected",
        "claimed",
        "in_progress",
        "running",
        "blocked",
    }
)


def _verify_live_schema_read_only(path: Path) -> dict[str, Any]:
    """Run the complete operational-profile verifier on one native RO handle."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        compute_schema_fingerprint,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        BOOKKEEPING_TABLES,
        DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
        DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION,
        DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
        DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
        DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES,
        DATASETS_SEMANTIC_TRUTH_RELATIONS,
        DIAGNOSTIC_VIEWS,
        JOIN_CRITICAL_IDENTITIES,
        LEASE_IDENTITY_COLUMNS,
        TASK_IDENTITY_COLUMNS,
        _assert_datasets_authoritative_database,
        _table_columns,
        load_datasets_authoritative_operational_catalog,
    )

    if not path.is_file():
        raise MaterializationError("control database is absent")
    try:
        import duckdb  # type: ignore

        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        raise MaterializationError("control schema cannot be opened read-only") from exc
    try:
        catalog = load_datasets_authoritative_operational_catalog()
        migration = catalog.get(DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION)
        relations, forbidden_contracts = _assert_datasets_authoritative_database(
            connection, operation="read-only live operational-profile verification"
        )
        required = set(BOOKKEEPING_TABLES) | set(DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES)
        missing = sorted(required - set(relations))
        missing_views = sorted(set(DIAGNOSTIC_VIEWS) - set(relations))
        if missing or missing_views:
            raise MaterializationError(
                "datasets-authoritative operational schema is incomplete: "
                + json.dumps({"tables": missing, "views": missing_views}, sort_keys=True)
            )
        row = connection.execute(
            "SELECT migration_id, checksum FROM schema_migrations WHERE version = ?",
            [DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION],
        ).fetchone()
        if row is None or str(row[0]) != DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID or str(
            row[1]
        ) != migration.checksum:
            raise MaterializationError("operational-profile migration identity/checksum mismatch")
        contract = connection.execute(
            """
            SELECT payload_schema, description FROM schema_contracts
            WHERE contract_id = 'contract:DatasetsAuthoritativeOperationalControlPlane@1'
            """
        ).fetchone()
        if (
            contract is None
            or str(contract[0]) != DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA
            or "operational" not in str(contract[1]).lower()
            or "ipfs_datasets_py" not in str(contract[1])
        ):
            raise MaterializationError("datasets-authoritative authority contract drifted")
        join_identities = [
            (table, column)
            for table, column in JOIN_CRITICAL_IDENTITIES
            if table in DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES
        ]
        missing_identities = [
            f"{table}.{column}"
            for table, column in join_identities
            if column not in _table_columns(connection, table)
            or column.casefold().endswith("_json")
        ]
        task_columns = _table_columns(connection, "tasks")
        lease_columns = _table_columns(connection, "leases")
        missing_task = [item for item in TASK_IDENTITY_COLUMNS if item not in task_columns]
        missing_lease = [item for item in LEASE_IDENTITY_COLUMNS if item not in lease_columns]
        if missing_identities or missing_task or missing_lease:
            raise MaterializationError(
                "operational-profile identity columns drifted: "
                + json.dumps(
                    {
                        "join": missing_identities,
                        "task": missing_task,
                        "lease": missing_lease,
                    },
                    sort_keys=True,
                )
            )
        report = {
            "valid": True,
            "database_path": str(path),
            "profile_id": DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
            "profile_schema": DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
            "migration_id": DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
            "migration_checksum": migration.checksum,
            "catalog_fingerprint": catalog.fingerprint(),
            "required_tables_ok": sorted(DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES),
            "views_ok": sorted(DIAGNOSTIC_VIEWS),
            "join_critical_ok": sorted(f"{table}.{column}" for table, column in join_identities),
            "task_columns_ok": list(TASK_IDENTITY_COLUMNS),
            "lease_columns_ok": list(LEASE_IDENTITY_COLUMNS),
            "forbidden_relations": sorted(set(relations) & set(DATASETS_SEMANTIC_TRUTH_RELATIONS)),
            "forbidden_contracts": list(forbidden_contracts),
            "operational_evidence": {
                "relation": "evidence_nodes",
                "purpose": "content-addressed operational completion/result receipts",
                "semantic_and_proof_authority": "ipfs_datasets_py",
            },
            "authority_contract": {
                "contract_id": "contract:DatasetsAuthoritativeOperationalControlPlane@1",
                "payload_schema": DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
                "operational_authority": "ipfs_accelerate_py",
                "semantic_and_proof_authority": "ipfs_datasets_py",
            },
            "schema_fingerprint": compute_schema_fingerprint(connection),
        }
        return report
    except MaterializationError:
        raise
    except Exception as exc:
        raise MaterializationError(
            "datasets-authoritative operational schema verification failed read-only"
        ) from exc
    finally:
        connection.close()


def _expected_task_static(task: Mapping[str, Any]) -> dict[str, Any]:
    outputs = [
        {
            "ordinal": index,
            "path": str(item.get("path") or ""),
            "effect": dict(item),
        }
        for index, item in enumerate(task["outputs"])
    ]
    acceptance = [
        {
            "ordinal": index,
            "criterion": str(
                item.get("criterion") if isinstance(item, Mapping) else item
            ),
            "evidence_policy": (
                dict(item.get("evidence_policy") or {})
                if isinstance(item, Mapping) and item.get("evidence_policy")
                else {
                    "criterion": str(
                        item.get("criterion") if isinstance(item, Mapping) else item
                    )
                }
            ),
        }
        for index, item in enumerate(task["acceptance"])
    ]
    validations = [
        {
            "ordinal": index,
            "argv": (
                [str(value) for value in item.get("argv") or ()]
                if isinstance(item, Mapping) and item.get("argv")
                else [str(item.get("command") if isinstance(item, Mapping) else item)]
            ),
            "policy": dict(item.get("policy") or {}) if isinstance(item, Mapping) else {},
        }
        for index, item in enumerate(task["validations"])
    ]
    return {
        "task_cid": str(task["task_cid"]),
        "task_alias": str(task["task_alias"]),
        "goal_cid": str(task["goal_cid"]),
        "plan_cid": str(task["plan_cid"]),
        "objective_id": str(task["objective_id"]),
        "ordinal": int(task["ordinal"]),
        "priority": str(task["priority"]),
        "identity": {
            "task_cid": str(task["task_cid"]),
            "task_alias": str(task["task_alias"]),
        },
        "body": {
            key: value for key, value in task.items() if key not in _TASK_BODY_TOP_LEVEL_FIELDS
        },
        "dependencies": sorted(str(item) for item in task["dependencies"]),
        "outputs": outputs,
        "acceptance": acceptance,
        "validations": validations,
    }


def _expected_task_spec_root(
    population: Mapping[str, Any], *, sealed: bool
) -> str:
    projection: list[dict[str, Any]] = []
    for task in population["tasks"]:
        spec = _expected_task_static(task)
        projection.append(
            {
                "task_cid": spec["task_cid"],
                "task_alias": spec["task_alias"],
                "goal_cid": spec["goal_cid"],
                "plan_cid": spec["plan_cid"],
                "objective_id": spec["objective_id"],
                "ordinal": spec["ordinal"],
                "priority": spec["priority"],
                "status": (
                    "completed"
                    if sealed and spec["task_alias"] == "LGSWF-006"
                    else str(task["status"])
                ),
                "dependencies": list(task["dependencies"]),
                "outputs": spec["outputs"],
                "acceptance": spec["acceptance"],
                "validations": spec["validations"],
                "body": spec["body"],
            }
        )
    return _identity(projection)


def _verify_live_control_read_only(
    config: Mapping[str, Any], population: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify sealed identities as a subset while admitting legal live revisions."""

    path = _paths(config)["control"]
    try:
        import duckdb  # type: ignore

        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        raise MaterializationError("live control projection cannot be opened read-only") from exc
    try:
        def bodies(sql: str, *, label: str) -> list[tuple[Any, ...]]:
            rows = connection.execute(sql).fetchall()
            return [
                (*row[:-1], _decode_projection_body(row[-1], label=f"{label} {row[0]} body"))
                for row in rows
            ]

        objective_rows = bodies(
            """SELECT objective_id, objective_alias, parent_objective_id, title,
                      status, priority, revision, body_json
               FROM objectives ORDER BY objective_id""",
            label="objective",
        )
        goal_rows = bodies(
            """SELECT goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                      title, status, revision, body_json FROM goals ORDER BY goal_cid""",
            label="goal",
        )
        plan_rows = bodies(
            """SELECT plan_cid, goal_cid, plan_alias, status, revision, body_json
               FROM plans ORDER BY plan_cid""",
            label="plan",
        )
        task_rows = connection.execute(
            """SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id,
                      ordinal, status, revision, priority, identity_json, body_json
               FROM tasks ORDER BY task_cid"""
        ).fetchall()
        tasks: dict[str, dict[str, Any]] = {}
        for row in task_rows:
            task_cid = str(row[0])
            tasks[task_cid] = {
                "task_cid": task_cid,
                "task_alias": str(row[1]),
                "goal_cid": str(row[2]),
                "plan_cid": str(row[3] or ""),
                "objective_id": str(row[4] or ""),
                "ordinal": int(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "priority": str(row[8]),
                "identity": _decode_projection_body(row[9], label=f"task {task_cid} identity"),
                "body": _decode_projection_body(row[10], label=f"task {task_cid} body"),
            }
        aliases = [item["task_alias"] for item in tasks.values()]
        if len(aliases) != len(set(aliases)):
            raise MaterializationError("live task projection contains duplicate aliases")

        objectives = {str(row[0]): row for row in objective_rows}
        goals = {str(row[0]): row for row in goal_rows}
        plans = {str(row[0]): row for row in plan_rows}
        if (
            len({str(row[1]) for row in objective_rows}) != len(objective_rows)
            or len({str(row[1]) for row in goal_rows}) != len(goal_rows)
            or len({str(row[2]) for row in plan_rows}) != len(plan_rows)
        ):
            raise MaterializationError("live control projection contains duplicate aliases")
        for objective_id, row in objectives.items():
            parent = str(row[2] or "")
            if parent and (parent not in objectives or parent == objective_id):
                raise MaterializationError("objective contains an impossible parent identity")
        for goal_cid, row in goals.items():
            objective_id, parent = str(row[2] or ""), str(row[3] or "")
            if (
                objective_id
                and objective_id not in objectives
                or parent
                and (parent not in goals or parent == goal_cid)
            ):
                raise MaterializationError("goal contains an impossible parent identity")
        if any(str(row[1]) not in goals for row in plan_rows):
            raise MaterializationError("plan contains an impossible goal identity")
        expected_control = _expected_control_population(population)
        for expected in expected_control["objectives"]:
            row = objectives.get(expected["objective_id"])
            static = (
                expected["objective_alias"],
                expected["parent_objective_id"],
                expected["title"],
                expected["priority"],
                expected["body"],
            )
            if row is None or (str(row[1]), str(row[2] or ""), str(row[3]), str(row[5]), row[7]) != static:
                raise MaterializationError(
                    f"sealed objective specification changed: {expected['objective_id']}"
                )
        for expected in expected_control["goals"]:
            row = goals.get(expected["goal_cid"])
            static = (
                expected["goal_alias"],
                expected["objective_id"],
                expected["parent_goal_cid"],
                expected["ordinal"],
                expected["title"],
                expected["body"],
            )
            if row is None or (
                str(row[1]),
                str(row[2] or ""),
                str(row[3] or ""),
                int(row[4]),
                str(row[5]),
                row[8],
            ) != static:
                raise MaterializationError(
                    f"sealed goal specification changed: {expected['goal_cid']}"
                )
        for expected in expected_control["plans"]:
            row = plans.get(expected["plan_cid"])
            if row is None or (
                str(row[1]),
                str(row[2]),
                row[5],
            ) != (expected["goal_cid"], expected["plan_alias"], expected["body"]):
                raise MaterializationError(
                    f"sealed plan specification changed: {expected['plan_cid']}"
                )

        goal_edges = {
            (str(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT parent_goal_cid, child_goal_cid, edge_kind FROM goal_edges"
            ).fetchall()
        }
        expected_goal_edges = {
            (item["parent_goal_cid"], item["child_goal_cid"], item["edge_kind"])
            for item in expected_control["goal_edges"]
        }
        if not expected_goal_edges.issubset(goal_edges) or any(
            parent not in goals or child not in goals or parent == child
            for parent, child, _kind in goal_edges
        ):
            raise MaterializationError("sealed goal dependency identities changed")

        dependency_rows = connection.execute(
            "SELECT task_cid, dependency_task_cid, kind FROM task_dependencies"
        ).fetchall()
        dependencies: dict[str, list[tuple[str, str]]] = {}
        for task_cid, dependency_cid, kind in dependency_rows:
            task_key, dependency_key = str(task_cid), str(dependency_cid)
            if task_key not in tasks or dependency_key not in tasks or task_key == dependency_key:
                raise MaterializationError("task dependency contains an impossible identity")
            dependencies.setdefault(task_key, []).append((dependency_key, str(kind)))

        def grouped(sql: str, decoder: Any) -> dict[str, list[dict[str, Any]]]:
            grouped_rows: dict[str, list[dict[str, Any]]] = {}
            for row in connection.execute(sql).fetchall():
                grouped_rows.setdefault(str(row[0]), []).append(decoder(row))
            return grouped_rows

        outputs = grouped(
            "SELECT task_cid, ordinal, path, effect_json FROM task_outputs ORDER BY task_cid, ordinal",
            lambda row: {
                "ordinal": int(row[1]),
                "path": str(row[2]),
                "effect": _decode_projection_body(row[3], label=f"task output {row[0]}:{row[1]}"),
            },
        )
        acceptance = grouped(
            """SELECT task_cid, ordinal, criterion, evidence_policy_json
               FROM task_acceptance ORDER BY task_cid, ordinal""",
            lambda row: {
                "ordinal": int(row[1]),
                "criterion": str(row[2]),
                "evidence_policy": _decode_projection_body(
                    row[3], label=f"task acceptance {row[0]}:{row[1]}"
                ),
            },
        )
        validations = grouped(
            """SELECT task_cid, ordinal, argv_json, policy_json
               FROM task_validations ORDER BY task_cid, ordinal""",
            lambda row: {
                "ordinal": int(row[1]),
                "argv": json.loads(str(row[2])),
                "policy": _decode_projection_body(
                    row[3], label=f"task validation {row[0]}:{row[1]}"
                ),
            },
        )

        expected_task_specs: list[dict[str, Any]] = []
        for item in population["tasks"]:
            expected = _expected_task_static(item)
            observed = tasks.get(expected["task_cid"])
            if observed is None:
                raise MaterializationError(
                    f"sealed task identity disappeared: {expected['task_alias']}"
                )
            static_keys = (
                "task_cid",
                "task_alias",
                "goal_cid",
                "plan_cid",
                "objective_id",
                "ordinal",
                "priority",
            )
            mismatches = [
                key for key in static_keys if observed.get(key) != expected.get(key)
            ]
            identity = dict(observed["identity"])
            expected_identity = {
                **expected["identity"],
                "repository_tree_id": str(population["repository_tree_id"]),
            }
            body = dict(observed["body"])
            body_mismatches = [
                key for key, value in expected["body"].items() if body.get(key) != value
            ]
            unexpected_body = set(body) - set(expected["body"]) - {"completion_receipt"}
            observed_dependencies = sorted(dependencies.get(expected["task_cid"], []))
            expected_dependencies = sorted(
                (dependency, "depends_on") for dependency in expected["dependencies"]
            )
            if (
                mismatches
                or identity != expected_identity
                or body_mismatches
                or unexpected_body
                or observed_dependencies != expected_dependencies
                or outputs.get(expected["task_cid"], []) != expected["outputs"]
                or acceptance.get(expected["task_cid"], []) != expected["acceptance"]
                or validations.get(expected["task_cid"], []) != expected["validations"]
            ):
                raise MaterializationError(
                    f"sealed task specification changed: {expected['task_alias']}"
                )
            expected_task_specs.append(expected)

        task_histories: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
        for task_cid, revision, status, body_json in connection.execute(
            """SELECT task_cid, revision, status, body_json FROM task_revisions
               ORDER BY task_cid, revision"""
        ).fetchall():
            task_histories.setdefault(str(task_cid), []).append(
                (
                    int(revision),
                    str(status),
                    _decode_projection_body(
                        body_json, label=f"task revision {task_cid}:{revision}"
                    ),
                )
            )
        for task_cid, task in tasks.items():
            if (
                task["goal_cid"] not in goals
                or (task["plan_cid"] and task["plan_cid"] not in plans)
                or (task["objective_id"] and task["objective_id"] not in objectives)
                or task["status"] not in _LIVE_TASK_STATUSES
                or int(task["revision"]) < 1
                or task["identity"].get("task_cid") != task_cid
                or task["identity"].get("task_alias") != task["task_alias"]
            ):
                raise MaterializationError(f"live task has an impossible identity: {task_cid}")
            history = task_histories.get(task_cid, [])
            revisions = [item[0] for item in history]
            if (
                revisions != list(range(1, int(task["revision"]) + 1))
                or not history
                or history[-1][1:] != (task["status"], task["body"])
                or any(status not in _LIVE_TASK_STATUSES for _revision, status, _body in history)
            ):
                raise MaterializationError(f"live task revision history is impossible: {task_cid}")
        for item in population["tasks"]:
            expected = _expected_task_static(item)
            first = task_histories[expected["task_cid"]][0]
            if first != (1, str(item["status"]), expected["body"]):
                raise MaterializationError(
                    f"sealed task revision origin changed: {expected['task_alias']}"
                )

        def verify_revision_chain(
            table: str,
            identity_column: str,
            current: Mapping[str, tuple[Any, ...]],
            revision_index: int,
            body_index: int,
            *,
            has_status: bool,
            status_index: int = -1,
        ) -> dict[str, list[tuple[Any, ...]]]:
            select = (
                f"SELECT {identity_column}, revision, status, body_json FROM {table}"
                if has_status
                else f"SELECT {identity_column}, revision, body_json FROM {table}"
            )
            histories: dict[str, list[tuple[Any, ...]]] = {}
            for row in connection.execute(select + f" ORDER BY {identity_column}, revision").fetchall():
                decoded_body_index = 3 if has_status else 2
                histories.setdefault(str(row[0]), []).append(
                    (
                        *row[1:decoded_body_index],
                        _decode_projection_body(row[decoded_body_index], label=table),
                    )
                )
            for identity, row in current.items():
                revision = int(row[revision_index])
                entries = histories.get(identity, [])
                latest_matches = bool(entries) and entries[-1][-1] == row[body_index]
                if has_status:
                    latest_matches = latest_matches and entries[-1][1] == str(
                        row[status_index]
                    )
                if (
                    [int(item[0]) for item in entries] != list(range(1, revision + 1))
                    or not latest_matches
                ):
                    raise MaterializationError(
                        f"{table} contains impossible revision history: {identity}"
                    )
            return histories

        objective_histories = verify_revision_chain(
            "objective_revisions",
            "objective_id",
            objectives,
            6,
            7,
            has_status=True,
            status_index=4,
        )
        plan_histories = verify_revision_chain(
            "plan_revisions", "plan_cid", plans, 4, 5, has_status=False
        )
        for expected in expected_control["objective_revisions"]:
            if objective_histories.get(expected["objective_id"], [None])[0] != (
                expected["revision"],
                expected["status"],
                expected["body"],
            ):
                raise MaterializationError(
                    f"sealed objective revision origin changed: {expected['objective_id']}"
                )
        for expected in expected_control["plan_revisions"]:
            if plan_histories.get(expected["plan_cid"], [None])[0] != (
                expected["revision"],
                expected["body"],
            ):
                raise MaterializationError(
                    f"sealed plan revision origin changed: {expected['plan_cid']}"
                )

        current_projection = {
            "objectives": len(objectives),
            "goals": len(goals),
            "plans": len(plans),
            "tasks": len(tasks),
            "task_revisions": sum(len(items) for items in task_histories.values()),
            "task_dependencies": len(dependency_rows),
        }
        immutable = {
            "control": expected_control,
            "tasks": expected_task_specs,
        }
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/lgswf-live-control-projection@1",
            "immutable_population_root": _identity(immutable),
            "task_spec_root": _identity(expected_task_specs),
            "counts": current_projection,
            "task_aliases": sorted(aliases),
            "tasks": tasks,
        }
    except MaterializationError:
        raise
    except Exception as exc:
        raise MaterializationError("live control authority cannot be projected read-only") from exc
    finally:
        connection.close()


def _verify_live_coordination_projection(
    projection: Mapping[str, Any],
    population: Mapping[str, Any],
    control_tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate live coordination history without requiring quiescence."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        LeaseKind,
        exclusive_scope_key,
    )

    registered = {
        str(item.get("task_cid") or ""): dict(item)
        for item in projection.get("tasks") or ()
    }
    for expected in population["tasks"]:
        task_cid = str(expected["task_cid"])
        item = registered.get(task_cid)
        item_body = item.get("body") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or item.get("task_id") != str(expected["task_alias"])
            or not isinstance(item_body, Mapping)
            or item_body.get("task_alias") != str(expected["task_alias"])
        ):
            raise MaterializationError(
                f"sealed coordination task specification changed: {expected['task_alias']}"
            )
    for task_cid, item in registered.items():
        control = control_tasks.get(task_cid)
        if (
            not task_cid
            or not isinstance(control, Mapping)
            or str(item.get("task_id") or "") != str(control.get("task_alias") or "")
        ):
            raise MaterializationError("coordination registry contains a foreign task identity")

    edges = [dict(item) for item in projection.get("dependency_edges") or ()]
    by_task: dict[str, set[str]] = {}
    for item in edges:
        task_cid = str(item.get("task_cid") or "")
        dependency = str(item.get("dependency_task_cid") or "")
        if (
            task_cid not in registered
            or dependency not in registered
            or task_cid not in control_tasks
            or dependency not in control_tasks
            or task_cid == dependency
        ):
            raise MaterializationError("coordination dependency contains an impossible identity")
        by_task.setdefault(task_cid, set()).add(dependency)
    for expected in population["tasks"]:
        if by_task.get(str(expected["task_cid"]), set()) != {
            str(item) for item in expected["dependencies"]
        }:
            raise MaterializationError(
                f"sealed coordination dependency changed: {expected['task_alias']}"
            )

    claims = {
        str(item.get("claim_id") or ""): dict(item)
        for item in projection.get("task_claims") or ()
    }
    attempts = {
        str(item.get("attempt_id") or ""): dict(item)
        for item in projection.get("task_attempts") or ()
    }
    leases = {
        str(item.get("lease_id") or ""): dict(item)
        for item in projection.get("fenced_leases") or ()
    }
    task_lease_ids: set[str] = set()
    seen_attempt_numbers: set[tuple[str, int]] = set()
    seen_task_fences: set[tuple[str, int, int]] = set()
    linked_fields = (
        "task_cid",
        "attempt_id",
        "attempt_number",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
    )
    for claim_id, claim in claims.items():
        attempt = attempts.get(str(claim.get("attempt_id") or ""))
        lease = leases.get(str(claim.get("lease_id") or ""))
        task_cid = str(claim.get("task_cid") or "")
        attempt_number = int(claim.get("attempt_number") or 0)
        fence = (
            task_cid,
            int(claim.get("fence_epoch") or 0),
            int(claim.get("fencing_token") or 0),
        )
        if (
            not claim_id
            or task_cid not in registered
            or not isinstance(attempt, Mapping)
            or not isinstance(lease, Mapping)
            or lease.get("lease_kind") != "task"
            or lease.get("mode") != "exclusive"
            or lease.get("scope") != task_cid
            or lease.get("scope_key") != f"task:{task_cid}"
            or lease.get("claim_id") != claim_id
            or any(claim.get(field) != attempt.get(field) for field in linked_fields)
            or any(
                claim.get(field) != lease.get(field)
                for field in (*linked_fields, "claim_id", "lease_id", "idempotency_key", "body")
            )
            or attempt_number <= 0
            or fence[1] <= 0
            or fence[2] <= 0
            or int(claim.get("revision") or 0) <= 0
            or int(attempt.get("revision") or 0) <= 0
            or int(lease.get("revision") or 0) <= 0
            or (task_cid, attempt_number) in seen_attempt_numbers
            or fence in seen_task_fences
        ):
            raise MaterializationError("coordination claim history has impossible identities")
        seen_attempt_numbers.add((task_cid, attempt_number))
        seen_task_fences.add(fence)
        task_lease_ids.add(str(claim.get("lease_id") or ""))
        state_triple = (
            str(claim.get("state") or ""),
            str(attempt.get("status") or ""),
            str(lease.get("state") or ""),
        )
        legal_triples = {
            ("accepted", "running", "accepted"),
            ("expired", "expired", "expired"),
            ("expired", "expired", "superseded"),
            ("released", "released", "released"),
            ("released", "succeeded", "released"),
            ("completed", "succeeded", "completed"),
        }
        if state_triple not in legal_triples:
            raise MaterializationError("coordination claim state history is impossible")
    if set(attempts) != {str(item.get("attempt_id") or "") for item in claims.values()}:
        raise MaterializationError("coordination attempt history contains an unbound identity")

    resource_claims = {
        str(item.get("claim_id") or ""): dict(item)
        for item in projection.get("resource_claims") or ()
    }
    resource_lease_ids: set[str] = set()
    resource_linked = (
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "task_cid",
        "resource_kind",
        "resource_id",
        "repository_id",
        "path",
        "worktree_id",
        "mode",
        "body",
    )
    for claim_id, claim in resource_claims.items():
        lease = leases.get(str(claim.get("lease_id") or ""))
        task_cid = str(claim.get("task_cid") or "")
        resource_kind = str(claim.get("resource_kind") or "")
        expected_kind = {
            "path": "path",
            "provider": "provider_capacity",
            "prover": "prover_capacity",
            "merge": "merge",
        }.get(resource_kind, "resource")
        expected_scope = (
            str(claim.get("path") or claim.get("resource_id") or "")
            if expected_kind == "path"
            else str(claim.get("resource_id") or "")
        )
        expected_scope_key = exclusive_scope_key(
            lease_kind=LeaseKind(expected_kind),
            scope=expected_scope,
            resource_kind=resource_kind,
            resource_id=str(claim.get("resource_id") or ""),
            repository_id=str(claim.get("repository_id") or ""),
            path=str(claim.get("path") or ""),
            task_cid=task_cid,
        )
        state_pair = (
            (str(claim.get("state") or ""), str(lease.get("state") or ""))
            if isinstance(lease, Mapping)
            else ("", "")
        )
        if (
            not claim_id
            or not isinstance(lease, Mapping)
            or lease.get("lease_kind") != expected_kind
            or lease.get("claim_id") != claim_id
            or lease.get("scope") != expected_scope
            or lease.get("scope_key") != expected_scope_key
            or any(claim.get(field) != lease.get(field) for field in resource_linked)
            or (task_cid and task_cid not in control_tasks)
            or int(claim.get("fence_epoch") or 0) <= 0
            or int(claim.get("fencing_token") or 0) <= 0
            or int(claim.get("revision") or 0) <= 0
            or int(lease.get("revision") or 0) <= 0
            or state_pair
            not in {
                ("accepted", "accepted"),
                ("released", "released"),
                ("expired", "expired"),
                ("expired", "superseded"),
                ("completed", "completed"),
            }
        ):
            raise MaterializationError("coordination resource history has impossible identities")
        resource_lease_ids.add(str(claim.get("lease_id") or ""))
    maintenance_lease_ids: set[str] = set()
    for item in projection.get("maintenance_leases") or ():
        maintenance = dict(item)
        lease_id = str(maintenance.get("lease_id") or "")
        lease = leases.get(lease_id)
        if (
            not lease_id
            or not isinstance(lease, Mapping)
            or lease.get("lease_kind") != "maintenance"
            or any(
                maintenance.get(field) != lease.get(field)
                for field in (
                    "lease_id",
                    "scope",
                    "owner_session_id",
                    "fencing_token",
                    "fence_epoch",
                    "body",
                )
            )
            or lease.get("scope_key")
            != exclusive_scope_key(
                lease_kind=LeaseKind.MAINTENANCE,
                scope=str(maintenance.get("scope") or ""),
            )
            or (
                str(maintenance.get("state") or ""),
                str(lease.get("state") or ""),
            )
            not in {
                ("accepted", "accepted"),
                ("released", "released"),
                ("expired", "expired"),
                ("expired", "superseded"),
                ("completed", "completed"),
            }
            or int(maintenance.get("fencing_token") or 0) <= 0
            or int(maintenance.get("fence_epoch") or 0) <= 0
            or int(maintenance.get("revision") or 0) <= 0
        ):
            raise MaterializationError("coordination maintenance history is impossible")
        maintenance_lease_ids.add(lease_id)
    known_lease_ids = task_lease_ids | resource_lease_ids | maintenance_lease_ids
    foreign_leases = [
        item
        for lease_id, item in leases.items()
        if lease_id not in known_lease_ids
    ]
    legal_lease_kinds = {
        "task",
        "resource",
        "path",
        "merge",
        "maintenance",
        "provider_capacity",
        "prover_capacity",
    }
    invalid_kinds = [
        item for item in leases.values() if item.get("lease_kind") not in legal_lease_kinds
    ]
    if foreign_leases or invalid_kinds:
        raise MaterializationError("coordination lease history contains an unbound identity")

    completions = [dict(item) for item in projection.get("logical_completions") or ()]
    for completion in completions:
        task_cid = str(completion.get("task_cid") or "")
        body = completion.get("body")
        body_map = dict(body) if isinstance(body, Mapping) else {}
        claim = claims.get(str(body_map.get("claim_id") or ""))
        attempt = attempts.get(str(body_map.get("attempt_id") or ""))
        lease = leases.get(str(body_map.get("lease_id") or ""))
        if (
            task_cid not in registered
            or completion.get("status") not in {"prepared", "succeeded"}
            or not isinstance(claim, Mapping)
            or not isinstance(attempt, Mapping)
            or not isinstance(lease, Mapping)
            or any(
                body_map.get(field) != claim.get(field)
                for field in (
                    "claim_id",
                    "attempt_id",
                    "attempt_number",
                    "lease_id",
                    "owner_session_id",
                    "fencing_token",
                    "fence_epoch",
                )
            )
            or claim.get("task_cid") != task_cid
        ):
            raise MaterializationError("coordination completion has an impossible claim identity")

    counts = dict(projection.get("counts") or {})
    return {
        "schema": str(projection.get("schema") or ""),
        "projection_root": str(projection.get("projection_root") or ""),
        "counts": counts,
        "active_task_claims": int(counts.get("active_task_claims") or 0),
        "active_resource_claims": int(counts.get("active_resource_claims") or 0),
        "active_fenced_leases": int(counts.get("active_fenced_leases") or 0),
        "projection": dict(projection),
    }


def _verify_live_execution_read_only(
    config: Mapping[str, Any],
    schema_profile: Mapping[str, Any],
    control_tasks: Mapping[str, Mapping[str, Any]],
    coordination_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify live daemon rows and their control/coordination identities."""

    path = _paths(config)["execution"]
    if not path.is_file():
        raise MaterializationError("daemon execution store is absent")
    try:
        import duckdb  # type: ignore

        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        raise MaterializationError("daemon execution store cannot be opened read-only") from exc
    required_tables = {
        "daemon_execution_metadata",
        "database_task_attempts",
        "attempt_phases",
        "provider_invocations",
        "effect_claims",
        "daemon_execution_events",
    }
    try:
        installed = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        if installed != required_tables:
            raise MaterializationError(
                "live daemon execution schema differs from its closed profile"
            )
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM daemon_execution_metadata ORDER BY key"
            ).fetchall()
        }
        expected_metadata = {
            "interface": "DatabaseImplementationDaemon@1",
            "schema": "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1",
            "authority_mode": "embedded",
            "state_schema_revision": str(config["database_program"]["schema_revision"]),
            "control_schema_profile_id": str(schema_profile.get("profile_id") or ""),
            "control_schema_fingerprint": str(schema_profile.get("schema_fingerprint") or ""),
        }
        if (
            any(metadata.get(key) != value for key, value in expected_metadata.items())
            or not metadata.get("logical_owner_session_id")
            or not metadata.get("process_instance_id")
            or set(metadata)
            != {*expected_metadata, "logical_owner_session_id", "process_instance_id"}
        ):
            raise MaterializationError("live daemon execution metadata is stale")

        coordination_attempts = {
            str(item.get("attempt_id") or ""): dict(item)
            for item in coordination_projection.get("task_attempts") or ()
        }
        coordination_claims = {
            str(item.get("attempt_id") or ""): dict(item)
            for item in coordination_projection.get("task_claims") or ()
        }
        attempts: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            """SELECT attempt_id, claim_id, task_cid, task_alias, attempt_number,
                      owner_session_id, fencing_token, fence_epoch, lease_id,
                      committed_phase, status, revision
               FROM database_task_attempts ORDER BY attempt_id"""
        ).fetchall():
            item = {
                "attempt_id": str(row[0]),
                "claim_id": str(row[1]),
                "task_cid": str(row[2]),
                "task_alias": str(row[3]),
                "attempt_number": int(row[4]),
                "owner_session_id": str(row[5]),
                "fencing_token": int(row[6]),
                "fence_epoch": int(row[7]),
                "lease_id": str(row[8]),
                "committed_phase": str(row[9]),
                "status": str(row[10]),
                "revision": int(row[11]),
            }
            control = control_tasks.get(item["task_cid"])
            coordination_attempt = coordination_attempts.get(item["attempt_id"])
            coordination_claim = coordination_claims.get(item["attempt_id"])
            shared = (
                "attempt_id",
                "task_cid",
                "attempt_number",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
            )
            if (
                not isinstance(control, Mapping)
                or item["task_alias"] != control.get("task_alias")
                or not isinstance(coordination_attempt, Mapping)
                or not isinstance(coordination_claim, Mapping)
                or any(item[field] != coordination_attempt.get(field) for field in shared)
                or any(
                    item[field] != coordination_claim.get(field)
                    for field in (*shared, "claim_id", "lease_id")
                )
                or item["attempt_number"] <= 0
                or item["fencing_token"] <= 0
                or item["fence_epoch"] <= 0
                or item["revision"] <= 0
            ):
                raise MaterializationError("daemon execution attempt has an impossible identity")
            attempts[item["attempt_id"]] = item

        phase_names = {
            "claimed",
            "context",
            "provider",
            "effect",
            "validation",
            "complete",
            "failed",
            "blocked",
        }
        phases_by_attempt: dict[str, dict[str, int]] = {}
        for attempt_id, phase, _committed_at, token, epoch, revision, body_json in connection.execute(
            """SELECT attempt_id, phase, committed_at_ms, fencing_token, fence_epoch,
                      revision, body_json
               FROM attempt_phases"""
        ).fetchall():
            attempt = attempts.get(str(attempt_id))
            if (
                not isinstance(attempt, Mapping)
                or str(phase) not in phase_names
                or int(token) != attempt["fencing_token"]
                or int(epoch) != attempt["fence_epoch"]
                or int(revision) <= 0
            ):
                raise MaterializationError("daemon attempt phase has an impossible identity")
            _decode_projection_body(
                body_json, label=f"daemon attempt phase {attempt_id}:{phase}"
            )
            phases_by_attempt.setdefault(str(attempt_id), {})[str(phase)] = int(revision)
        phase_status = {
            "claimed": "running",
            "context": "running",
            "provider": "running",
            "effect": "running",
            "validation": "running",
            "complete": "succeeded",
            "failed": "failed",
            "blocked": "blocked",
        }
        for attempt_id, attempt in attempts.items():
            phases = phases_by_attempt.get(attempt_id, {})
            current_phase = attempt["committed_phase"]
            if (
                "claimed" not in phases
                or current_phase not in phases
                or phases[current_phase] != attempt["revision"]
                or phase_status.get(current_phase) != attempt["status"]
            ):
                raise MaterializationError("daemon attempt has an impossible phase/status history")

        for table in ("provider_invocations", "effect_claims"):
            result_column = "result_json"
            for attempt_id, task_cid, owner_id, result_json in connection.execute(
                f"SELECT attempt_id, task_cid, owner_session_id, {result_column} FROM {table}"
            ).fetchall():
                attempt = attempts.get(str(attempt_id))
                if (
                    not isinstance(attempt, Mapping)
                    or str(task_cid) != attempt["task_cid"]
                    or str(owner_id) != attempt["owner_session_id"]
                ):
                    raise MaterializationError(
                        f"{table} contains an impossible attempt identity"
                    )
                _decode_projection_body(
                    result_json, label=f"{table} result {attempt_id}"
                )
        for attempt_id, task_cid, body_json in connection.execute(
            "SELECT attempt_id, task_cid, body_json FROM daemon_execution_events"
        ).fetchall():
            attempt_key, task_key = str(attempt_id or ""), str(task_cid or "")
            if (
                attempt_key
                and attempt_key not in attempts
                or task_key
                and task_key not in control_tasks
                or attempt_key
                and task_key
                and attempts[attempt_key]["task_cid"] != task_key
            ):
                raise MaterializationError("daemon event contains an impossible identity")
            _decode_projection_body(body_json, label=f"daemon event {attempt_key or task_key}")
        row_counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in sorted(required_tables - {"daemon_execution_metadata"})
        }
        report = {
            "required_tables": sorted(required_tables),
            "metadata": metadata,
            "row_counts": row_counts,
        }
        report["live_execution_root"] = _identity(report)
        return report
    except MaterializationError:
        raise
    except Exception as exc:
        raise MaterializationError("live daemon execution authority is unreadable") from exc
    finally:
        connection.close()


def materialize(config: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    source_verification = _assert_population_source_current(config, population)
    paths = _paths(config)
    qualification_receipt = _load_qualification_receipt(config, population)
    immutable_receipt_paths = (
        _bootstrap_receipt_path(config, "duckdb-materialization.json"),
        _bootstrap_receipt_path(config, "duckdb-seal.json"),
    )
    existing = [
        path.relative_to(ROOT).as_posix()
        for path in (*paths.values(), *immutable_receipt_paths)
        if path.exists()
    ]
    if existing:
        raise MaterializationError(
            "refusing to overwrite an existing control plane: " + ", ".join(existing)
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    schema_install = install_datasets_authoritative_operational_schema(
        paths["control"],
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="lgswf-materializer:operational-schema",
    )
    schema_env = "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"
    prior_schema_revision = os.environ.get(schema_env)
    os.environ[schema_env] = str(config["database_program"]["schema_revision"])
    daemon: DatabaseImplementationDaemon | None = None
    try:
        daemon = DatabaseImplementationDaemon(
            database_path=paths["control"],
            coordination_path=paths["coordination"],
            execution_path=paths["execution"],
            owner_session_id="lgswf-materializer:single-writer",
            authority_mode="embedded",
            task_source_kind="duckdb",
            install_schema=False,
        )
        database_receipt = daemon.materialize_population(
            population,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
    finally:
        if daemon is not None:
            daemon.close()
        if prior_schema_revision is None:
            os.environ.pop(schema_env, None)
        else:
            os.environ[schema_env] = prior_schema_revision
    task_source_receipt = (
        database_receipt.get("task_source") if isinstance(database_receipt, Mapping) else None
    )
    expected_task_cids = [str(item["task_cid"]) for item in population["tasks"]]
    expected_materialization = {
        "task_count": len(expected_task_cids),
        "goal_count": len(population["objectives"]),
        "goal_edge_count": len(population.get("goal_edges") or ()),
        "plan_count": len(population["plans"]),
        "task_cids": expected_task_cids,
    }
    if (
        not isinstance(task_source_receipt, Mapping)
        or any(
            task_source_receipt.get(key) != value for key, value in expected_materialization.items()
        )
        or list(database_receipt.get("registered_task_cids") or []) != expected_task_cids
    ):
        raise MaterializationError(
            "database materialization did not preserve the exact task/goal graph"
        )
    verified = _verify_store(config, population, expected_stage="unsealed")
    receipt = {
        "schema": SCHEMA,
        "authority_mode": "embedded",
        "maximum_writer_processes": 1,
        "task_source_kind": "duckdb",
        "schema_revision": config["database_program"]["schema_revision"],
        "schema_profile": config["database_program"]["schema_profile"],
        "semantic_truth_authority": "ipfs_datasets_py",
        "operational_coordination_authority": "ipfs_accelerate_py",
        "qualification_receipt_cid": qualification_receipt["receipt_cid"],
        "source_verification": source_verification,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "database_paths": {
            key: path.relative_to(ROOT).as_posix() for key, path in sorted(paths.items())
        },
        "schema_install": (
            schema_install.to_dict()
            if callable(getattr(schema_install, "to_dict", None))
            else dict(schema_install)
        ),
        "materialization": dict(database_receipt),
        "verification": verified,
        "population_cid": _identity(population),
    }
    receipt["receipt_cid"] = _identity(receipt)
    evidence_root = _relative_path(
        config["runtime_paths"]["evidence"], field="runtime_paths.evidence"
    )
    receipt_path = evidence_root / "bootstrap" / "duckdb-materialization.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=receipt_path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, receipt_path)
    return _receipt_result(
        operation="materialize",
        receipt=receipt,
        path=receipt_path,
        replayed=False,
    )


def _bootstrap_receipt_path(config: Mapping[str, Any], name: str) -> Path:
    evidence_root = _relative_path(
        config["runtime_paths"]["evidence"], field="runtime_paths.evidence"
    )
    return evidence_root / "bootstrap" / name


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _receipt_result(
    *,
    operation: str,
    receipt: Mapping[str, Any],
    path: Path,
    replayed: bool,
) -> dict[str, Any]:
    """Return an invocation envelope without mislabelling it as the receipt."""

    return {
        "schema": RECEIPT_RESULT_SCHEMA,
        "operation": operation,
        "canonical_receipt": dict(receipt),
        "canonical_receipt_path": path.relative_to(ROOT).as_posix(),
        "operation_replayed": bool(replayed),
    }


def _qualification_commands() -> tuple[tuple[str, tuple[str, ...], int | None], ...]:
    python = str(Path(sys.executable).resolve())
    pytest_prefix = (
        python,
        "-E",
        "-P",
        "-m",
        "pytest",
        "-q",
        "--color=no",
        "-o",
        "console_output_style=classic",
        "-p",
        "no:cacheprovider",
    )
    return (
        (
            "operational_schema",
            (
                *pytest_prefix,
                "test/api/test_agent_supervisor_control_plane_schema.py",
                "test/api/test_agent_supervisor_datasets_authoritative_operational_schema.py",
            ),
            14,
        ),
        (
            "intent_repository",
            (
                *pytest_prefix,
                "test/api/test_agent_supervisor_intent_repository.py",
            ),
            12,
        ),
        (
            "semantic_and_proof_writer_guards",
            (
                *pytest_prefix,
                "test/api/test_agent_supervisor_datasets_authoritative_writer_guards.py",
                "test/api/test_agent_supervisor_database_evidence_stores.py",
                "test/api/test_agent_supervisor_database_symbolic_repair.py",
                "test/api/test_agent_supervisor_datasets_authoritative_proof_writer_guards.py",
            ),
            62,
        ),
        (
            "coordination_daemon_portal_runner",
            (
                *pytest_prefix,
                "test/api/test_agent_supervisor_database_coordination.py",
                "test/api/test_agent_supervisor_database_implementation_daemon.py",
                "test/api/test_agent_supervisor_database_portal_bridge.py",
                "test/api/test_agent_supervisor_database_runner_propagation.py",
            ),
            89,
        ),
        (
            "configured_board_live_seal_gate",
            (
                *pytest_prefix,
                "test/api/test_agent_supervisor_configured_board_scheduler.py::test_configured_board_live_seal_dry_profile_binds_target_and_no_go",
                "test/api/test_agent_supervisor_configured_board_scheduler.py::test_configured_board_live_seal_target_swap_changes_dry_profile",
                "test/api/test_agent_supervisor_configured_board_scheduler.py::test_configured_board_live_seal_no_go_is_zero_popen_and_zero_io",
                "test/api/test_agent_supervisor_configured_board_scheduler.py::test_configured_board_live_seal_profile_and_flag_remain_bidirectional",
                "test/api/test_agent_supervisor_configured_board_scheduler.py::test_configured_board_live_seal_start_and_detach_are_zero_effect_no_go",
                "test/api/test_agent_supervisor_configured_board_scheduler.py::test_configured_board_live_seal_real_birth_rejects_before_startup_hook",
            ),
            7,
        ),
        (
            "bootstrap_seal",
            (
                *pytest_prefix,
                "test/api/test_agent_supervisor_lgswf_materialization_seal.py",
            ),
            34,
        ),
        (
            "board_structure",
            (
                python,
                "-E",
                "-P",
                "scripts/validate_logic_governed_semantic_work_fabric_board.py",
            ),
            None,
        ),
    )


def _read_self_addressed_receipt(path: Path, *, label: str) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"{label} receipt is absent or unreadable") from exc
    if not isinstance(receipt, dict):
        raise MaterializationError(f"{label} receipt is not an object")
    claimed_cid = str(receipt.get("receipt_cid") or "")
    unsigned = dict(receipt)
    unsigned.pop("receipt_cid", None)
    if not claimed_cid or _identity(unsigned) != claimed_cid:
        raise MaterializationError(f"{label} receipt CID does not verify")
    return receipt


def _load_qualification_receipt(
    config: Mapping[str, Any], population: Mapping[str, Any]
) -> dict[str, Any]:
    path = _bootstrap_receipt_path(config, "qualification.json")
    receipt = _read_self_addressed_receipt(path, label="bootstrap qualification")
    source_verification = _assert_population_source_current(config, population)
    expected = {
        "schema": QUALIFICATION_SCHEMA,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": _identity(population),
        "source_verification": source_verification,
        "command_argv": [list(argv) for _label, argv, _expected in _qualification_commands()],
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise MaterializationError(
            "bootstrap qualification receipt is stale: " + ", ".join(mismatches)
        )
    results = receipt.get("results")
    commands = _qualification_commands()
    result_bindings_valid = isinstance(results, list) and len(results) == len(commands)
    if result_bindings_valid:
        for item, (label, argv, expected_passed) in zip(results, commands, strict=True):
            if (
                not isinstance(item, Mapping)
                or item.get("label") != label
                or item.get("argv") != list(argv)
                or item.get("returncode") != 0
                or item.get("expected_passed") != expected_passed
                or item.get("required_outcomes_valid") is not True
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item.get("stdout_sha256") or ""))
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item.get("stderr_sha256") or ""))
            ):
                result_bindings_valid = False
                break
    if receipt.get("qualified") is not True or not result_bindings_valid:
        raise MaterializationError("bootstrap qualification did not pass")
    return receipt


def qualify(config: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    """Run deterministic local authorities and persist their exact-tree receipt."""

    source_verification = _assert_population_source_current(config, population)
    path = _bootstrap_receipt_path(config, "qualification.json")
    if path.exists():
        receipt = _read_self_addressed_receipt(path, label="bootstrap qualification")
        # A failed or stale receipt is immutable evidence; callers must advance
        # to a fresh namespace rather than overwrite it.
        _load_qualification_receipt(config, population)
        return _receipt_result(operation="qualify", receipt=receipt, path=path, replayed=True)

    results: list[dict[str, Any]] = []
    for label, argv, expected_passed in _qualification_commands():
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(argv),
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=900,
            )
            returncode = int(completed.returncode)
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "") + "\nqualification command timed out"
        observed_passed: int | None = None
        required_outcomes_valid = returncode == 0
        if expected_passed is not None:
            plain_stdout = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout)
            passed_matches = re.findall(
                r"(?m)^(?:=+\s*)?(\d+) passed\b",
                plain_stdout,
            )
            if not passed_matches:
                passed_matches = re.findall(r"\b(\d+) passed\b", plain_stdout)
            observed_passed = int(passed_matches[-1]) if passed_matches else None
            summary_lines = [
                line
                for line in plain_stdout.splitlines()
                if re.search(r"\b\d+ passed\b", line)
                or re.search(r"=+\s+.+\s+=+", line)
            ]
            summary_text = "\n".join(summary_lines).casefold()
            prohibited_outcomes = re.search(
                r"(?:^|[ ,=])(?:[1-9][0-9]* )?(?:failed|error|errors|skipped|xfailed|xpassed)(?:[,\s=]|$)",
                summary_text,
            )
            required_outcomes_valid = (
                required_outcomes_valid
                and observed_passed == expected_passed
                and prohibited_outcomes is None
            )
        else:
            try:
                board_report = json.loads(stdout)
            except json.JSONDecodeError:
                board_report = None
            required_outcomes_valid = (
                required_outcomes_valid
                and isinstance(board_report, Mapping)
                and board_report.get("valid") is True
            )
        results.append(
            {
                "label": label,
                "argv": list(argv),
                "returncode": returncode,
                "expected_passed": expected_passed,
                "observed_passed": observed_passed,
                "required_outcomes_valid": required_outcomes_valid,
                "duration_ms": max(0, int((time.monotonic() - started) * 1000)),
                "stdout_sha256": _identity(stdout.encode("utf-8")),
                "stderr_sha256": _identity(stderr.encode("utf-8")),
                "stdout_tail": stdout[-2048:],
                "stderr_tail": stderr[-2048:],
            }
        )
    receipt: dict[str, Any] = {
        "schema": QUALIFICATION_SCHEMA,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": _identity(population),
        "source_verification": source_verification,
        "command_argv": [list(argv) for _label, argv, _expected in _qualification_commands()],
        "qualified": all(item["required_outcomes_valid"] is True for item in results),
        "results": results,
    }
    receipt["receipt_cid"] = _identity(receipt)
    _write_receipt(path, receipt)
    if receipt["qualified"] is not True:
        raise MaterializationError(
            "bootstrap qualification failed; immutable receipt: "
            + path.relative_to(ROOT).as_posix()
        )
    return _receipt_result(operation="qualify", receipt=receipt, path=path, replayed=False)


def _load_materialization_receipt(
    config: Mapping[str, Any], population: Mapping[str, Any]
) -> dict[str, Any]:
    path = _bootstrap_receipt_path(config, "duckdb-materialization.json")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError("verified materialization receipt is absent") from exc
    if not isinstance(receipt, dict):
        raise MaterializationError("materialization receipt is not an object")
    claimed_cid = str(receipt.get("receipt_cid") or "")
    unsigned = dict(receipt)
    unsigned.pop("receipt_cid", None)
    if not claimed_cid or _identity(unsigned) != claimed_cid:
        raise MaterializationError("materialization receipt CID does not verify")
    qualification_receipt = _load_qualification_receipt(config, population)
    database_paths = {
        key: path.relative_to(ROOT).as_posix() for key, path in sorted(_paths(config).items())
    }
    expected = {
        "schema": SCHEMA,
        "authority_mode": "embedded",
        "maximum_writer_processes": 1,
        "task_source_kind": "duckdb",
        "semantic_truth_authority": "ipfs_datasets_py",
        "operational_coordination_authority": "ipfs_accelerate_py",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": _identity(population),
        "schema_revision": config["database_program"]["schema_revision"],
        "schema_profile": config["database_program"]["schema_profile"],
        "qualification_receipt_cid": qualification_receipt["receipt_cid"],
        "source_verification": qualification_receipt["source_verification"],
        "database_paths": database_paths,
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise MaterializationError("materialization receipt is stale: " + ", ".join(mismatches))
    verification = receipt.get("verification")
    if not isinstance(verification, Mapping) or verification.get("bootstrap_stage") != "unsealed":
        raise MaterializationError("materialization receipt is not the unsealed stage")
    materialization = receipt.get("materialization")
    task_source_receipt = (
        materialization.get("task_source") if isinstance(materialization, Mapping) else None
    )
    expected_task_cids = [str(item["task_cid"]) for item in population["tasks"]]
    expected_materialization = {
        "task_count": len(expected_task_cids),
        "goal_count": len(population["objectives"]),
        "goal_edge_count": len(population.get("goal_edges") or ()),
        "plan_count": len(population["plans"]),
        "task_cids": expected_task_cids,
    }
    if (
        not isinstance(task_source_receipt, Mapping)
        or any(
            task_source_receipt.get(key) != value for key, value in expected_materialization.items()
        )
        or not isinstance(materialization, Mapping)
        or list(materialization.get("registered_task_cids") or []) != expected_task_cids
    ):
        raise MaterializationError(
            "materialization receipt does not bind the exact task/goal graph"
        )
    return receipt


def _load_existing_seal_receipt(
    config: Mapping[str, Any], population: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _bootstrap_receipt_path(config, "duckdb-seal.json")
    if not path.is_file():
        return None
    receipt = _read_self_addressed_receipt(path, label="bootstrap seal")
    materialization_receipt = _load_materialization_receipt(config, population)
    qualification_receipt = _load_qualification_receipt(config, population)
    launch_plan = _render_launch_plan_evidence(config)
    current = _verify_store(config, population, expected_stage="sealed")
    expected = {
        "schema": "ipfs_accelerate_py/agent-supervisor/lgswf-bootstrap-seal@1",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": _identity(population),
        "materialization_receipt_cid": materialization_receipt["receipt_cid"],
        "qualification_receipt_cid": qualification_receipt["receipt_cid"],
        "launch_plan": launch_plan,
        "accepted_result_cid": current["accepted_result_cid"],
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise MaterializationError("existing bootstrap seal is stale: " + ", ".join(mismatches))
    post = receipt.get("post_verification")
    if (
        not isinstance(post, Mapping)
        or post.get("accepted_result_cid") != current["accepted_result_cid"]
        or post.get("task_spec_root") != current.get("task_spec_root")
        or post.get("completion_binding") != current.get("completion_binding")
    ):
        raise MaterializationError("existing bootstrap seal post-verification is stale or forged")
    binding = current.get("completion_binding")
    authority_keys = (
        "claim",
        "preparation",
        "validation_receipt",
        "seal_basis_evidence_receipt",
        "control_cas",
        "coordination_promotion",
        "cross_store_guard",
        "settled_lease",
        "writer_reservation",
        "writer_release",
    )
    authority = {key: receipt.get(key) for key in authority_keys}
    if any(not isinstance(value, Mapping) for value in authority.values()) or receipt.get(
        "authority_root"
    ) != _identity(authority):
        raise MaterializationError("bootstrap seal durable authority is incomplete")
    durable_authority = _durable_seal_authority(
        config,
        population,
        current,
        writer_reservation=dict(authority["writer_reservation"]),
    )
    if any(authority[key] != durable_authority.get(key) for key in authority_keys):
        raise MaterializationError(
            "bootstrap seal durable authority differs from current attempt-bound evidence"
        )
    claim = authority["claim"]
    preparation = authority["preparation"]
    validation = authority["validation_receipt"]
    basis_evidence = authority["seal_basis_evidence_receipt"]
    control_cas = authority["control_cas"]
    promotion = authority["coordination_promotion"]
    settled = authority["settled_lease"]
    bound_fields = (
        "claim_id",
        "attempt_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
    )
    if (
        not isinstance(binding, Mapping)
        or any(claim.get(field) != binding.get(field) for field in bound_fields)
        or any(preparation.get(field) != binding.get(field) for field in bound_fields)
        or any(promotion.get(field) != binding.get(field) for field in bound_fields)
        or any(settled.get(field) != binding.get(field) for field in bound_fields)
        or preparation.get("preparation_digest") != binding.get("preparation_digest")
        or preparation.get("evidence_digest") != current["accepted_result_cid"]
        or claim.get("state") not in {"released", "completed"}
        or settled.get("state") != claim.get("state")
        or promotion.get("status") != "succeeded"
        or validation.get("outcome") != "passed"
        or validation.get("evidence_digest") != qualification_receipt["receipt_cid"]
        or basis_evidence.get("digest") != current["accepted_result_cid"]
        or control_cas.get("previous_status") != preparation.get("control_expected_status")
        or control_cas.get("task", {}).get("status") != "completed"
    ):
        raise MaterializationError(
            "bootstrap seal detailed authority differs from promoted completion"
        )
    return receipt


def _render_launch_plan_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        configured_board_launch_plan,
        load_configured_board,
    )

    board = load_configured_board(CONFIG_PATH, repo_root=ROOT)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        duration_seconds=60.0,
        stamp="lgswf-bootstrap-seal",
    )
    program = plan.get("database_program")
    if not isinstance(program, Mapping):
        raise MaterializationError("launch plan omitted its database program")
    expected_program = {
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "schema_revision": "datasets-authoritative-operational-v1",
    }
    mismatches = [key for key, value in expected_program.items() if program.get(key) != value]
    if mismatches:
        raise MaterializationError(
            "launch plan lost operational authority fields: " + ", ".join(mismatches)
        )
    environment = plan.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION")
        != expected_program["schema_revision"]
    ):
        raise MaterializationError("launch plan did not propagate the operational schema revision")
    # This bootstrap deliberately uses the existing configured-board profile,
    # whose one embedded daemon claims from DuckDB but does not yet consume a
    # PlanRevisionStore execution slice.  LGSWF-005 owns that immutable
    # plan-bound cutover.  Record this limitation exactly instead of claiming
    # that the legacy profile is plan-bound.
    if (
        plan.get("implement") is not True
        or int(plan.get("lanes") or 0) != 1
        or int(plan.get("admitted_lanes") or 0) != 1
        or plan.get("plan_bound_dispatch") is not False
        or plan.get("effective_strict_task_sharding") is not True
    ):
        raise MaterializationError("launch plan is not the bounded implementation profile")
    return {
        "launch_plan_cid": _identity(plan),
        "schema": str(plan.get("schema") or ""),
        "authority_mode": str(program.get("authority_mode") or ""),
        "task_source_kind": str(program.get("task_source_kind") or ""),
        "schema_revision": str(program.get("schema_revision") or ""),
        "configured_schema_profile": config["database_program"]["schema_profile"],
        "semantic_relations_permitted": config["database_program"]["semantic_relations_permitted"],
        "lanes": int(plan["lanes"]),
        "admitted_lanes": int(plan["admitted_lanes"]),
        "plan_bound_dispatch": False,
        "effective_strict_task_sharding": True,
        "plan_bound_promotion_task": "LGSWF-005",
        "implement": True,
        "process_started": False,
    }


def _git_blob_cid(source_head: str, path: Path) -> str:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    completed = subprocess.run(
        ["git", "show", f"{source_head}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise MaterializationError(f"sealed source blob is unavailable: {relative}")
    return _identity(completed.stdout)


def _control_bundle_cid(config: Mapping[str, Any], population: Mapping[str, Any]) -> str:
    """Hash control inputs from the bound commit, never mutable worktree bytes."""

    source_head = str(population["source_head"])
    controls = {
        "config": CONFIG_PATH,
        "baseline": _relative_path(
            (config.get("source_binding") or {}).get("baseline_path"),
            field="source_binding.baseline_path",
        ),
        "board": _relative_path(config.get("taskboard_path"), field="taskboard_path"),
        "objectives": _relative_path(config.get("objectives_path"), field="objectives_path"),
        "plan": _relative_path(config.get("plan_path"), field="plan_path"),
    }
    return _identity(
        {key: _git_blob_cid(source_head, path) for key, path in sorted(controls.items())}
    )


def _build_seal_basis(
    *,
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    qualification_receipt: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    verification = materialization_receipt.get("verification")
    if not isinstance(verification, Mapping):
        raise MaterializationError("materialization receipt lost its verification")
    schema_profile = verification.get("schema_profile")
    if not isinstance(schema_profile, Mapping):
        raise MaterializationError("materialization receipt lost its schema profile")
    task_source_snapshot = verification.get("task_source_snapshot")
    execution_store = verification.get("execution_store")
    control_population = verification.get("control_population")
    coordination_registry = verification.get("coordination_registry")
    if (
        not isinstance(task_source_snapshot, Mapping)
        or not isinstance(execution_store, Mapping)
        or not isinstance(control_population, Mapping)
        or not isinstance(coordination_registry, Mapping)
    ):
        raise MaterializationError(
            "materialization receipt lost a control, coordination or execution snapshot"
        )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/lgswf-bootstrap-seal-basis@1",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": _identity(population),
        "materialization_receipt_cid": materialization_receipt["receipt_cid"],
        "qualification_receipt_cid": qualification_receipt["receipt_cid"],
        "control_bundle_cid": _control_bundle_cid(config, population),
        "launch_plan": dict(launch_plan),
        "schema_profile_fingerprint": schema_profile.get("schema_fingerprint"),
        "task_source_snapshot": dict(task_source_snapshot),
        "task_spec_root": verification.get("task_spec_root"),
        "control_population_root": control_population.get("population_root"),
        "execution_store_root": execution_store.get("execution_store_root"),
        "coordination_projection_root": coordination_registry.get("projection_root"),
        "coordination_registry_spec_root": coordination_registry.get("registry_spec_root"),
        "ready_task_aliases": list(verification.get("ready_task_aliases") or []),
    }


def _unsettled_completion_projection(
    projection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Derive unsettled barriers from a non-mutating coordination projection."""

    claims = {str(item.get("claim_id") or ""): item for item in projection.get("task_claims") or ()}
    attempts = {
        str(item.get("attempt_id") or ""): item for item in projection.get("task_attempts") or ()
    }
    leases = {
        str(item.get("lease_id") or ""): item for item in projection.get("fenced_leases") or ()
    }
    result: list[dict[str, Any]] = []
    for item in projection.get("logical_completions") or ():
        status = str(item.get("status") or "")
        body = item.get("body")
        body_map = dict(body) if isinstance(body, Mapping) else {}
        claim = claims.get(str(body_map.get("claim_id") or ""), {})
        attempt = attempts.get(str(body_map.get("attempt_id") or ""), {})
        lease = leases.get(str(body_map.get("lease_id") or ""), {})
        settled = (
            status == "succeeded"
            and claim.get("state") == "released"
            and attempt.get("status") == "succeeded"
            and lease.get("state") == "released"
        )
        if settled:
            continue
        result.append(
            {
                **body_map,
                "task_cid": str(item.get("task_cid") or body_map.get("task_cid") or ""),
                "status": status,
                "lease_state": str(lease.get("state") or ""),
                "owner_session_id": str(claim.get("owner_session_id") or ""),
                "claim_state": str(claim.get("state") or ""),
                "attempt_status": str(attempt.get("status") or ""),
            }
        )
    return result


def _manual_seal_attempt_binding(claim: Any) -> dict[str, Any]:
    """Return the exact attempt/fence fields governing seal evidence."""

    return {
        "task_cid": str(claim.task_cid),
        "claim_id": str(claim.claim_id),
        "attempt_id": str(claim.attempt_id),
        "lease_id": str(claim.lease_id),
        "fencing_token": int(claim.fencing_token),
        "fence_epoch": int(claim.fence_epoch),
    }


def _manual_seal_validation_body(
    *,
    binding: Mapping[str, Any],
    qualification_receipt_cid: str,
    seal_basis_cid: str,
    superseded_partial_evidence: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {
        "qualification_receipt_cid": qualification_receipt_cid,
        "seal_basis_cid": seal_basis_cid,
        "live_supervisor_process_started": False,
        "daemon_process_started": False,
        "qualification_subprocesses_started": True,
        "stage_guard_policy": {
            "schema": _MANUAL_SEAL_STAGE_GUARD_POLICY_SCHEMA,
            "stage": "validation",
            "required": True,
        },
        **dict(binding),
    }
    if superseded_partial_evidence:
        result["superseded_partial_evidence"] = [
            dict(item) for item in superseded_partial_evidence
        ]
    return result


def _manual_seal_basis_evidence_body(
    *,
    binding: Mapping[str, Any],
    qualification_receipt_cid: str,
    materialization_receipt_cid: str,
    superseded_partial_evidence: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {
        "qualification_receipt_cid": qualification_receipt_cid,
        "materialization_receipt_cid": materialization_receipt_cid,
        "stage_guard_policy": {
            "schema": _MANUAL_SEAL_STAGE_GUARD_POLICY_SCHEMA,
            "stage": "basis_evidence",
            "required": True,
        },
        **dict(binding),
    }
    if superseded_partial_evidence:
        result["superseded_partial_evidence"] = [
            dict(item) for item in superseded_partial_evidence
        ]
    return result


def _read_manual_seal_evidence(
    *,
    control_path: Path,
    task_cid: str,
    qualification_receipt_cid: str,
    seal_basis_cid: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project retry-relevant validation and basis evidence without mutation."""

    try:
        import duckdb  # type: ignore

        connection = duckdb.connect(str(control_path), read_only=True)
    except Exception as exc:
        raise MaterializationError("manual seal evidence cannot be opened read-only") from exc
    try:
        validation_rows = connection.execute(
            """
            SELECT result.result_id, result.run_id, result.task_cid,
                   result.ordinal, result.outcome, result.evidence_digest,
                   result.body_json, run.attempt_id, run.status,
                   run.command_digest, run.body_json
            FROM validation_results AS result
            JOIN validation_runs AS run ON run.run_id = result.run_id
            WHERE result.task_cid = ? AND result.evidence_digest = ?
            ORDER BY result.result_id
            """,
            [task_cid, qualification_receipt_cid],
        ).fetchall()
        evidence_rows = connection.execute(
            """
            SELECT evidence_id, parent_evidence_id, task_cid, evidence_kind,
                   digest, body_json
            FROM evidence_nodes
            WHERE task_cid = ? AND evidence_kind = 'bootstrap_seal_basis'
                  AND digest = ?
            ORDER BY evidence_id
            """,
            [task_cid, seal_basis_cid],
        ).fetchall()
    finally:
        connection.close()
    validations = [
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/durable-validation-projection@1",
            "result_id": str(row[0]),
            "run_id": str(row[1]),
            "task_cid": str(row[2]),
            "ordinal": int(row[3]),
            "outcome": str(row[4]),
            "evidence_digest": str(row[5]),
            "body": _decode_projection_body(row[6], label="bootstrap validation result body"),
            "attempt_id": str(row[7] or ""),
            "run_status": str(row[8]),
            "command_digest": str(row[9]),
            "run_body": _decode_projection_body(
                row[10], label="bootstrap validation run body"
            ),
        }
        for row in validation_rows
    ]
    evidence = [
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/durable-evidence-projection@1",
            "evidence_id": str(row[0]),
            "parent_evidence_id": str(row[1] or ""),
            "task_cid": str(row[2]),
            "evidence_kind": str(row[3]),
            "digest": str(row[4]),
            "body": _decode_projection_body(row[5], label="bootstrap seal-basis evidence body"),
        }
        for row in evidence_rows
    ]
    return validations, evidence


def _validate_manual_seal_evidence(
    *,
    validations: list[Mapping[str, Any]],
    evidence: list[Mapping[str, Any]],
    binding: Mapping[str, Any],
    qualification_receipt_cid: str,
    materialization_receipt_cid: str,
    seal_basis_cid: str,
    require_validation: bool,
    require_evidence: bool,
    superseded_partial_evidence: list[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select only exact current-attempt evidence; history is checked separately."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    evidence_binding = {
        field: binding.get(field)
        for field in (
            "task_cid",
            "claim_id",
            "attempt_id",
            "lease_id",
            "fencing_token",
            "fence_epoch",
        )
    }
    argv = ["verified-content-addressed-bootstrap-qualification"]
    validation_body = _manual_seal_validation_body(
        binding=evidence_binding,
        qualification_receipt_cid=qualification_receipt_cid,
        seal_basis_cid=seal_basis_cid,
        superseded_partial_evidence=superseded_partial_evidence,
    )
    evidence_body = _manual_seal_basis_evidence_body(
        binding=evidence_binding,
        qualification_receipt_cid=qualification_receipt_cid,
        materialization_receipt_cid=materialization_receipt_cid,
        superseded_partial_evidence=superseded_partial_evidence,
    )
    validation_rows = [
        dict(validation)
        for validation in validations
        if validation.get("attempt_id") == evidence_binding.get("attempt_id")
        or (
            isinstance(validation.get("body"), Mapping)
            and validation["body"].get("claim_id") == evidence_binding.get("claim_id")
        )
    ]
    exact_validations = [
        validation
        for validation in validation_rows
        if not (
            validation.get("task_cid") != evidence_binding.get("task_cid")
            or validation.get("attempt_id") != evidence_binding.get("attempt_id")
            or validation.get("ordinal") != 0
            or validation.get("outcome") != "passed"
            or validation.get("run_status") != "passed"
            or validation.get("evidence_digest") != qualification_receipt_cid
            or validation.get("command_digest") != content_identity({"argv": argv})
            or validation.get("body") != validation_body
            or validation.get("run_body") != {"argv": argv, **validation_body}
        )
    ]
    if len(validation_rows) != len(exact_validations) or len(exact_validations) > 1:
        raise MaterializationError(
            "manual seal validation is bound to a different attempt or fence"
        )
    evidence_rows = [
        dict(basis_evidence)
        for basis_evidence in evidence
        if isinstance(basis_evidence.get("body"), Mapping)
        and (
            basis_evidence["body"].get("attempt_id") == evidence_binding.get("attempt_id")
            or basis_evidence["body"].get("claim_id") == evidence_binding.get("claim_id")
        )
    ]
    exact_evidence = [
        basis_evidence
        for basis_evidence in evidence_rows
        if not (
            basis_evidence.get("task_cid") != evidence_binding.get("task_cid")
            or basis_evidence.get("parent_evidence_id") != ""
            or basis_evidence.get("evidence_kind") != "bootstrap_seal_basis"
            or basis_evidence.get("digest") != seal_basis_cid
            or basis_evidence.get("body") != evidence_body
        )
    ]
    if len(evidence_rows) != len(exact_evidence) or len(exact_evidence) > 1:
        raise MaterializationError(
            "manual seal basis evidence is bound to a different attempt or fence"
        )
    if require_validation and len(exact_validations) != 1:
        raise MaterializationError("manual seal does not have one exact validation receipt")
    if require_evidence and len(exact_evidence) != 1:
        raise MaterializationError("manual seal does not have one exact basis evidence receipt")
    return (
        exact_validations[0] if exact_validations else None,
        exact_evidence[0] if exact_evidence else None,
    )


def _manual_seal_evidence_binding(body: Mapping[str, Any]) -> dict[str, Any]:
    """Project the exact task authority carried by one evidence row."""

    fields = (
        "task_cid",
        "claim_id",
        "attempt_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
    )
    binding = {field: body.get(field) for field in fields}
    if (
        any(not str(binding[field] or "") for field in fields[:4])
        or type(binding["fencing_token"]) is not int
        or type(binding["fence_epoch"]) is not int
        or int(binding["fencing_token"]) < 1
        or int(binding["fence_epoch"]) < 1
    ):
        raise MaterializationError("manual seal evidence omitted its exact task fence")
    return binding


def _manual_seal_stage_receipt(
    *, stage: str, evidence_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in {"validation", "basis_evidence"}:
        raise MaterializationError(f"unsupported manual seal evidence stage {stage!r}")
    return {
        "schema": _MANUAL_SEAL_STAGE_RECEIPT_SCHEMA,
        "stage": stage,
        "evidence_receipt": dict(evidence_receipt),
    }


def _read_manual_seal_guard_events(coordination_path: Path) -> list[dict[str, Any]]:
    """Read immutable cross-store guard events without mutating coordination."""

    try:
        import duckdb  # type: ignore

        connection = duckdb.connect(str(coordination_path), read_only=True)
    except Exception as exc:
        raise MaterializationError("manual seal guard history cannot be opened read-only") from exc
    try:
        rows = connection.execute(
            """
            SELECT event_id, lease_id, scope_key, event_type, fencing_token,
                   fence_epoch, observed_at_ms, body_json
            FROM lease_events
            WHERE event_type = 'cross_store_fence_guard_succeeded'
            ORDER BY observed_at_ms, event_id
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "event_id": str(row[0]),
            "lease_id": str(row[1]),
            "scope_key": str(row[2]),
            "event_type": str(row[3]),
            "fencing_token": int(row[4]),
            "fence_epoch": int(row[5]),
            "observed_at_ms": int(row[6]),
            "body": _decode_projection_body(row[7], label=f"lease event {row[0]} body"),
        }
        for row in rows
    ]


def _manual_seal_stage_guard(
    *,
    stage: str,
    evidence_receipt: Mapping[str, Any],
    binding: Mapping[str, Any],
    projection: Mapping[str, Any],
    guard_events: list[Mapping[str, Any]],
    expected_writer: Mapping[str, Any],
    preparation_digest: str = "",
) -> dict[str, Any] | None:
    """Return one exact durable guard admitting a validation/evidence stage."""

    expected_result_digest = _identity(
        _manual_seal_stage_receipt(stage=stage, evidence_receipt=evidence_receipt)
    )
    resource_claims = {
        str(item.get("claim_id") or ""): item
        for item in projection.get("resource_claims") or ()
    }
    leases = {
        str(item.get("lease_id") or ""): item
        for item in projection.get("fenced_leases") or ()
    }
    matches: list[dict[str, Any]] = []
    for event in guard_events:
        body = event.get("body")
        if not isinstance(body, Mapping):
            continue
        if (
            event.get("event_type") != "cross_store_fence_guard_succeeded"
            or event.get("lease_id") != binding.get("lease_id")
            or event.get("fencing_token") != binding.get("fencing_token")
            or event.get("fence_epoch") != binding.get("fence_epoch")
            or body.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/cross-store-fence-guard@1"
            or body.get("task_cid") != binding.get("task_cid")
            or body.get("claim_id") != binding.get("claim_id")
            or body.get("attempt_id") != binding.get("attempt_id")
            or body.get("lease_id") != binding.get("lease_id")
            or body.get("fencing_token") != binding.get("fencing_token")
            or body.get("fence_epoch") != binding.get("fence_epoch")
            or body.get("control_result_digest") != expected_result_digest
            or (preparation_digest and body.get("preparation_digest") != preparation_digest)
        ):
            continue
        writer_claim = resource_claims.get(str(body.get("writer_claim_id") or ""))
        writer_lease = leases.get(str(body.get("writer_lease_id") or ""))
        if (
            not isinstance(writer_claim, Mapping)
            or not isinstance(writer_lease, Mapping)
            or writer_claim.get("lease_id") != writer_lease.get("lease_id")
            or writer_claim.get("owner_session_id") != body.get("writer_owner_session_id")
            or writer_claim.get("task_cid") != binding.get("task_cid")
            or writer_claim.get("resource_kind") != body.get("writer_resource_kind")
            or writer_claim.get("resource_id") != body.get("writer_resource_id")
            or writer_claim.get("mode") != body.get("writer_mode")
            or writer_claim.get("fencing_token") != body.get("writer_fencing_token")
            or writer_claim.get("fence_epoch") != body.get("writer_fence_epoch")
            or any(
                writer_claim.get(field) != expected_writer.get(field)
                for field in (
                    "owner_session_id",
                    "task_cid",
                    "resource_kind",
                    "resource_id",
                    "repository_id",
                    "path",
                    "worktree_id",
                    "mode",
                    "body",
                )
            )
            or any(
                writer_claim.get(field) != writer_lease.get(field)
                for field in (
                    "claim_id",
                    "lease_id",
                    "owner_session_id",
                    "task_cid",
                    "resource_kind",
                    "resource_id",
                    "repository_id",
                    "path",
                    "worktree_id",
                    "mode",
                    "fencing_token",
                    "fence_epoch",
                    "state",
                    "revision",
                    "body",
                )
            )
            or writer_claim.get("state") not in {"accepted", "released", "expired"}
        ):
            continue
        matches.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "guard_digest": _identity(body),
                "preparation_digest": str(body.get("preparation_digest") or ""),
                "control_result_digest": expected_result_digest,
                "writer_claim_id": str(writer_claim.get("claim_id") or ""),
                "writer_lease_id": str(writer_lease.get("lease_id") or ""),
                "writer_fencing_token": int(writer_claim.get("fencing_token") or 0),
                "writer_fence_epoch": int(writer_claim.get("fence_epoch") or 0),
            }
        )
    if not matches:
        return None
    return sorted(matches, key=lambda item: item["event_id"])[0]


def _manual_seal_partial_link(
    *,
    stage: str,
    evidence_receipt: Mapping[str, Any],
    binding: Mapping[str, Any],
    stage_guard: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": _MANUAL_SEAL_PARTIAL_LINK_SCHEMA,
        "stage": stage,
        "receipt_cid": _identity(evidence_receipt),
        **dict(binding),
        "fence_admission": "guarded" if stage_guard is not None else "post_fence_failed",
        "guard_event_id": str(stage_guard.get("event_id") or "") if stage_guard else "",
        "guard_digest": str(stage_guard.get("guard_digest") or "") if stage_guard else "",
    }


def _verify_manual_seal_partial_history(
    *,
    control_path: Path,
    projection: Mapping[str, Any],
    guard_events: list[Mapping[str, Any]],
    population: Mapping[str, Any],
    task_cid: str,
    owner_id: str,
    idempotency_key: str,
    accepted_result_cid: str,
    qualification_receipt_cid: str,
    materialization_receipt_cid: str,
    baseline_event_cursor: int,
    current_binding: Mapping[str, Any] | None = None,
    current_preparation_digest: str = "",
    strict_unsealed_events: bool,
) -> dict[str, Any]:
    """Authenticate immutable partial evidence and derive explicit supersession links."""

    validations, evidence = _read_manual_seal_evidence(
        control_path=control_path,
        task_cid=task_cid,
        qualification_receipt_cid=qualification_receipt_cid,
        seal_basis_cid=accepted_result_cid,
    )
    claims = {
        str(item.get("claim_id") or ""): item for item in projection.get("task_claims") or ()
    }
    attempts = {
        str(item.get("attempt_id") or ""): item
        for item in projection.get("task_attempts") or ()
    }
    leases = {
        str(item.get("lease_id") or ""): item
        for item in projection.get("fenced_leases") or ()
        if item.get("lease_kind") == "task"
    }
    expected_writer = {
        "owner_session_id": owner_id,
        "task_cid": task_cid,
        "resource_kind": "database_writer",
        "resource_id": (
            "lgswf-control-store:"
            + _identity(
                {
                    "plan_root_cid": population["plan_root_cid"],
                    "repository_tree_id": population["repository_tree_id"],
                }
            ).split(":", 1)[1]
        ),
        "repository_id": population["source_head"],
        "path": "",
        "worktree_id": "",
        "mode": "exclusive",
        "body": {
            "kind": "trusted_manual_bootstrap_writer",
            "accepted_result_cid": accepted_result_cid,
            "plan_root_cid": population["plan_root_cid"],
        },
    }
    records: list[dict[str, Any]] = []
    seen_stages: set[tuple[str, str]] = set()
    for stage, rows in (("validation", validations), ("basis_evidence", evidence)):
        for raw in rows:
            row = dict(raw)
            body = row.get("body")
            if not isinstance(body, Mapping):
                raise MaterializationError("manual seal partial evidence body is unavailable")
            binding = _manual_seal_evidence_binding(body)
            claim = claims.get(str(binding["claim_id"]))
            attempt = attempts.get(str(binding["attempt_id"]))
            lease = leases.get(str(binding["lease_id"]))
            expected_claim_body = {
                "kind": "trusted_manual_bootstrap_seal",
                "accepted_result_cid": accepted_result_cid,
            }
            if (
                not isinstance(claim, Mapping)
                or not isinstance(attempt, Mapping)
                or not isinstance(lease, Mapping)
                or claim.get("task_cid") != task_cid
                or claim.get("owner_session_id") != owner_id
                or claim.get("idempotency_key") != idempotency_key
                or claim.get("body") != expected_claim_body
                or any(
                    claim.get(field) != binding.get(field)
                    for field in (
                        "task_cid",
                        "claim_id",
                        "attempt_id",
                        "lease_id",
                        "fencing_token",
                        "fence_epoch",
                    )
                )
                or any(
                    claim.get(field) != attempt.get(field)
                    for field in (
                        "task_cid",
                        "attempt_id",
                        "attempt_number",
                        "owner_session_id",
                        "fencing_token",
                        "fence_epoch",
                    )
                )
                or any(
                    claim.get(field) != lease.get(field)
                    for field in (
                        "task_cid",
                        "claim_id",
                        "attempt_id",
                        "attempt_number",
                        "lease_id",
                        "owner_session_id",
                        "fencing_token",
                        "fence_epoch",
                        "idempotency_key",
                        "state",
                        "body",
                    )
                )
            ):
                raise MaterializationError(
                    "manual seal evidence is bound to a different attempt or fence"
                )
            transition = (claim.get("state"), attempt.get("status"), lease.get("state"))
            if transition not in {
                ("accepted", "running", "accepted"),
                ("expired", "expired", "expired"),
                ("released", "succeeded", "released"),
                ("completed", "succeeded", "completed"),
            }:
                raise MaterializationError(
                    "manual seal partial evidence names an invalid task transition"
                )
            stage_key = (str(binding["attempt_id"]), stage)
            if stage_key in seen_stages:
                raise MaterializationError("manual seal contains duplicate per-attempt evidence")
            seen_stages.add(stage_key)
            guard = _manual_seal_stage_guard(
                stage=stage,
                evidence_receipt=row,
                binding=binding,
                projection=projection,
                guard_events=guard_events,
                expected_writer=expected_writer,
                preparation_digest=(
                    current_preparation_digest
                    if current_binding is not None
                    and all(binding.get(key) == current_binding.get(key) for key in binding)
                    else ""
                ),
            )
            if guard is None and transition not in {
                ("accepted", "running", "accepted"),
                ("expired", "expired", "expired"),
            }:
                raise MaterializationError(
                    "terminal manual seal evidence is bound to a different attempt or fence: "
                    "no exact stage fence guard"
                )
            records.append(
                {
                    "stage": stage,
                    "row": row,
                    "binding": binding,
                    "guard": guard,
                    "link": _manual_seal_partial_link(
                        stage=stage,
                        evidence_receipt=row,
                        binding=binding,
                        stage_guard=guard,
                    ),
                }
            )

    records.sort(
        key=lambda item: (
            int(item["binding"]["fence_epoch"]),
            int(item["binding"]["fencing_token"]),
            item["stage"],
            item["link"]["receipt_cid"],
        )
    )
    for record in records:
        binding = record["binding"]
        prior_links = [
            dict(item["link"])
            for item in records
            if (
                int(item["binding"]["fence_epoch"]),
                int(item["binding"]["fencing_token"]),
            )
            < (int(binding["fence_epoch"]), int(binding["fencing_token"]))
        ]
        row = record["row"]
        if record["stage"] == "validation":
            expected_body = _manual_seal_validation_body(
                binding=binding,
                qualification_receipt_cid=qualification_receipt_cid,
                seal_basis_cid=accepted_result_cid,
                superseded_partial_evidence=prior_links,
            )
            if (
                row.get("task_cid") != task_cid
                or row.get("attempt_id") != binding.get("attempt_id")
                or row.get("ordinal") != 0
                or row.get("outcome") != "passed"
                or row.get("run_status") != "passed"
                or row.get("evidence_digest") != qualification_receipt_cid
                or row.get("body") != expected_body
                or row.get("run_body")
                != {"argv": ["verified-content-addressed-bootstrap-qualification"], **expected_body}
            ):
                raise MaterializationError(
                    "manual seal validation is bound to a different attempt or fence"
                )
        else:
            expected_body = _manual_seal_basis_evidence_body(
                binding=binding,
                qualification_receipt_cid=qualification_receipt_cid,
                materialization_receipt_cid=materialization_receipt_cid,
                superseded_partial_evidence=prior_links,
            )
            if (
                row.get("task_cid") != task_cid
                or row.get("parent_evidence_id") != ""
                or row.get("evidence_kind") != "bootstrap_seal_basis"
                or row.get("digest") != accepted_result_cid
                or row.get("body") != expected_body
            ):
                raise MaterializationError(
                    "manual seal basis evidence is bound to a different attempt or fence"
                )

    expected_events = sorted(
        (
            "intent.validation_recorded" if item["stage"] == "validation" else "intent.evidence_recorded",
            str(
                item["row"].get("result_id")
                if item["stage"] == "validation"
                else item["row"].get("evidence_id")
            ),
            str(item["binding"]["attempt_id"]) if item["stage"] == "validation" else "",
        )
        for item in records
    )
    try:
        import duckdb  # type: ignore

        connection = duckdb.connect(str(control_path), read_only=True)
    except Exception as exc:
        raise MaterializationError("manual seal partial history cannot be opened read-only") from exc
    try:
        validation_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM validation_results WHERE task_cid = ?",
                [task_cid],
            ).fetchone()[0]
        )
        evidence_rows = connection.execute(
            "SELECT evidence_kind, digest FROM evidence_nodes WHERE task_cid = ?",
            [task_cid],
        ).fetchall()
        domain_rows = connection.execute(
            """
            SELECT event_type, json_extract_string(body_json, '$.subject_id'),
                   attempt_id, task_cid
            FROM domain_events
            WHERE global_sequence > ?
            ORDER BY global_sequence
            """,
            [int(baseline_event_cursor)],
        ).fetchall()
    finally:
        connection.close()
    if validation_count != len(validations):
        raise MaterializationError("manual seal contains foreign validation history")
    allowed_partial_nodes = [
        (str(row[0]), str(row[1]))
        for row in evidence_rows
        if str(row[0]) in {"validation", "bootstrap_seal_basis"}
    ]
    expected_partial_nodes = sorted(
        [("validation", qualification_receipt_cid)] * len(validations)
        + [("bootstrap_seal_basis", accepted_result_cid)] * len(evidence)
    )
    if sorted(allowed_partial_nodes) != expected_partial_nodes:
        raise MaterializationError("manual seal contains foreign partial evidence nodes")
    all_events = [
        (str(row[0]), str(row[1]), str(row[2] or ""), str(row[3] or ""))
        for row in domain_rows
    ]
    observed_events = sorted(
        (event_type, subject_id, attempt_id)
        for event_type, subject_id, attempt_id, event_task_cid in all_events
        if event_task_cid == task_cid
    )
    if len(evidence_rows) != len(expected_partial_nodes):
        raise MaterializationError("manual seal contains foreign post-materialization evidence")
    expected_post_materialization_events = list(expected_events)
    if not strict_unsealed_events:
        expected_post_materialization_events.append(
            ("intent.completion_recorded", task_cid, "")
        )
        expected_post_materialization_events.sort()
    if (
        observed_events != expected_post_materialization_events
        or (strict_unsealed_events and len(all_events) != len(observed_events))
    ):
        raise MaterializationError("manual seal partial evidence lost its immutable event")

    current = dict(current_binding) if current_binding is not None else None
    current_records = [
        item
        for item in records
        if current is not None
        and all(item["binding"].get(key) == current.get(key) for key in item["binding"])
    ]
    superseded = [
        dict(item["link"])
        for item in records
        if item not in current_records
    ]
    by_stage = {item["stage"]: item for item in current_records}
    return {
        "validations": validations,
        "evidence": evidence,
        "superseded_partial_evidence": superseded,
        "current_validation": by_stage.get("validation"),
        "current_basis_evidence": by_stage.get("basis_evidence"),
        "has_partial_evidence": bool(records),
    }


def _writer_fence_authority(
    *,
    projection: Mapping[str, Any],
    population: Mapping[str, Any],
    accepted_result_cid: str,
    task_claim: Mapping[str, Any],
    cross_store_guard: Mapping[str, Any],
    writer_reservation: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the guarded writer and the writer used for recovery.

    The writer that executed the guarded control CAS is immutable history.  It
    may have been released normally or expired after the guard was committed.
    Receipt construction/reconstruction runs under a currently admitted writer;
    when that is a successor, its fence must be strictly later on the exact same
    resource scope and its terminal release is bound into the authority root.
    """

    resource_claims = {
        str(item.get("claim_id") or ""): dict(item)
        for item in projection.get("resource_claims") or ()
    }
    leases = {
        str(item.get("lease_id") or ""): dict(item)
        for item in projection.get("fenced_leases") or ()
    }
    guarded_claim = resource_claims.get(
        str(cross_store_guard.get("writer_claim_id") or "")
    )
    guarded_lease = leases.get(
        str(cross_store_guard.get("writer_lease_id") or "")
    )
    recovery_claim = resource_claims.get(
        str(writer_reservation.get("claim_id") or "")
    )
    recovery_lease = leases.get(
        str(writer_reservation.get("lease_id") or "")
    )
    expected_resource_id = (
        "lgswf-control-store:"
        + _identity(
            {
                "plan_root_cid": population["plan_root_cid"],
                "repository_tree_id": population["repository_tree_id"],
            }
        ).split(":", 1)[1]
    )
    expected_body = {
        "kind": "trusted_manual_bootstrap_writer",
        "accepted_result_cid": accepted_result_cid,
        "plan_root_cid": population["plan_root_cid"],
    }
    shared_fields = (
        "resource_kind",
        "resource_id",
        "owner_session_id",
        "task_cid",
        "repository_id",
        "path",
        "worktree_id",
        "mode",
        "body",
    )
    reservation_fields = (
        "claim_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        *shared_fields,
    )
    claim_lease_fields = (
        "claim_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        *shared_fields,
        "state",
        "revision",
    )
    reservation_authority = {
        field: writer_reservation.get(field)
        for field in (
            "schema",
            "interface",
            *reservation_fields,
            "state",
            "revision",
        )
    }
    if (
        not isinstance(guarded_claim, Mapping)
        or not isinstance(guarded_lease, Mapping)
        or not isinstance(recovery_claim, Mapping)
        or not isinstance(recovery_lease, Mapping)
        or writer_reservation.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/resource-claim@1"
        or writer_reservation.get("interface") != "ResourceClaim@1"
        or writer_reservation.get("state") != "accepted"
        or int(writer_reservation.get("revision") or 0) < 1
        or guarded_claim.get("claim_id")
        != cross_store_guard.get("writer_claim_id")
        or guarded_claim.get("lease_id")
        != cross_store_guard.get("writer_lease_id")
        or guarded_claim.get("fencing_token")
        != cross_store_guard.get("writer_fencing_token")
        or guarded_claim.get("fence_epoch")
        != cross_store_guard.get("writer_fence_epoch")
        or guarded_claim.get("state") not in {"released", "expired"}
        or guarded_lease.get("state") != guarded_claim.get("state")
        or guarded_claim.get("resource_kind") != "database_writer"
        or guarded_claim.get("resource_id") != expected_resource_id
        or guarded_claim.get("owner_session_id")
        != task_claim.get("owner_session_id")
        or guarded_claim.get("task_cid") != task_claim.get("task_cid")
        or guarded_claim.get("repository_id") != population["source_head"]
        or guarded_claim.get("mode") != "exclusive"
        or guarded_claim.get("body") != expected_body
        or any(
            guarded_claim.get(field) != guarded_lease.get(field)
            for field in claim_lease_fields
        )
        or any(
            writer_reservation.get(field) != recovery_claim.get(field)
            for field in reservation_fields
        )
        or recovery_claim.get("state") != "released"
        or recovery_lease.get("state") != "released"
        or int(recovery_claim.get("revision") or 0)
        != int(writer_reservation.get("revision") or 0) + 1
        or any(
            recovery_claim.get(field) != recovery_lease.get(field)
            for field in claim_lease_fields
        )
        or any(
            guarded_claim.get(field) != recovery_claim.get(field)
            for field in shared_fields
        )
    ):
        raise MaterializationError(
            "accepted seal writer fences cannot be reconstructed exactly"
        )
    same_writer = guarded_claim["claim_id"] == recovery_claim["claim_id"]
    guarded_fence = (
        int(guarded_claim["fence_epoch"]),
        int(guarded_claim["fencing_token"]),
    )
    recovery_fence = (
        int(recovery_claim["fence_epoch"]),
        int(recovery_claim["fencing_token"]),
    )
    if (
        same_writer
        and (
            guarded_claim.get("state") != "released"
            or recovery_fence != guarded_fence
        )
    ) or (
        not same_writer
        and (
            recovery_fence[0] < guarded_fence[0]
            or recovery_fence[1] <= guarded_fence[1]
        )
    ):
        raise MaterializationError(
            "accepted seal recovery writer is not the exact current or later fence"
        )
    return {
        "writer_reservation": reservation_authority,
        "writer_release": dict(recovery_lease),
    }


def _durable_seal_authority(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    writer_reservation: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the full accepted seal evidence from durable authorities."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    paths = _paths(config)
    binding = verification.get("completion_binding")
    accepted_result_cid = str(verification.get("accepted_result_cid") or "")
    if not isinstance(binding, Mapping) or not accepted_result_cid:
        raise MaterializationError("sealed verification omitted its durable completion binding")
    projection = read_coordination_registry_projection(paths["coordination"])
    claim = next(
        (
            dict(item)
            for item in projection.get("task_claims") or ()
            if item.get("claim_id") == binding.get("claim_id")
        ),
        None,
    )
    settled_lease = (
        next(
            (
                dict(item)
                for item in projection.get("fenced_leases") or ()
                if item.get("lease_id") == claim.get("lease_id")
            ),
            None,
        )
        if isinstance(claim, Mapping)
        else None
    )
    completion = next(
        (
            dict(item)
            for item in projection.get("logical_completions") or ()
            if item.get("task_cid") == binding.get("task_cid")
        ),
        None,
    )
    promoted_body = dict(completion.get("body") or {}) if isinstance(completion, Mapping) else {}
    control_completion = promoted_body.get("control_completion")
    cross_store_guard = promoted_body.get("cross_store_guard")
    preparation = {
        key: value
        for key, value in promoted_body.items()
        if key not in {"control_completion", "cross_store_guard"}
    }
    preparation.update({"status": "prepared", "replayed": False})
    if (
        not isinstance(claim, Mapping)
        or not isinstance(settled_lease, Mapping)
        or not isinstance(completion, Mapping)
        or completion.get("status") != "succeeded"
        or not isinstance(control_completion, Mapping)
        or not isinstance(cross_store_guard, Mapping)
        or claim.get("state") not in {"released", "completed"}
        or settled_lease.get("state") != claim.get("state")
        or any(
            claim.get(field) != binding.get(field)
            for field in (
                "task_cid",
                "claim_id",
                "attempt_id",
                "fencing_token",
                "fence_epoch",
            )
        )
        or preparation.get("preparation_digest") != binding.get("preparation_digest")
        or preparation.get("evidence_digest") != accepted_result_cid
        or not isinstance(preparation.get("body"), Mapping)
        or preparation["body"].get("requires_cross_store_fence_guard") is not True
    ):
        raise MaterializationError("accepted seal history cannot be reconstructed exactly")

    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id="lgswf-materializer:durable-seal-read",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    try:
        task = task_source.get(str(binding["task_cid"]))
        if task is None:
            raise MaterializationError(
                "accepted control task disappeared during receipt reconstruction"
            )
        task_projection = task.to_dict()
    finally:
        task_source.close()

    expected_control_digest = _identity(task_projection)
    if (
        cross_store_guard.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/cross-store-fence-guard@1"
        or cross_store_guard.get("preparation_digest")
        != preparation.get("preparation_digest")
        or cross_store_guard.get("control_result_digest")
        != expected_control_digest
        or control_completion.get("receipt_digest") != expected_control_digest
    ):
        raise MaterializationError(
            "accepted seal cross-store fence guard cannot be reconstructed exactly"
        )
    writer_authority = _writer_fence_authority(
        projection=projection,
        population=population,
        accepted_result_cid=accepted_result_cid,
        task_claim=claim,
        cross_store_guard=cross_store_guard,
        writer_reservation=writer_reservation,
    )

    qualification_cid = str(preparation["body"]["seal_basis"]["qualification_receipt_cid"])
    materialization_cid = str(
        preparation["body"]["seal_basis"]["materialization_receipt_cid"]
    )
    history = _verify_manual_seal_partial_history(
        control_path=paths["control"],
        projection=projection,
        guard_events=_read_manual_seal_guard_events(paths["coordination"]),
        population=population,
        task_cid=str(binding["task_cid"]),
        owner_id=str(binding.get("owner_session_id") or claim.get("owner_session_id") or ""),
        idempotency_key=str(claim.get("idempotency_key") or ""),
        accepted_result_cid=accepted_result_cid,
        qualification_receipt_cid=qualification_cid,
        materialization_receipt_cid=materialization_cid,
        baseline_event_cursor=int(
            preparation["body"]["seal_basis"]["task_source_snapshot"]["event_cursor"]
        ),
        current_binding=binding,
        current_preparation_digest=str(preparation.get("preparation_digest") or ""),
        strict_unsealed_events=False,
    )
    validation_receipt, basis_evidence = _validate_manual_seal_evidence(
        validations=history["validations"],
        evidence=history["evidence"],
        binding=binding,
        qualification_receipt_cid=qualification_cid,
        materialization_receipt_cid=materialization_cid,
        seal_basis_cid=accepted_result_cid,
        require_validation=True,
        require_evidence=True,
        superseded_partial_evidence=history["superseded_partial_evidence"],
    )
    if (
        history["current_validation"] is None
        or history["current_validation"].get("guard") is None
        or history["current_basis_evidence"] is None
        or history["current_basis_evidence"].get("guard") is None
        or validation_receipt is None
        or basis_evidence is None
    ):
        raise MaterializationError("accepted seal evidence lacks exact stage fence guards")
    control_cas = {
        "schema": "ipfs_accelerate_py/agent-supervisor/durable-control-completion@1",
        "changed": True,
        "previous_status": str(preparation["control_expected_status"]),
        "revision": int(control_completion["revision"]),
        "receipt_cid": str(control_completion["receipt_cid"]),
        "receipt_digest": str(control_completion["receipt_digest"]),
        "task": task_projection,
    }
    promotion = {
        "task_cid": str(binding["task_cid"]),
        "claim_id": str(binding["claim_id"]),
        "attempt_id": str(binding["attempt_id"]),
        "lease_id": str(claim["lease_id"]),
        "fencing_token": int(binding["fencing_token"]),
        "fence_epoch": int(binding["fence_epoch"]),
        "status": "succeeded",
        "control_completion": dict(control_completion),
    }
    authority = {
        "claim": claim,
        "preparation": preparation,
        "validation_receipt": validation_receipt,
        "seal_basis_evidence_receipt": basis_evidence,
        "control_cas": control_cas,
        "coordination_promotion": promotion,
        "cross_store_guard": dict(cross_store_guard),
        "settled_lease": settled_lease,
        **writer_authority,
    }
    authority["authority_root"] = _identity(authority)
    return authority


def _verify_live_sealed_task_receipt(
    task: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    accepted_result_cid: str,
    preparation: Mapping[str, Any],
) -> None:
    """Authenticate the immutable task snapshot captured by the seal receipt."""

    spec = _expected_task_static(expected)
    exact = {
        key: spec[key]
        for key in (
            "task_cid",
            "task_alias",
            "goal_cid",
            "plan_cid",
            "objective_id",
            "ordinal",
            "priority",
            "dependencies",
            "outputs",
            "acceptance",
            "validations",
        )
    }
    if any(task.get(key) != value for key, value in exact.items()):
        raise MaterializationError("bootstrap seal captured a foreign LGSWF-006 task")
    body = task.get("body")
    if not isinstance(body, Mapping):
        raise MaterializationError("bootstrap seal captured no LGSWF-006 task body")
    body_map = dict(body)
    if (
        any(body_map.get(key) != value for key, value in spec["body"].items())
        or set(body_map) - set(spec["body"]) != {"completion_receipt"}
        or task.get("status") != "completed"
        or int(task.get("revision") or 0) != 2
    ):
        raise MaterializationError("bootstrap seal captured a forged LGSWF-006 specification")
    completion_receipt = body_map.get("completion_receipt")
    if (
        not isinstance(completion_receipt, Mapping)
        or completion_receipt.get("accepted_result_cid") != accepted_result_cid
    ):
        raise MaterializationError("bootstrap seal task snapshot lost its accepted result")
    captured_preparation = completion_receipt.get("coordination_preparation")
    preparation_fields = (
        "schema",
        "task_cid",
        "attempt_id",
        "attempt_number",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "control_expected_revision",
        "control_expected_status",
        "evidence_digest",
        "prepared_at_ms",
        "body",
        "preparation_digest",
    )
    if not isinstance(captured_preparation, Mapping) or any(
        captured_preparation.get(field) != preparation.get(field)
        for field in preparation_fields
    ):
        raise MaterializationError("bootstrap seal task snapshot has a foreign preparation")


def _verify_live_seal_receipt(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    schema_profile: Mapping[str, Any],
    control_projection: Mapping[str, Any],
    coordination_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify immutable bootstrap receipts against the exact durable seal fence."""

    materialization = _load_materialization_receipt(config, population)
    qualification = _load_qualification_receipt(config, population)
    receipt_path = _bootstrap_receipt_path(config, "duckdb-seal.json")
    if not receipt_path.is_file():
        raise MaterializationError("accepted bootstrap seal receipt is absent")
    receipt = _read_self_addressed_receipt(receipt_path, label="bootstrap seal")
    launch_plan = _render_launch_plan_evidence(config)
    seal_basis = _build_seal_basis(
        config=config,
        population=population,
        materialization_receipt=materialization,
        qualification_receipt=qualification,
        launch_plan=launch_plan,
    )
    accepted_result_cid = _identity(seal_basis)
    expected = {
        "schema": "ipfs_accelerate_py/agent-supervisor/lgswf-bootstrap-seal@1",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": _identity(population),
        "materialization_receipt_cid": materialization["receipt_cid"],
        "qualification_receipt_cid": qualification["receipt_cid"],
        "launch_plan": launch_plan,
        "accepted_result_cid": accepted_result_cid,
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise MaterializationError("live bootstrap seal is stale: " + ", ".join(mismatches))
    materialized_verification = materialization.get("verification")
    materialized_schema = (
        materialized_verification.get("schema_profile")
        if isinstance(materialized_verification, Mapping)
        else None
    )
    if (
        not isinstance(materialized_schema, Mapping)
        or materialized_schema.get("schema_fingerprint")
        != schema_profile.get("schema_fingerprint")
        or seal_basis.get("schema_profile_fingerprint")
        != schema_profile.get("schema_fingerprint")
    ):
        raise MaterializationError("live schema no longer matches the materialization seal")

    post = receipt.get("post_verification")
    binding = post.get("completion_binding") if isinstance(post, Mapping) else None
    if (
        not isinstance(post, Mapping)
        or not isinstance(binding, Mapping)
        or post.get("accepted_result_cid") != accepted_result_cid
        or binding.get("task_cid")
        != population["task_cids_by_alias"]["LGSWF-006"]
        or binding.get("seal_basis_cid") != accepted_result_cid
        or materialized_verification.get("task_spec_root")
        != _expected_task_spec_root(population, sealed=False)
        or post.get("task_spec_root")
        != _expected_task_spec_root(population, sealed=True)
    ):
        raise MaterializationError("bootstrap seal post-verification is stale or forged")

    authority_keys = (
        "claim",
        "preparation",
        "validation_receipt",
        "seal_basis_evidence_receipt",
        "control_cas",
        "coordination_promotion",
        "cross_store_guard",
        "settled_lease",
        "writer_reservation",
        "writer_release",
    )
    authority = {key: receipt.get(key) for key in authority_keys}
    if any(not isinstance(item, Mapping) for item in authority.values()) or receipt.get(
        "authority_root"
    ) != _identity(authority):
        raise MaterializationError("bootstrap seal durable authority CID does not verify")
    claim = dict(authority["claim"])
    preparation = dict(authority["preparation"])
    validation = dict(authority["validation_receipt"])
    basis_evidence = dict(authority["seal_basis_evidence_receipt"])
    control_cas = dict(authority["control_cas"])
    promotion = dict(authority["coordination_promotion"])
    cross_store_guard = dict(authority["cross_store_guard"])
    settled = dict(authority["settled_lease"])
    writer_reservation = dict(authority["writer_reservation"])
    writer_release = dict(authority["writer_release"])
    bound_fields = (
        "task_cid",
        "claim_id",
        "attempt_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
    )
    if (
        any(claim.get(field) != binding.get(field) for field in bound_fields)
        or any(preparation.get(field) != binding.get(field) for field in bound_fields)
        or any(promotion.get(field) != binding.get(field) for field in bound_fields)
        or any(settled.get(field) != binding.get(field) for field in bound_fields)
        or preparation.get("preparation_digest") != binding.get("preparation_digest")
        or preparation.get("evidence_digest") != accepted_result_cid
        or preparation.get("body", {}).get("seal_basis") != seal_basis
        or preparation.get("body", {}).get("requires_cross_store_fence_guard")
        is not True
    ):
        raise MaterializationError("bootstrap seal receipt has a foreign completion binding")

    persisted_claim = next(
        (
            dict(item)
            for item in coordination_projection.get("task_claims") or ()
            if item.get("claim_id") == binding.get("claim_id")
        ),
        None,
    )
    persisted_attempt = next(
        (
            dict(item)
            for item in coordination_projection.get("task_attempts") or ()
            if item.get("attempt_id") == binding.get("attempt_id")
        ),
        None,
    )
    persisted_lease = next(
        (
            dict(item)
            for item in coordination_projection.get("fenced_leases") or ()
            if item.get("lease_id") == binding.get("lease_id")
        ),
        None,
    )
    completion = next(
        (
            dict(item)
            for item in coordination_projection.get("logical_completions") or ()
            if item.get("task_cid") == binding.get("task_cid")
        ),
        None,
    )
    completion_body = (
        dict(completion.get("body") or {}) if isinstance(completion, Mapping) else {}
    )
    control_completion = completion_body.get("control_completion")
    durable_cross_store_guard = completion_body.get("cross_store_guard")
    durable_preparation = {
        key: value
        for key, value in completion_body.items()
        if key not in {"control_completion", "cross_store_guard"}
    }
    durable_preparation.update({"status": "prepared", "replayed": False})
    durable_promotion = {
        "task_cid": str(binding["task_cid"]),
        "claim_id": str(binding["claim_id"]),
        "attempt_id": str(binding["attempt_id"]),
        "lease_id": str(binding["lease_id"]),
        "fencing_token": int(binding["fencing_token"]),
        "fence_epoch": int(binding["fence_epoch"]),
        "status": "succeeded",
        "control_completion": dict(control_completion or {}),
    }
    if (
        persisted_claim != claim
        or persisted_lease != settled
        or not isinstance(persisted_attempt, Mapping)
        or not isinstance(completion, Mapping)
        or completion.get("status") != "succeeded"
        or not isinstance(control_completion, Mapping)
        or not isinstance(durable_cross_store_guard, Mapping)
        or durable_preparation != preparation
        or durable_promotion != promotion
        or dict(durable_cross_store_guard) != cross_store_guard
        or persisted_attempt.get("status") != "succeeded"
        or claim.get("state") not in {"released", "completed"}
        or settled.get("state") not in {"released", "completed"}
    ):
        raise MaterializationError("accepted LGSWF-006 seal history cannot be reconstructed")

    qualification_cid = str(qualification["receipt_cid"])
    history = _verify_manual_seal_partial_history(
        control_path=_paths(config)["control"],
        projection=coordination_projection,
        guard_events=_read_manual_seal_guard_events(_paths(config)["coordination"]),
        population=population,
        task_cid=str(binding["task_cid"]),
        owner_id=str(persisted_claim.get("owner_session_id") or ""),
        idempotency_key=str(persisted_claim.get("idempotency_key") or ""),
        accepted_result_cid=accepted_result_cid,
        qualification_receipt_cid=qualification_cid,
        materialization_receipt_cid=str(materialization["receipt_cid"]),
        baseline_event_cursor=int(
            preparation["body"]["seal_basis"]["task_source_snapshot"]["event_cursor"]
        ),
        current_binding=binding,
        current_preparation_digest=str(preparation.get("preparation_digest") or ""),
        strict_unsealed_events=False,
    )
    current_validation, current_evidence = _validate_manual_seal_evidence(
        validations=history["validations"],
        evidence=history["evidence"],
        binding=binding,
        qualification_receipt_cid=qualification_cid,
        materialization_receipt_cid=str(materialization["receipt_cid"]),
        seal_basis_cid=accepted_result_cid,
        require_validation=True,
        require_evidence=True,
        superseded_partial_evidence=history["superseded_partial_evidence"],
    )
    if (
        history["current_validation"] is None
        or history["current_validation"].get("guard") is None
        or history["current_basis_evidence"] is None
        or history["current_basis_evidence"].get("guard") is None
        or current_validation != validation
        or current_evidence != basis_evidence
    ):
        raise MaterializationError("bootstrap seal receipt differs from durable evidence")

    captured_task = control_cas.get("task")
    lgswf006 = next(
        item for item in population["tasks"] if item["task_alias"] == "LGSWF-006"
    )
    if not isinstance(captured_task, Mapping):
        raise MaterializationError("bootstrap seal omitted its control task snapshot")
    _verify_live_sealed_task_receipt(
        captured_task,
        lgswf006,
        accepted_result_cid=accepted_result_cid,
        preparation=preparation,
    )
    if (
        control_cas.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/durable-control-completion@1"
        or control_cas.get("changed") is not True
        or control_cas.get("previous_status") != preparation.get("control_expected_status")
        or control_cas.get("revision") != control_completion.get("revision")
        or control_cas.get("receipt_cid") != control_completion.get("receipt_cid")
        or control_cas.get("receipt_digest") != control_completion.get("receipt_digest")
    ):
        raise MaterializationError("bootstrap seal control CAS receipt is forged")

    expected_control_digest = _identity(dict(captured_task))
    if (
        cross_store_guard.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/cross-store-fence-guard@1"
        or cross_store_guard.get("preparation_digest")
        != preparation.get("preparation_digest")
        or cross_store_guard.get("control_result_digest")
        != expected_control_digest
        or control_completion.get("receipt_digest") != expected_control_digest
    ):
        raise MaterializationError(
            "bootstrap seal cross-store guard is stale or forged"
        )
    durable_writer_authority = _writer_fence_authority(
        projection=coordination_projection,
        population=population,
        accepted_result_cid=accepted_result_cid,
        task_claim=claim,
        cross_store_guard=cross_store_guard,
        writer_reservation=writer_reservation,
    )
    if (
        durable_writer_authority["writer_reservation"] != writer_reservation
        or durable_writer_authority["writer_release"] != writer_release
    ):
        raise MaterializationError(
            "bootstrap seal recovery-writer release is stale or forged"
        )

    current_task = control_projection.get("tasks", {}).get(str(binding["task_cid"]))
    current_body = current_task.get("body") if isinstance(current_task, Mapping) else None
    current_completion = (
        current_body.get("completion_receipt") if isinstance(current_body, Mapping) else None
    )
    sealed_completion = captured_task.get("body", {}).get("completion_receipt")
    if (
        not isinstance(current_task, Mapping)
        or current_task.get("status") not in {"completed", "complete", "done"}
        or current_completion != sealed_completion
    ):
        raise MaterializationError("current LGSWF-006 control authority lost the accepted seal")
    return {
        "receipt_cid": str(receipt["receipt_cid"]),
        "accepted_result_cid": accepted_result_cid,
        "completion_binding": dict(binding),
        "authority_root": str(receipt["authority_root"]),
        "materialization_receipt_cid": str(materialization["receipt_cid"]),
        "qualification_receipt_cid": qualification_cid,
    }


def _verify_live_store(
    config: Mapping[str, Any], population: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a live post-bootstrap authority using read-only database handles."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )

    paths = _paths(config)
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise MaterializationError(f"control-plane files missing: {missing}")
    schema_profile = _verify_live_schema_read_only(paths["control"])
    control = _verify_live_control_read_only(config, population)
    try:
        coordination_projection = read_coordination_registry_projection(paths["coordination"])
    except Exception as exc:
        raise MaterializationError("live coordination authority cannot be read safely") from exc
    coordination = _verify_live_coordination_projection(
        coordination_projection,
        population,
        control["tasks"],
    )
    execution = _verify_live_execution_read_only(
        config,
        schema_profile,
        control["tasks"],
        coordination_projection,
    )
    seal = _verify_live_seal_receipt(
        config,
        population,
        schema_profile=schema_profile,
        control_projection=control,
        coordination_projection=coordination_projection,
    )
    return {
        "verification_mode": "live",
        "schema_profile": schema_profile,
        "control": {key: value for key, value in control.items() if key != "tasks"},
        "coordination": {key: value for key, value in coordination.items() if key != "projection"},
        "execution": execution,
        "seal": seal,
        "database_identities": {
            key: _sha256_file(path) for key, path in sorted(paths.items())
        },
    }


_MANUAL_SEAL_PREPARATION_FIELDS = (
    "schema",
    "task_cid",
    "attempt_id",
    "attempt_number",
    "claim_id",
    "lease_id",
    "owner_session_id",
    "fencing_token",
    "fence_epoch",
    "control_expected_revision",
    "control_expected_status",
    "evidence_digest",
    "prepared_at_ms",
    "body",
    "preparation_digest",
)


def _manual_seal_preparation_binding(
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the stable preparation fields persisted in the control receipt."""

    return {
        field: preparation.get(field) for field in _MANUAL_SEAL_PREPARATION_FIELDS
    }


def _manual_seal_control_receipt(
    *,
    accepted_result_cid: str,
    qualification_receipt_cid: str,
    preparation: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic control receipt used by fresh CAS and replay."""

    return {
        "operation": "trusted_manual_bootstrap_seal",
        "accepted_result_cid": accepted_result_cid,
        "qualification_receipt_cid": qualification_receipt_cid,
        "coordination_preparation": _manual_seal_preparation_binding(preparation),
        "launch_plan": dict(launch_plan),
    }


def _validated_manual_seal_control_task(
    task: Any,
    *,
    task_cid: str,
    expected_revision: int,
    expected_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one stable completed TaskRecord projection or fail closed."""

    projection = task.to_dict() if callable(getattr(task, "to_dict", None)) else dict(task)
    body = projection.get("body")
    completion_receipt = body.get("completion_receipt") if isinstance(body, Mapping) else None
    if (
        projection.get("task_cid") != task_cid
        or str(projection.get("status") or "").strip().lower()
        not in {"completed", "complete", "done"}
        or int(projection.get("revision") or 0) != int(expected_revision) + 1
        or not isinstance(completion_receipt, Mapping)
        or dict(completion_receipt) != dict(expected_receipt)
    ):
        raise MaterializationError(
            "manual seal control completion is not the exact prepared result"
        )
    return projection


def _reconcile_manual_seal(
    *,
    task_source: Any,
    coordinator: Any,
    task_cid: str,
    owner_id: str,
    idempotency_key: str,
    accepted_result_cid: str,
    seal_basis: Mapping[str, Any],
    qualification_receipt_cid: str,
    launch_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Repair only exact manual-seal crash windows from durable truth."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD,
        DatabaseCoordinationNotReadyError,
    )

    outcomes: list[dict[str, Any]] = []
    # This is an explicit writer/recovery path with an already-admitted
    # coordinator.  The projection method itself performs no sweep or write;
    # opening a second read-only DuckDB connection while the writer is open is
    # not supported by DuckDB's configuration lock.
    projection = coordinator.coordination_registry_projection()
    unsettled = _unsettled_completion_projection(projection)
    foreign = [item for item in unsettled if item.get("task_cid") != task_cid]
    if foreign:
        raise MaterializationError("foreign unsettled task completion blocks sealing")
    if not unsettled:
        return outcomes
    barrier = dict(unsettled[0])
    claim = coordinator.get_task_claim(str(barrier.get("claim_id") or ""))
    claim_lease = (
        coordinator.get_lease(str(claim.lease_id)) if claim is not None else None
    )
    barrier_body = barrier.get("body")
    persisted_basis = barrier_body.get("seal_basis") if isinstance(barrier_body, Mapping) else None
    if (
        claim is None
        or claim_lease is None
        or claim.task_cid != task_cid
        or claim.owner_session_id != owner_id
        or claim.idempotency_key != idempotency_key
        or claim.body.get("kind") != "trusted_manual_bootstrap_seal"
        or claim.body.get("accepted_result_cid") != accepted_result_cid
        or barrier.get("attempt_id") != claim.attempt_id
        or barrier.get("lease_id") != claim.lease_id
        or barrier.get("fencing_token") != claim.fencing_token
        or barrier.get("fence_epoch") != claim.fence_epoch
        or claim_lease.task_cid != claim.task_cid
        or claim_lease.claim_id != claim.claim_id
        or claim_lease.attempt_id != claim.attempt_id
        or claim_lease.owner_session_id != claim.owner_session_id
        or claim_lease.fencing_token != claim.fencing_token
        or claim_lease.fence_epoch != claim.fence_epoch
        or claim_lease.expires_at_ms != claim.expires_at_ms
        or barrier.get("evidence_digest") != accepted_result_cid
        or not isinstance(persisted_basis, Mapping)
        or dict(persisted_basis) != dict(seal_basis)
        or _identity(persisted_basis) != accepted_result_cid
        or barrier_body.get(CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD) is not True
    ):
        raise MaterializationError(
            "unsettled completion is not the exact deterministic manual seal"
        )
    task = task_source.get(task_cid)
    if task is None:
        raise MaterializationError("manual seal task disappeared during recovery")
    task_status = str(task.status or "").strip().lower()
    barrier_status = str(barrier.get("status") or "").strip().lower()
    lease_state = claim_lease.state.value
    claim_state = claim.state.value
    now_ms = time.time_ns() // 1_000_000
    effectively_expired = (
        claim_state == "expired"
        or lease_state == "expired"
        or claim.expires_at_ms <= now_ms
    )
    if task_status in {"completed", "complete", "done"}:
        expected_receipt = _manual_seal_control_receipt(
            accepted_result_cid=accepted_result_cid,
            qualification_receipt_cid=qualification_receipt_cid,
            preparation=barrier,
            launch_plan=launch_plan,
        )
        control_receipt = _validated_manual_seal_control_task(
            task,
            task_cid=task_cid,
            expected_revision=int(barrier.get("control_expected_revision") or 0),
            expected_receipt=expected_receipt,
        )
        if barrier_status == "prepared":
            try:
                if effectively_expired:
                    outcome = coordinator.recover_prepared_task_completion(
                        task_cid,
                        control_completion_receipt=control_receipt,
                    )
                else:
                    outcome = coordinator.complete_task_claim(
                        barrier,
                        control_completion_receipt=control_receipt,
                    )
                    outcome = coordinator.reconcile_promoted_task_completion(
                        task_cid,
                        control_completion_receipt=control_receipt,
                    )
            except DatabaseCoordinationNotReadyError as exc:
                reason = str(dict(getattr(exc, "evidence", {}) or {}).get("reason") or "")
                if reason != "cross_store_fence_guard_missing":
                    raise
                if effectively_expired:
                    raise MaterializationError(
                        "expired manual seal completion has no durable cross-store fence guard"
                    ) from exc
                # The external control CAS committed, but the coordinator
                # transaction did not durably record its post-fence receipt.
                # Leave the live PREPARED barrier untouched; the main path
                # will replay the idempotent completed-row callback while the
                # exact task and writer fences are still current.
                outcomes.append(
                    {
                        "task_cid": task_cid,
                        "claim_id": str(barrier.get("claim_id") or ""),
                        "attempt_id": str(barrier.get("attempt_id") or ""),
                        "status": "guard_replay_required",
                        "reason": reason,
                    }
                )
                return outcomes
        elif barrier_status == "succeeded":
            outcome = coordinator.reconcile_promoted_task_completion(
                task_cid,
                control_completion_receipt=control_receipt,
            )
        else:
            raise MaterializationError(f"unsupported manual-seal barrier status {barrier_status!r}")
        outcomes.append(dict(outcome))
        return outcomes
    if task_status != "todo" or barrier_status != "prepared":
        raise MaterializationError("manual seal control and coordination states disagree")
    if effectively_expired:
        outcomes.append(
            dict(
                coordinator.abort_prepared_task_completion(
                    task_cid,
                    control_task_observation=task.to_dict(),
                    reason="manual_seal_control_cas_absent",
                )
            )
        )
    return outcomes


def _verify_resumable_manual_claim(
    *,
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    task_cid: str,
    owner_id: str,
    idempotency_key: str,
    accepted_result_cid: str,
    seal_basis: Mapping[str, Any],
    permitted_writer_lease: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recognize only the exact live manual claim; reject every other lease."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD,
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    paths = _paths(config)
    coordination_projection = read_coordination_registry_projection(paths["coordination"])
    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id="lgswf-materializer:resume-check",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    try:
        active = [
            dict(item)
            for item in coordination_projection.get("fenced_leases") or ()
            if item.get("state") == "accepted"
        ]
        writer_fields = (
            "lease_id",
            "lease_kind",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "state",
            "task_cid",
            "resource_kind",
            "resource_id",
            "repository_id",
            "claim_id",
        )
        writer_matches = [
            item
            for item in active
            if all(item.get(field) == permitted_writer_lease.get(field) for field in writer_fields)
        ]
        if len(writer_matches) != 1:
            raise MaterializationError("exact embedded-writer reservation is not active")
        task_leases = [item for item in active if item["lease_id"] != writer_matches[0]["lease_id"]]
        all_resource_claims = list(coordination_projection.get("resource_claims") or ())
        all_resource_leases = [
            item
            for item in coordination_projection.get("fenced_leases") or ()
            if item.get("lease_kind") == "resource"
        ]
        if (
            not all_resource_claims
            or len(all_resource_claims) != len(all_resource_leases)
            or coordination_projection.get("maintenance_leases")
        ):
            raise MaterializationError(
                "manual recovery contains foreign resource or maintenance history"
            )
        leases_by_id = {str(item.get("lease_id") or ""): item for item in all_resource_leases}
        expected_writer_body = {
            "kind": "trusted_manual_bootstrap_writer",
            "accepted_result_cid": accepted_result_cid,
            "plan_root_cid": population["plan_root_cid"],
        }
        expected_resource_id = (
            "lgswf-control-store:"
            + _identity(
                {
                    "plan_root_cid": population["plan_root_cid"],
                    "repository_tree_id": population["repository_tree_id"],
                }
            ).split(":", 1)[1]
        )
        for writer_claim in all_resource_claims:
            writer_lease = leases_by_id.get(str(writer_claim.get("lease_id") or ""))
            if (
                not isinstance(writer_lease, Mapping)
                or writer_claim.get("resource_kind") != "database_writer"
                or writer_claim.get("resource_id") != expected_resource_id
                or writer_claim.get("owner_session_id") != owner_id
                or writer_claim.get("task_cid") != task_cid
                or writer_claim.get("repository_id") != str(population["source_head"])
                or writer_claim.get("mode") != "exclusive"
                or writer_claim.get("body") != expected_writer_body
                or writer_claim.get("state") not in {"accepted", "released", "expired"}
                or any(
                    writer_claim.get(field) != writer_lease.get(field)
                    for field in (
                        "lease_id",
                        "owner_session_id",
                        "fencing_token",
                        "fence_epoch",
                        "resource_kind",
                        "resource_id",
                        "task_cid",
                        "repository_id",
                        "mode",
                        "body",
                        "state",
                    )
                )
            ):
                raise MaterializationError(
                    "manual recovery contains foreign resource or maintenance history"
                )

        all_task_claims = list(coordination_projection.get("task_claims") or ())
        attempts_by_id = {
            str(item.get("attempt_id") or ""): item
            for item in coordination_projection.get("task_attempts") or ()
        }
        task_leases_by_id = {
            str(item.get("lease_id") or ""): item
            for item in coordination_projection.get("fenced_leases") or ()
            if item.get("lease_kind") == "task"
        }
        if not (len(all_task_claims) == len(attempts_by_id) == len(task_leases_by_id)):
            raise MaterializationError("manual recovery task history is incomplete")
        active_claim_ids: list[str] = []
        for historical_claim in all_task_claims:
            historical_attempt = attempts_by_id.get(str(historical_claim.get("attempt_id") or ""))
            historical_lease = task_leases_by_id.get(str(historical_claim.get("lease_id") or ""))
            expected_body = {
                "kind": "trusted_manual_bootstrap_seal",
                "accepted_result_cid": accepted_result_cid,
            }
            if (
                not isinstance(historical_attempt, Mapping)
                or not isinstance(historical_lease, Mapping)
                or historical_claim.get("task_cid") != task_cid
                or historical_claim.get("owner_session_id") != owner_id
                or historical_claim.get("idempotency_key") != idempotency_key
                or historical_claim.get("body") != expected_body
                or any(
                    historical_claim.get(field) != historical_attempt.get(field)
                    for field in (
                        "task_cid",
                        "attempt_id",
                        "attempt_number",
                        "owner_session_id",
                        "fencing_token",
                        "fence_epoch",
                    )
                )
                or any(
                    historical_claim.get(field) != historical_lease.get(field)
                    for field in (
                        "task_cid",
                        "claim_id",
                        "attempt_id",
                        "attempt_number",
                        "lease_id",
                        "owner_session_id",
                        "fencing_token",
                        "fence_epoch",
                        "idempotency_key",
                        "body",
                    )
                )
            ):
                raise MaterializationError("manual recovery contains a foreign task history")
            transition = (
                historical_claim.get("state"),
                historical_attempt.get("status"),
                historical_lease.get("state"),
            )
            if transition == ("accepted", "running", "accepted"):
                active_claim_ids.append(str(historical_claim.get("claim_id") or ""))
            elif transition != ("expired", "expired", "expired"):
                raise MaterializationError("manual recovery contains a foreign task transition")
        if len(active_claim_ids) != len(task_leases):
            raise MaterializationError("manual recovery active task history is inconsistent")
        if not task_leases:
            return None
        if len(task_leases) != 1:
            raise MaterializationError("foreign or multiple task leases block sealing")
        lease = task_leases[0]
        expected_lease = {
            "lease_kind": "task",
            "task_cid": task_cid,
            "owner_session_id": owner_id,
            "idempotency_key": idempotency_key,
        }
        mismatches = [key for key, value in expected_lease.items() if lease.get(key) != value]
        body = lease.get("body")
        if (
            mismatches
            or lease.get("claim_id") not in active_claim_ids
            or not isinstance(body, Mapping)
            or body.get("kind") != "trusted_manual_bootstrap_seal"
            or body.get("accepted_result_cid") != accepted_result_cid
        ):
            raise MaterializationError(
                "active lease is not the exact deterministic manual seal claim"
            )

        current_snapshot = task_source.snapshot().to_dict()
        page = task_source.list_tasks(limit=100)
        successor = next(
            (item for item in page.tasks if item.task_cid == task_cid),
            None,
        )
        if successor is None:
            raise MaterializationError("manual seal task disappeared during recovery")
        successor_status = str(successor.status or "").strip().lower()
        control_completed = successor_status in {"completed", "complete", "done"}
        if successor_status not in {"todo", "completed", "complete", "done"}:
            raise MaterializationError(
                "manual seal control state is not resumable"
            )
        task_spec_root = _verify_task_records(
            list(page.tasks),
            population,
            expected_stage="sealed" if control_completed else "unsealed",
        )
        if not control_completed and task_spec_root != seal_basis.get("task_spec_root"):
            raise MaterializationError("task specifications changed during manual seal recovery")
        aliases = [task.task_alias for task in page.tasks]
        expected_aliases = [task["task_alias"] for task in population["tasks"]]
        expected_statuses = {
            item.task_alias: (
                "completed"
                if control_completed and item.task_alias == "LGSWF-006"
                else "todo"
            )
            for item in page.tasks
        }
        if aliases != expected_aliases or any(
            str(item.status).lower() != expected_statuses[item.task_alias]
            for item in page.tasks
        ):
            raise MaterializationError(
                "control task population changed during manual seal recovery"
            )
        ready_aliases = [task.task_alias for task in task_source.ready_tasks(limit=100).tasks]
        expected_ready = (
            ["LGSWF-001", "LGSWF-002", "LGSWF-003"]
            if control_completed
            else ["LGSWF-006"]
        )
        if ready_aliases != expected_ready:
            raise MaterializationError(
                "control ready projection changed during manual seal recovery"
            )

        unsettled = _unsettled_completion_projection(coordination_projection)
        if len(unsettled) > 1 or (unsettled and unsettled[0].get("task_cid") != task_cid):
            raise MaterializationError("foreign completion barrier blocks sealing")
        if unsettled:
            barrier = unsettled[0]
            barrier_body = barrier.get("body")
            persisted_basis = (
                barrier_body.get("seal_basis") if isinstance(barrier_body, Mapping) else None
            )
            if (
                barrier.get("status") != "prepared"
                or barrier.get("lease_state") != "accepted"
                or barrier.get("owner_session_id") != owner_id
                or barrier.get("evidence_digest") != accepted_result_cid
                or not isinstance(persisted_basis, Mapping)
                or dict(persisted_basis) != dict(seal_basis)
                or _identity(persisted_basis) != accepted_result_cid
                or barrier_body.get(CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD)
                is not True
            ):
                raise MaterializationError(
                    "live preparation is not the exact deterministic seal basis"
                )
            if control_completed:
                expected_receipt = _manual_seal_control_receipt(
                    accepted_result_cid=accepted_result_cid,
                    qualification_receipt_cid=str(
                        seal_basis["qualification_receipt_cid"]
                    ),
                    preparation=barrier,
                    launch_plan=dict(seal_basis["launch_plan"]),
                )
                _validated_manual_seal_control_task(
                    successor,
                    task_cid=task_cid,
                    expected_revision=int(
                        barrier.get("control_expected_revision") or 0
                    ),
                    expected_receipt=expected_receipt,
                )
        elif current_snapshot != seal_basis.get("task_source_snapshot"):
            raise MaterializationError(
                "claim-only recovery no longer matches the materialized task snapshot"
            )

        schema = verify_datasets_authoritative_operational_schema(paths["control"])
        verification = materialization_receipt.get("verification")
        recorded_profile = (
            verification.get("schema_profile") if isinstance(verification, Mapping) else None
        )
        if (
            schema.get("valid") is not True
            or not isinstance(recorded_profile, Mapping)
            or schema.get("schema_fingerprint") != recorded_profile.get("schema_fingerprint")
        ):
            raise MaterializationError("operational schema changed during manual seal recovery")
        execution = _verify_execution_store(config, schema)
        if execution.get("execution_store_root") != seal_basis.get("execution_store_root"):
            raise MaterializationError("execution store changed during manual seal recovery")
        coordination_spec = {
            "tasks": [
                {key: value for key, value in item.items() if key != "ready"}
                for item in coordination_projection["tasks"]
            ],
            "dependency_edges": coordination_projection["dependency_edges"],
        }
        if _identity(coordination_spec) != seal_basis.get("coordination_registry_spec_root"):
            raise MaterializationError("coordination registry changed during manual seal recovery")
        report = dict(verification)
        report["resumed_active_claim"] = {
            "claim_id": lease.get("claim_id"),
            "attempt_id": lease.get("attempt_id"),
            "fencing_token": lease.get("fencing_token"),
            "fence_epoch": lease.get("fence_epoch"),
            "prepared": bool(unsettled),
            "control_completed": control_completed,
        }
        return report
    finally:
        task_source.close()


def _acquire_bootstrap_writer(
    coordinator: Any,
    *,
    population: Mapping[str, Any],
    task_cid: str,
    owner_id: str,
    accepted_result_cid: str,
) -> Any:
    """Reserve the embedded control store under the existing resource authority."""

    resource_id = (
        "lgswf-control-store:"
        + _identity(
            {
                "plan_root_cid": population["plan_root_cid"],
                "repository_tree_id": population["repository_tree_id"],
            }
        ).split(":", 1)[1]
    )
    body = {
        "kind": "trusted_manual_bootstrap_writer",
        "accepted_result_cid": accepted_result_cid,
        "plan_root_cid": population["plan_root_cid"],
    }
    claim = coordinator.claim_resource(
        resource_kind="database_writer",
        resource_id=resource_id,
        owner_session_id=owner_id,
        lease_ms=300_000,
        task_cid=task_cid,
        repository_id=str(population["source_head"]),
        body=body,
    )
    if (
        claim.resource_kind != "database_writer"
        or claim.resource_id != resource_id
        or claim.owner_session_id != owner_id
        or claim.task_cid != task_cid
        or claim.repository_id != str(population["source_head"])
        or claim.state.value != "accepted"
        or claim.mode.value != "exclusive"
        or dict(claim.body) != body
    ):
        raise MaterializationError("bootstrap writer reservation changed its exact authority")
    return claim


def _release_bootstrap_writer(coordinator: Any, writer_claim: Any) -> dict[str, Any]:
    """Release the exact writer fence, or return its prior terminal record."""

    stored = coordinator.get_lease(str(writer_claim.lease_id))
    if stored is None:
        raise MaterializationError("bootstrap writer lease disappeared")
    exact_fields = (
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "task_cid",
        "resource_kind",
        "resource_id",
        "repository_id",
        "claim_id",
    )
    expected = writer_claim.as_fenced_lease().to_dict()
    observed = stored.to_dict()
    if any(observed.get(field) != expected.get(field) for field in exact_fields):
        raise MaterializationError("bootstrap writer lease lost its exact fence")
    if stored.state.value == "accepted":
        stored = coordinator.release(
            stored,
            reason="trusted_manual_bootstrap_seal_finished",
            expected_fencing_token=int(writer_claim.fencing_token),
            expected_fence_epoch=int(writer_claim.fence_epoch),
        )
    if stored.state.value not in {"released", "completed"}:
        raise MaterializationError(f"bootstrap writer lease ended as {stored.state.value!r}")
    return stored.to_dict()


def _seal_with_writer(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    writer_claim: Any,
) -> dict[str, Any]:
    """Accept the corrected manual bootstrap through claim, prepare and CAS."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD,
        open_database_coordinator,
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    materialization_receipt = _load_materialization_receipt(config, population)
    qualification_receipt = _load_qualification_receipt(config, population)
    launch_plan = _render_launch_plan_evidence(config)
    seal_basis = _build_seal_basis(
        config=config,
        population=population,
        materialization_receipt=materialization_receipt,
        qualification_receipt=qualification_receipt,
        launch_plan=launch_plan,
    )
    accepted_result_cid = _identity(seal_basis)
    owner_id = "lgswf-bootstrap-seal:" + _identity(population).split(":", 1)[1][:24]
    idempotency_key = "manual-seal:" + accepted_result_cid.split(":", 1)[1]
    paths = _paths(config)
    task_cid = population["task_cids_by_alias"]["LGSWF-006"]
    writer_lease = writer_claim.as_fenced_lease().to_dict()
    if (
        writer_claim.owner_session_id != owner_id
        or writer_claim.task_cid != task_cid
        or writer_claim.body.get("accepted_result_cid") != accepted_result_cid
    ):
        raise MaterializationError("bootstrap writer reservation is bound to a different seal")
    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id="lgswf-materializer:seal",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    coordinator = open_database_coordinator(paths["coordination"])
    recovery: list[dict[str, Any]] = []
    try:
        recovery = _reconcile_manual_seal(
            task_source=task_source,
            coordinator=coordinator,
            task_cid=task_cid,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            accepted_result_cid=accepted_result_cid,
            seal_basis=seal_basis,
            qualification_receipt_cid=str(qualification_receipt["receipt_cid"]),
            launch_plan=launch_plan,
        )
    finally:
        coordinator.close()
        task_source.close()

    current = DatabaseTaskSource(
        paths["control"],
        owner_id="lgswf-materializer:seal-check",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    try:
        observed = current.get(task_cid)
        if observed is None:
            raise MaterializationError("manual bootstrap seal task is absent")
        already_completed = str(observed.status).lower() in {
            "completed",
            "complete",
            "done",
        }
    finally:
        current.close()
    partial_history: dict[str, Any] | None = None
    if not already_completed:
        partial_history = _verify_manual_seal_partial_history(
            control_path=paths["control"],
            projection=read_coordination_registry_projection(paths["coordination"]),
            guard_events=_read_manual_seal_guard_events(paths["coordination"]),
            population=population,
            task_cid=task_cid,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            accepted_result_cid=accepted_result_cid,
            qualification_receipt_cid=str(qualification_receipt["receipt_cid"]),
            materialization_receipt_cid=str(materialization_receipt["receipt_cid"]),
            baseline_event_cursor=int(seal_basis["task_source_snapshot"]["event_cursor"]),
            strict_unsealed_events=True,
        )
    guard_replay_required = any(
        item.get("status") == "guard_replay_required" for item in recovery
    )
    if already_completed and not guard_replay_required:
        writer_held_verification = _verify_store(
            config,
            population,
            expected_stage="sealed",
            permitted_writer_lease=writer_lease,
        )
        release_coordinator = open_database_coordinator(paths["coordination"])
        try:
            _release_bootstrap_writer(release_coordinator, writer_claim)
        finally:
            release_coordinator.close()
        post_verification = _verify_store(config, population, expected_stage="sealed")
        durable_authority = _durable_seal_authority(
            config,
            population,
            post_verification,
            writer_reservation=writer_claim.to_dict(),
        )
        result = {
            "schema": "ipfs_accelerate_py/agent-supervisor/lgswf-bootstrap-seal@1",
            "accepted_result_cid": post_verification["accepted_result_cid"],
            "reconciled": bool(recovery),
            "recovery": recovery,
            "source_head": population["source_head"],
            "repository_tree_id": population["repository_tree_id"],
            "plan_root_cid": population["plan_root_cid"],
            "population_cid": _identity(population),
            "materialization_receipt_cid": materialization_receipt["receipt_cid"],
            "qualification_receipt_cid": qualification_receipt["receipt_cid"],
            "launch_plan": launch_plan,
            "writer_held_verification": writer_held_verification,
            "post_verification": post_verification,
            **durable_authority,
        }
        result["receipt_cid"] = _identity(result)
        path = _bootstrap_receipt_path(config, "duckdb-seal.json")
        _write_receipt(path, result)
        return _receipt_result(operation="seal", receipt=result, path=path, replayed=False)

    premature_seal_path = _bootstrap_receipt_path(config, "duckdb-seal.json")
    if premature_seal_path.exists():
        raise MaterializationError("pre-existing seal receipt blocks an unaccepted namespace")

    pre_verification = _verify_resumable_manual_claim(
        config=config,
        population=population,
        materialization_receipt=materialization_receipt,
        task_cid=task_cid,
        owner_id=owner_id,
        idempotency_key=idempotency_key,
        accepted_result_cid=accepted_result_cid,
        seal_basis=seal_basis,
        permitted_writer_lease=writer_lease,
    )
    if pre_verification is None:
        pre_verification = _verify_store(
            config,
            population,
            expected_stage="unsealed",
            permitted_writer_lease=writer_lease,
        )
        if (
            pre_verification.get("ready_task_aliases") != seal_basis["ready_task_aliases"]
            or pre_verification.get("schema_profile", {}).get("schema_fingerprint")
            != seal_basis["schema_profile_fingerprint"]
            or (
                pre_verification.get("task_source_snapshot")
                != seal_basis["task_source_snapshot"]
                and not bool(partial_history and partial_history["has_partial_evidence"])
            )
            or pre_verification.get("task_spec_root") != seal_basis["task_spec_root"]
            or pre_verification.get("control_population", {}).get("population_root")
            != seal_basis["control_population_root"]
            or pre_verification.get("execution_store", {}).get("execution_store_root")
            != seal_basis["execution_store_root"]
            or pre_verification.get("coordination_registry", {}).get("registry_spec_root")
            != seal_basis["coordination_registry_spec_root"]
        ):
            raise MaterializationError(
                "current unsealed authority differs from the materialization receipt"
            )
    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id=owner_id,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    coordinator = open_database_coordinator(paths["coordination"])
    try:
        task = task_source.get(task_cid)
        task_status = str(task.status).lower() if task is not None else ""
        if task is None or task_status not in {"todo", "completed", "complete", "done"}:
            raise MaterializationError("manual bootstrap seal is not resumable")
        control_already_completed = task_status in {"completed", "complete", "done"}
        if control_already_completed and not guard_replay_required:
            raise MaterializationError(
                "completed manual seal control state lacks an admitted replay path"
            )
        resumed = pre_verification.get("resumed_active_claim")
        if isinstance(resumed, Mapping):
            claim = coordinator.get_task_claim(str(resumed.get("claim_id") or ""))
            if claim is None:
                raise MaterializationError("resumable manual claim disappeared")
            coordinator.protect_task_claim(
                claim,
                expected_task_cid=task_cid,
                expected_attempt_id=str(resumed.get("attempt_id") or ""),
                expected_owner_session_id=owner_id,
                expected_fencing_token=int(resumed.get("fencing_token") or 0),
                expected_fence_epoch=int(resumed.get("fence_epoch") or 0),
                allow_logically_completed=True,
            )
        else:
            claim = coordinator.claim_task(
                task_cid=task_cid,
                owner_session_id=owner_id,
                lease_ms=300_000,
                idempotency_key=idempotency_key,
                body={
                    "kind": "trusted_manual_bootstrap_seal",
                    "accepted_result_cid": accepted_result_cid,
                },
            )
        if (
            claim.owner_session_id != owner_id
            or claim.idempotency_key != idempotency_key
            or claim.body.get("kind") != "trusted_manual_bootstrap_seal"
            or claim.body.get("accepted_result_cid") != accepted_result_cid
        ):
            raise MaterializationError("manual claim replay changed its sealed identity")
        current_writer = coordinator.get_lease(str(writer_claim.lease_id))
        if (
            current_writer is None
            or current_writer.state.value != "accepted"
            or current_writer.fencing_token != writer_claim.fencing_token
            or current_writer.fence_epoch != writer_claim.fence_epoch
            or current_writer.owner_session_id != writer_claim.owner_session_id
        ):
            raise MaterializationError(
                "embedded writer lost authority before completion preparation"
            )
        preparation = coordinator.prepare_task_completion(
            claim,
            control_expected_revision=(
                int(task.revision) - 1 if control_already_completed else int(task.revision)
            ),
            control_expected_status="todo",
            evidence_digest=accepted_result_cid,
            body={
                CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD: True,
                "seal_basis": seal_basis,
            },
        )
        preparation_body = preparation.get("body")
        persisted_basis = (
            preparation_body.get("seal_basis") if isinstance(preparation_body, Mapping) else None
        )
        if (
            preparation.get("evidence_digest") != accepted_result_cid
            or not isinstance(persisted_basis, Mapping)
            or dict(persisted_basis) != seal_basis
            or _identity(persisted_basis) != accepted_result_cid
            or preparation_body.get(CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD)
            is not True
        ):
            raise MaterializationError("manual preparation changed its seal basis")
        coordinator.protect_task_claim(
            claim,
            expected_task_cid=task_cid,
            expected_attempt_id=claim.attempt_id,
            expected_owner_session_id=owner_id,
            expected_fencing_token=claim.fencing_token,
            expected_fence_epoch=claim.fence_epoch,
            allow_logically_completed=True,
        )
        evidence_binding = _manual_seal_attempt_binding(claim)
        qualification_cid = str(qualification_receipt["receipt_cid"])
        materialization_cid = str(materialization_receipt["receipt_cid"])

        def inspect_partial_history() -> dict[str, Any]:
            return _verify_manual_seal_partial_history(
                control_path=paths["control"],
                projection=coordinator.coordination_registry_projection(),
                guard_events=coordinator.lease_events(limit=10_000),
                population=population,
                task_cid=task_cid,
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                accepted_result_cid=accepted_result_cid,
                qualification_receipt_cid=qualification_cid,
                materialization_receipt_cid=materialization_cid,
                baseline_event_cursor=int(seal_basis["task_source_snapshot"]["event_cursor"]),
                current_binding=evidence_binding,
                current_preparation_digest=str(preparation["preparation_digest"]),
                strict_unsealed_events=not control_already_completed,
            )

        partial = inspect_partial_history()
        superseded = partial["superseded_partial_evidence"]
        validation_body = _manual_seal_validation_body(
            binding=evidence_binding,
            qualification_receipt_cid=qualification_cid,
            seal_basis_cid=accepted_result_cid,
            superseded_partial_evidence=superseded,
        )
        basis_evidence_body = _manual_seal_basis_evidence_body(
            binding=evidence_binding,
            qualification_receipt_cid=qualification_cid,
            materialization_receipt_cid=materialization_cid,
            superseded_partial_evidence=superseded,
        )

        def guarded_validation_stage() -> dict[str, Any]:
            validations, basis_evidence_rows = _read_manual_seal_evidence(
                control_path=paths["control"],
                task_cid=task_cid,
                qualification_receipt_cid=qualification_cid,
                seal_basis_cid=accepted_result_cid,
            )
            validation, _basis = _validate_manual_seal_evidence(
                validations=validations,
                evidence=basis_evidence_rows,
                binding=evidence_binding,
                qualification_receipt_cid=qualification_cid,
                materialization_receipt_cid=materialization_cid,
                seal_basis_cid=accepted_result_cid,
                require_validation=False,
                require_evidence=False,
                superseded_partial_evidence=superseded,
            )
            if validation is None:
                task_source.record_validation_result(
                    task_cid=task_cid,
                    outcome="passed",
                    evidence_digest=qualification_cid,
                    argv=["verified-content-addressed-bootstrap-qualification"],
                    attempt_id=str(claim.attempt_id),
                    body=validation_body,
                )
                validations, basis_evidence_rows = _read_manual_seal_evidence(
                    control_path=paths["control"],
                    task_cid=task_cid,
                    qualification_receipt_cid=qualification_cid,
                    seal_basis_cid=accepted_result_cid,
                )
                validation, _basis = _validate_manual_seal_evidence(
                    validations=validations,
                    evidence=basis_evidence_rows,
                    binding=evidence_binding,
                    qualification_receipt_cid=qualification_cid,
                    materialization_receipt_cid=materialization_cid,
                    seal_basis_cid=accepted_result_cid,
                    require_validation=True,
                    require_evidence=False,
                    superseded_partial_evidence=superseded,
                )
            assert validation is not None
            return _manual_seal_stage_receipt(
                stage="validation", evidence_receipt=validation
            )

        if partial["current_validation"] is None or partial["current_validation"]["guard"] is None:
            coordinator.execute_with_task_and_resource_fences(
                claim,
                writer_claim,
                guarded_validation_stage,
                allow_logically_completed=True,
            )
        partial = inspect_partial_history()
        if partial["current_validation"] is None or partial["current_validation"]["guard"] is None:
            raise MaterializationError("manual seal validation was not fence-admitted")

        def guarded_basis_evidence_stage() -> dict[str, Any]:
            validations, basis_evidence_rows = _read_manual_seal_evidence(
                control_path=paths["control"],
                task_cid=task_cid,
                qualification_receipt_cid=qualification_cid,
                seal_basis_cid=accepted_result_cid,
            )
            _validation, basis_evidence = _validate_manual_seal_evidence(
                validations=validations,
                evidence=basis_evidence_rows,
                binding=evidence_binding,
                qualification_receipt_cid=qualification_cid,
                materialization_receipt_cid=materialization_cid,
                seal_basis_cid=accepted_result_cid,
                require_validation=True,
                require_evidence=False,
                superseded_partial_evidence=superseded,
            )
            if basis_evidence is None:
                task_source.record_evidence(
                    task_cid=task_cid,
                    evidence_kind="bootstrap_seal_basis",
                    digest=accepted_result_cid,
                    body=basis_evidence_body,
                )
                validations, basis_evidence_rows = _read_manual_seal_evidence(
                    control_path=paths["control"],
                    task_cid=task_cid,
                    qualification_receipt_cid=qualification_cid,
                    seal_basis_cid=accepted_result_cid,
                )
                _validation, basis_evidence = _validate_manual_seal_evidence(
                    validations=validations,
                    evidence=basis_evidence_rows,
                    binding=evidence_binding,
                    qualification_receipt_cid=qualification_cid,
                    materialization_receipt_cid=materialization_cid,
                    seal_basis_cid=accepted_result_cid,
                    require_validation=True,
                    require_evidence=True,
                    superseded_partial_evidence=superseded,
                )
            assert basis_evidence is not None
            return _manual_seal_stage_receipt(
                stage="basis_evidence", evidence_receipt=basis_evidence
            )

        if (
            partial["current_basis_evidence"] is None
            or partial["current_basis_evidence"]["guard"] is None
        ):
            coordinator.execute_with_task_and_resource_fences(
                claim,
                writer_claim,
                guarded_basis_evidence_stage,
                allow_logically_completed=True,
            )
        partial = inspect_partial_history()
        current_validation = partial["current_validation"]
        current_basis = partial["current_basis_evidence"]
        if (
            current_validation is None
            or current_validation["guard"] is None
            or current_basis is None
            or current_basis["guard"] is None
        ):
            raise MaterializationError("manual seal evidence stages were not fence-admitted")
        expected_control_revision = int(preparation["control_expected_revision"])
        completion_receipt = _manual_seal_control_receipt(
            accepted_result_cid=accepted_result_cid,
            qualification_receipt_cid=qualification_cid,
            preparation=preparation,
            launch_plan=launch_plan,
        )

        def guarded_control_completion() -> dict[str, Any]:
            observed_task = task_source.get(task_cid)
            if observed_task is None:
                raise MaterializationError(
                    "manual seal control task disappeared during guarded completion"
                )
            observed_status = str(observed_task.status or "").strip().lower()
            if observed_status == "todo":
                if int(observed_task.revision) != expected_control_revision:
                    raise MaterializationError(
                        "manual seal control revision changed before guarded CAS"
                    )
                cas = task_source.compare_and_set_status(
                    task_cid,
                    expected_revision=expected_control_revision,
                    status="completed",
                    receipt=completion_receipt,
                    evidence_digests=[accepted_result_cid, qualification_cid],
                )
                if (
                    cas.changed is not True
                    or cas.previous_status != "todo"
                    or int(cas.revision) != expected_control_revision + 1
                ):
                    raise MaterializationError(
                        "manual seal control CAS returned an unexpected transition"
                    )
                observed_task = cas.task
            elif observed_status not in {"completed", "complete", "done"}:
                raise MaterializationError(
                    "manual seal control task entered an incompatible state"
                )
            return _validated_manual_seal_control_task(
                observed_task,
                task_cid=task_cid,
                expected_revision=expected_control_revision,
                expected_receipt=completion_receipt,
            )

        control_receipt = coordinator.execute_with_task_and_resource_fences(
            claim,
            writer_claim,
            guarded_control_completion,
            allow_logically_completed=True,
        )
        coordinator.complete_task_claim(
            claim,
            control_completion_receipt=control_receipt,
        )
        coordinator.settle_task_claim(
            claim,
            reason="trusted_manual_bootstrap_seal",
        )
    finally:
        coordinator.close()
        task_source.close()
    writer_held_verification = _verify_store(
        config,
        population,
        expected_stage="sealed",
        permitted_writer_lease=writer_lease,
    )
    release_coordinator = open_database_coordinator(paths["coordination"])
    try:
        _release_bootstrap_writer(release_coordinator, writer_claim)
    finally:
        release_coordinator.close()
    post_verification = _verify_store(config, population, expected_stage="sealed")
    durable_authority = _durable_seal_authority(
        config,
        population,
        post_verification,
        writer_reservation=writer_claim.to_dict(),
    )
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/lgswf-bootstrap-seal@1",
        "accepted_result_cid": accepted_result_cid,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": _identity(population),
        "materialization_receipt_cid": materialization_receipt["receipt_cid"],
        "qualification_receipt_cid": qualification_receipt["receipt_cid"],
        "launch_plan": launch_plan,
        "writer_held_verification": writer_held_verification,
        **durable_authority,
        "post_verification": post_verification,
        "recovery": recovery,
    }
    receipt["receipt_cid"] = _identity(receipt)
    path = _bootstrap_receipt_path(config, "duckdb-seal.json")
    _write_receipt(path, receipt)
    return _receipt_result(operation="seal", receipt=receipt, path=path, replayed=False)


def seal(config: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    """Run or replay the seal under one explicit embedded-writer reservation."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        open_database_coordinator,
    )

    _assert_population_source_current(config, population)
    path = _bootstrap_receipt_path(config, "duckdb-seal.json")
    if path.is_file():
        existing = _load_existing_seal_receipt(config, population)
        assert existing is not None
        return _receipt_result(operation="seal", receipt=existing, path=path, replayed=True)

    materialization_receipt = _load_materialization_receipt(config, population)
    qualification_receipt = _load_qualification_receipt(config, population)
    seal_basis = _build_seal_basis(
        config=config,
        population=population,
        materialization_receipt=materialization_receipt,
        qualification_receipt=qualification_receipt,
        launch_plan=_render_launch_plan_evidence(config),
    )
    accepted_result_cid = _identity(seal_basis)
    owner_id = "lgswf-bootstrap-seal:" + _identity(population).split(":", 1)[1][:24]
    task_cid = population["task_cids_by_alias"]["LGSWF-006"]
    coordinator = open_database_coordinator(_paths(config)["coordination"])
    try:
        writer_claim = _acquire_bootstrap_writer(
            coordinator,
            population=population,
            task_cid=task_cid,
            owner_id=owner_id,
            accepted_result_cid=accepted_result_cid,
        )
    finally:
        coordinator.close()

    try:
        result = _seal_with_writer(
            config,
            population,
            writer_claim=writer_claim,
        )
    except Exception:
        cleanup = open_database_coordinator(_paths(config)["coordination"])
        try:
            stored = cleanup.get_lease(str(writer_claim.lease_id))
            if stored is not None and stored.state.value == "accepted":
                try:
                    _release_bootstrap_writer(cleanup, writer_claim)
                except Exception:
                    # Preserve the primary failure. The bounded lease remains
                    # recoverable evidence and expires under coordinator policy.
                    pass
        finally:
            cleanup.close()
        raise

    check = open_database_coordinator(_paths(config)["coordination"])
    try:
        stored = check.get_lease(str(writer_claim.lease_id))
        if stored is None or stored.state.value not in {"released", "completed"}:
            raise MaterializationError(
                "bootstrap seal returned before releasing its writer reservation"
            )
    finally:
        check.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "population",
            "qualify",
            "materialize",
            "verify-unsealed",
            "seal",
            "verify",
            "verify-live",
        ),
        nargs="?",
        default="verify",
    )
    args = parser.parse_args(argv)
    try:
        config = _load_config()
        population = build_population(config)
        if args.command == "population":
            result: Mapping[str, Any] = {
                "schema": population["schema"],
                "population_cid": _identity(population),
                "plan_root_cid": population["plan_root_cid"],
                "repository_tree_id": population["repository_tree_id"],
                "task_count": len(population["tasks"]),
                "goal_count": len(population["objectives"]),
            }
        elif args.command == "qualify":
            result = qualify(config, population)
        elif args.command == "materialize":
            result = materialize(config, population)
        elif args.command == "verify-unsealed":
            result = {
                "schema": SCHEMA,
                "valid": True,
                "plan_root_cid": population["plan_root_cid"],
                "verification": _verify_store(config, population, expected_stage="unsealed"),
            }
        elif args.command == "seal":
            result = seal(config, population)
        elif args.command == "verify-live":
            result = {
                "schema": SCHEMA,
                "valid": True,
                "plan_root_cid": population["plan_root_cid"],
                "verification": _verify_live_store(config, population),
            }
        else:
            verification = _verify_store(config, population, expected_stage="sealed")
            seal_receipt = _load_existing_seal_receipt(config, population)
            if seal_receipt is None:
                raise MaterializationError("accepted bootstrap seal receipt is absent")
            if seal_receipt.get("accepted_result_cid") != verification.get("accepted_result_cid"):
                raise MaterializationError(
                    "bootstrap seal receipt disagrees with control authority"
                )
            result = {
                "schema": SCHEMA,
                "valid": True,
                "plan_root_cid": population["plan_root_cid"],
                "verification": verification,
                "seal_receipt_cid": seal_receipt["receipt_cid"],
            }
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        json.dump(
            {
                "schema": SCHEMA,
                "valid": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
