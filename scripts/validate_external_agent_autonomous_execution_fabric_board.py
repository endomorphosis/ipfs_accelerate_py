#!/usr/bin/env python3
"""Fail-closed validator for the ExternalAgentAutonomousExecutionFabric board."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
SOURCE_PATH = CAMPAIGN_DIR / "source_reconciliation_manifest.json"
STACK_PATH = CAMPAIGN_DIR / "stack_compatibility_manifest.json"
BOARD_PATH = CAMPAIGN_DIR / "task_board.json"
MARKDOWN_PATH = CAMPAIGN_DIR / "TASK_BOARD.md"
OBJECTIVES_PATH = CAMPAIGN_DIR / "OBJECTIVES.md"
GENERATOR_PATH = ROOT / "scripts/generate_external_agent_autonomous_execution_fabric_board.py"
SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-board-validation@1"
)
EXPECTED_REPOSITORIES = {
    "ipfs_accelerate_py",
    "ipfs_datasets_py",
    "ipfs_kit_py",
    "Mcp-Plus-Plus",
}
REQUIRED_TASK_FIELDS = {
    "schema",
    "stable_task_id",
    "parent_goal_id",
    "subgoal_id",
    "epic",
    "title",
    "objective",
    "owning_repository",
    "owned_files",
    "execution_owned_files",
    "integration_conflict_keys",
    "source_revisions",
    "source_semantic_state_root",
    "source_control_plane_schema_version",
    "dependencies",
    "read_scope",
    "write_scope",
    "external_effect_scope",
    "required_capsules",
    "context_artifacts",
    "resource_request",
    "container_profile",
    "model_route",
    "provider_policy",
    "test_requirements",
    "proof_requirements",
    "completion_contract",
    "lease_and_fencing_requirements",
    "idempotency_key",
    "rollback_or_compensation",
    "required_evidence",
    "terminal_status",
    "allowed_terminal_statuses",
    "outcome",
    "allowed_outcomes",
    "outcome_status_mapping",
    "final_artifact_identities",
    "status",
    "completion_mode",
    "is_schedulable",
    "initial_population",
    "population_state",
    "blocked_reason",
    "priority",
    "track",
    "plan_revision",
    "board_namespace",
    "permitted_effects",
    "prohibited_effects",
    "outputs",
    "execution_outputs",
    "validation",
    "execution_validation",
    "acceptance",
    "conflict_and_merge_contract",
    "task_spec_cid",
}
RESOURCE_FIELDS = {
    "cpu_millicores",
    "ram_mib",
    "gpu_count",
    "disk_mib",
    "network",
    "supervisor_processes",
    "worktree_slots",
    "container_slots",
    "merge_slots",
    "provider_concurrency",
    "model_input_token_ceiling",
    "model_output_token_ceiling",
    "prover_concurrency",
    "timeout_seconds",
}
GOAL_REQUIRED_FIELDS = {
    "schema",
    "goal_id",
    "parent_goal_id",
    "epic",
    "title",
    "dependencies",
    "completion_contract",
    "desired_postconditions",
    "prohibited_outcomes",
    "scope",
    "resource_budget",
    "authority_ceiling",
    "verification_requirements",
    "proof_requirements",
    "human_review_requirements",
    "completion_evidence",
}
RECONCILIATION_RECORD_FIELDS = {
    "repository",
    "branch",
    "head",
    "merge_base",
    "files_changed",
    "schemas_changed",
    "public_apis_changed",
    "tests",
    "dependencies",
    "superseded",
    "safe_to_cherry_pick",
    "real_merge_required",
    "conflict_risk",
    "recommended_destination",
    "qualification_status",
}
EXECUTION_PREFIXES = {
    "ipfs_accelerate_py": "",
    "ipfs_datasets_py": "ipfs_datasets_py",
    "ipfs_kit_py": "ipfs_kit_py",
    "Mcp-Plus-Plus": "ipfs_accelerate_py/mcplusplus",
}


def _canonical_cid(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load_object(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.relative_to(ROOT)} must contain an object")
        return {}
    return value


def _git_object_exists(repository: Path, oid: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{oid}^{{commit}}"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _git_tree(repository: Path, oid: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{oid}^{{tree}}"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _repository_paths() -> dict[str, Path]:
    return {
        "ipfs_accelerate_py": ROOT,
        "ipfs_datasets_py": ROOT / "ipfs_datasets_py",
        "ipfs_kit_py": ROOT / "ipfs_kit_py",
        "Mcp-Plus-Plus": ROOT / "ipfs_accelerate_py/mcplusplus",
    }


def _validate_source(errors: list[str], warnings: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load_object(SOURCE_PATH, errors)
    stack = _load_object(STACK_PATH, errors)
    if source.get("schema") != "SourceReconciliationManifest@1":
        errors.append("source manifest schema mismatch")
    if stack.get("schema") != "StackCompatibilityManifest@1":
        errors.append("stack manifest schema mismatch")
    payload = source.get("source_forest_payload")
    if not isinstance(payload, dict) or _canonical_cid(payload) != source.get("source_forest_root"):
        errors.append("source_forest_root does not match its canonical payload")
    selected = source.get("selected_integration_roots")
    roots = stack.get("integration_roots")
    if not isinstance(selected, dict) or set(selected) != EXPECTED_REPOSITORIES:
        errors.append("source manifest does not select exactly the four repositories")
        selected = {}
    if not isinstance(roots, dict) or set(roots) != EXPECTED_REPOSITORIES:
        errors.append("stack manifest does not bind exactly the four repositories")
        roots = {}
    repositories = _repository_paths()
    for name in sorted(EXPECTED_REPOSITORIES):
        source_root = selected.get(name)
        stack_root = roots.get(name)
        if not isinstance(source_root, dict) or not isinstance(stack_root, dict):
            continue
        for field in ("commit", "tree", "integration_branch"):
            if str(source_root.get(field) or "") != str(stack_root.get(field) or ""):
                errors.append(f"{name}: source/stack {field} mismatch")
        commit = str(stack_root.get("commit") or "")
        tree = str(stack_root.get("tree") or "")
        if len(commit) != 40 or len(tree) != 40:
            errors.append(f"{name}: integration root is not full commit/tree identity")
            continue
        path = repositories[name]
        if not _git_object_exists(path, commit):
            errors.append(f"{name}: integration commit is not present locally: {commit}")
        elif _git_tree(path, commit) != tree:
            errors.append(f"{name}: integration tree differs from recorded tree")
    qualification = stack.get("qualification_state")
    if isinstance(qualification, dict):
        no_go = [key for key, value in qualification.items() if str(value).startswith("no_go")]
        if no_go:
            warnings.append("current fail-closed promotion gates: " + ", ".join(sorted(no_go)))
    else:
        errors.append("stack qualification_state is missing")
    repository_records = source.get("repositories")
    if not isinstance(repository_records, dict):
        errors.append("source manifest repository reconciliation records are missing")
    else:
        for repository in sorted(EXPECTED_REPOSITORIES):
            record = repository_records.get(repository)
            if not isinstance(record, dict):
                errors.append(f"{repository}: reconciliation record is missing")
                continue
            inventory = record.get("inventory")
            if not isinstance(inventory, dict) or not inventory:
                errors.append(f"{repository}: inventory evidence is missing")
            for index, branch in enumerate(record.get("relevant_unmerged") or ()):  # closed A2 rows
                label = f"{repository}.relevant_unmerged[{index}]"
                if not isinstance(branch, dict):
                    errors.append(f"{label}: must be an object")
                    continue
                missing = sorted(RECONCILIATION_RECORD_FIELDS - set(branch))
                if missing:
                    errors.append(f"{label}: missing closed A2 fields {missing}")
                for oid_field in ("head", "merge_base"):
                    oid = str(branch.get(oid_field) or "")
                    if len(oid) != 40 or any(ch not in "0123456789abcdef" for ch in oid):
                        errors.append(f"{label}: {oid_field} is not a full lowercase Git OID")
                changed = branch.get("files_changed")
                if not isinstance(changed, dict):
                    errors.append(f"{label}: files_changed must be a complete-set descriptor")
                else:
                    digest = str(changed.get("sha256") or "")
                    count = changed.get("count")
                    if (
                        not isinstance(count, int)
                        or count < 0
                        or len(digest) != 64
                        or any(ch not in "0123456789abcdef" for ch in digest)
                        or changed.get("complete") is not True
                    ):
                        errors.append(f"{label}: files_changed descriptor is incomplete")
    return source, stack


def _load_generator(errors: list[str]):
    spec = importlib.util.spec_from_file_location("eaaef_board_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        errors.append("unable to load board generator")
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"board generator import failed: {exc}")
        return None
    return module


def _validate_generated_parity(errors: list[str]) -> dict[str, Any]:
    board = _load_object(BOARD_PATH, errors)
    module = _load_generator(errors)
    if module is None:
        return board
    try:
        expected_board, expected_markdown, expected_objectives = module._build()
    except Exception as exc:
        errors.append(f"board regeneration failed: {exc}")
        return board
    if board != expected_board:
        errors.append("task_board.json differs from deterministic generator output")
    try:
        if MARKDOWN_PATH.read_text(encoding="utf-8") != expected_markdown:
            errors.append("TASK_BOARD.md differs from deterministic generator output")
        if OBJECTIVES_PATH.read_text(encoding="utf-8") != expected_objectives:
            errors.append("OBJECTIVES.md differs from deterministic generator output")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"generated Markdown is unreadable: {exc}")
    return board


def _validate_native_markdown_projection(
    board: dict[str, Any], errors: list[str]
) -> None:
    """Prove the human projection round-trips through the real supervisor parser."""

    try:
        root_text = str(ROOT)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
            normalize_task_header_prefix,
            parse_task_text,
        )

        parsed = parse_task_text(
            MARKDOWN_PATH.read_text(encoding="utf-8"),
            path=MARKDOWN_PATH,
            task_header_prefix=normalize_task_header_prefix("EAAEF-"),
        )
    except Exception as exc:
        errors.append(f"native supervisor Markdown parse failed: {exc}")
        return
    expected = board.get("tasks")
    if not isinstance(expected, list):
        return
    if len(parsed) != len(expected):
        errors.append("native supervisor Markdown task count differs from JSON")
        return
    for native, task in zip(parsed, expected):
        task_id = str(task.get("stable_task_id") or "")
        if native.task_id != task_id:
            errors.append(f"{task_id}: native Markdown task identity differs")
        if list(native.depends_on) != list(task.get("dependencies") or ()):
            errors.append(f"{task_id}: native Markdown dependencies differ from JSON")
        if list(native.outputs) != list(task.get("outputs") or ()):
            errors.append(f"{task_id}: native Markdown outputs differ from JSON")
        if native.status != str(task.get("status") or ""):
            errors.append(f"{task_id}: native Markdown status differs from JSON")


def _validate_board(board: dict[str, Any], source: dict[str, Any], stack: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if board.get("schema") != (
        "ipfs_accelerate_py/agent-supervisor/"
        "external-agent-autonomous-execution-fabric-board@1"
    ):
        errors.append("board schema mismatch")
    if board.get("board_namespace") != "external-agent-autonomous-execution-fabric-v1":
        errors.append("board namespace mismatch")
    stored_board_cid = board.get("board_cid")
    cid_projection = dict(board)
    cid_projection.pop("board_cid", None)
    if stored_board_cid != _canonical_cid(cid_projection):
        errors.append("board_cid mismatch")
    if board.get("source_reconciliation_manifest_cid") != _canonical_cid(source):
        errors.append("board source manifest CID mismatch")
    if board.get("stack_compatibility_manifest_cid") != _canonical_cid(stack):
        errors.append("board stack manifest CID mismatch")
    if board.get("source_forest_root") != source.get("source_forest_root"):
        errors.append("board source forest root mismatch")
    control = board.get("control_plane")
    control_text = json.dumps(control, sort_keys=True).casefold()
    for required in ("duckdb", "quack", "ducklake", "fenced", "never current"):
        if required not in control_text:
            errors.append(f"control-plane declaration omits {required!r}")

    goals = board.get("goals")
    tasks = board.get("tasks")
    if not isinstance(goals, list) or len(goals) != 19:
        errors.append("board must contain root plus 18 epic goals")
        goals = []
    if not isinstance(tasks, list) or len(tasks) != 104:
        errors.append("board must contain the frozen 104-task population")
        tasks = []
    goal_ids = [str(goal.get("goal_id") or "") for goal in goals if isinstance(goal, dict)]
    if len(goal_ids) != len(set(goal_ids)):
        errors.append("duplicate goal IDs")
    goal_set = set(goal_ids)
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            errors.append(f"goal[{index}] is not an object")
            continue
        goal_id = str(goal.get("goal_id") or f"goal[{index}]")
        missing_goal_fields = sorted(GOAL_REQUIRED_FIELDS - set(goal))
        if missing_goal_fields:
            errors.append(f"{goal_id}: missing H1 goal fields {missing_goal_fields}")
        for field in (
            "desired_postconditions",
            "prohibited_outcomes",
            "authority_ceiling",
            "verification_requirements",
            "proof_requirements",
            "human_review_requirements",
            "completion_evidence",
        ):
            if not isinstance(goal.get(field), list) or not goal.get(field):
                errors.append(f"{goal_id}: {field} must be a nonempty list")
        if not isinstance(goal.get("scope"), dict) or not goal.get("scope"):
            errors.append(f"{goal_id}: scope must be a nonempty object")
        if not isinstance(goal.get("resource_budget"), dict) or not goal.get("resource_budget"):
            errors.append(f"{goal_id}: resource_budget must be a nonempty object")
    task_ids: list[str] = []
    path_owners: dict[tuple[str, str], list[str]] = defaultdict(list)
    roots = stack.get("integration_roots") if isinstance(stack.get("integration_roots"), dict) else {}
    expected_revisions = {
        name: {
            "commit": str(value.get("commit") or ""),
            "tree": str(value.get("tree") or ""),
            "integration_branch": str(value.get("integration_branch") or ""),
        }
        for name, value in roots.items()
        if isinstance(value, dict)
    }
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"task[{index}] is not an object")
            continue
        task_id = str(task.get("stable_task_id") or "")
        task_ids.append(task_id)
        missing_fields = sorted(REQUIRED_TASK_FIELDS - set(task))
        if missing_fields:
            errors.append(f"{task_id}: missing fields {missing_fields}")
        if task.get("parent_goal_id") != "EAAEF-G000":
            errors.append(f"{task_id}: parent goal mismatch")
        if task.get("subgoal_id") not in goal_set:
            errors.append(f"{task_id}: unknown subgoal")
        repository = str(task.get("owning_repository") or "")
        if repository not in EXPECTED_REPOSITORIES:
            errors.append(f"{task_id}: unsupported owning repository {repository!r}")
        if task.get("source_revisions") != expected_revisions:
            errors.append(f"{task_id}: source revisions differ from stack manifest")
        resource = task.get("resource_request")
        if not isinstance(resource, dict) or set(resource) != RESOURCE_FIELDS:
            errors.append(f"{task_id}: resource request field set mismatch")
        if not str(task.get("container_profile") or "").startswith("ContainerExecutionProfile@1:"):
            errors.append(f"{task_id}: container profile is not versioned")
        owned = task.get("owned_files")
        if not isinstance(owned, list) or not owned:
            errors.append(f"{task_id}: owned_files must be nonempty")
            owned = []
        if owned != task.get("write_scope") or owned != task.get("outputs"):
            errors.append(f"{task_id}: owned/write/output path sets differ")
        prefix = EXECUTION_PREFIXES.get(repository, "")
        expected_execution = [
            f"{prefix}/{item}" if prefix else str(item)
            for item in owned
        ]
        if task.get("execution_owned_files") != expected_execution:
            errors.append(f"{task_id}: execution ownership is not repository-qualified")
        if task.get("execution_outputs") != expected_execution:
            errors.append(f"{task_id}: execution outputs differ from qualified ownership")
        validation = str(task.get("validation") or "")
        expected_execution_validation = [
            {
                "working_directory": prefix or ".",
                "argv": shlex.split(command.strip()),
            }
            for command in validation.split(";")
            if command.strip()
        ]
        if task.get("execution_validation") != expected_execution_validation:
            errors.append(f"{task_id}: execution validation is not a deterministic bounded cwd/argv projection")
        conflict_keys = task.get("integration_conflict_keys")
        expected_conflicts = [] if not prefix else [f"serialized-superproject-gitlink:{prefix}"]
        if conflict_keys != expected_conflicts:
            errors.append(f"{task_id}: integration conflict keys differ from repository projection")
        for item in owned:
            path = PurePosixPath(str(item))
            if path.is_absolute() or ".." in path.parts or not str(path):
                errors.append(f"{task_id}: unsafe owned path {item!r}")
            path_owners[(repository, str(path))].append(task_id)
        initial = bool(task.get("initial_population"))
        number = int(task_id.split("-")[-1]) if task_id.startswith("EAAEF-") else 9999
        if initial != (number < 10):
            errors.append(f"{task_id}: initial population marker mismatch")
        expected_status = "todo" if initial else "blocked"
        if task.get("status") != expected_status:
            errors.append(f"{task_id}: bootstrap/template status mismatch")
        if bool(task.get("is_schedulable")) is not initial:
            errors.append(f"{task_id}: only the bootstrap population may be schedulable")
        expected_population_state = (
            "materialized_bootstrap" if initial else "template_only_awaiting_plan_r2"
        )
        if task.get("population_state") != expected_population_state:
            errors.append(f"{task_id}: population state mismatch")
        expected_reason = "" if initial else "awaiting_EAAEF-009_plan_revision"
        if task.get("blocked_reason") != expected_reason:
            errors.append(f"{task_id}: held-task blocked reason mismatch")
        expected_completion_mode = "manual" if task_id == "EAAEF-000" else "auto"
        if task.get("completion_mode") != expected_completion_mode:
            errors.append(f"{task_id}: completion mode must be {expected_completion_mode}")
        if task_id == "EAAEF-000":
            if not isinstance(resource, dict) or any(
                resource.get(field) != 0
                for field in (
                    "supervisor_processes",
                    "container_slots",
                    "provider_concurrency",
                    "model_input_token_ceiling",
                    "model_output_token_ceiling",
                )
            ):
                errors.append("EAAEF-000: host bootstrap admission cannot reserve a supervisor, container or model/provider route")
            bootstrap_text = json.dumps(task, sort_keys=True).casefold()
            for required in (
                "signed eaaef-scoped provider authorization",
                "signed oci image and sbom",
                "default-deny network",
                "materialization",
                "quack authority",
                "no supervisor",
            ):
                if required not in bootstrap_text:
                    errors.append(f"EAAEF-000: bootstrap contract omits {required!r}")
        expected_root = (
            source.get("source_forest_root")
            if initial
            else "REBIND_REQUIRED_BY_EAAEF-009"
        )
        if task.get("source_semantic_state_root") != expected_root:
            errors.append(f"{task_id}: semantic-root bootstrap/rebind policy mismatch")
        if task.get("terminal_status") != "not_terminal":
            errors.append(f"{task_id}: planning task falsely declares a terminal status")
        if task.get("outcome") != "pending":
            errors.append(f"{task_id}: planning task outcome must be pending")
        if task.get("allowed_terminal_statuses") != [
            "completed",
            "cancelled",
            "failed",
            "quarantined",
        ]:
            errors.append(f"{task_id}: database terminal status set is incompatible")
        allowed_outcomes = task.get("allowed_outcomes")
        if not isinstance(allowed_outcomes, list) or "accepted" not in allowed_outcomes:
            errors.append(f"{task_id}: typed outcome set is missing accepted")
            allowed_outcomes = []
        outcome_mapping = task.get("outcome_status_mapping")
        if not isinstance(outcome_mapping, dict):
            errors.append(f"{task_id}: typed outcome/database status mapping is missing")
        elif set(outcome_mapping) != set(allowed_outcomes):
            errors.append(f"{task_id}: every typed outcome must map to one database terminal status")
        elif any(
            status not in task.get("allowed_terminal_statuses", ())
            for status in outcome_mapping.values()
        ):
            errors.append(f"{task_id}: typed outcome maps to an inadmissible database status")
        artifacts = task.get("final_artifact_identities")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{task_id}: final artifact identity roles are missing")
        tests = task.get("test_requirements")
        if not isinstance(tests, dict) or set(tests) != {
            "pre_change",
            "focused",
            "affected_integration",
            "required_result",
        }:
            errors.append(f"{task_id}: phased test requirements are incomplete")
        if not str(task.get("idempotency_key") or "").startswith("sha256:"):
            errors.append(f"{task_id}: idempotency key is not content-addressed")
        task_projection = dict(task)
        task_projection.pop("task_spec_cid", None)
        if task.get("task_spec_cid") != _canonical_cid(task_projection):
            errors.append(f"{task_id}: task_spec_cid mismatch")
    if len(task_ids) != len(set(task_ids)):
        errors.append("duplicate task IDs")
    task_set = set(task_ids)
    indegree = {task_id: 0 for task_id in task_ids}
    children: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("stable_task_id") or "")
        dependencies = task.get("dependencies")
        if not isinstance(dependencies, list):
            errors.append(f"{task_id}: dependencies must be a list")
            continue
        for dependency in dependencies:
            dependency = str(dependency)
            if dependency not in task_set:
                errors.append(f"{task_id}: unknown dependency {dependency}")
                continue
            indegree[task_id] += 1
            children[dependency].append(task_id)
    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    visited: list[str] = []
    while queue:
        current = queue.popleft()
        visited.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) != len(task_ids):
        errors.append("task dependency graph is cyclic")
    collisions = {
        f"{repository}:{path}": owners
        for (repository, path), owners in path_owners.items()
        if len(owners) > 1
    }
    if collisions:
        errors.append(f"owned-file collisions require explicit split: {collisions}")
    initial_ids = [task_id for task_id in task_ids if int(task_id.split("-")[-1]) < 10]
    if board.get("initial_population_task_ids") != initial_ids:
        errors.append("initial population list differs from task order")
    ready_ids = [
        str(task.get("stable_task_id") or "")
        for task in tasks
        if isinstance(task, dict)
        and task.get("status") == "todo"
        and task.get("is_schedulable") is True
        and not task.get("dependencies")
    ]
    if ready_ids != ["EAAEF-000"]:
        errors.append(f"initial ready frontier must contain only EAAEF-000, got {ready_ids}")
    return {
        "goal_count": len(goals),
        "task_count": len(tasks),
        "initial_population_count": len(initial_ids),
        "owned_path_count": len(path_owners),
        "dependency_edge_count": sum(len(task.get("dependencies") or []) for task in tasks if isinstance(task, dict)),
    }


def validate(*, source_only: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    source, stack = _validate_source(errors, warnings)
    counts: dict[str, Any] = {}
    board: dict[str, Any] = {}
    if not source_only:
        board = _validate_generated_parity(errors)
        _validate_native_markdown_projection(board, errors)
        counts = _validate_board(board, source, stack, errors)
    return {
        "schema": SCHEMA,
        "valid": not errors,
        "source_only": source_only,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
        "source_forest_root": source.get("source_forest_root"),
        "board_cid": board.get("board_cid"),
        "qualification_status": "planning_and_bootstrap_only",
        "live_launch_allowed": False,
        "live_launch_blockers": [
            "bootstrap OCI worker identity is not admitted",
            "single-supervisor provider route and rootless no-network container policy are not admitted",
            "configured multi-supervisor launch requires an immutable accepted control-plane capsule",
            "continuous Quack and live DuckLake exact profiles remain unqualified",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--check-all", action="store_true", help="structural/source checks; execution suites remain task-owned")
    args = parser.parse_args(argv)
    report = validate(source_only=args.source_only)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
