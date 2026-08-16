#!/usr/bin/env python3
"""Dependency-free fail-closed validator for the LGSWF control bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgswf-board-validation@1"
TASK_RE = re.compile(r"^## (LGSWF-(\d{3})) (.+)$", re.MULTILINE)
GOAL_RE = re.compile(r"^## (LGSWF-G\d{3}) (.+)$", re.MULTILINE)
META_RE = re.compile(r"^- ([^:\n]+):(?: (.*))?$", re.MULTILINE)
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
BOOTSTRAP_BASE = "b6dc155c3d779a4166a8ee92c0e0214e0157e2e2"
ACCELERATOR_BASE = "3a07f2b9273161ce805feff98414ef3c66eae7cc"
DATASETS_BASE = "0691203550c0f316852c74d293d8fc3c4ce130a6"
BOARD_NAMESPACE = "logic-governed-semantic-work-fabric-actual-v1"
EXPECTED_READY = ["LGSWF-001", "LGSWF-002", "LGSWF-003"]
EXPECTED_COMPLETED = ["LGSWF-000"]
EXPECTED_GOALS = ["LGSWF-G000"] + [f"LGSWF-G{i:03d}" for i in range(10, 151, 10)]

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/architecture/LOGIC_GOVERNED_SEMANTIC_WORK_FABRIC_PLAN.md"
OBJECTIVES = ROOT / "docs/architecture/logic_governed_semantic_work_fabric.objectives.md"
BOARD = ROOT / "docs/architecture/logic_governed_semantic_work_fabric.todo.md"
CONFIG = ROOT / "config/logic_governed_semantic_work_fabric_scheduler.json"
BASELINE = ROOT / "config/logic_governed_semantic_work_fabric_baseline.json"
MATERIALIZER = ROOT / "scripts/materialize_logic_governed_semantic_work_fabric_control_plane.py"

REQUIRED_TASK_FIELDS = (
    "Stable task ID",
    "Status",
    "Completion",
    "Is schedulable",
    "Review only",
    "Priority",
    "Track",
    "Goal id",
    "Parent goal ID",
    "Subgoal ID",
    "Owning repository",
    "Owned paths",
    "Base revision",
    "Base semantic-state root",
    "Base plan revision",
    "Objective",
    "Depends on",
    "Read scope",
    "Write scope",
    "External effect scope",
    "Relevant symbol IDs",
    "Capsule CIDs",
    "Contract and obligation CIDs",
    "Resource demand",
    "Model-route class",
    "Permitted effects",
    "Prohibited effects",
    "Completion contract",
    "Validation requirements",
    "Proof requirements",
    "Lease requirements",
    "Rollback or compensation procedure",
    "Required evidence",
    "Final result identity",
    "Outputs",
    "Validation",
    "Acceptance",
    "Board namespace",
    "Parallel lane",
    "Predicted files",
    "Conflict policy",
    "Raw-source requirements",
)

RESOURCE_FIELDS = {
    "cpu_ms",
    "cpu_concurrency",
    "ram_mib",
    "gpu_memory_mib",
    "gpu_compute_class",
    "disk_mib",
    "disk_bandwidth_mib_s",
    "network",
    "network_bandwidth_kib_s",
    "subprocesses",
    "worktree_slots",
    "model_input_tokens",
    "model_output_tokens",
    "provider_quota_units",
    "provider_concurrency",
    "prover_class",
    "prover_concurrency",
    "exclusive_keys",
    "merge_slots",
    "persistence_kib_s",
}
RESOURCE_INTEGER_FIELDS = RESOURCE_FIELDS - {
    "gpu_compute_class",
    "network",
    "prover_class",
    "exclusive_keys",
}


def _identity(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _load_text(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"missing/unreadable {path.relative_to(ROOT)}: {exc}")
        return ""


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"JSON root is not an object: {path.relative_to(ROOT)}")
        return {}
    return value


def _blocks(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str, str]]:
    matches = list(pattern.finditer(text))
    result: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append((match.group(1), match.group(match.lastindex or 1), text[match.end():end]))
    return result


def _metadata(body: str, *, record_id: str, errors: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in META_RE.finditer(body):
        key = match.group(1).strip()
        value = (match.group(2) or "").strip()
        if key in result:
            errors.append(f"{record_id}: duplicate field {key!r}")
        result[key] = value
    return result


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_paths(value: str, *, task_id: str, field: str, errors: list[str]) -> list[str]:
    paths = _csv(value)
    if not paths:
        errors.append(f"{task_id}: {field} is empty")
    for item in paths:
        path = PurePosixPath(item)
        if path.is_absolute() or ".." in path.parts or item.startswith("-") or any(ch in item for ch in "*?[]\x00"):
            errors.append(f"{task_id}: unsafe {field} path {item!r}")
    return paths


def _resource_vector(value: str, *, task_id: str, errors: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for segment in value.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if "=" not in segment:
            errors.append(f"{task_id}: malformed resource segment {segment!r}")
            continue
        key, raw = (part.strip() for part in segment.split("=", 1))
        if key in result:
            errors.append(f"{task_id}: duplicate resource field {key}")
        result[key] = raw
    missing = sorted(RESOURCE_FIELDS - set(result))
    extra = sorted(set(result) - RESOURCE_FIELDS)
    if missing:
        errors.append(f"{task_id}: resource vector missing {missing}")
    if extra:
        errors.append(f"{task_id}: resource vector has unknown {extra}")
    for key in sorted(RESOURCE_INTEGER_FIELDS & set(result)):
        if not re.fullmatch(r"\d+", result[key]):
            errors.append(f"{task_id}: resource {key} must be a nonnegative integer")
    return result


def _dependency_closure(task_id: str, deps: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(deps.get(task_id, []))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(deps.get(current, []))
    return seen


def _overlap(left: str, right: str) -> bool:
    a = PurePosixPath(left).parts
    b = PurePosixPath(right).parts
    return a[: len(b)] == b or b[: len(a)] == a


def _git(args: list[str], cwd: Path = ROOT) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True,
            check=False, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _command(args: list[str], cwd: Path = ROOT) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def validate(*, require_database: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    plan_text = _load_text(PLAN, errors)
    objectives_text = _load_text(OBJECTIVES, errors)
    board_text = _load_text(BOARD, errors)
    config = _load_json(CONFIG, errors)
    baseline = _load_json(BASELINE, errors)

    tasks: dict[str, dict[str, str]] = {}
    titles: dict[str, str] = {}
    for match in TASK_RE.finditer(board_text):
        task_id, _number, title = match.group(1), match.group(2), match.group(3).strip()
        if task_id in tasks:
            errors.append(f"duplicate task heading: {task_id}")
            continue
        next_match = TASK_RE.search(board_text, match.end())
        body = board_text[match.end(): next_match.start() if next_match else len(board_text)]
        tasks[task_id] = _metadata(body, record_id=task_id, errors=errors)
        titles[task_id] = title

    goals: dict[str, dict[str, str]] = {}
    for match in GOAL_RE.finditer(objectives_text):
        goal_id, title = match.group(1), match.group(2).strip()
        if goal_id in goals:
            errors.append(f"duplicate goal heading: {goal_id}")
            continue
        next_match = GOAL_RE.search(objectives_text, match.end())
        body = objectives_text[match.end(): next_match.start() if next_match else len(objectives_text)]
        goals[goal_id] = _metadata(body, record_id=goal_id, errors=errors)
        if not title:
            errors.append(f"{goal_id}: empty title")

    if len(tasks) != 47:
        errors.append(f"expected 47 tasks, found {len(tasks)}")
    if sorted(goals) != EXPECTED_GOALS:
        errors.append(f"goal population mismatch: {sorted(goals)}")

    dependencies: dict[str, list[str]] = {}
    owned: dict[str, list[str]] = {}
    for task_id, metadata in sorted(tasks.items()):
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{task_id}: missing fields {missing}")
        for field in REQUIRED_TASK_FIELDS:
            if field == "Depends on":
                continue
            if field in metadata and not metadata[field]:
                errors.append(f"{task_id}: empty required field {field!r}")
        if metadata.get("Stable task ID") != task_id:
            errors.append(f"{task_id}: Stable task ID mismatch")
        if metadata.get("Board namespace") != BOARD_NAMESPACE:
            errors.append(f"{task_id}: board namespace mismatch")
        if metadata.get("Parent goal ID") != "LGSWF-G000":
            errors.append(f"{task_id}: parent goal must be LGSWF-G000")
        subgoal = metadata.get("Subgoal ID", "")
        goal_id = metadata.get("Goal id", "")
        if subgoal not in goals:
            errors.append(f"{task_id}: unknown subgoal {subgoal!r}")
        if goal_id not in goals:
            errors.append(f"{task_id}: unknown Goal id {goal_id!r}")
        if task_id != "LGSWF-000" and subgoal != goal_id:
            errors.append(f"{task_id}: Goal id/Subgoal ID mismatch")
        status = metadata.get("Status", "")
        if status not in {"todo", "completed"}:
            errors.append(f"{task_id}: unsupported initial status {status!r}")
        if metadata.get("Is schedulable") != "true" or metadata.get("Review only") != "false":
            errors.append(f"{task_id}: schedulable/review metadata mismatch")
        owner = metadata.get("Owning repository")
        base = metadata.get("Base revision", "")
        expected_base = (
            DATASETS_BASE
            if owner == "ipfs_datasets_py"
            else BOOTSTRAP_BASE
            if task_id == "LGSWF-000"
            else ACCELERATOR_BASE
        )
        if not HEX40_RE.fullmatch(base) or base != expected_base:
            errors.append(f"{task_id}: wrong exact base revision for {owner}: {base!r}")
        dependencies[task_id] = _csv(metadata.get("Depends on", ""))
        owned[task_id] = _safe_paths(metadata.get("Owned paths", ""), task_id=task_id, field="Owned paths", errors=errors)
        _safe_paths(metadata.get("Predicted files", ""), task_id=task_id, field="Predicted files", errors=errors)
        _safe_paths(metadata.get("Outputs", ""), task_id=task_id, field="Outputs", errors=errors)
        _resource_vector(metadata.get("Resource demand", ""), task_id=task_id, errors=errors)
        if "lease" not in metadata.get("Lease requirements", "").lower():
            errors.append(f"{task_id}: lease requirements do not name a lease")

    for task_id, task_deps in dependencies.items():
        for dep in task_deps:
            if dep not in tasks:
                errors.append(f"{task_id}: unknown dependency {dep}")
            if dep == task_id:
                errors.append(f"{task_id}: self dependency")

    indegree = {task_id: 0 for task_id in tasks}
    children: dict[str, list[str]] = defaultdict(list)
    for task_id, task_deps in dependencies.items():
        for dep in task_deps:
            if dep in tasks:
                indegree[task_id] += 1
                children[dep].append(task_id)
    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    topo: list[str] = []
    while queue:
        current = queue.popleft()
        topo.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(topo) != len(tasks):
        errors.append("task dependency graph contains a cycle")

    completed = sorted(task_id for task_id, data in tasks.items() if data.get("Status") == "completed")
    ready = sorted(
        task_id for task_id, data in tasks.items()
        if data.get("Status") == "todo" and all(dep in completed for dep in dependencies.get(task_id, []))
    )
    if completed != EXPECTED_COMPLETED:
        errors.append(f"completed task population mismatch: {completed}")
    if ready != EXPECTED_READY:
        errors.append(f"ready task population mismatch: {ready}")

    for task_id, data in tasks.items():
        number = int(task_id.rsplit("-", 1)[1])
        if task_id not in {"LGSWF-000", "LGSWF-001", "LGSWF-002", "LGSWF-003", "LGSWF-004", "LGSWF-005"}:
            if "LGSWF-005" not in _dependency_closure(task_id, dependencies):
                errors.append(f"{task_id}: post-A task does not depend transitively on LGSWF-005")
            if data.get("Base semantic-state root") != "REBIND_REQUIRED_BY_LGSWF-005":
                errors.append(f"{task_id}: post-A semantic-root sentinel mismatch")
            if data.get("Base plan revision") != "LGSWF-PLAN-R2-required":
                errors.append(f"{task_id}: post-A plan revision is not R2-required")
        _ = number

    for goal_id, data in sorted(goals.items()):
        for field in ("Status", "Parent", "Depends on", "Priority", "Track", "Goal", "Completion contract", "Evidence", "Acceptance criteria", "Outputs", "Validation", "Acceptance", "Gap task"):
            if field not in data:
                errors.append(f"{goal_id}: missing field {field!r}")
            elif field not in {"Parent", "Depends on"} and not data[field]:
                errors.append(f"{goal_id}: empty field {field!r}")
        parent = data.get("Parent", "")
        if goal_id == "LGSWF-G000":
            if parent:
                errors.append("root goal must not have a parent")
        elif parent != "LGSWF-G000":
            errors.append(f"{goal_id}: parent must be LGSWF-G000")

    if config.get("board_namespace") != BOARD_NAMESPACE:
        errors.append("config board namespace mismatch")
    if config.get("task_prefix") != "LGSWF-" or config.get("goal_prefix") != "LGSWF-G":
        errors.append("config task/goal prefix mismatch")
    if config.get("max_lanes") != 1 or config.get("strict_task_sharding") is not True:
        errors.append("bootstrap config must use one strict single-writer lane")
    if config.get("qualification_target_max_lanes") != 3:
        errors.append("qualification target must retain three conflict-aware lanes")
    writer_policy = config.get("bootstrap_writer_policy")
    if not isinstance(writer_policy, dict) or writer_policy.get("maximum_processes") != 1:
        errors.append("bootstrap writer policy must prohibit multiple DuckDB file writers")
    source_binding = config.get("source_binding")
    if not isinstance(source_binding, dict) or source_binding.get("bootstrap_task_source") != "duckdb":
        errors.append("DuckDB must be the explicit bootstrap task authority")
    database_program = config.get("database_program")
    if not isinstance(database_program, dict):
        errors.append("database_program is required")
    else:
        if database_program.get("authority_mode") != "embedded":
            errors.append("bootstrap database authority must be bounded embedded mode")
        if database_program.get("task_source_kind") != "duckdb":
            errors.append("bootstrap task source must be duckdb")
        if database_program.get("failover_policy") != "fail_closed":
            errors.append("database authority must fail closed")
        store_id = str(database_program.get("store_id") or "")
        if not store_id.endswith("/control.duckdb") or ".." in PurePosixPath(store_id).parts:
            errors.append("database store_id must be the sealed repository-relative control.duckdb")
    projection = config.get("initial_projection") if isinstance(config.get("initial_projection"), dict) else {}
    if projection.get("task_count") != len(tasks):
        errors.append("config initial task count mismatch")
    if sorted(projection.get("completed_task_ids") or []) != completed:
        errors.append("config completed projection mismatch")
    if sorted(projection.get("ready_task_ids") or []) != ready:
        errors.append("config ready projection mismatch")
    if projection.get("goal_count") != len(goals) or projection.get("root_goal_id") != "LGSWF-G000":
        errors.append("config goal projection mismatch")
    if projection.get("terminal_task_id") != "LGSWF-141":
        errors.append("config terminal task mismatch")
    if config.get("objective_refill_enabled") is not False or config.get("codebase_refill_enabled") is not False:
        errors.append("bootstrap refill must remain disabled until Epic I acceptance")
    provider = config.get("provider") if isinstance(config.get("provider"), dict) else {}
    if provider.get("provider_id") != "codex" or provider.get("model_id") != "gpt-5.6-terra" or provider.get("max_concurrency") != 3:
        errors.append("provider must be direct bounded codex/gpt-5.6-terra concurrency 3")

    expected_control_paths = {
        PLAN.relative_to(ROOT).as_posix(), OBJECTIVES.relative_to(ROOT).as_posix(),
        BOARD.relative_to(ROOT).as_posix(), CONFIG.relative_to(ROOT).as_posix(),
        BASELINE.relative_to(ROOT).as_posix(), Path(__file__).resolve().relative_to(ROOT).as_posix(),
        MATERIALIZER.relative_to(ROOT).as_posix(),
    }
    protected = set(config.get("protected_paths") or [])
    if not expected_control_paths <= protected:
        errors.append(f"config does not protect all controls: {sorted(expected_control_paths - protected)}")

    waves = config.get("waves") if isinstance(config.get("waves"), list) else []
    wave_population: list[str] = []
    for wave in waves:
        if not isinstance(wave, dict) or not isinstance(wave.get("task_ids"), list):
            errors.append("malformed wave entry")
            continue
        ids = [str(item) for item in wave["task_ids"]]
        wave_population.extend(ids)
        for i, left_id in enumerate(ids):
            for right_id in ids[i + 1:]:
                if left_id not in tasks or right_id not in tasks:
                    continue
                ordered = left_id in _dependency_closure(right_id, dependencies) or right_id in _dependency_closure(left_id, dependencies)
                if ordered:
                    continue
                for left_path in owned[left_id]:
                    for right_path in owned[right_id]:
                        if _overlap(left_path, right_path):
                            errors.append(f"unsafe same-wave owned-path overlap: {left_id}:{left_path} <> {right_id}:{right_path}")
    if sorted(wave_population) != sorted(tasks):
        errors.append("waves do not contain each task exactly once")

    selected = baseline.get("actual_accelerator_snapshot") if isinstance(baseline.get("actual_accelerator_snapshot"), dict) else {}
    semantic = baseline.get("semantic_authority") if isinstance(baseline.get("semantic_authority"), dict) else {}
    if selected.get("duckdb_integration_commit") != BOOTSTRAP_BASE or selected.get("duckdb_integration_tree") != "1313cf18fecd969f654f0233f6678c2d851116e8":
        errors.append("baseline selected accelerator identity mismatch")
    if selected.get("control_contract_commit") != ACCELERATOR_BASE:
        errors.append("baseline control-contract identity mismatch")
    if semantic.get("head") != DATASETS_BASE or semantic.get("semantic_state_root_status") != "unavailable" or semantic.get("semantic_state_root") is not None:
        errors.append("baseline datasets identity/root status mismatch")
    commitments = baseline.get("intervening_change_commitments")
    if not isinstance(commitments, list) or len(commitments) != 4:
        errors.append("baseline must contain four intervening-change commitments")
    else:
        for index, item in enumerate(commitments):
            if not isinstance(item, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("ordered_log_sha256") or "")):
                errors.append(f"baseline change commitment {index} is incomplete")

    required_plan_phrases = (
        "`ipfs_datasets_py` is authoritative", "`ipfs_accelerate_py` is authoritative",
        "SupervisorWorldSnapshot@1", "SemanticWorkBinding@1", "SemanticWorkGraph@1",
        "deterministic conflict-free", "fixed-point", "three supervisors and ten daemons",
        "A: one supervisor/one daemon/serial", "explicit continuous-operation go/no-go",
    )
    for phrase in required_plan_phrases:
        if phrase not in plan_text:
            errors.append(f"plan missing required phrase: {phrase!r}")

    branch_code, branch, branch_err = _git(["branch", "--show-current"])
    if branch_code != 0 or branch != "agent/logic-governed-semantic-work-fabric-actual-v1":
        errors.append(f"wrong launch branch: {branch or branch_err}")
    ancestor_code, _, ancestor_err = _git(["merge-base", "--is-ancestor", ACCELERATOR_BASE, "HEAD"])
    if ancestor_code != 0:
        errors.append(f"selected accelerator base is not an ancestor: {ancestor_err}")
    for relative, expected in (
        ("ipfs_datasets_py", DATASETS_BASE),
        ("ipfs_kit_py", "e164bb21c7a73b722a83aea7623e5677391bce54"),
        ("ipfs_accelerate_py/mcplusplus", "15c1816d6c63a2b11edd505704f6a04a9abc6167"),
    ):
        path = ROOT / relative
        code, head, error = _git(["rev-parse", "HEAD"], cwd=path)
        if code != 0 or head != expected:
            errors.append(f"submodule {relative} identity mismatch: {head or error}")

    if require_database:
        code, output, error = _command(
            [sys.executable, str(MATERIALIZER), "verify"], cwd=ROOT
        )
        if code != 0:
            errors.append(
                "DuckDB control-plane verification failed: "
                + (output or error).strip()[:2000]
            )

    artifacts = {}
    for path in (PLAN, OBJECTIVES, BOARD, CONFIG, BASELINE, Path(__file__).resolve()):
        try:
            artifacts[path.relative_to(ROOT).as_posix()] = _identity(path.read_bytes())
        except OSError:
            pass
    return {
        "schema": SCHEMA,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "board_namespace": BOARD_NAMESPACE,
        "task_count": len(tasks),
        "goal_count": len(goals),
        "completed_task_ids": completed,
        "ready_task_ids": ready,
        "topological_task_ids": topo,
        "initial_lane_assignments": {task_id: 0 for task_id in ready},
        "control_artifact_identities": artifacts,
        "source": {
            "branch": branch,
            "accelerator_planning_revision": ACCELERATOR_BASE,
            "datasets_planning_revision": DATASETS_BASE,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-all", action="store_true", help="also require the materialized DuckDB authority")
    args = parser.parse_args()
    report = validate(require_database=args.check_all)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
