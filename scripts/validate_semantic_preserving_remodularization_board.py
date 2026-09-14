#!/usr/bin/env python3
"""Fail-closed validator for the SPAR goals, board, controls, and scheduler."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC_PATH = ROOT / "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py"
CONFIG_PATH = ROOT / "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json"
TASK_PATH = ROOT / "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md"
GOAL_PATH = ROOT / "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md"

REQUIRED_TASK_FIELDS = frozenset({
    "stable_task_id", "status", "completion", "is_schedulable", "review_only", "priority", "track", "goal_id", "parent_goal_id", "subgoal_id", "owning_repository", "board_namespace", "base_revision", "base_repository_tree", "base_plan_revision", "objective", "depends_on", "owned_paths", "predicted_files", "predicted_symbols", "read_scope", "write_scope", "external_effect_scope", "authority_impact", "effect_class", "public_api_impact", "state_impact", "preconditions", "declared_effects", "permitted_effects", "prohibited_effects", "resource_class", "timeout", "provider_role", "context_budget", "token_budget", "resource_demand", "model_route_class", "no_model_route", "model_fallback", "autonomy_tier", "rollout_mode", "parallel_lane", "concurrency_group", "conflict_policy", "lease_and_fencing", "interfaces", "acceptance_subset", "completion_contract", "validation", "proof_requirements", "rollback", "required_evidence", "evidence", "final_result_identity", "outputs", "raw_source_requirements", "protected_paths", "limitations", "capability_blockers"
})
REQUIRED_GOAL_FIELDS = frozenset({"status", "parent", "depends_on", "priority", "track", "goal", "completion_contract", "evidence", "acceptance_criteria", "outputs", "validation", "acceptance", "gap_task"})


class ValidationError(RuntimeError):
    pass


def _load_spec() -> Any:
    spec = importlib.util.spec_from_file_location("spar_control_spec_board_validator", SPEC_PATH)
    if spec is None or spec.loader is None:
        raise ValidationError("SPAR control specification unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise ValidationError(f"{path.name} must contain one object")
    return value


def _blocks(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str, dict[str, str]]]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import normalize_metadata_key
    matches = list(pattern.finditer(text))
    result: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for line in text[match.end():end].splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            key, value = stripped[2:].split(":", 1)
            normalized = normalize_metadata_key(key)
            if normalized in fields:
                raise ValidationError(f"{match.group(1)} duplicate field {normalized}")
            fields[normalized] = value.strip()
        result.append((match.group(1), match.group(2).strip(), fields))
    return result


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def validate() -> dict[str, Any]:
    spec = _load_spec()
    config = _load_json(CONFIG_PATH)
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(name)

    render_report = spec.render(ROOT, check=True)
    check("deterministic_controls", render_report["valid"], render_report["mismatches"])

    tasks = _blocks(TASK_PATH.read_text(encoding="utf-8"), re.compile(r"^## (SPAR-\d{3}) (.+)$", re.MULTILINE))
    goals = _blocks(GOAL_PATH.read_text(encoding="utf-8"), re.compile(r"^## (SPAR-G\d{3}) (.+)$", re.MULTILINE))
    expected_tasks = {task.task_id: task for task in spec.TASKS}
    expected_goals = {goal.goal_id: goal for goal in spec.GOALS}
    check("task_ids", tuple(row[0] for row in tasks) == tuple(expected_tasks), tuple(row[0] for row in tasks))
    check("goal_ids", tuple(row[0] for row in goals) == tuple(expected_goals), tuple(row[0] for row in goals))

    task_errors: list[str] = []
    for task_id, title, fields in tasks:
        expected = expected_tasks.get(task_id)
        if expected is None:
            task_errors.append(f"{task_id}:unknown")
            continue
        missing = sorted(REQUIRED_TASK_FIELDS - fields.keys())
        if missing:
            task_errors.append(f"{task_id}:missing:{','.join(missing)}")
        if title != expected.title or fields.get("stable_task_id") != task_id:
            task_errors.append(f"{task_id}:identity")
        if fields.get("status") != "todo" or fields.get("priority") != "P0":
            task_errors.append(f"{task_id}:initial-state")
        operator = task_id == "SPAR-000"
        if fields.get("completion") != ("operator" if operator else "auto") or fields.get("is_schedulable") != ("false" if operator else "true") or fields.get("review_only") != ("true" if operator else "false"):
            task_errors.append(f"{task_id}:completion-mode")
        if fields.get("board_namespace") != spec.PROGRAM or fields.get("base_revision") != spec.BASE_REVISION or fields.get("base_repository_tree") != spec.BASE_TREE or fields.get("base_plan_revision") != spec.PLAN_REVISION:
            task_errors.append(f"{task_id}:base-binding")
        if fields.get("goal_id") != expected.goal_id or fields.get("subgoal_id") != expected.goal_id or fields.get("owning_repository") != expected.owner:
            task_errors.append(f"{task_id}:ownership")
        if _csv(fields.get("depends_on", "")) != expected.dependencies:
            task_errors.append(f"{task_id}:dependencies")
        if fields.get("final_result_identity") != "pending; only current authority derives it after validation, merge, and post-merge acceptance":
            task_errors.append(f"{task_id}:result-authority")
        if "vector/model authority" not in fields.get("prohibited_effects", "") or "self-approval" not in fields.get("prohibited_effects", ""):
            task_errors.append(f"{task_id}:prohibitions")
    check("task_schema_and_bindings", not task_errors, task_errors[:100])

    goal_errors: list[str] = []
    for goal_id, title, fields in goals:
        expected = expected_goals.get(goal_id)
        if expected is None:
            goal_errors.append(f"{goal_id}:unknown")
            continue
        missing = sorted(REQUIRED_GOAL_FIELDS - fields.keys())
        if missing:
            goal_errors.append(f"{goal_id}:missing:{','.join(missing)}")
        if title != expected.title or fields.get("status") != "active" or fields.get("parent", "") != expected.parent or _csv(fields.get("depends_on", "")) != expected.depends_on:
            goal_errors.append(f"{goal_id}:binding")
    check("goal_schema_and_bindings", not goal_errors, goal_errors)

    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import load_configured_board
        board = load_configured_board(CONFIG_PATH, repo_root=ROOT)
        scheduler_detail: Any = {"namespace": board.board_namespace, "lanes": board.max_lanes, "authority": board.resolved_database_program().authority_mode}
        scheduler_valid = board.board_namespace == spec.PROGRAM and board.task_prefix == "SPAR-" and board.max_lanes == 3 and board.resolved_database_program().authority_mode == "quack" and board.resolved_database_program().task_source_kind == "duckdb"
    except Exception as exc:
        scheduler_valid, scheduler_detail = False, type(exc).__name__
    check("configured_board_loader", scheduler_valid, scheduler_detail)

    seal = _load_json(ROOT / config.get("dependency_seal_path", ""))
    projection = config.get("initial_projection") if isinstance(config.get("initial_projection"), dict) else {}
    check("initial_projection", projection == {"task_count": 51, "task_dependency_count": seal.get("dependency_count"), "task_dependency_root_cid": seal.get("dependency_root_cid"), "completed_task_ids": ["SPAR-000"], "ready_task_ids": ["SPAR-001"], "blocked_task_ids": [], "terminal_task_id": "SPAR-050", "goal_count": 32, "root_goal_id": "SPAR-G000"}, projection)
    check("authority_policy", config.get("authority_policy", {}).get("duckdb_transactional_authority") is True and config.get("authority_policy", {}).get("quack_exclusive_state_owner_transport") is True and config.get("authority_policy", {}).get("ducklake_projection_authority") is False and config.get("authority_policy", {}).get("vector_similarity_is_authority") is False and config.get("authority_policy", {}).get("worker_self_approval") is False, config.get("authority_policy"))
    check("ducklake_non_authority", config.get("ducklake_projection_program", {}).get("authority") is False and config.get("ducklake_projection_program", {}).get("scheduling_prerequisite") is False and config.get("ducklake_projection_program", {}).get("completion_prerequisite") is False, config.get("ducklake_projection_program"))
    protected = set(config.get("protected_paths", ()))
    required_protected = set(spec.render_payloads()) | {"scripts/validate_semantic_preserving_remodularization_dependencies.py", "scripts/validate_semantic_preserving_remodularization_board.py", "scripts/materialize_semantic_preserving_remodularization_program.py", "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py", "test/api/semantic_refactoring/test_bootstrap_controls.py"}
    check("protected_controls", not sorted(required_protected - protected), sorted(required_protected - protected))

    configured_waves = tuple((item.get("id"), tuple(item.get("task_ids", ()))) for item in config.get("waves", ()))
    check("waves", configured_waves == spec.WAVES, {"count": len(configured_waves)})
    wave_conflicts: list[str] = []
    for wave, ids in spec.WAVES:
        paths: dict[str, str] = {}
        for task_id in ids:
            task = expected_tasks[task_id]
            for path in task.paths:
                for seen, owner in paths.items():
                    if path == seen or path.startswith(seen.rstrip("/") + "/") or seen.startswith(path.rstrip("/") + "/"):
                        wave_conflicts.append(f"{wave}:{owner}:{task_id}:{seen}:{path}")
                paths[path] = task_id
    check("parallel_path_disjointness", not wave_conflicts, wave_conflicts)

    plan = (ROOT / config["plan_path"]).read_text(encoding="utf-8")
    terms = ("exact state and receipt reuse", "SCC condensation", "compatibility façade", "DuckDB/`DatabaseTaskSource@1`", "Quack", "DuckLake", "shadow_plan", "shadow_apply", "guarded", "required", "100k", "false completion")
    check("plan_coverage", all(term in plan for term in terms), [term for term in terms if term not in plan])
    return {"schema": "spar/board-validation@1", "valid": not errors, "program": spec.PROGRAM, "task_count": len(tasks), "goal_count": len(goals), "wave_count": len(spec.WAVES), "errors": errors, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true")
    parser.parse_args()
    try:
        report = validate()
    except Exception as exc:
        report = {"schema": "spar/board-validation@1", "valid": False, "errors": [type(exc).__name__], "checks": []}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
