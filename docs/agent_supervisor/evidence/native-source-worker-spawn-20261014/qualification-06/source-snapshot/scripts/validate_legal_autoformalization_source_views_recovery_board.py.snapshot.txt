#!/usr/bin/env python3
"""Validate the native source-views board and frozen production/test inputs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def module(root, name):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-all", action="store_true")
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for key, value in {"IPFS_ACCEL_SKIP_CORE": "1", "IPFS_AUTO_INSTALL": "false",
                       "IPFS_DATASETS_AUTO_INSTALL": "false", "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1"}.items():
        os.environ.setdefault(key, value)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_file
    from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import parse_goal_heap
    config = json.loads((root / "config/legal_autoformalization_source_views_recovery_scheduler.json").read_text())
    sources_path = root / "config/legal_autoformalization_source_views_recovery_sources.json"
    sources = json.loads(sources_path.read_text())
    tasks = parse_task_file(root / config["taskboard_path"], "## LAC-S8R1-")
    goals = parse_goal_heap((root / config["objectives_path"]).read_text())
    expected = {"LAC-S8R1-011", "LAC-S8R1-024", "LAC-S8R1-044"}
    errors = []
    if len(tasks) != 3 or {t.task_id for t in tasks} != expected:
        errors.append("Exact three-task source-views population differs")
    if len(goals) != 4:
        errors.append("Four-objective source-views hierarchy differs")
    if config["board_namespace"] != sources["campaign_id"]:
        errors.append("Board/source campaign binding differs")
    if config.get("max_lanes") != 3 or config.get("objective_refill_enabled") or config.get("codebase_refill_enabled"):
        errors.append("Bounded parallelism/refill contract differs")
    runtime = "data/agent_supervisor/legal_autoformalization_source_views_recovery/runtime/generation-01"
    wanted_runtime = {"root": runtime, **{key: runtime + "/" + key
        for key in ("state", "worktrees", "merge_queue", "logs", "evidence")}}
    if config.get("runtime_paths") != wanted_runtime:
        errors.append("Owned phase8 generation runtime differs")
    for task in tasks:
        output = "data/legal_autoformalization_source_views_recovery/" + task.task_id + "/result.json"
        command = "python3 scripts/legal_autoformalization_source_views_recovery.py materialize --task " + task.task_id
        if task.metadata.get("outputs") != output or task.metadata.get("predicted files") != output:
            errors.append(task.task_id + ": disjoint exact output contract differs")
        if task.metadata.get("validation") != command or task.depends_on:
            errors.append(task.task_id + ": independent reviewed command differs")
        if task.metadata.get("provider role") != "deterministic-only" or task.metadata.get("context budget tokens") != "0":
            errors.append(task.task_id + ": zero-provider execution contract differs")
    if any(not (root / path).is_file() for path in config["protected_paths"]):
        errors.append("Protected source-views control file missing")
    try:
        driver = module(root, "legal_autoformalization_source_views_recovery")
        profiles = {"LAC-S8R1-011": "document_views", "LAC-S8R1-024": "source_view_inventory",
                    "LAC-S8R1-044": "producer_lineage"}
        for task in sorted(expected):
            _, _, job, _ = driver.bindings(task, sources_path)
            if job.get("real_input", {}).get("spec", {}).get("kind") != profiles[task]:
                errors.append(task + ": exact phase8 real-input profile differs")
            else:
                driver.expected_real_input(job)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append("Frozen source validation refused: " + str(error))
    completed = sorted(task.task_id for task in tasks if task.status == "completed")
    ready = sorted(task.task_id for task in tasks if task.status == "todo")
    print(json.dumps({"schema": "legal-autoformalization-source-views-recovery-board-validation/v1",
        "valid": not errors, "task_count": len(tasks), "goal_count": len(goals),
        "completed_task_ids": completed, "ready_task_ids": ready, "errors": errors,
        "datasets_commit": sources["datasets"]["commit"],
        "scope": "bounded versioned comparison views, occurrence inventories and producer-lineage qualification; original full-plan acceptance remains partial"}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
