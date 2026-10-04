#!/usr/bin/env python3
"""Fail-closed validator for the semantic-compression harness board.

Confirms the SCH markdown board is parseable by the agent-supervisor task
parser and that the scheduler document matches the configured-board schema
the supervisor consumes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)

TODO_PATH = REPO_ROOT / "docs/architecture/semantic_compression_harness.todo.md"
SCHEDULER_PATH = (
    REPO_ROOT / "config/agent_supervisor_semantic_compression_harness_scheduler.json"
)
SEAL_PATH = REPO_ROOT / "config/semantic_state_dependencies.seal.json"
OPS_LAUNCHER = (
    REPO_ROOT / "scripts/ops/agent_supervisor/semantic_compression_harness_scheduler.py"
)
SEALED_PINS = {
    "kit": "df2f9cc092456329de9724c45a50c54b410875d1",
    "datasets": "1330038f626ef92993f03d46f21e1a57719e9c25",
    "accelerate": "271e331af802f37d759c000666282631a99f7aab",
    "mcp": "dc3164653a48d059ae9812078359daeafb451c07",
}
BOARD_NAMESPACE = "semantic-compression-harness-v1"
TASK_PREFIX = "## SCH-"
TASK_IDS = tuple(f"SCH-{index:03d}" for index in range(19))
TERMINAL_TASK = "SCH-018"
SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.semantic_compression_harness.scheduler_config@1"
)
SCHEDULER_SCHEMA_PATTERN = re.compile(
    r"^ipfs_accelerate_py\.agent_supervisor\.[a-z0-9_.-]+\.scheduler_config@1$"
)


def _closed_object(pairs):
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def validate_scheduler_document(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    schema = payload.get("schema")
    if schema != SCHEDULER_SCHEMA:
        errors.append(f"schema must equal {SCHEDULER_SCHEMA!r}")
    if not isinstance(schema, str) or SCHEDULER_SCHEMA_PATTERN.fullmatch(schema) is None:
        errors.append(f"schema is not a configured-board scheduler_config@1: {schema!r}")
    for field in (
        "taskboard_path",
        "objectives_path",
        "plan_path",
        "validator_path",
        "task_prefix",
        "board_namespace",
        "merge_target_branch",
    ):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            errors.append(f"{field} must be a nonempty string")
    if payload.get("task_prefix") != "SCH-":
        errors.append("task_prefix must be SCH-")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"board_namespace must be {BOARD_NAMESPACE}")
    if payload.get("taskboard_path") != "docs/architecture/semantic_compression_harness.todo.md":
        errors.append("taskboard_path must point at the SCH todo")
    if payload.get("max_lanes") != 3:
        errors.append("max_lanes must be 3")
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 3:
        errors.append("lanes must contain exactly 3 entries")
    else:
        for index, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                errors.append(f"lanes[{index}] must be an object")
                continue
            if lane.get("index") != index or lane.get("strict_shard_remainder") != index:
                errors.append(f"lanes[{index}] shard remainder mismatch")
    protected = payload.get("protected_paths")
    config_rel = "config/agent_supervisor_semantic_compression_harness_scheduler.json"
    if not isinstance(protected, list) or config_rel not in protected:
        errors.append("scheduler must protect its own source path")
    runtime = payload.get("runtime_paths")
    if not isinstance(runtime, dict):
        errors.append("runtime_paths must be an object")
    else:
        for field in ("root", "state", "worktrees", "merge_queue", "logs"):
            if not isinstance(runtime.get(field), str) or not runtime.get(field):
                errors.append(f"runtime_paths.{field} is required")
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        errors.append("provider must be an object")
    else:
        if provider.get("primary_provider_id") != "grok_cli":
            errors.append("provider.primary_provider_id must be grok_cli")
        if provider.get("max_concurrency") != 3:
            errors.append("provider.max_concurrency must equal max_lanes")
    projection = payload.get("initial_projection")
    if not isinstance(projection, dict):
        errors.append("initial_projection must be an object")
    else:
        completed = tuple(projection.get("completed_task_ids") or ())
        if completed != TASK_IDS:
            errors.append("completed_task_ids must list SCH-000 through SCH-018")
        if projection.get("terminal_task_id") != TERMINAL_TASK:
            errors.append("terminal_task_id must be SCH-018")
        if projection.get("ready_task_ids") not in ([], ()):
            errors.append("completed board must have no ready tasks")
        if projection.get("root_goal_id") != "SCH-G000":
            errors.append("root_goal_id must be SCH-G000")
    return errors


def validate_board() -> list[str]:
    errors: list[str] = []
    tasks = parse_task_file(TODO_PATH, task_header_prefix=TASK_PREFIX)
    by_id = {task.task_id: task for task in tasks}
    if tuple(task.task_id for task in tasks) != TASK_IDS:
        errors.append(
            "task ids must be exactly SCH-000 through SCH-018 in order; "
            f"got {tuple(task.task_id for task in tasks)}"
        )
    visiting: set[str] = set()
    seen: set[str] = set()

    def walk(task_id: str) -> None:
        if task_id in seen:
            return
        if task_id in visiting:
            errors.append(f"dependency cycle at {task_id}")
            return
        visiting.add(task_id)
        task = by_id.get(task_id)
        if task is None:
            errors.append(f"unknown dependency {task_id}")
            visiting.remove(task_id)
            return
        for dep in task.depends_on:
            walk(dep)
        visiting.remove(task_id)
        seen.add(task_id)

    for task in tasks:
        if task.board_namespace != BOARD_NAMESPACE:
            errors.append(
                f"{task.task_id} board_namespace is {task.board_namespace!r}"
            )
        if task.status != "completed":
            errors.append(f"{task.task_id} status is {task.status!r}, expected completed")
        walk(task.task_id)

    try:
        payload = json.loads(
            SCHEDULER_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=_closed_object,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"scheduler is unreadable: {exc}"]
    if not isinstance(payload, dict):
        return ["scheduler root must be an object"]
    errors.extend(validate_scheduler_document(payload))
    if not OPS_LAUNCHER.is_file():
        errors.append("omitted SCH supervisor launcher is missing")
    try:
        seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"dependency seal unreadable: {exc}")
        return errors
    authorities = {
        str(item.get("role")): str(item.get("commit"))
        for item in seal.get("authorities", [])
        if isinstance(item, dict)
    }
    if authorities.get("kit_state_roots") != SEALED_PINS["kit"]:
        errors.append("scheduler/seal kit pin drift")
    if authorities.get("semantic_state_contracts") != SEALED_PINS["datasets"]:
        errors.append("scheduler/seal datasets pin drift")
    if authorities.get("accelerate_harness") != SEALED_PINS["accelerate"]:
        errors.append("scheduler/seal accelerate pin drift")
    binding = payload.get("source_binding")
    if isinstance(binding, dict):
        if binding.get("ipfs_kit_planning_revision") != authorities.get("kit_state_roots"):
            errors.append("source_binding kit pin does not match SCH-000 seal")
        if binding.get("ipfs_datasets_planning_revision") != authorities.get(
            "semantic_state_contracts"
        ):
            errors.append("source_binding datasets pin does not match SCH-000 seal")
        if binding.get("accelerator_required_ancestor") != authorities.get(
            "accelerate_harness"
        ):
            errors.append("source_binding accelerate pin does not match SCH-000 seal")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true")
    parser.parse_args(argv)
    errors = validate_board()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"valid": True, "board_namespace": BOARD_NAMESPACE, "tasks": 19}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
