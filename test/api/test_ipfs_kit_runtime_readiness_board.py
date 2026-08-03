"""Supervisor-ingestion tests for the IPFS Kit runtime-readiness board."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    parse_task_file,
)
from scripts.validate_ipfs_kit_runtime_readiness_board import (
    SEALED_TASKBOARD_DEFINITION_SHA256,
    _taskboard_definition_sha256,
    _validate_operational_repair_tasks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TODO_PATH = (
    REPO_ROOT / "docs/architecture/ipfs_kit_runtime_readiness.todo.md"
)
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/ipfs_kit_runtime_readiness.objectives.md"
)
SCHEDULER_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json"
)
VALIDATOR_PATH = (
    REPO_ROOT / "scripts/validate_ipfs_kit_runtime_readiness_board.py"
)


def _with_task_status(text: str, task_id: str, status: str) -> str:
    lines = text.splitlines(keepends=True)
    current_task_id = ""
    for index, line in enumerate(lines):
        if line.startswith("## KITA-"):
            current_task_id = line[3:].strip().split(" ", 1)[0]
        elif current_task_id == task_id and line.startswith("- Status:"):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f"- Status: {status}{newline}"
            return "".join(lines)
    raise AssertionError(f"missing status for {task_id}")


def _task_populations():
    parsed = parse_task_file(TODO_PATH, task_header_prefix="## KITA-")
    canonical = [
        task
        for task in parsed
        if task.metadata.get("canonical board task") != "false"
    ]
    operational = [
        task
        for task in parsed
        if task.metadata.get("canonical board task") == "false"
    ]
    return canonical, operational


def test_production_parsers_consume_exact_task_and_goal_populations() -> None:
    tasks, operational = _task_populations()
    goals = parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))

    assert [task.task_id for task in tasks] == [
        f"KITA-{index:03d}" for index in range(48)
    ]
    assert [goal.goal_id for goal in goals] == [
        "KITA-G000",
        "KITA-G010",
        "KITA-G020",
        "KITA-G030",
        "KITA-G040",
        "KITA-G050",
        "KITA-G060",
        "KITA-G070",
        "KITA-G080",
        "KITA-G090",
        "KITA-G100",
        "KITA-G110",
    ]
    completed = {
        task.task_id for task in tasks if task.status == "completed"
    }
    assert "KITA-000" in completed
    assert all(task.status in {"todo", "completed"} for task in tasks)
    assert all(
        set(task.depends_on).issubset(completed)
        for task in tasks
        if task.status == "completed"
    )
    assert all(
        task.board_namespace == "ipfs-kit-runtime-readiness-v1"
        for task in tasks
    )
    assert all(
        int(task.metadata["llm context budget bytes"]) > 0
        for task in tasks
    )
    assert [task.task_id for task in operational] == ["KITA-048"]
    assert operational[0].metadata["retry repair source"] == "KITA-030"
    assert all(
        not output.startswith("data/agent_supervisor/")
        for output in operational[0].outputs
    )


def test_operational_failure_paths_remain_diagnostic_read_only() -> None:
    tasks, operational = _task_populations()
    canonical_by_id = {task.task_id: task for task in tasks}
    failure_path = (
        "ipfs_kit_py/ipfs_kit_py/mcp_server/tests_e2e_interop.py"
    )
    metadata = {
        **operational[0].metadata,
        "validation failure paths": failure_path,
        "validation failure path authority": "diagnostic-read-only",
    }
    repair = replace(operational[0], metadata=metadata)

    errors: list[str] = []
    assert _validate_operational_repair_tasks(
        [repair],
        canonical_by_id=canonical_by_id,
        errors=errors,
    ) == ("KITA-048",)
    assert errors == []

    overbroad = replace(
        repair,
        outputs=[*repair.outputs, failure_path],
    )
    errors = []
    _validate_operational_repair_tasks(
        [overbroad],
        canonical_by_id=canonical_by_id,
        errors=errors,
    )
    assert any(
        "validation failure paths do not grant write authority" in error
        for error in errors
    )


def test_every_task_has_a_production_admissible_context_budget(tmp_path) -> None:
    tasks, _operational = _task_populations()
    daemon = PortalImplementationDaemon(
        todo_path=TODO_PATH,
        state_path=tmp_path / "task_state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=REPO_ROOT,
        task_header_prefix="## KITA-",
        implement=False,
    )

    assert [
        daemon._task_llm_context_budget_bytes(task) for task in tasks
    ] == [
        int(task.metadata["llm context budget bytes"]) for task in tasks
    ]


def test_taskboard_seal_excludes_only_monotonic_status_progress() -> None:
    text = TODO_PATH.read_text(encoding="utf-8")

    assert (
        _taskboard_definition_sha256(text)
        == SEALED_TASKBOARD_DEFINITION_SHA256
    )

    launch_projection = _with_task_status(text, "KITA-001", "todo")
    progressed = _with_task_status(
        launch_projection, "KITA-001", "completed"
    )
    assert launch_projection != progressed
    assert (
        _taskboard_definition_sha256(launch_projection)
        == SEALED_TASKBOARD_DEFINITION_SHA256
    )
    assert (
        _taskboard_definition_sha256(progressed)
        == SEALED_TASKBOARD_DEFINITION_SHA256
    )

    tampered = text.replace(
        "- Track: foundations-inventory",
        "- Track: foundations-inventory-tampered",
        1,
    )
    assert tampered != text
    assert (
        _taskboard_definition_sha256(tampered)
        != SEALED_TASKBOARD_DEFINITION_SHA256
    )


def test_scheduler_seals_launch_projection_and_validator_reports_progress() -> None:
    scheduler = json.loads(SCHEDULER_PATH.read_text(encoding="utf-8"))
    tasks, operational = _task_populations()
    completed = {
        task.task_id for task in tasks if task.status == "completed"
    }
    blocked = {
        task.task_id for task in tasks if task.status == "blocked"
    }
    ready = [
        task.task_id
        for task in tasks
        if task.task_id not in completed
        and task.task_id not in blocked
        and all(dependency in completed for dependency in task.depends_on)
    ]

    assert scheduler["initial_projection"]["task_count"] == 48
    assert scheduler["initial_projection"]["completed_task_ids"] == [
        "KITA-000"
    ]
    assert scheduler["initial_projection"]["ready_task_ids"] == [
        "KITA-001",
        "KITA-002",
        "KITA-003",
        "KITA-004",
    ]
    assert scheduler["max_lanes"] == 4
    assert {
        lane["initial_task_ids"][0] for lane in scheduler["lanes"]
    } == {"KITA-001", "KITA-002", "KITA-003", "KITA-004"}

    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "--check-all"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["task_count"] == 48
    assert report["operational_task_count"] == len(operational)
    assert report["operational_task_ids"] == ["KITA-048"]
    assert report["goal_count"] == 12
    assert report["completed_task_ids"] == sorted(completed)
    assert report["ready_task_ids"] == ready
    assert report["waiting_task_count"] == (
        48 - len(completed) - len(ready) - len(blocked)
    )
    assert report["blocked_task_ids"] == sorted(blocked)
    assert (
        report["taskboard_definition_sha256"]
        == SEALED_TASKBOARD_DEFINITION_SHA256
    )
