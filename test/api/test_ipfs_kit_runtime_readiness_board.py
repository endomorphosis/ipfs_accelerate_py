"""Supervisor-ingestion tests for the IPFS Kit runtime-readiness board."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    parse_task_file,
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


def test_production_parsers_consume_exact_task_and_goal_populations() -> None:
    tasks = parse_task_file(TODO_PATH, task_header_prefix="## KITA-")
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
    assert tasks[0].status == "completed"
    assert all(task.status == "todo" for task in tasks[1:])
    assert all(
        task.board_namespace == "ipfs-kit-runtime-readiness-v1"
        for task in tasks
    )
    assert all(
        int(task.metadata["llm context budget bytes"]) > 0
        for task in tasks
    )


def test_every_task_has_a_production_admissible_context_budget(tmp_path) -> None:
    tasks = parse_task_file(TODO_PATH, task_header_prefix="## KITA-")
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


def test_scheduler_and_validator_agree_on_initial_projection() -> None:
    scheduler = json.loads(SCHEDULER_PATH.read_text(encoding="utf-8"))
    completed = {"KITA-000"}
    tasks = parse_task_file(TODO_PATH, task_header_prefix="## KITA-")
    ready = [
        task.task_id
        for task in tasks
        if task.status == "todo"
        and all(dependency in completed for dependency in task.depends_on)
    ]

    assert scheduler["initial_projection"]["task_count"] == 48
    assert scheduler["initial_projection"]["completed_task_ids"] == [
        "KITA-000"
    ]
    assert ready == ["KITA-001", "KITA-002", "KITA-003", "KITA-004"]
    assert scheduler["initial_projection"]["ready_task_ids"] == ready
    assert scheduler["max_lanes"] == 4
    assert {
        lane["initial_task_ids"][0] for lane in scheduler["lanes"]
    } == set(ready)

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
    assert report["goal_count"] == 12
    assert report["completed_task_ids"] == ["KITA-000"]
    assert report["ready_task_ids"] == ready
    assert report["waiting_task_count"] == 43
    assert report["blocked_task_ids"] == []
