"""Supervisor-ingestion tests for the MCP++ 1.0 gap-closure board."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TODO_PATH = REPO_ROOT / "docs/architecture/mcplusplus_1_0_gap_closure.todo.md"
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/mcplusplus_1_0_gap_closure.objectives.md"
)
SCHEDULER_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json"
)
VALIDATOR_PATH = (
    REPO_ROOT / "scripts/validate_mcplusplus_1_0_gap_closure_board.py"
)
PLAN_PATH = REPO_ROOT / "docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md"


def test_production_parsers_consume_exact_task_and_goal_populations() -> None:
    parsed = parse_task_file(TODO_PATH, task_header_prefix="## MCPP-")
    tasks = [
        task
        for task in parsed
        if task.metadata.get("canonical board task") != "false"
    ]
    goals = parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))

    assert [task.task_id for task in tasks] == [
        f"MCPP-{index:03d}" for index in range(84)
    ]
    assert [goal.goal_id for goal in goals] == [
        "MCPP-G000",
        "MCPP-G010",
        "MCPP-G020",
        "MCPP-G030",
        "MCPP-G040",
        "MCPP-G050",
        "MCPP-G060",
        "MCPP-G070",
        "MCPP-G080",
        "MCPP-G090",
        "MCPP-G100",
        "MCPP-G110",
        "MCPP-G120",
        "MCPP-G130",
        "MCPP-G140",
        "MCPP-G150",
        "MCPP-G160",
        "MCPP-G170",
    ]
    assert tasks[0].status == "completed"
    assert tasks[1].status == "todo"
    assert tasks[1].metadata.get("depends on") == "MCPP-000"


def test_scheduler_loads_and_protects_control_artifacts() -> None:
    board = load_configured_board(SCHEDULER_PATH, repo_root=REPO_ROOT)
    assert board.board_namespace == "mcplusplus-1-0-gap-closure-v1"
    assert board.max_lanes == 6
    assert board.merge_target_branch == "codex/mcplusplus-1.0-gap-closure"
    protected = set(board.protected_paths)
    assert "config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json" in protected
    assert "docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md" in protected
    assert PLAN_PATH.is_file()


def test_validator_check_all_is_green() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "--check-all"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert payload["tasks"] == 84
    assert payload["ready"] == ["MCPP-001"]
