"""Contract tests for the DuckDB + Quack migration bootstrap board."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    materialize_task_dependency_dag,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = (
    REPO_ROOT
    / "scripts/validate_agent_supervisor_duckdb_quack_control_plane_board.py"
)
TODO = (
    REPO_ROOT
    / "docs/architecture/agent_supervisor_duckdb_quack_control_plane.todo.md"
)
OBJECTIVES = (
    REPO_ROOT
    / "docs/architecture/agent_supervisor_duckdb_quack_control_plane.objectives.md"
)


def _validator_module():
    spec = importlib.util.spec_from_file_location("dqp_board_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_reports_complete_valid_program() -> None:
    report = _validator_module().validate_program()
    assert report["valid"] is True, report["errors"]
    assert report["task_count"] == 40
    assert report["goal_count"] == 10
    assert report["initial_ready_task_ids"] == [
        "DQP-001",
        "DQP-002",
        "DQP-003",
        "DQP-004",
        "DQP-009",
    ]
    assert set(report["initial_ready_by_shard"]) == {"0", "1", "2", "3"}
    assert all(report["initial_ready_by_shard"].values())
    assert report["terminal_task_id"] == "DQP-039"


def test_validator_cli_emits_pure_json() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--check-all"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert completed.stderr == ""


def test_production_parsers_preserve_population_and_acyclicity() -> None:
    tasks = parse_task_text(
        TODO.read_text(encoding="utf-8"),
        path=TODO,
        task_header_prefix="## DQP-",
    )
    goals = parse_goal_heap(OBJECTIVES.read_text(encoding="utf-8"))
    graph = materialize_task_dependency_dag(tasks)
    assert [task.task_id for task in tasks] == [
        f"DQP-{index:03d}" for index in range(40)
    ]
    assert [goal.goal_id for goal in goals] == [
        "DQP-G000",
        "DQP-G010",
        "DQP-G020",
        "DQP-G030",
        "DQP-G040",
        "DQP-G050",
        "DQP-G060",
        "DQP-G070",
        "DQP-G080",
        "DQP-G090",
    ]
    assert graph.invalid_task_cids == []
    assert len(graph.nodes) == 40
    assert graph.edges
