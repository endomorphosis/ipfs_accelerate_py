"""Contract tests for the generated EAAEF supervisor board."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"


def _load_validator():
    path = ROOT / "scripts/validate_external_agent_autonomous_execution_fabric_board.py"
    spec = importlib.util.spec_from_file_location("eaaef_validator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _board() -> dict:
    return json.loads((CAMPAIGN / "task_board.json").read_text(encoding="utf-8"))


def test_generated_board_passes_fail_closed_validator() -> None:
    report = _load_validator().validate()
    assert report["valid"] is True, report["errors"]
    assert report["counts"] == {
        "goal_count": 19,
        "task_count": 104,
        "initial_population_count": 10,
        "owned_path_count": 228,
        "dependency_edge_count": 237,
    }


def test_bootstrap_is_the_only_initial_ready_task() -> None:
    board = _board()
    tasks = {task["stable_task_id"]: task for task in board["tasks"]}
    ready = [
        task["stable_task_id"]
        for task in board["tasks"]
        if task["status"] == "todo"
        and task["is_schedulable"]
        and not task["dependencies"]
    ]
    assert ready == ["EAAEF-000"]
    assert tasks["EAAEF-000"]["completion_mode"] == "manual"
    for number in range(1, 6):
        assert "EAAEF-000" in tasks[f"EAAEF-{number:03d}"]["dependencies"]


def test_future_population_is_held_until_plan_r2() -> None:
    for task in _board()["tasks"]:
        if int(task["stable_task_id"].split("-")[-1]) < 10:
            continue
        assert task["status"] == "blocked"
        assert task["is_schedulable"] is False
        assert task["population_state"] == "template_only_awaiting_plan_r2"
        assert task["blocked_reason"] == "awaiting_EAAEF-009_plan_revision"
        assert task["source_semantic_state_root"] == "REBIND_REQUIRED_BY_EAAEF-009"


def test_cross_repository_execution_validation_has_explicit_cwd_and_argv() -> None:
    expected_cwd = {
        "ipfs_accelerate_py": ".",
        "ipfs_datasets_py": "ipfs_datasets_py",
        "ipfs_kit_py": "ipfs_kit_py",
        "Mcp-Plus-Plus": "ipfs_accelerate_py/mcplusplus",
    }
    for task in _board()["tasks"]:
        commands = task["execution_validation"]
        assert commands
        for command in commands:
            assert command["working_directory"] == expected_cwd[task["owning_repository"]]
            assert isinstance(command["argv"], list) and command["argv"]
            assert ";" not in command["argv"]
