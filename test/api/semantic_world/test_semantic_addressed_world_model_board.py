"""Focused tests for the operator-owned SAWM R2 controls.

These tests inspect and render controls only. They do not open the live
authority database, start Quack, probe a provider, or launch a supervisor.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(relative: str, name: str) -> ModuleType:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_static_board_gate_is_valid() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_validator_test",
    )
    report = validator.validate_program(REPO_ROOT)
    assert report["valid"] is True, report["errors"]
    assert report["task_count"] == 45
    assert report["goal_count"] == 29
    assert report["markdown_completion_is_authority"] is False


def test_rendered_population_has_exact_operator_frontier() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_test",
    )
    population = materializer.build_population(REPO_ROOT)
    tasks = population["taskboard"]
    goals = population["objectives"]
    assert len(tasks) == 45
    assert len(goals) == 29
    assert [task["task_id"] for task in tasks] == [
        f"SAWM-{index:03d}" for index in range(45)
    ]
    assert tasks[0]["status"] == "ready"
    assert all(task["status"] == "todo" for task in tasks[1:])
    assert tasks[1]["depends_on"] == [tasks[0]["task_cid"]]
    assert all(str(goal["goal_cid"]).startswith("sha256:") for goal in goals)
    assert population["plan_revision"] == "SAWM-PLAN-R2"


def test_scheduler_keeps_ducklake_non_authoritative() -> None:
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    assert config["max_lanes"] == 1
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["failover_policy"] == "fail_closed"
    ducklake = config["ducklake_history_projection"]
    assert ducklake["load_local_only"] is True
    assert ducklake["install_or_network_forbidden"] is True
    assert ducklake["authority"] is False
    assert ducklake["scheduling_prerequisite"] is False
    assert ducklake["completion_prerequisite"] is False
