"""Supervisor consumption tests for the SCH configured board."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPO_ROOT / "config/agent_supervisor_semantic_compression_harness_scheduler.json"
)
TODO_PATH = REPO_ROOT / "docs/architecture/semantic_compression_harness.todo.md"
VALIDATOR = REPO_ROOT / "scripts/validate_semantic_compression_harness_board.py"
OPS_ENTRY = REPO_ROOT / "scripts/ops/agent_supervisor/configured_board_scheduler.py"
SCH_LAUNCHER = (
    REPO_ROOT / "scripts/ops/agent_supervisor/semantic_compression_harness_scheduler.py"
)


def test_supervisor_parses_completed_sch_taskboard() -> None:
    tasks = parse_task_file(TODO_PATH, task_header_prefix="## SCH-")
    assert [task.task_id for task in tasks] == [f"SCH-{index:03d}" for index in range(19)]
    assert {task.status for task in tasks} == {"completed"}
    assert {task.board_namespace for task in tasks} == {"semantic-compression-harness-v1"}
    assert tasks[-1].task_id == "SCH-018"


def test_scheduler_document_is_configured_board_schema() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "ipfs_accelerate_py.agent_supervisor.semantic_compression_harness.scheduler_config@1"
    )
    assert payload["task_prefix"] == "SCH-"
    assert payload["board_namespace"] == "semantic-compression-harness-v1"
    assert payload["taskboard_path"] == "docs/architecture/semantic_compression_harness.todo.md"
    assert payload["validator_path"] == "scripts/validate_semantic_compression_harness_board.py"
    assert payload["max_lanes"] == 3
    assert payload["strict_task_sharding"] is True
    assert payload["merge_target_branch"] == "main"
    assert payload["provider"]["primary_provider_id"] == "grok_cli"
    assert payload["initial_projection"]["terminal_task_id"] == "SCH-018"
    assert payload["initial_projection"]["completed_task_ids"] == [
        f"SCH-{index:03d}" for index in range(19)
    ]
    assert (
        "config/agent_supervisor_semantic_compression_harness_scheduler.json"
        in payload["protected_paths"]
    )
    assert OPS_ENTRY.is_file()
    assert SCH_LAUNCHER.is_file()
    assert payload["source_binding"]["ipfs_kit_planning_revision"] == (
        "df2f9cc092456329de9724c45a50c54b410875d1"
    )
    assert payload["source_binding"]["accelerator_required_ancestor"] == (
        "271e331af802f37d759c000666282631a99f7aab"
    )


def test_board_validator_accepts_completed_supervisor_board() -> None:
    completed = subprocess.run(
        ["python3.12", str(VALIDATOR), "--check-all"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert payload["tasks"] == 19


def test_sch_supervisor_launcher_preflight_and_dry_run() -> None:
    preflight = subprocess.run(
        ["python3.12", str(SCH_LAUNCHER), "preflight"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert preflight.returncode == 0, preflight.stderr
    admitted = json.loads(preflight.stdout)
    assert admitted["valid"] is True
    assert admitted["terminal_task_id"] == "SCH-018"
    dry = subprocess.run(
        ["python3.12", str(SCH_LAUNCHER), "launch", "--dry-run", "--implement"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert dry.returncode == 0, dry.stderr
    plan = json.loads(dry.stdout)
    assert "implementation_supervisor_entry.py" in " ".join(plan["argv"])
    assert "--task-prefix" in plan["argv"]
    assert "## SCH-" in plan["argv"]
