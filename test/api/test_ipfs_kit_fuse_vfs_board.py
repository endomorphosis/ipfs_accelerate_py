"""Supervisor-ingestion tests for the IPFS Kit kernel-VFS/FUSE board."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import load_configured_board
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_file
from scripts.validate_ipfs_kit_fuse_vfs_board import (
    GOAL_IDS,
    INITIAL_READY,
    INITIAL_SHARDS,
    TASK_IDS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TODO_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_fuse_vfs.todo.md"
OBJECTIVE_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_fuse_vfs.objectives.md"
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json"
VALIDATOR_PATH = REPO_ROOT / "scripts/validate_ipfs_kit_fuse_vfs_board.py"


def test_declared_validator_accepts_the_sealed_projection() -> None:
    result = subprocess.run(
        (sys.executable, str(VALIDATOR_PATH), "--check-all"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["task_count"] == 40
    assert report["goal_count"] == 9
    assert report["ready_task_ids"] == list(INITIAL_READY)
    assert report["initial_shards"] == {str(key): value for key, value in INITIAL_SHARDS.items()}


def test_production_parsers_consume_exact_tasks_goals_and_dependencies() -> None:
    tasks = parse_task_file(TODO_PATH, task_header_prefix="## KVFS-")
    goals = parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))
    assert tuple(task.task_id for task in tasks) == TASK_IDS
    assert tuple(goal.goal_id for goal in goals) == GOAL_IDS
    assert all(task.metadata["goal id"] in GOAL_IDS for task in tasks)
    by_id = {task.task_id: task for task in tasks}
    completed = {task_id for task_id, task in by_id.items() if task.status == "completed"}
    ready = tuple(
        task_id for task_id in TASK_IDS
        if by_id[task_id].status == "todo"
        and all(dependency in completed for dependency in by_id[task_id].depends_on)
    )
    assert ready == INITIAL_READY


def test_native_scheduler_loader_and_strict_shards_match() -> None:
    board = load_configured_board(CONFIG_PATH, repo_root=REPO_ROOT)
    assert board.max_lanes == 4
    assert board.strict_task_sharding is True
    assert board.worktree_submodule_paths == ("ipfs_kit_py",)
    assert board.payload["initial_projection"]["ready_task_ids"] == list(INITIAL_READY)
    for task_id in INITIAL_READY:
        shard = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 4
        assert INITIAL_SHARDS[shard] == task_id
