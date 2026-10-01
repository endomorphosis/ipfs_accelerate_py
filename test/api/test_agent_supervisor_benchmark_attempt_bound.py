"""Configured runners must not silently discard a finite attempt budget."""
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_args
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)


def test_legacy_configured_runner_preserves_attempt_bound(tmp_path: Path):
    board = tmp_path / "tasks.md"
    board.write_text("# Tasks\n")
    args = parse_args(["--todo-path", str(board), "--state-dir", str(tmp_path / "state"),
                       "--task-source-kind", "legacy-markdown", "--explicit-legacy-task-source",
                       "--max-task-attempts", "2", "--once"])
    daemon, _ = build_portal_implementation_daemon_from_args(args, repo_root=tmp_path)
    assert daemon.max_task_attempts == 2
