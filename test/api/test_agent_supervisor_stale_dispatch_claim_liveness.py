"""Leftover dispatch claims must not stay active after their state_dir is gone."""

from __future__ import annotations

import os
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
    PortalImplementationDaemon,
)


def test_task_claim_inactive_when_state_dir_disposed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    worktree_root = tmp_path / "worktrees"
    worktree_root.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        implementation_log_dir=state_dir / "logs",
        repo_root=repo,
        worktree_root=worktree_root,
        implement=True,
        implementation_command="must-not-run",
    )
    disposed = tmp_path / "disposed-attempt-b04376e4"
    metadata = {
        "kind": IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
        "pid": os.getpid(),
        "repo_root": str(repo.resolve()),
        "state_dir": str(disposed),
        "task_id": "ASEH-062",
    }
    assert daemon._implementation_task_claim_owner_is_active(metadata) is False

    live_state = {
        **metadata,
        "state_dir": str(state_dir.resolve()),
    }
    assert daemon._implementation_task_claim_owner_is_active(live_state) is True
