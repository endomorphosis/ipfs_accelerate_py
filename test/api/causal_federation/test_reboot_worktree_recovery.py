from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def test_cleanup_skips_locked_registered_worktree_missing_after_reboot(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")

    branch_name = "implementation/accel-002-attempt-1-interrupted"
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "accel-002-attempt-1-interrupted"
    _git(repo, "worktree", "add", "-b", branch_name, str(worktree_path), "HEAD")
    _git(repo, "worktree", "lock", "--reason", "initializing", str(worktree_path))
    shutil.rmtree(worktree_path)

    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        worktree_root=worktree_root,
        merged_worktree_cleanup_max=5,
    )

    result = daemon._cleanup_already_merged_worktrees()

    assert result["removed_count"] == 0
    assert result["skipped"] == [
        {
            "worktree_path": str(worktree_path),
            "branch": branch_name,
            "reason": "worktree_missing",
        }
    ]
    assert str(worktree_path) in _git(repo, "worktree", "list", "--porcelain")
