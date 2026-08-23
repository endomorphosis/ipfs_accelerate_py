from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("fence_kind", "owner_source"),
    (
        ("task_state", "task_state"),
        ("active_snapshot", "protected_path_snapshot"),
    ),
)
def test_cleanup_preserves_nested_database_portal_attempt_worktree(
    tmp_path: Path,
    fence_kind: str,
    owner_source: str,
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

    branch = f"implementation/nested-portal-{fence_kind}"
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / fence_kind
    _git(repo, "branch", branch)
    _git(repo, "worktree", "add", str(worktree_path), branch)

    state_root = repo / "state"
    own_state_dir = state_root / "lane-0"
    attempt_dir = (
        state_root
        / "lane-1"
        / "lane_1_database_portal_attempts"
        / "attempt-identity"
    )
    own_state_dir.mkdir(parents=True)
    attempt_dir.mkdir(parents=True)
    if fence_kind == "task_state":
        fence_path = attempt_dir / "portal-task-state.json"
        PortalTaskState(
            active_task_id="PCAR-001",
            active_attempt=1,
            active_phase="validating",
            active_worktree_path=str(worktree_path),
            active_branch=branch,
            implementation_in_progress=True,
        ).save(fence_path)
    else:
        fence_path = (
            attempt_dir / IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME
        )
        fence_path.write_text(
            json.dumps(
                {
                    "schema": "implementation-protected-path-active-v1",
                    "task_id": "PCAR-001",
                    "attempt": 1,
                    "workspace_path": str(worktree_path),
                    "protected_paths": ["README.md"],
                }
            ),
            encoding="utf-8",
        )

    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=own_state_dir / "lane_0_task_state.json",
            strategy_path=own_state_dir / "strategy.json",
            events_path=own_state_dir / "events.jsonl",
            state_dir=own_state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
            merge_target_branch="main",
        )
    )
    supervisor._list_process_commands = lambda: []  # type: ignore[method-assign]

    result = supervisor.cleanup_backlogged_worktrees()

    assert result["removed_count"] == 0
    skip = next(
        item
        for item in result["skipped"]
        if item["reason"] == "active_peer_state_worktree"
    )
    assert skip["owner_source"] == owner_source
    assert skip[
        "owner_state_path"
        if fence_kind == "task_state"
        else "owner_snapshot_path"
    ] == str(fence_path)
    assert skip["owner_task_id"] == "PCAR-001"
    assert worktree_path.exists()
    assert _git(worktree_path, "branch", "--show-current") == branch
