"""Interrupted validation snapshots skip a second Grok provider call."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.interrupted_validation_checkpoint import (
    restore_interrupted_validation,
    snapshot_implementation_workspace,
    task_id_from_workspace,
    workspace_is_fresh,
)


def _git(workspace: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)


def _seed_worktree(root: Path, *, branch: str) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "keep.txt").write_text("keep\n", encoding="utf-8")
    _git(repo, "add", "keep.txt")
    _git(repo, "commit", "-m", "seed")
    worktrees = repo / "data" / "aseh" / "worktrees"
    worktrees.mkdir(parents=True)
    workspace = worktrees / "workspace_one"
    _git(repo, "branch", branch)
    _git(repo, "worktree", "add", str(workspace), branch)
    return workspace


def test_snapshot_and_restore_skips_fresh_worktree(tmp_path: Path) -> None:
    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    first = _seed_worktree(tmp_path / "one", branch=branch)
    repo = first.parent.parent.parent.parent
    target = first / "ipfs_accelerate_py" / "agent_supervisor" / "runtime"
    target.mkdir(parents=True)
    (target / "quack_state_server.py").write_text("changed\n", encoding="utf-8")
    (first / "docs").mkdir()
    (first / "docs" / "note.md").write_text("migration\n", encoding="utf-8")

    assert task_id_from_workspace(first) == "ASEH-061"
    payload = snapshot_implementation_workspace(first)
    assert payload is not None
    assert payload["task_id"] == "ASEH-061"
    assert payload["file_count"] >= 2

    second = repo / "data" / "aseh" / "worktrees" / "workspace_two"
    _git(repo, "worktree", "add", "--detach", str(second), "HEAD")
    _git(second, "checkout", "-B", branch)
    assert workspace_is_fresh(second)
    assert restore_interrupted_validation(second) is True
    assert (
        (second / "ipfs_accelerate_py" / "agent_supervisor" / "runtime" / "quack_state_server.py")
        .read_text(encoding="utf-8")
        == "changed\n"
    )
    assert (second / "docs" / "note.md").read_text(encoding="utf-8") == "migration\n"


def test_restore_does_not_clobber_dirty_worktree(tmp_path: Path) -> None:
    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    first = _seed_worktree(tmp_path / "one", branch=branch)
    (first / "a.py").write_text("first\n", encoding="utf-8")
    snapshot_implementation_workspace(first)

    second = _seed_worktree(tmp_path / "two", branch=branch)
    (second / "a.py").write_text("live\n", encoding="utf-8")
    assert restore_interrupted_validation(second) is False
    assert (second / "a.py").read_text(encoding="utf-8") == "live\n"
