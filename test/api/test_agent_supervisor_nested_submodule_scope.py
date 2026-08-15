"""Focused regression for complete nested-submodule proposal materialization."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
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


def _seed_parent_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    child_source = tmp_path / "child-source"
    child_source.mkdir()
    _git(child_source, "init")
    _git(child_source, "checkout", "-b", "main")
    _git(child_source, "config", "user.name", "Test User")
    _git(child_source, "config", "user.email", "test@example.invalid")
    (child_source / "base.txt").write_text("base\n", encoding="utf-8")
    _git(child_source, "add", "base.txt")
    _git(child_source, "commit", "-m", "child base")

    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init")
    _git(parent, "checkout", "-b", "main")
    _git(parent, "config", "user.name", "Test User")
    _git(parent, "config", "user.email", "test@example.invalid")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "libs/child",
    )
    _git(parent, "add", ".gitmodules", "libs/child")
    _git(parent, "commit", "-m", "add child submodule")

    child = parent / "libs" / "child"
    _git(child, "checkout", "main")
    _git(child, "config", "user.name", "Test User")
    _git(child, "config", "user.email", "test@example.invalid")
    return parent, child


def _daemon(parent: Path, tmp_path: Path) -> TodoImplementationDaemon:
    state_dir = tmp_path / "state"
    return TodoImplementationDaemon(
        todo_path=parent / "todo.md",
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["libs/child"],
    )


def _task() -> PortalTask:
    return PortalTask(
        task_id="SCOPE-001",
        title="Change one declared child file",
        status="todo",
        completion="manual",
        priority="P0",
        track="scope",
        outputs=["libs/child/allowed.txt"],
        validation=["python -m pytest"],
        acceptance="Only the declared child path may change.",
    )


def test_mixed_declared_and_undeclared_child_files_reject(tmp_path: Path) -> None:
    parent, child = _seed_parent_with_submodule(tmp_path)
    baseline = _git(parent, "rev-parse", "HEAD")
    (child / "allowed.txt").write_text("declared\n", encoding="utf-8")
    # Reproduce VGO-001 exactly: an empty undeclared untracked sibling.
    (child / "undeclared-smoke.py").write_text("", encoding="utf-8")

    result = _daemon(parent, tmp_path)._validate_implementation_patch(
        parent,
        _task(),
        baseline_ref=baseline,
    )

    assert result.accepted is False
    assert result.proposal.changed_paths == (
        "libs/child/allowed.txt",
        "libs/child/undeclared-smoke.py",
    )
    assert "path_outside_scope" in {
        finding.code.value for finding in result.findings
    }
