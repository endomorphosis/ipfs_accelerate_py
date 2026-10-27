"""Physical Git checkout failures retain their original SDK diagnostic.

Private synthetic repositories only. An actual post-checkout hook fails after
Git has created the task branch; no filesystem/SDK result is manufactured.
"""
from pathlib import Path
import subprocess
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon


def _git(repo, *args):
    p = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)
    return p.stdout.strip()


def _init(repo):
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Synthetic fixture")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    (repo / "source.txt").write_text("synthetic immutable source\n")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "synthetic fixture")


def _world(tmp_path, *, missing=False):
    parent, origin = tmp_path / "parent", tmp_path / "origin"
    _init(parent); _init(origin)
    _git(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(origin), "deps/child")
    source = parent / "deps/child"
    pinned = _git(source, "rev-parse", "HEAD")
    if missing:
        pinned = "f" * 40
        _git(parent, "update-index", "--cacheinfo", "160000," + pinned + ",deps/child")
    _git(parent, "commit", "-m", "synthetic pinned child")
    worktree = tmp_path / "worker"
    _git(parent, "worktree", "add", "-b", "implementation/synthetic", str(worktree), "main")
    daemon = PortalImplementationDaemon(todo_path=parent / "todo.md",
        state_path=tmp_path / "state/task_state.json", strategy_path=tmp_path / "state/strategy.json",
        events_path=tmp_path / "state/events.jsonl", repo_root=parent,
        worktree_root=tmp_path / "workers", worktree_submodule_paths=["deps/child"])
    return daemon, source, worktree, pinned


def test_actual_failed_checkout_preserves_original_error_and_created_branch(tmp_path):
    daemon, source, worker, pinned = _world(tmp_path)
    hooks = tmp_path / "hooks"; hooks.mkdir()
    invoked = tmp_path / "actual-hook-invoked"
    hook = hooks / "post-checkout"
    hook.write_text("#!/bin/sh\nprintf invoked > '" + str(invoked) + "'\nprintf synthetic-original-checkout-failure >&2\nexit 73\n")
    hook.chmod(0o700)
    _git(source, "config", "core.hooksPath", str(hooks))
    with pytest.raises(RuntimeError) as refused:
        daemon._create_local_submodule_worktree(worker, "deps/child", branch_name="implementation/synthetic")
    # Git really created the branch and checkout before the physical hook failed.
    branch = daemon._submodule_worktree_branch_name("implementation/synthetic", "deps/child")
    assert invoked.read_text() == "invoked"
    assert _git(source, "rev-parse", branch) == pinned
    assert _git(worker / "deps/child", "rev-parse", "HEAD") == pinned
    assert "synthetic-original-checkout-failure" in str(refused.value)
    assert "already exists" not in str(refused.value)


@pytest.mark.parametrize("existing_branch", [False, True])
def test_actual_normal_and_existing_branch_checkout_keep_exact_pin(tmp_path, existing_branch):
    daemon, source, worker, pinned = _world(tmp_path)
    branch = daemon._submodule_worktree_branch_name("implementation/synthetic", "deps/child")
    if existing_branch:
        _git(source, "branch", branch, pinned)
    assert daemon._create_local_submodule_worktree(worker, "deps/child", branch_name="implementation/synthetic") is True
    assert _git(worker / "deps/child", "rev-parse", "HEAD") == pinned
    assert _git(worker / "deps/child", "branch", "--show-current") == branch


def test_actual_missing_recorded_pin_is_not_replaced_offline(tmp_path):
    daemon, source, worker, _pinned = _world(tmp_path, missing=True)
    with pytest.raises(RuntimeError, match="offline submodule gitlink unavailable"):
        daemon._create_local_submodule_worktree(worker, "deps/child", branch_name="implementation/synthetic",
            offline_local_only=True)
    branch = daemon._submodule_worktree_branch_name("implementation/synthetic", "deps/child")
    p = subprocess.run(["git", "show-ref", "--verify", "refs/heads/" + branch], cwd=source,
        text=True, capture_output=True, check=False)
    assert p.returncode != 0



def test_actual_missing_registered_checkout_reuses_branch_and_preserves_orphan(tmp_path):
    daemon, source, worker, pinned = _world(tmp_path)
    branch = "implementation/synthetic"
    assert daemon._create_local_submodule_worktree(worker, "deps/child", branch_name=branch) is True
    target = worker / "deps/child"
    orphan = worker / "deps/preserved-orphan"
    target.rename(orphan)
    original = (orphan / "source.txt").read_bytes()
    assert str(target) in _git(source, "worktree", "list", "--porcelain")
    assert daemon._create_local_submodule_worktree(worker, "deps/child", branch_name=branch) is True
    assert _git(target, "rev-parse", "HEAD") == pinned
    assert (orphan / "source.txt").read_bytes() == original
    assert _git(source, "worktree", "list", "--porcelain").count("worktree " + str(target)) == 1
