"""Native reconciliation cannot rewrite retained unknown-callback workspaces."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import workspace_quarantine as q
from ipfs_accelerate_py.agent_supervisor.merge.quarantine_validation import (
    QuarantineDenied,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor
from test.api.test_agent_supervisor_reconciliation_auto_unblock import _git, _supervisor
from test.api.test_workspace_root_quarantine import seed


def rescue(supervisor, workspace):
    return supervisor._rescue_dirty_worktree(
        workspace,
        branch=_git(workspace, "branch", "--show-current"),
        head=_git(workspace, "rev-parse", "HEAD"),
        target_ref="HEAD",
        status_lines=_git(workspace, "status", "--porcelain").splitlines(),
        reason="retained callback custody",
    )


def git_preimage(workspace):
    return (
        _git(workspace, "branch", "--show-current"),
        _git(workspace, "rev-parse", "HEAD"),
        _git(workspace, "status", "--porcelain"),
        (workspace / "README").read_bytes(),
    )


def test_frozen_native_rescue_keeps_branch_index_content_and_lifecycle(tmp_path):
    repo, root, _, lease, lifecycle, record = seed(tmp_path)
    supervisor = _supervisor(repo, worktree_root=root)
    (lease.path / "README").write_bytes(b"unresolved dirty callback output\n")
    before_git = git_preimage(lease.path)
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        rescue(supervisor, lease.path)
    assert git_preimage(lease.path) == before_git
    assert lifecycle.load_workspace(lease.path) == record
    assert q.verify(repo, root) == frozen


def test_fresh_native_rescue_keeps_existing_git_behavior(tmp_path):
    repo, root, pool, lease, _, _ = seed(tmp_path)
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    fresh_root = Path(frozen["fresh_root"])
    fresh_pool = type(pool)(repo_root=repo, worktree_root=fresh_root)
    fresh = fresh_pool.acquire(cache_key="fresh", branch_name="implementation/fresh")
    supervisor = _supervisor(repo, worktree_root=fresh_root)
    (fresh.path / "README").write_bytes(b"independent dirty output\n")
    old_head = _git(fresh.path, "rev-parse", "HEAD")
    result = rescue(supervisor, fresh.path)
    assert result["preserved"] is True
    assert result["reason"] == "dirty_worktree_committed_to_rescue_branch"
    assert _git(fresh.path, "rev-parse", "HEAD") != old_head
    assert _git(fresh.path, "status", "--porcelain") == ""
    assert (fresh.path / "README").read_bytes() == b"independent dirty output\n"
    assert q.verify(repo, root) == frozen
    assert lease.path.is_dir()


def test_native_backlog_reconciliation_defers_before_scanning_frozen_git(tmp_path):
    repo, root, _, _, _, _ = seed(tmp_path)
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    # A distinct configured sibling still shares repository-wide maintenance.
    owner = SimpleNamespace(config=SimpleNamespace(repo_root=repo))
    result = implementation_supervisor.PortalImplementationSupervisor.reconcile_backlogged_worktrees(owner)
    assert result == {
        "attempted": False, "skipped": True, "reason": "retained_workspace_scope",
    }
    assert q.verify(repo, root) == frozen


def test_native_rescue_holds_shared_custody_through_actual_git_effect(tmp_path, monkeypatch):
    repo, root, _, lease, _, _ = seed(tmp_path)
    supervisor = _supervisor(repo, worktree_root=root)
    (lease.path / "README").write_bytes(b"dirty callback output\n")
    before = q.census(repo, root)
    entered, release, frozen = threading.Event(), threading.Event(), threading.Event()
    results, freezes = [], []
    original = implementation_supervisor.subprocess.run

    def paused_run(command, *args, **kwargs):
        if list(command[:3]) == ["git", "checkout", "-B"]:
            entered.set()
            assert release.wait(5)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(implementation_supervisor.subprocess, "run", paused_run)

    def writer():
        try:
            results.append(rescue(supervisor, lease.path))
        except BaseException as error:
            results.append(error)

    def freezer():
        try:
            freezes.append(q.freeze(repo, root, expected=before))
        except BaseException as error:
            freezes.append(error)
        finally:
            frozen.set()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    assert entered.wait(5)
    freeze_thread = threading.Thread(target=freezer)
    freeze_thread.start()
    try:
        assert not frozen.wait(0.05)
    finally:
        release.set()
        writer_thread.join(5)
        freeze_thread.join(5)
    assert not writer_thread.is_alive() and not freeze_thread.is_alive()
    assert len(results) == 1 and isinstance(results[0], dict), results
    assert results[0]["preserved"] is True
    assert len(freezes) == 1 and isinstance(freezes[0], dict), freezes
    assert q.verify(repo, root) == freezes[0]
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        rescue(supervisor, lease.path)


def test_nested_supervisor_rescue_preserves_parent_frozen_workspace(tmp_path):
    from test.api.test_workspace_root_quarantine import nested_repository
    from test.api.test_agent_supervisor_reconciliation_auto_unblock import _supervisor

    repo, root, module = nested_repository(tmp_path)
    workspace = root / "nested-native-workspace"
    _git(module, "worktree", "add", "-b", "attempt/nested-retained", str(workspace), "HEAD")
    supervisor = _supervisor(module, worktree_root=root)
    (workspace / "README").write_bytes(b"retained unknown nested callback output\n")

    def preimage():
        return (
            _git(workspace, "branch", "--show-current"),
            _git(workspace, "rev-parse", "HEAD"),
            _git(workspace, "status", "--porcelain"),
            (workspace / "README").read_bytes(),
        )

    branch, head, status, _ = before = preimage()
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        supervisor._rescue_dirty_worktree(
            workspace, branch=branch, head=head, target_ref="HEAD",
            status_lines=status.splitlines(), reason="retained callback custody",
        )
    assert preimage() == before
    assert q.verify(repo, root) == frozen


@pytest.fixture
def completed_workspace(tmp_path, monkeypatch):
    from test.api.test_agent_supervisor_callback_worktree_retention import completed_workspace as native_fixture
    return native_fixture.__wrapped__(tmp_path, monkeypatch)


def test_native_completed_rescue_prune_respects_frozen_workspace(completed_workspace):
    case = completed_workspace
    head = _git(case.workspace, "rev-parse", "HEAD")
    before = _git(case.workspace, "status", "--porcelain")
    frozen = q.freeze(case.repo, case.workspace.parent,
                      expected=q.census(case.repo, case.workspace.parent))
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        case.supervisor._prune_completed_leftover_worktree(
            case.workspace, case.branch, expected_head=head,
        )
    assert case.workspace.is_dir()
    assert _git(case.workspace, "rev-parse", "HEAD") == head
    assert _git(case.workspace, "status", "--porcelain") == before
    assert q.verify(case.repo, case.workspace.parent) == frozen
