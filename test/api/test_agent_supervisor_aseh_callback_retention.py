"""The older runtime must preserve peer callback source until native settlement."""
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)
from test.api.test_agent_supervisor_reconciliation_auto_unblock import (
    _database_program, _git, _init_repo, _supervisor,
)


@pytest.mark.parametrize("missing_workspace", [False, True])
def test_database_cleanup_retains_merged_callback_and_registration(tmp_path, missing_workspace):
    repo = _init_repo(tmp_path / "repo")
    (repo / "README.md").write_text("baseline\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "baseline")
    workspace = repo / "worktrees" / "callback"
    branch = "implementation/aseh-061-unknown-callback"
    _git(repo, "worktree", "add", "-b", branch, str(workspace), "main")
    if missing_workspace:
        workspace.rename(repo / "retained-callback")
    supervisor = _supervisor(
        repo, worktree_root=workspace.parent, database_program=_database_program(),
    )
    result = supervisor.cleanup_backlogged_worktrees()
    assert result["reason"] == "canonical_cleanup_requires_guarded_runtime"
    assert result["removed_count"] == 0
    assert str(workspace) in _git(repo, "worktree", "list", "--porcelain")
    assert _git(repo, "rev-parse", branch) == _git(repo, "rev-parse", "main")
    assert (repo / "retained-callback" if missing_workspace else workspace).is_dir()


def test_task_projection_cannot_enter_peer_cleanup_mutation():
    peer = SimpleNamespace(isolate_merge_queue_to_task_projection=True)
    result = PortalImplementationDaemon._cleanup_already_merged_worktrees(peer)
    assert result == {
        "attempted": False, "removed_count": 0,
        "reason": "task_projection_has_no_peer_cleanup_authority",
    }
