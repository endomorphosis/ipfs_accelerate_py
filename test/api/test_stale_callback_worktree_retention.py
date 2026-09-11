"""Age and a missing workspace cannot close a peer callback's effects."""
from pathlib import Path
import shutil
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import ProcessBirthIdentity
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import TodoImplementationDaemon


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.mark.parametrize("condition", ["dirty_unknown_callback", "missing_source", "terminal_dead_owner"])
def test_stale_peer_callback_is_retained_for_canonical_reconciliation(tmp_path, monkeypatch, condition):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    (repo / "baseline").write_text("baseline\n")
    git(repo, "add", "baseline")
    git(repo, "commit", "-m", "baseline")
    branch = "implementation/pctdd-006-retained-callback"
    workspace = repo / "worktrees" / "peer"
    git(repo, "worktree", "add", "-b", branch, str(workspace), "main")
    (workspace / "committed-callback").write_text("unaccepted callback\n")
    git(workspace, "add", "committed-callback")
    git(workspace, "commit", "-m", "preserve callback history")
    head = git(repo, "rev-parse", branch)
    dirty = workspace / "uncommitted-proof"
    dirty.write_text("unvalidated result\n")
    peer_state = tmp_path / "peer-state"
    peer_state.mkdir()
    snapshot = peer_state / "implementation-protected-path-active.json"
    snapshot.write_text('{"callback_outcome":"unknown"}\n')
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md", state_path=tmp_path / "own" / "state.json",
        strategy_path=tmp_path / "own" / "strategy.json",
        events_path=tmp_path / "own" / "events.jsonl",
        repo_root=repo, worktree_root=workspace.parent, worktree_pool_enabled=False,
    )
    record = None
    if condition == "terminal_dead_owner":
        record = daemon.worktree_lifecycle.begin_preparing(
            task_id="PCTDD-006", canonical_task_cid="task:unknown", attempt=1,
            lane_id="peer", state_dir=str(peer_state), workspace_path=workspace,
            branch=branch, merge_target="main",
            owner=ProcessBirthIdentity(pid=2**30, start_time_ticks=1, boot_id="dead"),
        )
        record = daemon.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
            workspace, expected_state_dir=record.state_dir,
        )
        assert record.is_terminal
    if condition == "missing_source":
        shutil.rmtree(workspace)  # Disposable fixture; ref and peer state survive.
    monkeypatch.setattr(daemon, "_list_process_commands", lambda: [])
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.time.time",
        lambda: 4_000_000_000.0,
    )
    def forbidden(*args, **kwargs):
        pytest.fail("stale peer cleanup attempted mutation without canonical completion proof")
    monkeypatch.setattr(daemon, "_cleanup_merged_worktree", forbidden)
    result = daemon._cleanup_stale_worktrees(max_age_seconds=1)
    assert result["removed_count"] == 0
    assert result["delegated_count"] == 1
    assert git(repo, "rev-parse", branch) == head
    assert snapshot.read_text() == '{"callback_outcome":"unknown"}\n'
    if condition != "missing_source":
        assert dirty.read_text() == "unvalidated result\n"
    if record is not None:
        assert daemon.worktree_lifecycle.load_workspace(workspace) == record
