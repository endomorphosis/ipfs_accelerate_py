"""Read-side index healing cannot replace a newer workspace identity."""
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import OwnershipError
from test.api.test_agent_supervisor_runtime_authority_vectors import _repository
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore


def test_index_healer_rechecks_captured_terminal_before_publication(tmp_path):
    repo = tmp_path / "repo"
    _repository(repo)
    store = WorktreeLifecycleStore(repo)
    workspace = tmp_path / "workspace"
    original = store.begin_preparing(
        task_id="ORIGINAL", canonical_task_cid="cid:original", attempt=1,
        lane_id="first", workspace_path=workspace, branch="first", merge_target="main",
    )
    terminal = store.mark_terminal(workspace, lease_id=original.lease_id, expected_fence=original.fence)
    assert store.compare_and_delete(workspace, lease_id=terminal.lease_id, expected_fence=terminal.fence)
    replacement = store.begin_preparing(
        task_id="ORIGINAL", canonical_task_cid="cid:original", attempt=1,
        lane_id="replacement", workspace_path=workspace, branch="replacement", merge_target="main",
    )
    index_path = store.task_index_path_for(canonical_task_cid="cid:original", task_id="ORIGINAL", attempt=1)
    before = index_path.read_bytes()
    with pytest.raises(OwnershipError, match="publication"):
        store._publish_task_index(terminal)
    assert index_path.read_bytes() == before
    assert store.load_workspace(workspace) == replacement


def test_current_terminal_index_can_still_be_healed(tmp_path):
    repo = tmp_path / "repo"
    _repository(repo)
    store = WorktreeLifecycleStore(repo)
    workspace = tmp_path / "workspace"
    record = store.begin_preparing(
        task_id="ORIGINAL", canonical_task_cid="cid:original", attempt=1,
        lane_id="first", workspace_path=workspace, branch="first", merge_target="main",
    )
    index_path = store.task_index_path_for(canonical_task_cid="cid:original", task_id="ORIGINAL", attempt=1)
    old_index = index_path.read_bytes()
    terminal = store.mark_terminal(workspace, lease_id=record.lease_id, expected_fence=record.fence)
    index_path.write_bytes(old_index)
    assert store.heal_stale_task_indexes() == [terminal]
    assert json.loads(index_path.read_bytes())["state"] == "terminal"
    assert store.load_task_attempt(canonical_task_cid="cid:original", task_id="ORIGINAL", attempt=1) == terminal
