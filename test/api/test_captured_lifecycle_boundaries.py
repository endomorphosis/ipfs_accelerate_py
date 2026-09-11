"""Captured workspace custody must survive stale validation/cleanup attempts."""
import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnershipError, WorktreeLifecycleStore, WorkspaceLifecycleState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import TodoImplementationDaemon


@pytest.fixture
def custody(tmp_path):
    daemon = TodoImplementationDaemon.__new__(TodoImplementationDaemon)
    store = WorktreeLifecycleStore(tmp_path, store_dir=tmp_path / 'lifecycle')
    record = store.begin_preparing(
        task_id='A', canonical_task_cid='cid:a', attempt=1, lane_id='lane',
        workspace_path=tmp_path / 'workspace', branch='implementation/a', merge_target='main',
    )
    daemon.worktree_lifecycle = store
    daemon._active_worktree_lifecycle = record
    return daemon, store, record


def replacement(store, record):
    terminal = store.mark_terminal(record.workspace_path, lease_id=record.lease_id,
                                   expected_fence=record.fence)
    assert store.compare_and_delete(record.workspace_path, lease_id=terminal.lease_id,
                                    expected_fence=terminal.fence)
    new = store.begin_preparing(
        task_id='B', canonical_task_cid='cid:b', attempt=1, lane_id='other',
        workspace_path=record.workspace_path, branch='implementation/b', merge_target='main',
        lease_id=record.lease_id,
    )
    assert new.fence == record.fence
    return new


@pytest.mark.parametrize('boundary', ['settling', 'finalize'])
def test_reused_token_cannot_adopt_replacement(custody, boundary):
    daemon, store, record = custody
    new = replacement(store, record)
    before = store.workspace_path_for(record.workspace_path).read_bytes()
    if boundary == 'settling':
        with pytest.raises(OwnershipError, match='before validation'):
            daemon._mark_worktree_lifecycle_settling(record.workspace_path)
    else:
        result = daemon._finalize_worktree_lifecycle(record.workspace_path)
        assert result['finalized'] is False
    assert store.load_workspace(record.workspace_path) == new
    assert store.workspace_path_for(record.workspace_path).read_bytes() == before
    assert daemon._active_worktree_lifecycle == record


@pytest.mark.parametrize('boundary', ['settling', 'finalize'])
def test_replacement_inside_mutation_boundary_is_preserved(custody, monkeypatch, boundary):
    daemon, store, record = custody
    original = store.transition
    observed = {}
    def raced(*args, **kwargs):
        monkeypatch.setattr(store, 'transition', original)
        observed['new'] = replacement(store, record)
        observed['bytes'] = store.workspace_path_for(record.workspace_path).read_bytes()
        return original(*args, **kwargs)
    monkeypatch.setattr(store, 'transition', raced)
    if boundary == 'settling':
        with pytest.raises(OwnershipError, match='captured record'):
            daemon._mark_worktree_lifecycle_settling(record.workspace_path)
    else:
        calls = []
        result = daemon._finalize_exact_worktree_lifecycle(record, reason='cleanup',
            terminal_callback=lambda *_: calls.append(True))
        assert result['finalized'] is False
        assert calls == []
    assert store.load_workspace(record.workspace_path) == observed['new']
    assert store.workspace_path_for(record.workspace_path).read_bytes() == observed['bytes']
    assert daemon._active_worktree_lifecycle == record


def test_current_capture_settles_and_finalizes(custody):
    daemon, store, record = custody
    settled = daemon._mark_worktree_lifecycle_settling(record.workspace_path)
    assert settled.state == WorkspaceLifecycleState.SETTLING
    assert daemon._active_worktree_lifecycle == settled
    assert daemon._mark_worktree_lifecycle_settling(record.workspace_path) == settled
    assert daemon._finalize_worktree_lifecycle(record.workspace_path)['finalized'] is True
    assert daemon._active_worktree_lifecycle is None


@pytest.mark.parametrize('boundary', ['settling', 'finalize'])
def test_no_capture_does_not_adopt_path(custody, boundary):
    daemon, store, record = custody
    daemon._active_worktree_lifecycle = None
    if boundary == 'settling':
        assert daemon._mark_worktree_lifecycle_settling(record.workspace_path) is None
    else:
        assert daemon._finalize_worktree_lifecycle(record.workspace_path)['finalized'] is False
    assert store.load_workspace(record.workspace_path) == record
