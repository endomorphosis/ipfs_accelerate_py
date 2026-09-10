"""A terminal lifecycle alone cannot replace a required handoff callback."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorktreeLifecycleStore,
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)


@pytest.fixture
def lifecycle(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    store = WorktreeLifecycleStore(repo_root=repo, store_dir=tmp_path / "lifecycle")
    record = store.begin_preparing(
        task_id="TASK-001", canonical_task_cid="task:one", attempt=1,
        lane_id="lane-0", workspace_path=tmp_path / "workspace",
        branch="attempt/task-001", merge_target="main",
        owner=current_process_birth(), state_dir=str(tmp_path / "state"),
    )
    record = store.mark_settling(
        record.workspace_path, lease_id=record.lease_id, expected_fence=record.fence,
    )
    daemon = object.__new__(TodoImplementationDaemon)
    daemon.worktree_lifecycle = store
    daemon._active_worktree_lifecycle = record
    return SimpleNamespace(daemon=daemon, store=store, prior=record)


def _failed_handoff(case):
    calls = []

    def fail(prior, terminal):
        calls.append((prior, terminal))
        # This is the actual post-terminal-CAS/pre-deletion failure boundary.
        assert case.store.load_workspace(prior.workspace_path) == terminal
        raise OSError("receipt persistence unavailable")

    with pytest.raises(OSError, match="receipt persistence unavailable"):
        case.daemon._finalize_exact_worktree_lifecycle(
            case.prior, reason="merge_queue_handoff", terminal_callback=fail,
        )
    terminal = case.store.load_workspace(case.prior.workspace_path)
    assert terminal.is_terminal
    assert terminal.fence == case.prior.fence + 1
    assert calls == [(case.prior, terminal)]
    return terminal


def _bytes(case):
    return (
        case.store.workspace_path_for(case.prior.workspace_path).read_bytes(),
        case.store.task_index_path_for(
            task_id=case.prior.task_id,
            canonical_task_cid=case.prior.canonical_task_cid,
            attempt=case.prior.attempt,
        ).read_bytes(),
    )


def test_terminal_retry_preserves_record_when_required_callback_failed(lifecycle):
    case = lifecycle
    terminal = _failed_handoff(case)
    before = _bytes(case)
    case.daemon._active_worktree_lifecycle = terminal
    result = case.daemon._finalize_exact_worktree_lifecycle(
        terminal, reason="merge_queue_handoff",
        terminal_callback=lambda *_: pytest.fail("terminal record is not prior proof"),
    )
    assert result["finalized"] is False
    assert result["reason"] == "lifecycle_terminal_callback_recovery_required"
    assert result["attempt_consumed"] is False
    assert result["provider_call_allowed"] is False
    assert _bytes(case) == before
    assert case.daemon._active_worktree_lifecycle == terminal


def test_terminal_retry_cannot_publish_queue_or_release_pool(lifecycle):
    case = lifecycle
    terminal = _failed_handoff(case)
    before = _bytes(case)
    daemon = case.daemon
    daemon._active_worktree_lifecycle = terminal
    daemon._worktree_pool_effective_paths = {}
    daemon._worktree_pool_leases = {}
    daemon._reject_protected_merge_candidate = lambda **_: None
    daemon._mark_active_phase = lambda *_, **__: None
    daemon._release_pooled_worktree_lease = lambda *_, **__: pytest.fail("pool released")
    daemon._enqueue_merge_candidate = lambda **_: pytest.fail("queue published")
    daemon._consume_one_merge_candidate = lambda: pytest.fail("consumer dispatched")
    result = daemon._enqueue_validated_worktree(
        state=object(), task=SimpleNamespace(task_id=terminal.task_id), attempt=1,
        branch_name=terminal.branch, baseline_ref="a" * 40,
        worktree_path=Path(terminal.workspace_path),
        implementation_commit="b" * 40, commit_result={}, validation_result={},
        lifecycle_terminal_callback=lambda *_: pytest.fail("prior identity missing"),
    )
    assert result["queued"] is False
    assert result["merged"] is False
    assert result["reason"] == "worktree_lifecycle_handoff_failed"
    assert _bytes(case) == before


def test_normal_callback_observes_prior_and_terminal_before_delete(lifecycle):
    case = lifecycle
    calls = []

    def commit(prior, terminal):
        assert prior == case.prior
        assert case.store.load_workspace(prior.workspace_path) == terminal
        calls.append(terminal)
        return {"handoff_receipt_cid": "receipt:exact"}

    result = case.daemon._finalize_exact_worktree_lifecycle(
        case.prior, reason="merge_queue_handoff", terminal_callback=commit,
    )
    assert len(calls) == 1
    assert result["finalized"] is True
    assert result["terminal_callback"] == {"handoff_receipt_cid": "receipt:exact"}
    assert case.store.load_workspace(case.prior.workspace_path) is None


def test_terminal_cleanup_without_required_callback_remains_compatible(lifecycle):
    case = lifecycle
    terminal = case.store.mark_terminal(
        case.prior.workspace_path, lease_id=case.prior.lease_id,
        expected_fence=case.prior.fence, reason="cleanup_finished",
    )
    result = case.daemon._finalize_exact_worktree_lifecycle(
        terminal, reason="cleanup_finished",
    )
    assert result["finalized"] is True
    assert result["reason"] == "already_terminal_deleted"
