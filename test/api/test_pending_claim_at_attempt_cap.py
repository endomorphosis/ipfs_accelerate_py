"""Final consumed-attempt custody release never grants a fourth dispatch.

Synthetic history uses real Native claims, lifecycle transitions, real exited
Python births and Git. No provider, task completion or Docker worker runs.
"""
from dataclasses import asdict
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import pending_task_claim_recovery as recovery
from test.api.test_pending_task_claim_recovery import pending


def assert_no_new_dispatch(daemon, task, before):
    state = native.PortalTaskState.load(daemon.state_path)
    assert asdict(state) == before
    selectable, limited = daemon._partition_tasks_at_attempt_limit(
        [task], {task.task_id: "ready"}, state)
    assert selectable == [] and len(limited) == 1
    assert limited[0]["task_id"] == task.task_id
    assert limited[0]["attempt_count"] == daemon.max_task_attempts == 3
    assert daemon._task_attempt(state, task) == 4
    assert not any(event["type"] in recovery._OUTCOMES for event in daemon._iter_events())


@pytest.mark.parametrize("pending", [{"attempt": 3}], indirect=True)
def test_final_consumed_attempt_releases_exact_dead_claim_without_rearming(pending):
    daemon, task, workspace, claim = pending
    daemon.max_task_attempts = 3
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    board = daemon.todo_path.read_bytes()
    lifecycle = daemon.worktree_lifecycle.workspace_path_for(workspace).read_bytes()
    assert before["implementation_attempts"][task.task_id] == 3
    request = daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    result = daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["reconciled"] and not result["blocked"] and not claim.exists(), result
    assert result["counters_preserved"] and result["task_status_unchanged"]
    assert result["provider_outcome"] == "unknown"
    assert all(result[key] is False for key in recovery._FALSE)
    assert daemon.todo_path.read_bytes() == board
    assert daemon.worktree_lifecycle.workspace_path_for(workspace).read_bytes() == lifecycle
    assert_no_new_dispatch(daemon, task, before)
    again = daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert again["reason"] == "pending_claim_already_released"
    assert again["receipt_id"] == result["receipt_id"]
    assert_no_new_dispatch(daemon, task, before)


@pytest.mark.parametrize("pending", [{"attempt": 3, "candidate": True}], indirect=True)
def test_final_attempt_preserves_admitted_candidate_without_completion_or_dispatch(pending):
    daemon, task, workspace, claim = pending
    daemon.max_task_attempts = 3
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    board = daemon.todo_path.read_bytes()
    code = (workspace / "src/example.py").read_bytes()
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    assert request["base"]["attempt"] == 3
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["reconciled"] and not result["blocked"] and not claim.exists(), result
    assert result["reason"] == "preserved_candidate_pending_task_claim_released"
    assert result["provider_outcome"] == "unknown"
    assert all(result[key] is False for key in recovery._FALSE)
    assert not workspace.exists() and daemon.todo_path.read_bytes() == board
    preserved = result["preserved_candidate"]
    assert subprocess.check_output(["git", "show", preserved["commit"] + ":src/example.py"], cwd=daemon.repo_root) == code
    assert_no_new_dispatch(daemon, task, before)
    fresh = recovery.review_recovery(daemon, task_id=task.task_id)
    assert fresh == request
    assert recovery.reconcile_recovery(daemon, request=fresh) == result
    assert_no_new_dispatch(daemon, task, before)


@pytest.mark.parametrize("pending", [{"attempt": 4}, {"attempt": 4, "candidate": True}], indirect=True)
def test_over_cap_history_cannot_release_or_preserve_claim(pending, monkeypatch):
    daemon, task, workspace, claim = pending
    daemon.max_task_attempts = 3
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    raw = claim.read_bytes()
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", lambda *a, **k: pytest.fail("over-cap preservation"))
    with pytest.raises(ValueError, match="retry_budget_exhausted"):
        recovery.review_recovery(daemon, task_id=task.task_id)
    assert claim.read_bytes() == raw and workspace.exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    assert not (daemon.state_path.parent / recovery.DIRECTORY).exists()
