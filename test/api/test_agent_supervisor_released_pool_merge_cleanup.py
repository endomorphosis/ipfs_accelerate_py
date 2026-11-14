"""A released pool handoff cannot acquire workspace mutation authority later."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import workspace_quarantine
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


def git(repo, *args):
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def queued_handoff(tmp_path, *, status_lines="- Status: ready", protected=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Pool Handoff Test")
    git(repo, "config", "user.email", "pool@example.invalid")
    board = repo / "tasks.md"
    board.write_text("# Tasks\n\n## POOL-001 Retain source-bound output\n\n"
                     + status_lines + "\n- Completion: manual\n- Priority: P0\n"
                     "- Track: runtime\n- Outputs: result.txt\n")
    git(repo, "add", "tasks.md")
    git(repo, "commit", "-qm", "reviewed task")
    baseline = git(repo, "rev-parse", "HEAD")
    branch = "implementation/pool-001"
    git(repo, "checkout", "-qb", branch)
    (repo / "result.txt").write_text("validated output\n")
    git(repo, "add", "result.txt")
    git(repo, "commit", "-qm", "task output")
    candidate = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    state = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=board, repo_root=repo, task_header_prefix="## POOL-",
        state_path=state / "state.json", strategy_path=state / "strategy.json",
        events_path=state / "events.jsonl", implement=False,
        worktree_root=tmp_path / "fresh-workspaces",
        merge_queue_dir=tmp_path / "merge-queue",
        implementation_protected_paths=("tasks.md",) if protected else (),
    )
    task = daemon._load_tasks()[0]
    request, _ = daemon._enqueue_merge_candidate(
        task=task, attempt=1, branch_name=branch,
        implementation_commit=candidate, baseline_ref=baseline,
        worktree_path=None, worktree_pool_handoff=True,
        changed_submodule_paths=[],
        validation_result={"attempted": True, "passed": True, "returncode": 0,
                           "results": [], "selection": {"scope": "pre_merge"}},
    )
    frozen_root = tmp_path / "retained-workspaces"
    frozen_root.mkdir()
    snapshot = workspace_quarantine.plan(repo, frozen_root)["snapshot"]
    frozen = workspace_quarantine.freeze(repo, frozen_root, expected=snapshot)
    return daemon, repo, request, branch, candidate, frozen_root, frozen


@pytest.mark.parametrize("status_lines", [
    "- Status: ready",
    "- status: todo",
    "- Status: completed\n- status: todo",
    "- StAtUs: completed\n- STATUS: in_progress\n- status: todo",
])
@pytest.mark.parametrize("protected", [False, True])
def test_released_pool_handoff_settles_fenced_queue_under_retained_quarantine(tmp_path, monkeypatch, status_lines, protected):
    daemon, repo, request, branch, candidate, frozen_root, frozen = queued_handoff(
        tmp_path, status_lines=status_lines, protected=protected,
    )
    cleanup_calls = []
    original = daemon._cleanup_merged_worktree

    def cleanup(path, *args, **kwargs):
        cleanup_calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(daemon, "_cleanup_merged_worktree", cleanup)
    result = daemon._consume_one_merge_candidate()
    assert result is not None
    assert result["status"] in {"merged", "already_merged"}, result
    assert daemon.merge_queue.get(request.request_id).status == "completed"
    assert cleanup_calls == []
    receipt = result["merge_result"]["cleanup_result"]
    assert receipt["cleaned"] is True
    assert receipt["reason"] == "worktree_pool_handoff_already_released"
    assert receipt["removed_worktree"] is receipt["deleted_branch"] is False
    assert git(repo, "merge-base", "--is-ancestor", candidate, "main") == ""
    assert git(repo, "rev-parse", branch) == candidate
    assert workspace_quarantine.verify(repo, frozen_root) == frozen
    assert daemon._load_tasks()[0].status == "completed"
    statuses = [line for line in daemon.todo_path.read_text().splitlines()
                if line.split(":", 1)[0].lower() == "- status"]
    assert len(statuses) == len(status_lines.splitlines())
    assert all(line.split(":", 1)[1].strip() == "completed" for line in statuses)
    assert result["merge_result"]["todo_update_result"]["inserted_status_task_ids"] == []
    board_after = daemon.todo_path.read_bytes()
    repeated = daemon._mark_task_completed_in_todo(
        request.task_id,
        expected_task_cids=request.metadata["completion_task_cids"],
    )
    assert repeated["reason"] == "already_completed"
    assert daemon.todo_path.read_bytes() == board_after
    integrated = git(repo, "rev-parse", "main")
    assert daemon._consume_one_merge_candidate() is None
    assert git(repo, "rev-parse", "main") == integrated


def test_unbound_branch_cleanup_still_fails_closed_under_quarantine(tmp_path):
    daemon, repo, _request, branch, candidate, frozen_root, frozen = queued_handoff(tmp_path)
    with pytest.raises(workspace_quarantine.QuarantineDenied, match="workspace_mutation_path_unbound"):
        daemon._cleanup_merged_worktree(None, branch)
    assert git(repo, "rev-parse", branch) == candidate
    assert workspace_quarantine.verify(repo, frozen_root) == frozen


def test_completion_status_repair_cannot_cross_task_or_goal_sections():
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.task_header_prefix = "## POOL-"
    daemon.todo_path = Path("tasks.md")
    daemon._completion_receipts_for_task_ids = lambda _ids: []
    lines = ["## POOL-001 selected\n", "- Status: completed\n", "- status: todo\n",
             "## Goal ROOT\n", "- status: blocked\n",
             "## POOL-002 peer\n", "- status: in_progress\n"]
    result = daemon._mark_tasks_completed_in_locked_lines(
        lines, target_task_ids=["POOL-001"], primary_task_id="POOL-001",
        completion_reason="single_task", bundle_work_order=None,
    )
    assert result["updated_task_ids"] == ["POOL-001"]
    assert lines[1:3] == ["- Status: completed\n", "- status: completed\n"]
    assert lines[4] == "- status: blocked\n"
    assert lines[6] == "- status: in_progress\n"


@pytest.mark.parametrize("recovery_path", ["ordinary_consumer", "exact_request"])
def test_exact_integrated_quarantine_recovers_native_completion_without_remerging(tmp_path, monkeypatch, recovery_path):
    daemon, repo, request, _branch, candidate, frozen_root, frozen = queued_handoff(
        tmp_path, status_lines="- Status: completed\n- status: todo", protected=True,
    )
    original = daemon._mark_task_completed_in_todo

    def legacy_false_completion(task_id, **_kwargs):
        # Reproduce the old writer's promise while its trailing lowercase
        # declaration still parses as pending. The real callback must reject it.
        return {"updated": False, "durable": True, "reason": "already_completed",
                "completion_receipts": daemon._completion_receipts_for_task_ids([task_id])}

    monkeypatch.setattr(daemon, "_mark_task_completed_in_todo", legacy_false_completion)
    failed = daemon._consume_one_merge_candidate()
    assert failed["status"] == "quarantined", failed
    assert failed["reason"] == "merge_completion_receipt_invalid"
    assert daemon.merge_queue.get(request.request_id).status == "quarantined"
    git(repo, "merge-base", "--is-ancestor", candidate, "main")
    merge_count = git(repo, "rev-list", "--count", "--min-parents=2", "main")
    monkeypatch.setattr(daemon, "_mark_task_completed_in_todo", original)
    train = MergeTrain(
        repo_root=repo, queue=daemon.merge_queue,
        target_branch=daemon.resolved_merge_target_branch,
        max_attempts=daemon.merge_queue.max_attempts,
        merge_callback=daemon._merge_train_callback,
        formal_verification_policy=daemon.formal_verification_policy,
        proof_gate=daemon.proof_gate, proof_cache_dir=daemon.proof_cache_dir,
        decision_runtime=daemon.decision_runtime,
        decision_runtime_cancellation=daemon.implementation_cancelled,
    )
    recovered = (
        daemon._consume_one_merge_candidate()
        if recovery_path == "ordinary_consumer"
        else train.recover_one_integrated_quarantine(request_id=request.request_id)
    )
    assert recovered is not None
    assert recovered["status"] == "already_merged", recovered
    assert recovered["request_id"] == request.request_id
    assert daemon.merge_queue.get(request.request_id).status == "completed"
    assert daemon._load_tasks()[0].status == "completed"
    assert git(repo, "rev-list", "--count", "--min-parents=2", "main") == merge_count
    assert workspace_quarantine.verify(repo, frozen_root) == frozen
