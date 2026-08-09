from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DuckDBTaskCompletionEvidence,
    DuckDBValidationExecutionReceipt,
    PortalImplementationDaemon,
    PortalTask,
)
from test.api.test_agent_supervisor_task_source_e2e import _sources


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _integrated_commits(repo: Path) -> tuple[str, str, str, str, str]:
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Completion Evidence Test")
    _git(repo, "config", "user.email", "completion-evidence@example.test")
    tracked = repo / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    declared_test = repo / "test_declared.py"
    declared_test.write_text(
        "def test_declared_validation():\n    assert True\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tracked.txt", "test_declared.py")
    _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "implementation/completion-evidence")
    tracked.write_text("implemented\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "implementation")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    implementation_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    _git(repo, "checkout", "main")
    _git(
        repo,
        "merge",
        "--no-ff",
        "--no-edit",
        "implementation/completion-evidence",
    )
    merge_commit = _git(repo, "rev-parse", "HEAD")
    merge_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    return (
        baseline,
        implementation_commit,
        implementation_tree,
        merge_commit,
        merge_tree,
    )


def _ordinary_validation_result(
    daemon: PortalImplementationDaemon,
    task: PortalTask,
    workspace: Path,
) -> tuple[dict[str, object], PortalTask]:
    validated_task = replace(
        task,
        validation=["python -m pytest -q test_declared.py"],
    )
    result = daemon._run_validation_commands(
        workspace,
        validated_task,
        workspace / "declared-validation.log",
    )
    assert result["passed"] is True
    assert result["selection"]["scope"] == "pre_merge"
    assert result["results"]
    assert all(item.get("validation_result_digest") for item in result["results"])
    result["proposal_gate"] = {
        "accepted": True,
        "receipt_id": "proposal-receipt:exact",
    }
    return result, validated_task


def test_duckdb_completion_requires_exact_live_post_merge_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _markdown, database = _sources(tmp_path)
    (
        baseline,
        implementation_commit,
        implementation_tree,
        merge_commit,
        merge_tree,
    ) = _integrated_commits(tmp_path)
    runtime = tmp_path / "runtime"
    daemon = PortalImplementationDaemon(
        task_source=database,
        state_path=runtime / "state.json",
        strategy_path=runtime / "strategy.json",
        events_path=runtime / "events.jsonl",
        repo_root=tmp_path,
        merge_target_branch="main",
        worktree_pool_enabled=False,
        validation_cache_dir=runtime / "validation-cache",
        merge_queue_dir=runtime / "merge-queue",
    )
    task = daemon._load_tasks()[0]
    validation_result, task = _ordinary_validation_result(daemon, task, tmp_path)
    current = database.get(task.task_id)
    assert current is not None

    bare = daemon._mark_task_completed_in_todo(task.task_id)
    assert bare["reason"] == "completion_evidence_required"
    assert database.get(task.task_id).status != "completed"

    queued, _result = daemon._enqueue_merge_candidate(
        branch_name="implementation/completion-evidence",
        implementation_commit=implementation_commit,
        baseline_ref=baseline,
        worktree_path=None,
        task=task,
        attempt=1,
        validation_result=validation_result,
    )
    assert queued.metadata["task_source_identity"] == database.identity.to_dict()
    execution_receipt = DuckDBValidationExecutionReceipt.from_dict(
        queued.metadata["validation_proof"]["validation_execution_receipt"]
    )
    # Declared DuckDB merges now seal post_merge_evidence_input and therefore
    # retain both the impact validation receipt id and the execution receipt.
    receipt_ids = queued.metadata["validation_proof"]["validation_receipt_ids"]
    assert execution_receipt.receipt_id in receipt_ids
    post_merge = queued.metadata["validation_proof"].get("post_merge_evidence_input")
    assert isinstance(post_merge, dict)
    assert post_merge.get("packet_id")
    impact_receipt_id = str(
        post_merge.get("validation_receipt", {}).get("receipt_id") or ""
    )
    assert impact_receipt_id
    assert impact_receipt_id in receipt_ids
    assert all(execution_receipt.validation_ids)
    assert all(
        item["validation_id"] for item in queued.metadata["validation_proof"]["results"]
    )
    assert execution_receipt.validation_result_digests == tuple(
        item["validation_result_digest"] for item in validation_result["results"]
    )
    assert (
        queued.metadata["validation_proof"]["proposal_gate"]["receipt_id"]
        == "proposal-receipt:exact"
    )
    assert queued.metadata["candidate_tree"] == implementation_tree

    claim = daemon.merge_queue.dequeue("consumer:completion-evidence")
    assert claim is not None
    evidence_by_task = daemon._duckdb_completion_evidence_for_merge_claim(
        claim,
        {"merge_commit": merge_commit},
        (task.task_id,),
    )
    evidence = evidence_by_task[task.task_id]
    assert evidence.target_tree == merge_tree

    forged_values = (
        replace(evidence, task_cid="task:foreign"),
        replace(evidence, proposal_receipt_id="proposal-receipt:foreign"),
        replace(evidence, fencing_token=evidence.fencing_token + 1),
        replace(
            evidence,
            task_source_fencing_token=evidence.task_source_fencing_token + 1,
        ),
    )
    for forged in forged_values:
        rejected = daemon._mark_task_completed_in_todo(
            task.task_id,
            completion_evidence=forged,
            completion_claim=claim,
        )
        assert rejected["reason"] == "completion_evidence_invalid"
        assert database.get(task.task_id).status != "completed"

    monkeypatch.setattr(
        daemon,
        "_rehydrate_merge_request_branch",
        lambda **_kwargs: {"ready": True, "rehydrated": False},
    )
    monkeypatch.setattr(
        daemon,
        "_merge_branch_to_main",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "merged": True,
            "returncode": 0,
            "merge_commit": merge_commit,
            "submodule_merge_results": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    merged = daemon._merge_train_callback(claim)

    assert merged["merged"] is True
    assert merged["todo_update_result"]["updated_task_ids"] == [task.task_id]
    assert database.get(task.task_id).status == "completed"
    status_event = database.backend.events().events[-1]
    persisted = status_event["body"]["receipt"]["completion_evidence"]
    assert persisted["evidence_id"] == evidence.evidence_id
    assert DuckDBTaskCompletionEvidence.from_dict(persisted) == evidence

    replay = daemon._mark_task_completed_in_todo(task.task_id)
    assert replay["updated"] is False
    assert replay["reason"] == "already_completed"


def test_post_merge_crash_replays_on_fresh_daemon_and_new_claim(
    tmp_path: Path,
) -> None:
    _markdown, database = _sources(tmp_path)
    (
        baseline,
        implementation_commit,
        _implementation_tree,
        merge_commit,
        merge_tree,
    ) = _integrated_commits(tmp_path)
    runtime = tmp_path / "runtime-replay"
    now = [100.0]
    queue_dir = runtime / "merge-queue"
    producer_queue = MergeQueue(
        queue_dir,
        max_age_seconds=1,
        clock=lambda: now[0],
    )
    producer = PortalImplementationDaemon(
        task_source=database,
        state_path=runtime / "state.json",
        strategy_path=runtime / "strategy.json",
        events_path=runtime / "events.jsonl",
        repo_root=tmp_path,
        merge_target_branch="main",
        worktree_pool_enabled=False,
        validation_cache_dir=runtime / "validation-cache",
        merge_queue=producer_queue,
        merge_queue_dir=queue_dir,
    )
    task = producer._load_tasks()[0]
    validation_result, task = _ordinary_validation_result(producer, task, tmp_path)
    queued, _ = producer._enqueue_merge_candidate(
        branch_name="implementation/completion-evidence",
        implementation_commit=implementation_commit,
        baseline_ref=baseline,
        worktree_path=None,
        task=task,
        attempt=1,
        validation_result=validation_result,
    )
    crashed_claim = producer_queue.dequeue("consumer:crashed-before-cas")
    assert crashed_claim is not None
    assert database.get(task.task_id).status != "completed"
    assert _git(tmp_path, "rev-parse", "main") == merge_commit

    # The repository merge is durable, but the old owner disappears before
    # status CAS. A separately constructed queue/daemon recovers a new fence.
    now[0] += 2
    restart_queue = MergeQueue(
        queue_dir,
        max_age_seconds=1,
        clock=lambda: now[0],
    )
    restart = PortalImplementationDaemon(
        task_source=database,
        state_path=runtime / "restart-state.json",
        strategy_path=runtime / "restart-strategy.json",
        events_path=runtime / "restart-events.jsonl",
        repo_root=tmp_path,
        merge_target_branch="main",
        worktree_pool_enabled=False,
        validation_cache_dir=runtime / "restart-validation-cache",
        merge_queue=restart_queue,
        merge_queue_dir=queue_dir,
    )
    recovered_claim = restart_queue.dequeue("consumer:fresh-daemon")
    assert recovered_claim is not None
    assert recovered_claim.request_id == queued.request_id
    assert recovered_claim.claim_token != crashed_claim.claim_token
    assert recovered_claim.claim_generation > crashed_claim.claim_generation
    assert not restart_queue.owns_claim(crashed_claim)

    replayed = restart._merge_train_callback(recovered_claim)
    assert replayed["merged"] is True
    assert replayed.get("merge_commit") == merge_commit
    assert replayed["todo_update_result"]["updated_task_ids"] == [task.task_id]
    completed = database.get(task.task_id)
    assert completed is not None and completed.status == "completed"
    status_event = database.backend.events(cursor=0, limit=100).events[-1]
    evidence = DuckDBTaskCompletionEvidence.from_dict(
        status_event["body"]["receipt"]["completion_evidence"]
    )
    assert evidence.merge_commit == merge_commit
    assert evidence.target_tree == merge_tree
    assert evidence.lease_id == recovered_claim.claim_token
    assert evidence.fencing_token == recovered_claim.claim_generation

    second_restart = PortalImplementationDaemon(
        task_source=database,
        state_path=runtime / "second-restart-state.json",
        strategy_path=runtime / "second-restart-strategy.json",
        events_path=runtime / "second-restart-events.jsonl",
        repo_root=tmp_path,
        merge_target_branch="main",
        worktree_pool_enabled=False,
        validation_cache_dir=runtime / "second-restart-validation-cache",
        merge_queue=restart_queue,
        merge_queue_dir=queue_dir,
    )
    assert second_restart._load_tasks()[0].status == "completed"
    admitted = second_restart._mark_task_completed_in_todo(task.task_id)
    assert admitted["reason"] == "already_completed"
    assert (
        admitted["completion_receipts"][0]["completion_evidence"]["evidence_id"]
        == evidence.evidence_id
    )
