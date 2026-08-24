from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    FALSE_COMPLETION_RECOVERY_RECEIPT_SCHEMA,
    _FALSE_COMPLETION_REVIVAL_CAPABILITY,
    MergeQueue,
    MergeQueueFenceError,
    MergeRequest,
    completed_request_digest,
    false_completion_recovery_receipt_cid,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import (
    MergeResolverRegistry,
    conflict_fingerprint,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Merge Train Test")
    _git(repo, "config", "user.email", "merge-train@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    return repo


def _false_completion_recovery_receipt(
    completed: MergeRequest,
    *,
    observed_target_commit: str,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": FALSE_COMPLETION_RECOVERY_RECEIPT_SCHEMA,
        "request_id": completed.request_id,
        "canonical_task_id": completed.canonical_task_id,
        "canonical_task_key": completed.canonical_task_key,
        "dedupe_key": completed.dedupe_key,
        "candidate_commit": completed.commit_sha,
        "target_repository_id": completed.target_repository_id,
        "target_branch": completed.target_branch,
        "observed_target_commit": observed_target_commit,
        "candidate_integrated": False,
        "completed_claim_generation": completed.claim_generation,
        "completed_finished_at": completed.finished_at,
        "completed_row_digest": completed_request_digest(completed),
        "observation_method": "git_merge_base_is_ancestor",
        "observer_id": "test:merge-train:false-completion-recovery",
        "observed_at": completed.finished_at + 1.0,
        "reason": "exact candidate is absent from the target",
    }
    receipt["receipt_cid"] = false_completion_recovery_receipt_cid(receipt)
    return receipt


def test_queue_deduplicates_canonical_task_and_commit_across_lanes(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "queue")
    first = queue.enqueue(
        branch_name="implementation/ref-038-a",
        task_id="REF-038",
        canonical_task_id="canonical-ref-038",
        commit_sha="a" * 40,
        lane_id="lane-a",
    )
    duplicate = queue.enqueue(
        branch_name="implementation/alias-b",
        task_id="BOARD-912",
        canonical_task_id="canonical-ref-038",
        commit_sha="a" * 40,
        lane_id="lane-b",
    )

    assert duplicate.request_id == first.request_id
    assert queue.pending_count() == 1
    claimed = queue.dequeue(consumer_id="train")
    assert isinstance(claimed, MergeRequest)
    assert claimed.request_id == first.request_id


def test_queue_projects_active_and_completed_canonical_task_ids(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "queue")
    completed = queue.enqueue(
        branch_name="implementation/completed",
        task_id="LANE-001",
        canonical_task_id="canonical-completed",
        commit_sha="a" * 40,
    )
    claimed = queue.dequeue(consumer_id="merge-train:test")
    assert claimed is not None and claimed.request_id == completed.request_id
    queue.complete(claimed)
    queue.enqueue(
        branch_name="implementation/pending",
        task_id="LANE-002",
        canonical_task_id="canonical-pending",
        commit_sha="b" * 40,
    )

    assert queue.completed_canonical_task_ids() == {"canonical-completed"}
    assert queue.active_canonical_task_ids() == {"canonical-pending"}
    processing = queue.dequeue(consumer_id="merge-train:other")
    assert processing is not None
    assert queue.active_canonical_task_ids() == {"canonical-pending"}


def test_queue_combines_priority_with_age_fairness(tmp_path: Path) -> None:
    now = [0.0]
    queue = MergeQueue(
        tmp_path / "queue",
        clock=lambda: now[0],
        priority_aging_seconds=10,
        max_age_seconds=1_000,
    )
    old = queue.enqueue(
        branch_name="old-low", task_id="OLD", priority="P3", commit_sha="1" * 40
    )
    now[0] = 40.0
    queue.enqueue(
        branch_name="new-high", task_id="NEW", priority="P0", commit_sha="2" * 40
    )

    claimed = queue.dequeue()
    assert claimed is not None
    assert claimed.request_id == old.request_id


def test_pending_request_does_not_expire_without_consumer_claim(tmp_path: Path) -> None:
    now = [0.0]
    queue = MergeQueue(tmp_path / "queue", clock=lambda: now[0], max_age_seconds=10)
    request = queue.enqueue(
        branch_name="implementation/waiting",
        task_id="WAITING",
        commit_sha="a" * 40,
    )

    now[0] = 60.0
    claimed = queue.dequeue(consumer_id="merge-train:test")

    assert claimed is not None
    assert claimed.request_id == request.request_id
    assert claimed.status == "processing"
    assert claimed.attempt == 1
    assert claimed.failure_count == 0


def test_queue_can_revive_false_positive_quarantine(tmp_path: Path) -> None:
    now = [10.0]
    queue = MergeQueue(tmp_path / "queue", clock=lambda: now[0])
    request = queue.enqueue(
        branch_name="implementation/recoverable",
        task_id="RECOVERABLE",
        commit_sha="b" * 40,
    )
    quarantine_path = queue.quarantine(
        request,
        reason="pending request exceeded max age",
    )
    assert quarantine_path is not None and quarantine_path.exists()

    now[0] = 20.0
    revived = queue.revive_quarantined(
        request.request_id,
        reason="host resumed after suspension",
        reset_failures=True,
    )

    assert revived is not None
    assert revived.status == "pending"
    assert revived.attempt == 1
    assert revived.failure_count == 0
    assert revived.failure_reason == ""
    assert revived.file_path is not None and revived.file_path.parent == queue.pending_dir
    assert not quarantine_path.exists()
    assert revived.metadata["revivals"] == [
        {
            "at": 20.0,
            "reason": "host resumed after suspension",
            "previous_enqueued_at": 10.0,
            "previous_failure_count": 1,
            "previous_failure_reason": "pending request exceeded max age",
        }
    ]


def test_expired_processing_claim_is_recovered(tmp_path: Path) -> None:
    now = [10.0]
    queue = MergeQueue(
        tmp_path / "queue",
        clock=lambda: now[0],
        max_age_seconds=10,
        max_attempts=3,
    )
    request = queue.enqueue(
        branch_name="implementation/abandoned",
        task_id="ABANDONED",
        commit_sha="c" * 40,
    )
    first_claim = queue.dequeue(consumer_id="worker-that-exited")
    assert first_claim is not None

    now[0] = 30.0
    recovered = queue.dequeue(consumer_id="replacement-worker")

    assert recovered is not None
    assert recovered.request_id == request.request_id
    assert recovered.status == "processing"
    assert recovered.attempt == 2
    assert recovered.failure_count == 1
    assert recovered.failure_reason == "consumer claim expired; request recovered"


def test_train_rebases_candidate_on_latest_target_and_updates_target(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/ref-038")
    (repo / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    (repo / "target.txt").write_text("latest target\n", encoding="utf-8")
    _git(repo, "add", "target.txt")
    _git(repo, "commit", "-m", "advance target")
    target_before = _git(repo, "rev-parse", "HEAD")

    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/ref-038",
        task_id="REF-038",
        canonical_task_id="canonical-ref-038",
        commit_sha=candidate,
        metadata={"baseline_ref": base},
    )
    result = MergeTrain(repo, queue).run_once()

    assert result is not None
    assert result["status"] == "merged"
    assert result["rebased"] is True
    target_after = _git(repo, "rev-parse", "refs/heads/main")
    assert target_after != target_before
    assert _git(repo, "show", f"{target_after}:candidate.txt") == "candidate"
    assert _git(repo, "show", f"{target_after}:target.txt") == "latest target"
    assert queue.get(request.request_id).status == "completed"  # type: ignore[union-attr]


def test_existing_commit_validation_uses_identifier_worktree_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    commit = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/validation-worktree",
        task_id="VALIDATION-WORKTREE",
        commit_sha=commit,
    )
    train = MergeTrain(repo, queue)
    observed: list[Path] = []

    def validate(**kwargs: object) -> dict[str, object]:
        observed.append(Path(str(kwargs["workspace"])))
        return {"passed": False, "reason": "fixture_validation_stop"}

    monkeypatch.setattr(train, "_validate_synthesized_tree", validate)

    result = train._validate_existing_integrated_commit(
        request,
        commit=commit,
        candidate_commit=commit,
    )

    assert result["reason"] == "fixture_validation_stop"
    assert len(observed) == 1
    assert observed[0].name.startswith("validation_")
    assert observed[0].name.isidentifier()


def test_rebase_validation_uses_identifier_worktree_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    commit = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/candidate-worktree",
        task_id="CANDIDATE-WORKTREE",
        commit_sha=commit,
    )
    train = MergeTrain(
        repo,
        queue,
        post_merge_validation=lambda *_args, **_kwargs: {"passed": True},
    )
    observed: list[Path] = []

    def validate(**kwargs: object) -> dict[str, object]:
        observed.append(Path(str(kwargs["workspace"])))
        return {"passed": False, "reason": "fixture_validation_stop"}

    monkeypatch.setattr(train, "_validate_synthesized_tree", validate)

    result = train._rebase_and_integrate(
        request=request,
        canonical=request.canonical_identity,
        candidate=commit,
        target=commit,
    )

    assert result["reason"] == "fixture_validation_stop"
    assert len(observed) == 1
    assert observed[0].name.startswith("candidate_")
    assert observed[0].name.isidentifier()


def test_train_callback_runs_when_root_candidate_is_already_merged(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/ref-040",
        task_id="REF-040",
        canonical_task_id="canonical-ref-040",
        commit_sha=candidate,
    )
    callbacks: list[str] = []

    def finish_nested_handoff(claimed: MergeRequest) -> dict[str, object]:
        callbacks.append(claimed.request_id)
        return {"merged": True, "nested_handoff": "completed"}

    result = MergeTrain(repo, queue, merge_callback=finish_nested_handoff).run_once()

    assert result is not None
    assert result["status"] == "merged"
    assert result["merge_result"]["nested_handoff"] == "completed"
    assert callbacks == [request.request_id]
    assert queue.get(request.request_id).status == "completed"  # type: ignore[union-attr]


def test_portal_projection_does_not_equate_existing_path_with_candidate_blob(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/aseh-001")
    (repo / "base.txt").write_text("candidate revision\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "candidate declared output")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/aseh-001",
        task_id="ASEH-001",
        canonical_task_id="canonical-aseh-001",
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "todo_path": str(tmp_path / "task-projection.md"),
            "baseline_ref": base,
            "completion_task_cids": {
                "ASEH-001": "canonical-aseh-001",
            },
            "manual_completion_authority_task_ids": [],
            "manual_completion_authority_epoch_id": "",
            "task": {"outputs": ["base.txt"]},
        },
    )
    callbacks: list[str] = []

    def merge_candidate(claimed: MergeRequest) -> dict[str, object]:
        callbacks.append(claimed.request_id)
        _git(repo, "merge", "--ff-only", candidate)
        return {
            "merged": True,
            "target_commit": candidate,
            "merge_commit": candidate,
        }

    result = MergeTrain(
        repo,
        queue,
        merge_callback=merge_candidate,
    ).run_once()

    assert result is not None
    assert result["status"] == "merged"
    assert callbacks == [request.request_id]
    assert _git(repo, "show", "main:base.txt") == "candidate revision"


def test_portal_projection_shortcut_requires_exact_candidate_tree_entry(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/aseh-001",
        task_id="ASEH-001",
        canonical_task_id="canonical-aseh-001",
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "todo_path": str(tmp_path / "task-projection.md"),
            "completion_task_cids": {
                "ASEH-001": "canonical-aseh-001",
            },
            "manual_completion_authority_task_ids": [],
            "manual_completion_authority_epoch_id": "",
            "task": {"outputs": ["base.txt"]},
        },
    )
    callbacks: list[str] = []

    train = MergeTrain(
        repo,
        queue,
        merge_callback=lambda claimed: callbacks.append(claimed.request_id)
        or {"merged": True},
    )
    assert (
        train.completed_request_is_integrated(
            replace(request, commit_sha="HEAD")
        )
        is False
    )
    result = train.run_once()

    assert result is not None
    assert result["status"] == "already_merged"
    assert result["reason"] == "declared_outputs_already_on_target"
    assert result["mutation_short_circuited"] is True
    assert callbacks == []


def test_completed_queue_row_is_not_task_completion_before_integration(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """## REF-041R Qualify terminal queue bookkeeping

- Status: todo
- Completion: manual
- Outputs: base.txt
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "task projection")
    baseline = _git(repo, "rev-parse", "HEAD")
    target_branch = "agent/ref-041r-board"
    _git(repo, "switch", "-c", target_branch)
    branch_name = "implementation/ref-041r"
    _git(repo, "switch", "-c", branch_name)
    (repo / "base.txt").write_text(
        "candidate revision\n",
        encoding="utf-8",
    )
    _git(repo, "commit", "-am", "REF-041R: candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", target_branch)

    state_dir = tmp_path / "state"
    queue = MergeQueue(tmp_path / "queue")
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## REF-",
        merge_queue=queue,
        merge_target_branch=target_branch,
    )
    [task] = daemon._load_tasks()
    request, _enqueue_result = daemon._enqueue_merge_candidate(
        branch_name=branch_name,
        implementation_commit=candidate,
        baseline_ref=baseline,
        worktree_path=None,
        task=task,
        attempt=1,
    )
    claimed = queue.dequeue(consumer_id="seed-false-terminal-row")
    assert claimed is not None and claimed.request_id == request.request_id
    queue.complete(claimed)
    assert task.canonical_task_cid in queue.completed_canonical_task_ids()

    completed_cids, completed_bindings = (
        daemon._admitted_shared_merge_completions()
    )
    assert completed_cids == set()
    assert completed_bindings == {}

    _git(repo, "merge", "--ff-only", candidate)
    completed_cids, completed_bindings = (
        daemon._admitted_shared_merge_completions()
    )
    assert completed_cids == {task.canonical_task_cid}
    assert completed_bindings == {
        task.task_id: {task.canonical_task_cid},
    }


def test_integrated_pending_validation_row_is_not_completion_authority(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/pending-validation",
        task_id="PENDING-VALIDATION",
        canonical_task_id="canonical-pending-validation",
        commit_sha=candidate,
        metadata={
            "completion": {
                "status": "integrated_pending_validation",
                "accepted": False,
                "acceptance_pending": True,
            },
        },
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
    )

    assert (
        MergeTrain(repo, queue).completed_request_is_integrated(request)
        is False
    )


def test_queue_false_completion_revival_is_exact_audited_and_idempotent(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    now = [10.0]
    queue = MergeQueue(
        tmp_path / "queue",
        clock=lambda: now[0],
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/false-completion",
        task_id="FALSE-COMPLETION",
        canonical_task_id="canonical-false-completion",
        canonical_task_key="canonical-false-completion",
        commit_sha="a" * 40,
    )
    claimed = queue.dequeue(consumer_id="merge-train:seed-false-completion")
    assert claimed is not None
    queue.complete(
        claimed,
        metadata={"status": "merged", "accepted": True},
    )
    completed = queue.get(request.request_id)
    assert completed is not None and completed.status == "completed"
    recovery_receipt = _false_completion_recovery_receipt(
        completed,
        observed_target_commit=_git(repo, "rev-parse", "refs/heads/main"),
    )
    foreign = replace(
        completed,
        metadata={**completed.metadata, "target_branch": "foreign"},
    )
    train = MergeTrain(repo, queue)

    now[0] = 20.0
    with pytest.raises(
        MergeQueueFenceError,
        match="requires merge-train consumer authority",
    ):
        queue.revive_false_completed(
            completed,
            recovery_receipt=recovery_receipt,
        )
    with train._consumer_lease() as acquired:
        assert acquired is True
        with pytest.raises(MergeQueueFenceError, match="identity differs"):
            queue.revive_false_completed(
                foreign,
                recovery_receipt=recovery_receipt,
                _consumer_capability=_FALSE_COMPLETION_REVIVAL_CAPABILITY,
            )
        stale_generation = replace(
            completed,
            claim_generation=completed.claim_generation - 1,
        )
        with pytest.raises(
            MergeQueueFenceError,
            match="does not bind the supplied completed row",
        ):
            queue.revive_false_completed(
                stale_generation,
                recovery_receipt=recovery_receipt,
                _consumer_capability=_FALSE_COMPLETION_REVIVAL_CAPABILITY,
            )
        tampered_receipt = {
            **recovery_receipt,
            "reason": "tampered after content identification",
        }
        with pytest.raises(MergeQueueFenceError, match="content id is invalid"):
            queue.revive_false_completed(
                completed,
                recovery_receipt=tampered_receipt,
                _consumer_capability=_FALSE_COMPLETION_REVIVAL_CAPABILITY,
            )
        revived = queue.revive_false_completed(
            completed,
            recovery_receipt=recovery_receipt,
            _consumer_capability=_FALSE_COMPLETION_REVIVAL_CAPABILITY,
        )
        repeated = queue.revive_false_completed(
            completed,
            recovery_receipt=recovery_receipt,
            _consumer_capability=_FALSE_COMPLETION_REVIVAL_CAPABILITY,
        )
        different_proof = {
            **recovery_receipt,
            "reason": "a different observation must not replay",
        }
        different_proof["receipt_cid"] = (
            false_completion_recovery_receipt_cid(different_proof)
        )
        with pytest.raises(
            MergeQueueFenceError,
            match="pending request is not this false-completion revival",
        ):
            queue.revive_false_completed(
                completed,
                recovery_receipt=different_proof,
                _consumer_capability=_FALSE_COMPLETION_REVIVAL_CAPABILITY,
            )

    assert revived.status == repeated.status == "pending"
    assert revived.claim_generation == completed.claim_generation + 1
    assert repeated.claim_generation == revived.claim_generation
    assert repeated.metadata["completion"] == {
        "accepted": True,
        "status": "merged",
    }
    revivals = repeated.metadata["false_completion_revivals"]
    assert len(revivals) == 1
    revival = revivals[0]
    assert revival["at"] == 20.0
    assert revival["reason"] == recovery_receipt["reason"]
    assert revival["recovery_receipt_id"] == recovery_receipt["receipt_cid"]
    assert revival["recovery_receipt"] == recovery_receipt
    assert revival["previous_completed_row_digest"] == completed_request_digest(
        completed
    )
    assert revival["previous_finished_at"] == completed.finished_at == 10.0
    assert revival["previous_claim_generation"] == completed.claim_generation
    assert revival["previous_failure_count"] == completed.failure_count
    assert revival["previous_failure_reason"] == completed.failure_reason
    assert revival["previous_completion"] == {
        "accepted": True,
        "status": "merged",
    }


def test_train_recovers_one_exact_false_completion_through_existing_gates(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/false-completion")
    (repo / "recovered.txt").write_text("recovered\n", encoding="utf-8")
    _git(repo, "add", "recovered.txt")
    _git(repo, "commit", "-m", "false completion candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/false-completion",
        task_id="FALSE-COMPLETION",
        canonical_task_id="canonical-false-completion",
        canonical_task_key="canonical-false-completion",
        commit_sha=candidate,
        metadata={"baseline_ref": base, "changed_submodule_paths": []},
    )
    claimed = queue.dequeue(consumer_id="merge-train:seed-false-completion")
    assert claimed is not None
    queue.complete(
        claimed,
        metadata={"status": "merged", "accepted": True},
    )
    completed = queue.get(request.request_id)
    assert completed is not None and completed.status == "completed"
    recovery_receipt = _false_completion_recovery_receipt(
        completed,
        observed_target_commit=_git(repo, "rev-parse", "refs/heads/main"),
    )
    predicate_calls: list[str] = []
    context_events: list[str] = []
    after_results: list[str] = []
    train = MergeTrain(repo, queue)
    # The historical false terminal came from this shortcut.  The admitted
    # revival must pass through ordinary integration instead of repeating it.
    train._portal_projection_invalid_metadata_already_on_target = (
        lambda _request: True
    )

    def predicate(row: MergeRequest) -> bool:
        predicate_calls.append(row.request_id)
        return row.canonical_task_id == "canonical-false-completion"

    @contextmanager
    def processor_context(selected_train: MergeTrain):
        assert selected_train is train
        context_events.append("entered")
        try:
            yield
        finally:
            context_events.append("exited")

    result = train.recover_one_false_completion(
        request_id=request.request_id,
        request_filter=predicate,
        recovery_receipt=recovery_receipt,
        processor_context=processor_context,
        after_process=lambda _claimed, outcome: after_results.append(
            str(outcome.get("status") or "")
        ),
    )

    assert result is not None and result["status"] == "merged"
    assert predicate_calls == [request.request_id, request.request_id]
    assert context_events == ["entered", "exited"]
    assert after_results == ["merged"]
    settled = queue.get(request.request_id)
    assert settled is not None and settled.status == "completed"
    assert len(settled.metadata["false_completion_revivals"]) == 1
    target = _git(repo, "rev-parse", "refs/heads/main")
    assert _git(repo, "merge-base", "--is-ancestor", candidate, target) == ""
    assert _git(repo, "show", f"{target}:recovered.txt") == "recovered"


def test_train_denies_false_completion_recovery_when_candidate_is_integrated(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/already-integrated",
        task_id="ALREADY-INTEGRATED",
        canonical_task_id="canonical-already-integrated",
        canonical_task_key="canonical-already-integrated",
        commit_sha=candidate,
        metadata={"changed_submodule_paths": []},
    )
    claimed = queue.dequeue(consumer_id="merge-train:seed-integrated")
    assert claimed is not None
    queue.complete(
        claimed,
        metadata={"status": "merged", "accepted": True},
    )
    completed = queue.get(request.request_id)
    assert completed is not None and completed.status == "completed"
    recovery_receipt = _false_completion_recovery_receipt(
        completed,
        observed_target_commit=_git(repo, "rev-parse", "refs/heads/main"),
    )

    result = MergeTrain(repo, queue).recover_one_false_completion(
        request_id=request.request_id,
        request_filter=lambda _row: True,
        recovery_receipt=recovery_receipt,
    )

    assert result is None
    durable = queue.get(request.request_id)
    assert durable is not None and durable.status == "completed"
    assert "false_completion_revivals" not in durable.metadata


def test_train_denies_false_completion_when_negative_git_observation_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/unknown-observation")
    (repo / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/unknown-observation",
        task_id="UNKNOWN-OBSERVATION",
        canonical_task_id="canonical-unknown-observation",
        canonical_task_key="canonical-unknown-observation",
        commit_sha=candidate,
        metadata={"baseline_ref": base, "changed_submodule_paths": []},
    )
    claimed = queue.dequeue(consumer_id="merge-train:seed-unknown")
    assert claimed is not None
    queue.complete(claimed, metadata={"status": "merged", "accepted": True})
    completed = queue.get(request.request_id)
    assert completed is not None
    receipt = _false_completion_recovery_receipt(
        completed,
        observed_target_commit=_git(repo, "rev-parse", "refs/heads/main"),
    )
    train = MergeTrain(repo, queue)
    original_git = train._git

    def unknown_ancestry(*arguments: str) -> subprocess.CompletedProcess[str]:
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return subprocess.CompletedProcess(
                ["git", *arguments],
                128,
                "",
                "synthetic Git observation failure",
            )
        return original_git(*arguments)

    monkeypatch.setattr(train, "_git", unknown_ancestry)
    result = train.recover_one_false_completion(
        request_id=request.request_id,
        request_filter=lambda _row: True,
        recovery_receipt=receipt,
    )

    assert result is None
    durable = queue.get(request.request_id)
    assert durable is not None and durable.status == "completed"
    assert "false_completion_revivals" not in durable.metadata


def test_train_immediately_recovers_a_claim_abandoned_by_dead_consumer(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/ref-016",
        task_id="REF-016",
        canonical_task_id="canonical-ref-016",
        commit_sha=candidate,
    )
    abandoned = queue.dequeue(consumer_id="merge-train:999999:dead")
    assert abandoned is not None and abandoned.status == "processing"
    callbacks: list[str] = []

    result = MergeTrain(
        repo,
        queue,
        merge_callback=lambda claimed: callbacks.append(claimed.request_id) or {"merged": True},
    ).run_once()

    assert result is not None and result["status"] == "merged"
    assert callbacks == [request.request_id]
    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"
    assert stored.attempt == 2
    assert stored.failure_count == 1


def test_bounded_train_failures_create_durable_quarantine_receipt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "switch", "-c", "implementation/broken")
    (repo / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "candidate")
    commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    queue = MergeQueue(tmp_path / "queue", max_attempts=2)
    request = queue.enqueue(
        branch_name="implementation/broken",
        task_id="BROKEN-1",
        canonical_task_id="canonical-broken",
        commit_sha=commit,
    )
    train = MergeTrain(
        repo,
        queue,
        max_attempts=2,
        merge_callback=lambda _request: {"merged": False, "reason": "synthetic_conflict"},
    )

    # Advance the target independently so the candidate is not already merged.
    (repo / "advance.txt").write_text("advance\n", encoding="utf-8")
    _git(repo, "add", "advance.txt")
    _git(repo, "commit", "-m", "advance")
    assert train.run_once()["status"] == "retrying"  # type: ignore[index]
    terminal = train.run_once()

    assert terminal is not None
    assert terminal["status"] == "quarantined"
    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "quarantined"
    receipt = queue.quarantine_dir / f"{request.request_id}.json"
    assert receipt.exists()
    assert json.loads(receipt.read_text(encoding="utf-8"))["receipt_type"] == "merge_quarantine"
    assert queue.pending_count() == 0


def test_one_conflict_fingerprint_has_one_active_resolver_attempt(tmp_path: Path) -> None:
    registry = MergeResolverRegistry(tmp_path / "resolver", max_attempts=2)
    event = {
        "canonical_task_id": "canonical-ref-038",
        "branch": "implementation/ref-038",
        "target_branch": "main",
        "source_commit": "a" * 40,
        "target_commit": "b" * 40,
        "reason": "rebase_conflict",
        "unmerged_paths": ["one.py", "two.py"],
        "timestamp": "volatile-1",
    }
    same_conflict = {**event, "timestamp": "volatile-2", "attempt": 99}

    assert conflict_fingerprint(event) == conflict_fingerprint(same_conflict)
    first = registry.acquire(event, owner_id="resolver-a")
    assert first is not None
    assert registry.acquire(same_conflict, owner_id="resolver-b") is None
    registry.release(first, succeeded=False, error="still conflicted")
    second = registry.acquire(same_conflict, owner_id="resolver-b")
    assert second is not None and second.attempt == 2
    receipt = registry.release(second, succeeded=False, error="still conflicted")
    assert receipt is not None and receipt.exists()
    assert registry.status(event)["state"] == "quarantined"


def test_isolated_daemon_lanes_share_only_one_target_scoped_train(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    todo = repo / "tasks.md"
    todo.write_text("## REF-038 Merge train\n\n- Status: todo\n", encoding="utf-8")

    def daemon(lane: str) -> PortalImplementationDaemon:
        state_dir = tmp_path / lane
        return PortalImplementationDaemon(
            todo_path=todo,
            state_path=state_dir / "state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            repo_root=repo,
            task_header_prefix="## REF-",
        )

    lane_a = daemon("lane-a")
    lane_b = daemon("lane-b")
    assert lane_a.merge_queue.database_path == lane_b.merge_queue.database_path
    assert lane_a.merge_queue_dir.parent == repo / ".git" / "agent-merge-trains"
    assert lane_a.merge_queue.target_branch == "main"

    task = PortalTask(
        task_id="REF-038",
        title="Merge train",
        status="todo",
        completion="manual",
        priority="P0",
        track="g9",
    )
    commit = _git(repo, "rev-parse", "HEAD")
    identity = lane_a._identity_for_task(task)
    request, result = lane_a._enqueue_merge_candidate(
        branch_name="implementation/ref-038",
        implementation_commit=commit,
        baseline_ref=commit,
        worktree_path=repo,
        task=task,
        attempt=1,
    )

    assert result["queued"] is True
    assert request.commit_sha == commit
    assert request.target_repository_id == lane_a.merge_target_repository_id
    assert request.target_branch == "main"
    assert request.has_target_binding is True
    assert lane_b.merge_queue.has_pending_for_task(identity.canonical_task_cid, commit_sha=commit)

    _git(repo, "branch", "benchmark/semantic-roundtrip")
    benchmark_state = tmp_path / "benchmark-lane"
    benchmark_lane = PortalImplementationDaemon(
        todo_path=todo,
        state_path=benchmark_state / "state.json",
        strategy_path=benchmark_state / "strategy.json",
        events_path=benchmark_state / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## REF-",
        merge_target_branch="benchmark/semantic-roundtrip",
    )
    assert (
        benchmark_lane.merge_queue.database_path
        != lane_a.merge_queue.database_path
    )
    assert benchmark_lane.merge_queue.target_branch == (
        "benchmark/semantic-roundtrip"
    )
    assert benchmark_lane.merge_queue.pending_count() == 0
    foreign_request, _foreign_result = benchmark_lane._enqueue_merge_candidate(
        branch_name="implementation/ref-038-benchmark",
        implementation_commit=commit,
        baseline_ref=commit,
        worktree_path=repo,
        task=task,
        attempt=1,
    )

    rejected = lane_a._merge_train_callback(foreign_request)

    assert rejected["reason"] == "merge_target_binding_mismatch"
    assert rejected["expected_target_branch"] == "main"
    assert rejected["actual_target_branch"] == "benchmark/semantic-roundtrip"
    assert lane_a.merge_queue.pending_count() == 1
    assert benchmark_lane.merge_queue.pending_count() == 1


def test_cross_lane_completion_without_authority_policy_fails_closed(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "branch", "benchmark/semantic-roundtrip")
    _git(repo, "branch", "implementation/ref-039")
    producer_todo = repo / "producer-tasks.md"
    consumer_todo = repo / "consumer-tasks.md"
    task_text = "## REF-039 Cross-lane completion\n\n- Status: todo\n"
    producer_todo.write_text(task_text, encoding="utf-8")
    consumer_todo.write_text(task_text, encoding="utf-8")

    def daemon(todo_path: Path, lane: str) -> PortalImplementationDaemon:
        state_dir = tmp_path / lane
        return PortalImplementationDaemon(
            todo_path=todo_path,
            state_path=state_dir / "state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            repo_root=repo,
            task_header_prefix="## REF-",
            merge_target_branch="benchmark/semantic-roundtrip",
            worktree_pool_enabled=False,
        )

    producer = daemon(producer_todo, "producer")
    consumer = daemon(consumer_todo, "consumer")
    task = PortalTask(
        task_id="REF-039",
        title="Cross-lane completion",
        status="todo",
        completion="manual",
        priority="P0",
        track="g9",
    )
    commit = _git(repo, "rev-parse", "HEAD")
    request, _result = producer._enqueue_merge_candidate(
        branch_name="implementation/ref-039",
        implementation_commit=commit,
        baseline_ref=commit,
        worktree_path=None,
        task=task,
        attempt=1,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
            "selection": {"scope": "pre_merge"},
        },
    )

    result = consumer._merge_train_callback(request)

    assert result["merged"] is False
    assert result["reason"] == (
        "cross_board_manual_completion_authority_metadata_invalid"
    )
    assert result["request_todo_path"] == str(producer_todo)
    assert result["consumer_todo_path"] == str(consumer_todo)
    assert "- Status: todo" in producer_todo.read_text(encoding="utf-8")
    assert consumer.merge_queue.target_branch == "benchmark/semantic-roundtrip"


def test_same_git_worktree_todo_path_is_not_cross_board(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    todo = repo / "tasks.md"
    todo.write_text(
        "## REF-042 Same-board worktree copy\n\n- Status: todo\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tasks.md")
    _git(repo, "commit", "-m", "todo")
    worktree = tmp_path / "linked-worktree"
    _git(repo, "worktree", "add", str(worktree), "HEAD")
    worktree_todo = worktree / "tasks.md"
    other_todo = repo / "other-tasks.md"
    other_todo.write_text(
        "## OTHER-001 Foreign board\n\n- Status: todo\n",
        encoding="utf-8",
    )

    daemon = PortalImplementationDaemon(
        todo_path=todo,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## REF-",
        worktree_pool_enabled=False,
    )

    assert daemon._same_board_todo_path(worktree_todo) is True
    assert daemon._merge_request_is_cross_board(worktree_todo, {}) is False
    assert daemon._merge_request_is_cross_board(
        other_todo,
        {"task": {"task_id": "OTHER-001"}},
    ) is True


def test_merge_cleanup_failure_keeps_merged_and_completes_board(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    todo = repo / "tasks.md"
    todo.write_text(
        "## REF-041 Cleanup after merge\n\n- Status: todo\n- Completion: manual\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tasks.md")
    _git(repo, "commit", "-m", "todo")
    baseline = _git(repo, "rev-parse", "HEAD")
    branch = "implementation/ref-041"
    _git(repo, "checkout", "-b", branch)
    (repo / "feature.txt").write_text("landed\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "feature")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")

    daemon = PortalImplementationDaemon(
        todo_path=todo,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## REF-",
        worktree_pool_enabled=False,
    )
    task = daemon._load_tasks()[0]
    request, queued = daemon._enqueue_merge_candidate(
        branch_name=branch,
        implementation_commit=candidate,
        baseline_ref=baseline,
        worktree_path=tmp_path / "leftover-worktree",
        task=task,
        attempt=1,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
            "selection": {"scope": "pre_merge"},
        },
    )
    assert queued.get("queued") is True

    def integrate(*_args, **_kwargs):
        _git(repo, "merge", "--ff-only", branch)
        return {
            "merged": True,
            "returncode": 0,
            "merge_commit": _git(repo, "rev-parse", "HEAD"),
        }

    monkeypatch.setattr(daemon, "_merge_branch_to_main", integrate)
    monkeypatch.setattr(
        daemon,
        "_cleanup_merged_worktree",
        lambda *_args, **_kwargs: {
            "cleaned": False,
            "reason": "worktree_busy",
        },
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is True
    assert result.get("cleanup_failed") is True
    assert result.get("reason") != "merge_cleanup_failed"
    assert "- Status: completed" in todo.read_text(encoding="utf-8")
    assert _git(repo, "merge-base", "--is-ancestor", candidate, "main") == ""


def test_merge_train_rejects_a_mismatched_bound_queue_target(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "branch", "benchmark/semantic-roundtrip")
    repository_id = checkout_repository_id(repo)
    benchmark_queue = MergeQueue(
        tmp_path / "benchmark-queue",
        target_repository_id=repository_id,
        target_branch="benchmark/semantic-roundtrip",
        require_target_binding=True,
    )
    foreign_repo_queue = MergeQueue(
        tmp_path / "foreign-repo-queue",
        target_repository_id="repository:foreign",
        target_branch="main",
        require_target_binding=True,
    )

    with pytest.raises(ValueError, match="branch differs"):
        MergeTrain(repo, benchmark_queue, target_branch="main")
    with pytest.raises(ValueError, match="repository differs"):
        MergeTrain(repo, foreign_repo_queue, target_branch="main")


def test_bound_merge_train_receipts_are_namespaced_by_exact_target(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "branch", "Feature")
    _git(repo, "branch", "feature")
    repository_id = checkout_repository_id(repo)
    queue_path = tmp_path / "shared-queue"
    upper = MergeTrain(
        repo,
        MergeQueue(
            queue_path,
            target_repository_id=repository_id,
            target_branch="Feature",
            require_target_binding=True,
        ),
        target_branch="Feature",
    )
    lower = MergeTrain(
        repo,
        MergeQueue(
            queue_path,
            target_repository_id=repository_id,
            target_branch="feature",
            require_target_binding=True,
        ),
        target_branch="feature",
    )

    assert upper._dedupe_key("canonical-task", "a" * 40) != (
        lower._dedupe_key("canonical-task", "a" * 40)
    )
