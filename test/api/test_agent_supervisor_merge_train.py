from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.merge_queue import MergeQueue, MergeRequest
from ipfs_accelerate_py.agent_supervisor.merge_resolver import (
    MergeResolverRegistry,
    conflict_fingerprint,
)
from ipfs_accelerate_py.agent_supervisor.merge_train import MergeTrain
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


def test_isolated_daemon_lanes_enqueue_into_one_repo_wide_train(tmp_path: Path) -> None:
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
    assert lane_a.merge_queue_dir.parent == repo / ".git"

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
    assert lane_b.merge_queue.has_pending_for_task(identity.canonical_task_cid, commit_sha=commit)
