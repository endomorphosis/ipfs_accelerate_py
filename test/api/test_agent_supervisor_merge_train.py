from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue, MergeRequest
from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import (
    MergeResolverRegistry,
    conflict_fingerprint,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseTaskAttempt,
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


class _DatabaseProjectionTaskSource:
    def __init__(self, record: object) -> None:
        self.record = record

    def get_task(self, task_cid: str) -> object | None:
        return (
            self.record
            if task_cid == str(getattr(self.record, "task_cid", ""))
            else None
        )

    def snapshot(self) -> object:
        return SimpleNamespace(repository_tree_id="tree:merge-continuation")


def _database_projection_record(
    *, task_cid: str = "task:cid:ref-040", revision: int = 1
) -> SimpleNamespace:
    return SimpleNamespace(
        task_cid=task_cid,
        task_alias="REF-040",
        goal_cid="goal:ref-040",
        plan_cid="plan:merge-continuation",
        revision=revision,
        priority="P0",
        dependencies=(),
        outputs=({"path": "base.txt"},),
        validations=(),
        acceptance=({"criterion": "The exact candidate is integrated"},),
        body={
            "objective": "Continue one exact database task merge",
            "completion": "auto",
            "track": "implementation",
        },
    )


def _database_projection_attempt(
    *, attempt_id: str, claim_id: str, task_cid: str, attempt_number: int
) -> DatabaseTaskAttempt:
    return DatabaseTaskAttempt(
        attempt_id=attempt_id,
        claim_id=claim_id,
        task_cid=task_cid,
        task_alias="REF-040",
        attempt_number=attempt_number,
        owner_session_id="session:merge-continuation",
        fencing_token=attempt_number,
        fence_epoch=1,
        lease_id=f"lease:{attempt_number}",
        committed_phase="claimed",
        status="running",
        started_at_ms=attempt_number,
    )


def _database_projection_daemon(
    *,
    repo: Path,
    attempt_root: Path,
    merge_queue_dir: Path,
    attempt: DatabaseTaskAttempt,
    record: SimpleNamespace,
) -> tuple[PortalImplementationDaemon, object, dict[str, object]]:
    bridge = DatabasePortalExecutionBridge(
        task_source=_DatabaseProjectionTaskSource(record),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: None,
        repository_root=repo,
        task_header_prefix="## REF-",
    )
    paths, binding = bridge._ensure_attempt_projection(attempt, record)
    daemon = PortalImplementationDaemon(
        todo_path=paths.task_projection,
        state_path=paths.state,
        strategy_path=paths.strategy,
        events_path=paths.events,
        repo_root=repo,
        task_header_prefix="## REF-",
        merge_target_branch="main",
        merge_queue_dir=merge_queue_dir,
        worktree_pool_enabled=False,
    )
    return daemon, paths, dict(binding)


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


def test_database_portal_retry_continues_merge_into_current_projection(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    merge_queue_dir = tmp_path / "merge-queue"
    task_cid = "task:cid:ref-040"
    producer_attempt = _database_projection_attempt(
        attempt_id="attempt:producer",
        claim_id="claim:producer",
        task_cid=task_cid,
        attempt_number=1,
    )
    consumer_attempt = _database_projection_attempt(
        attempt_id="attempt:consumer",
        claim_id="claim:consumer",
        task_cid=task_cid,
        attempt_number=2,
    )
    producer, producer_paths, producer_binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=merge_queue_dir,
        attempt=producer_attempt,
        record=_database_projection_record(revision=2),
    )
    consumer, consumer_paths, consumer_binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=merge_queue_dir,
        attempt=consumer_attempt,
        record=_database_projection_record(revision=4),
    )
    _git(repo, "branch", "implementation/ref-040")
    task = producer._load_tasks()[0]
    commit = _git(repo, "rev-parse", "HEAD")
    request, _queued = producer._enqueue_merge_candidate(
        branch_name="implementation/ref-040",
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

    assert result.get("merged") is True or result.get("already_merged") is True
    continuation = result["database_portal_merge_continuation"]
    assert continuation["task_id"] == "REF-040"
    assert continuation["task_cid"] == task_cid
    assert continuation["producer_binding_id"] == producer_binding["binding_id"]
    assert continuation["consumer_binding_id"] == consumer_binding["binding_id"]
    assert "- Status: ready" in producer_paths.task_projection.read_text(
        encoding="utf-8"
    )
    assert "- Status: completed" in consumer_paths.task_projection.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "tamper",
    ("missing_binding", "projection_authority", "task_cid_mismatch"),
)
def test_database_portal_retry_continuation_fails_closed_on_binding_mismatch(
    tmp_path: Path,
    tamper: str,
) -> None:
    repo = _repo(tmp_path)
    merge_queue_dir = tmp_path / "merge-queue"
    producer_attempt = _database_projection_attempt(
        attempt_id="attempt:producer",
        claim_id="claim:producer",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    consumer_task_cid = (
        "task:cid:other" if tamper == "task_cid_mismatch" else "task:cid:ref-040"
    )
    consumer_attempt = _database_projection_attempt(
        attempt_id="attempt:consumer",
        claim_id="claim:consumer",
        task_cid=consumer_task_cid,
        attempt_number=2,
    )
    producer, producer_paths, _producer_binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=merge_queue_dir,
        attempt=producer_attempt,
        record=_database_projection_record(revision=2),
    )
    consumer, consumer_paths, _consumer_binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=merge_queue_dir,
        attempt=consumer_attempt,
        record=_database_projection_record(
            task_cid=consumer_task_cid,
            revision=4,
        ),
    )
    if tamper == "missing_binding":
        consumer_paths.binding.unlink()
    elif tamper == "projection_authority":
        binding = json.loads(consumer_paths.binding.read_text(encoding="utf-8"))
        binding.pop("binding_id")
        binding["projection_authority"] = True
        binding["binding_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                binding,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        consumer_paths.binding.write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _git(repo, "branch", "implementation/ref-040")
    task = producer._load_tasks()[0]
    commit = _git(repo, "rev-parse", "HEAD")
    request, _queued = producer._enqueue_merge_candidate(
        branch_name="implementation/ref-040",
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
    target_before = _git(repo, "rev-parse", "main")

    result = consumer._merge_train_callback(request)

    assert result["merged"] is False
    assert result["reason"] in {
        "cross_board_manual_completion_authority_metadata_invalid",
        "cross_board_manual_completion_authority_metadata_missing",
        "merge_target_binding_mismatch",
        "merge_candidate_primary_identity_mismatch",
    }
    assert _git(repo, "rev-parse", "main") == target_before
    assert "- Status: ready" in producer_paths.task_projection.read_text(
        encoding="utf-8"
    )
    assert "- Status: ready" in consumer_paths.task_projection.read_text(
        encoding="utf-8"
    )


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


def test_queue_claim_pending_request_never_claims_a_fairer_foreign_request(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    fairer = queue.enqueue(
        branch_name="implementation/fairer",
        task_id="FAIRER",
        priority="P0",
        commit_sha="a" * 40,
    )
    selected = queue.enqueue(
        branch_name="implementation/exact",
        task_id="EXACT",
        priority="P3",
        commit_sha="b" * 40,
    )

    claimed = queue.claim_pending_request(
        selected.request_id,
        consumer_id="request-routed-recovery",
    )

    assert claimed is not None
    assert claimed.request_id == selected.request_id
    assert claimed.consumer_id == "request-routed-recovery"
    assert queue.get(fairer.request_id).status == "pending"
    assert [item.request_id for item in queue.pending_requests()] == [
        fairer.request_id
    ]


def test_bound_quarantine_snapshot_filters_target_before_limit(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    producer = MergeQueue(queue_dir, max_queue_size=32)
    for index in range(6):
        foreign = producer.enqueue(
            branch_name=f"implementation/foreign-{index}",
            task_id=f"FOREIGN-{index}",
            commit_sha=f"{index + 1:040x}",
            target_repository_id="repository:foreign",
            target_branch="main",
        )
        producer.quarantine(foreign, reason="foreign terminal row")
    selected = producer.enqueue(
        branch_name="implementation/selected-target",
        task_id="SELECTED-TARGET",
        commit_sha="f" * 40,
        target_repository_id="repository:selected",
        target_branch="main",
    )
    producer.quarantine(
        selected,
        reason="post_merge_declared_outputs_missing",
    )
    consumer = MergeQueue(
        queue_dir,
        target_repository_id="repository:selected",
        target_branch="main",
        require_target_binding=True,
    )

    visible = consumer.quarantined_requests(limit=1)

    assert [request.request_id for request in visible] == [
        selected.request_id
    ]


def test_completed_recovery_snapshot_filters_and_paginates_before_limit(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id="repository:selected",
        target_branch="main",
        require_target_binding=True,
    )
    completion_schema = "repair-completion@test"
    completion_reason = "post_merge_declared_outputs_repaired"

    def complete(task_id: str, *, repair: bool) -> object:
        request = queue.enqueue(
            branch_name=f"implementation/{task_id.casefold()}",
            task_id=task_id,
            commit_sha=(task_id[-1].casefold() * 40),
        )
        claimed = queue.claim_pending_request(
            request.request_id,
            consumer_id="fixture",
        )
        assert claimed is not None
        queue.complete(
            claimed,
            metadata=(
                {
                    "schema": completion_schema,
                    "reason": completion_reason,
                }
                if repair
                else {"schema": "ordinary-completion@test"}
            ),
        )
        stored = queue.get(request.request_id)
        assert stored is not None
        return stored

    older = complete("TASK-A", repair=True)
    complete("TASK-B", repair=False)
    newer = complete("TASK-C", repair=True)

    first_page = queue.completed_requests(
        limit=1,
        completion_schema=completion_schema,
        completion_reason=completion_reason,
    )
    second_page = queue.completed_requests(
        limit=1,
        completion_schema=completion_schema,
        completion_reason=completion_reason,
        before_request_id=newer.request_id,
    )

    assert [request.request_id for request in first_page] == [
        newer.request_id
    ]
    assert [request.request_id for request in second_page] == [
        older.request_id
    ]


def test_active_recovery_snapshot_keyset_reaches_later_same_target_row(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id="repository:selected",
        target_branch="main",
        require_target_binding=True,
    )
    rows = []
    for index in range(3):
        request = queue.enqueue(
            branch_name=f"implementation/recovery-{index}",
            task_id=f"RECOVERY-{index}",
            commit_sha=f"{index + 1:040x}",
        )
        claimed = queue.claim_pending_request(
            request.request_id,
            consumer_id="fixture",
        )
        assert claimed is not None
        queue.quarantine(
            claimed,
            reason="post_merge_declared_outputs_missing",
        )
        stored = queue.get(request.request_id)
        assert stored is not None
        rows.append(stored)

    first_page = queue.quarantined_requests(
        limit=2,
        after_request_id="",
    )
    second_page = queue.quarantined_requests(
        limit=2,
        after_request_id=first_page[-1].request_id,
    )

    assert [request.request_id for request in first_page] == [
        rows[0].request_id,
        rows[1].request_id,
    ]
    assert [request.request_id for request in second_page] == [
        rows[2].request_id
    ]


def test_recover_one_integrated_quarantine_processes_only_filtered_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    repository_id = checkout_repository_id(repo)
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="main",
        require_target_binding=True,
    )
    selected = queue.enqueue(
        branch_name="implementation/selected",
        task_id="SELECTED",
        canonical_task_id="task:cid:selected",
        commit_sha=_git(repo, "rev-parse", "HEAD"),
        metadata={"changed_submodule_paths": []},
    )
    unrelated = queue.enqueue(
        branch_name="implementation/unrelated",
        task_id="UNRELATED",
        canonical_task_id="task:cid:unrelated",
        commit_sha=_git(repo, "rev-parse", "HEAD"),
        metadata={"changed_submodule_paths": []},
    )
    queue.quarantine(
        selected,
        reason="post_merge_declared_outputs_missing",
    )
    queue.quarantine(
        unrelated,
        reason="post_merge_declared_outputs_missing",
    )
    candidate = selected.commit_sha
    repair_commit = _git(repo, "rev-parse", "HEAD")

    train = MergeTrain(
        repo,
        queue,
        merge_callback=lambda request: {
            "merged": True,
            "reason": "post_merge_declared_outputs_repaired",
            "post_merge_declared_output_repair": {
                "passed": True,
                "reason": "post_merge_declared_outputs_repaired",
                "receipt": {
                    "schema": "repair@test",
                    "task_ids": [request.task_id],
                    "candidate_commit": request.commit_sha,
                    "repair_commit": repair_commit,
                    "receipt_id": "receipt:test",
                },
            },
        },
    )
    monkeypatch.setattr(
        queue,
        "pending_requests",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact recovery must not rescan pending rows")
        ),
    )
    monkeypatch.setattr(
        queue,
        "quarantined_requests",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact recovery must not rescan quarantined rows")
        ),
    )

    result = train.recover_one_integrated_quarantine(
        request_filter=lambda request: request.request_id
        == selected.request_id,
        request_id=selected.request_id,
    )

    assert result is not None
    assert result["status"] == "merged"
    completed = queue.get(selected.request_id)
    assert completed is not None and completed.status == "completed"
    assert completed.metadata["completion"]["candidate_commit"] == candidate
    untouched = queue.get(unrelated.request_id)
    assert untouched is not None and untouched.status == "quarantined"


def test_invalid_authority_metadata_quarantine_settles_when_outputs_on_target(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "switch", "-c", "implementation/side")
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    _git(repo, "add", "side.txt")
    _git(repo, "commit", "-m", "side")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    head_before = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/side",
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "todo_path": str(tmp_path / "attempts" / "x" / "task-projection.md"),
            "completion_task_cids": {"REF-040": "task:cid:ref-040"},
            "manual_completion_authority_task_ids": [],
            "manual_completion_authority_required_task_ids": [],
            "manual_completion_authority_epoch_id": "",
            "manual_completion_authority_revocation_generation": 0,
            "manual_completion_authority_context_id": "baguqeera-invalid",
            "task": {
                "task_id": "REF-040",
                "outputs": ["base.txt"],
            },
            "changed_submodule_paths": [],
        },
    )
    claimed = queue.dequeue(consumer_id="merge-train:test")
    assert claimed is not None
    queue.quarantine(
        claimed,
        reason="cross_board_manual_completion_authority_metadata_invalid",
    )

    callback_calls: list[str] = []

    def fail_closed(_request: object) -> dict[str, object]:
        callback_calls.append("called")
        return {
            "attempted": False,
            "merged": False,
            "returncode": 2,
            "reason": "cross_board_manual_completion_authority_metadata_invalid",
        }

    train = MergeTrain(repo, queue, merge_callback=fail_closed)
    result = train.run_once()

    assert result is not None
    assert result.get("status") == "already_merged"
    assert result.get("already_merged") is True
    assert result.get("reason") == "declared_outputs_already_on_target"
    assert callback_calls == []
    assert _git(repo, "rev-parse", "HEAD") == head_before
    side_probe = subprocess.run(
        ["git", "cat-file", "-e", "HEAD:side.txt"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert side_probe.returncode != 0
    completed = queue.get(request.request_id)
    assert completed is not None
    assert completed.status == "completed"


def test_custom_train_state_dirs_share_one_queue_consumer_lease(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/canonical-consumer-lease",
        task_id="CANONICAL-LEASE",
        commit_sha=_git(repo, "rev-parse", "HEAD"),
    )
    first = MergeTrain(repo, queue, state_dir=tmp_path / "lane-a")
    second = MergeTrain(repo, queue, state_dir=tmp_path / "lane-b")

    assert first.consumer_lock_path == second.consumer_lock_path
    with first._consumer_lease() as acquired:
        assert acquired is True
        claimed = queue.claim_pending_request(
            request.request_id,
            consumer_id=first.owner_id,
        )
        assert claimed is not None
        assert second.run_once() is None
        assert queue.owns_claim(claimed, consumer_id=first.owner_id)

    recovered = second.run_once()

    assert recovered is not None
    assert recovered["status"] == "already_merged"
    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"
    assert second.run_once() is None


def test_queue_quarantine_preserves_callback_failure_metadata(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/missing-outputs",
        task_id="MISSING-OUTPUTS",
        commit_sha=candidate,
        metadata={"changed_submodule_paths": []},
    )
    claimed = queue.dequeue(consumer_id="merge-train:test")
    assert claimed is not None and claimed.request_id == request.request_id
    failure = {
        "status": "quarantined",
        "reason": "post_merge_declared_outputs_missing",
        "merge_result": {
            "reason": "post_merge_declared_outputs_missing",
            "automatic_repair_attempted": True,
            "automatic_repair_terminal": True,
            "repair_reason": "declared_output_repair_conflict",
        },
    }

    MergeTrain._call_queue_failure(
        queue.quarantine,
        claimed,
        "post_merge_declared_outputs_missing",
        failure,
    )

    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "quarantined"
    assert stored.metadata["quarantine"] == failure


def test_terminal_missing_output_repair_quarantine_is_not_revived(
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
        branch_name="implementation/missing-outputs",
        task_id="MISSING-OUTPUTS",
        commit_sha=candidate,
        metadata={"changed_submodule_paths": []},
    )
    claimed = queue.dequeue(consumer_id="merge-train:test")
    assert claimed is not None and claimed.request_id == request.request_id
    queue.quarantine(
        claimed,
        reason="post_merge_declared_outputs_missing",
        metadata={
            "merge_result": {
                "reason": "post_merge_declared_outputs_missing",
                "automatic_repair_attempted": True,
                "automatic_repair_terminal": True,
                "repair_reason": "declared_output_repair_conflict",
            }
        },
    )
    train = MergeTrain(repo, queue)
    quarantined = queue.get(request.request_id)
    assert quarantined is not None
    assert train._quarantined_candidate_is_integrated(quarantined) is True

    assert train._recover_integrated_quarantines() == 0

    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "quarantined"
    assert "revivals" not in stored.metadata


def test_declared_output_recovery_may_revive_terminal_missing_output_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        branch_name="implementation/missing-outputs",
        task_id="MISSING-OUTPUTS",
        commit_sha=candidate,
        metadata={"changed_submodule_paths": []},
    )
    claimed = queue.dequeue(consumer_id="merge-train:test")
    assert claimed is not None and claimed.request_id == request.request_id
    queue.quarantine(
        claimed,
        reason="post_merge_declared_outputs_missing",
        metadata={
            "merge_result": {
                "reason": "post_merge_declared_outputs_missing",
                "automatic_repair_attempted": True,
                "automatic_repair_terminal": True,
                "repair_reason": "declared_output_repair_conflict",
            }
        },
    )
    train = MergeTrain(repo, queue)
    quarantined = queue.get(request.request_id)
    assert quarantined is not None
    assert train._quarantine_auto_recovery_allowed(quarantined) is False
    assert (
        train._quarantine_auto_recovery_allowed(
            quarantined,
            allow_post_merge_declared_output_recovery=True,
        )
        is True
    )

    processed: list[str] = []

    def process_claimed(_request: object) -> dict[str, object]:
        current = queue.get(request.request_id)
        assert current is not None
        processed.append(str(current.status))
        return {"merged": True, "reason": "declared_output_recovery_fixture"}

    monkeypatch.setattr(train, "_process_claimed", process_claimed)
    result = train.recover_one_integrated_quarantine(
        request_id=request.request_id,
        allow_post_merge_declared_output_recovery=True,
    )
    assert result is not None
    assert processed
    assert train.recover_one_integrated_quarantine(
        request_id=request.request_id,
    ) is None


def test_portal_projection_cross_board_quarantine_revives_when_outputs_landed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "switch", "-c", "implementation/side")
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    _git(repo, "add", "side.txt")
    _git(repo, "commit", "-m", "side")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/side",
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "todo_path": str(tmp_path / "attempts" / "x" / "task-projection.md"),
            "completion_task_cids": {"REF-040": "task:cid:ref-040"},
            "task": {
                "task_id": "REF-040",
                "outputs": ["base.txt"],
            },
            "changed_submodule_paths": [],
        },
    )
    claimed = queue.dequeue(consumer_id="merge-train:test")
    assert claimed is not None
    queue.quarantine(
        claimed,
        reason="cross_board_manual_completion_authority_metadata_invalid",
    )
    train = MergeTrain(repo, queue)
    quarantined = queue.get(request.request_id)
    assert quarantined is not None
    assert train._quarantined_candidate_is_integrated(quarantined) is False
    assert train._quarantine_auto_recovery_allowed(quarantined) is True
    assert train._quarantine_may_auto_recover(quarantined) is True

    processed: list[str] = []

    def process_claimed(_request: object) -> dict[str, object]:
        current = queue.get(request.request_id)
        assert current is not None
        processed.append(str(current.status))
        return {
            "already_merged": True,
            "reason": "declared_outputs_already_on_target",
        }

    monkeypatch.setattr(train, "_process_claimed", process_claimed)
    result = train.recover_one_integrated_quarantine(
        request_id=request.request_id,
    )
    assert result is not None
    assert processed
