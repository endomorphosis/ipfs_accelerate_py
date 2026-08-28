from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_lock_metadata,
    checkout_mutation_lock_path,
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA,
    MergeQueue,
    MergeRequest,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import (
    MergeResolverRegistry,
    conflict_fingerprint,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
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
    *,
    task_cid: str = "task:cid:ref-040",
    task_alias: str = "REF-040",
    goal_cid: str = "goal:ref-040",
    plan_cid: str = "plan:merge-continuation",
    revision: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_cid=task_cid,
        task_alias=task_alias,
        goal_cid=goal_cid,
        plan_cid=plan_cid,
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
    *,
    attempt_id: str,
    claim_id: str,
    task_cid: str,
    attempt_number: int,
    task_alias: str = "REF-040",
) -> DatabaseTaskAttempt:
    return DatabaseTaskAttempt(
        attempt_id=attempt_id,
        claim_id=claim_id,
        task_cid=task_cid,
        task_alias=task_alias,
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


def _quarantine_invalid_authority_projection(
    *,
    queue: MergeQueue,
    tmp_path: Path,
    candidate: str,
) -> MergeRequest:
    request = queue.enqueue(
        branch_name="implementation/side",
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "todo_path": str(
                tmp_path / "attempts" / "x" / "task-projection.md"
            ),
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
    return request


def test_database_projection_checkout_lease_exemption_is_fail_closed(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    attempt = _database_projection_attempt(
        attempt_id="attempt:lease-classification",
        claim_id="claim:lease-classification",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=repo / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(),
    )

    # A verified projection is still fenced until Git proves it is disposable.
    assert daemon._todo_mutation_requires_checkout_lease() is True

    (repo / ".gitignore").write_text("attempts/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore database attempt projections")
    assert daemon._todo_mutation_requires_checkout_lease() is False

    relative = paths.task_projection.resolve().relative_to(repo).as_posix()
    daemon.implementation_protected_paths = (relative,)
    assert daemon._todo_mutation_requires_checkout_lease() is True
    daemon.implementation_protected_paths = ()

    binding = json.loads(paths.binding.read_text(encoding="utf-8"))
    binding["projection_authority"] = True
    paths.binding.write_text(
        json.dumps(binding, sort_keys=True),
        encoding="utf-8",
    )
    assert daemon._todo_mutation_requires_checkout_lease() is True


def test_database_projection_checkout_lease_exemption_rejects_nested_repo(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("attempts/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore database attempt projections")
    attempt = _database_projection_attempt(
        attempt_id="attempt:nested-repository",
        claim_id="claim:nested-repository",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=repo / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(),
    )
    assert daemon._todo_mutation_requires_checkout_lease() is False

    nested_repo = paths.task_projection.parent
    _git(nested_repo, "init", "-b", "nested")
    _git(nested_repo, "config", "user.name", "Nested Projection Test")
    _git(
        nested_repo,
        "config",
        "user.email",
        "nested-projection@example.invalid",
    )
    _git(nested_repo, "add", "task-projection.md")
    _git(nested_repo, "commit", "-m", "track nested projection")

    # The outer repository still reports the path ignored, but it must not
    # authorize a lease exemption for a tracked file in another checkout.
    assert daemon._todo_mutation_requires_checkout_lease() is True


def test_database_projection_completion_survives_foreign_checkout_lease(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("attempts/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore database attempt projections")
    task_cid = "task:cid:ref-040"
    attempt = _database_projection_attempt(
        attempt_id="attempt:lease-contention",
        claim_id="claim:lease-contention",
        task_cid=task_cid,
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=repo / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(task_cid=task_cid),
    )
    assert daemon._todo_mutation_requires_checkout_lease() is False

    lock_path = checkout_mutation_lock_path(repo)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_payload = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="FOREIGN-MERGE",
        branch="implementation/foreign",
        owner_script="",
        extra={"operation": "merge_train_callback"},
    )
    lock_path.write_text(
        json.dumps(lock_payload, sort_keys=True),
        encoding="utf-8",
    )

    result = daemon._mark_task_completed_in_todo(
        "REF-040",
        expected_task_cids={"REF-040": task_cid},
    )

    assert result["updated"] is True
    assert result["completion_receipts"] == [
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "member_completion_receipt@1"
            ),
            "task_id": "REF-040",
            "canonical_task_key": result["completion_receipts"][0][
                "canonical_task_key"
            ],
            "canonical_task_cid": task_cid,
            "board_namespace": "task-projection.md",
            "status": "succeeded",
        }
    ]
    assert daemon._todo_completion_is_durable(result) is True
    assert "- Status: completed" in paths.task_projection.read_text(
        encoding="utf-8"
    )
    # The ignored attempt projection is durable without a Git commit.  The
    # no-change commit probe therefore never contends with the foreign
    # checkout lease.
    assert result["commit_result"]["reason"] == "no_changes"
    assert json.loads(lock_path.read_text(encoding="utf-8")) == lock_payload


def test_synchronous_callback_completes_ignored_projection_under_foreign_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("attempts/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore database attempt projections")
    task_cid = "task:cid:ref-040"
    attempt = _database_projection_attempt(
        attempt_id="attempt:synchronous-lock-contention",
        claim_id="claim:synchronous-lock-contention",
        task_cid=task_cid,
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=repo / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(task_cid=task_cid),
    )
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/synchronous-lock")
    (repo / "base.txt").write_text(
        "synchronous callback with foreign lease\n",
        encoding="utf-8",
    )
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "synchronous lock candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    [task] = daemon._load_tasks()
    request, _queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/synchronous-lock",
        implementation_commit=candidate,
        baseline_ref=baseline,
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
    lock_path = checkout_mutation_lock_path(repo)
    lock_payload = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="FOREIGN-MERGE",
        branch="implementation/foreign",
        owner_script="",
        extra={"operation": "merge_train_callback"},
    )
    original_completion = daemon._mark_task_completed_in_todo

    def complete_while_foreign_lease_is_live(*args: object, **kwargs: object):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(lock_payload, sort_keys=True),
            encoding="utf-8",
        )
        return original_completion(*args, **kwargs)

    monkeypatch.setattr(
        daemon,
        "_mark_task_completed_in_todo",
        complete_while_foreign_lease_is_live,
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is True
    assert result["merge_reconciliation_receipt"]["recorded"] is True
    receipts = result["todo_update_result"]["completion_receipts"]
    assert {
        receipt["task_id"]: receipt["canonical_task_cid"]
        for receipt in receipts
    } == {"REF-040": task_cid}
    assert daemon._todo_completion_is_durable(
        result["todo_update_result"]
    ) is True
    assert "- Status: completed" in paths.task_projection.read_text(
        encoding="utf-8"
    )
    assert json.loads(lock_path.read_text(encoding="utf-8")) == lock_payload
    events = daemon._iter_merge_lifecycle_events()
    event_types = [event["type"] for event in events]
    assert event_types.index("worktree_reconciliation_candidate_queued") < (
        event_types.index("merge_reconciled")
    )
    assert event_types.index("merge_reconciled") < event_types.index(
        "todo_status_updated"
    )


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


def test_merge_train_filter_leaves_incompatible_request_pending(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    incompatible = queue.enqueue(
        branch_name="implementation/ref-foreign",
        task_id="REF-FOREIGN",
        canonical_task_id="task:cid:foreign",
        commit_sha=candidate,
    )
    compatible = queue.enqueue(
        branch_name="implementation/ref-active",
        task_id="REF-ACTIVE",
        canonical_task_id="task:cid:active",
        commit_sha=candidate,
    )
    train = MergeTrain(
        repo,
        queue,
        request_filter=lambda request: request.task_id == "REF-ACTIVE",
        merge_callback=lambda _request: {
            "already_merged": True,
            "reason": "test_candidate_already_integrated",
        },
    )

    result = train.run_once()

    assert result is not None
    assert result["task_id"] == "REF-ACTIVE"
    untouched = queue.get(incompatible.request_id)
    assert untouched is not None
    assert untouched.status == "pending"
    assert untouched.attempt == 1
    assert untouched.failure_count == 0
    completed = queue.get(compatible.request_id)
    assert completed is not None
    assert completed.status == "completed"


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
    target_branch = lane_a.resolved_merge_target_branch
    assert lane_a.merge_queue.database_path == lane_b.merge_queue.database_path
    assert lane_a.merge_queue_dir.parent == repo / ".git" / "agent-merge-trains"
    assert target_branch == "implementation/tasks"
    assert lane_a.merge_queue.target_branch == target_branch

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
    assert request.target_branch == target_branch
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
    assert rejected["expected_target_branch"] == target_branch
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

    assert consumer._merge_request_matches_active_lane(request) is True
    result = consumer._merge_train_callback(request)

    assert result["merged"] is False
    assert result["reason"] == (
        "cross_board_manual_completion_authority_metadata_invalid"
    )
    assert result["request_todo_path"] == str(producer_todo)
    assert result["consumer_todo_path"] == str(consumer_todo)
    assert "- Status: todo" in producer_todo.read_text(encoding="utf-8")
    assert consumer.merge_queue.target_branch == "benchmark/semantic-roundtrip"


def test_database_portal_wrong_lane_leaves_request_for_compatible_consumer(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    merge_queue_dir = tmp_path / "merge-queue"
    shared_goal = "goal:shared-board"
    shared_plan = "plan:shared-board"
    producer_attempt = _database_projection_attempt(
        attempt_id="attempt:ref-040",
        claim_id="claim:ref-040",
        task_cid="task:cid:ref-040",
        task_alias="REF-040",
        attempt_number=1,
    )
    wrong_lane_attempt = _database_projection_attempt(
        attempt_id="attempt:ref-041",
        claim_id="claim:ref-041",
        task_cid="task:cid:ref-041",
        task_alias="REF-041",
        attempt_number=1,
    )
    producer, _producer_paths, _producer_binding = (
        _database_projection_daemon(
            repo=repo,
            attempt_root=repo / "attempts",
            merge_queue_dir=merge_queue_dir,
            attempt=producer_attempt,
            record=_database_projection_record(
                task_cid="task:cid:ref-040",
                task_alias="REF-040",
                goal_cid=shared_goal,
                plan_cid=shared_plan,
                revision=2,
            ),
        )
    )
    wrong_lane, _wrong_paths, _wrong_binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=merge_queue_dir,
        attempt=wrong_lane_attempt,
        record=_database_projection_record(
            task_cid="task:cid:ref-041",
            task_alias="REF-041",
            goal_cid=shared_goal,
            plan_cid=shared_plan,
            revision=3,
        ),
    )
    _git(repo, "branch", "implementation/ref-040")
    task = producer._load_tasks()[0]
    commit = _git(repo, "rev-parse", "HEAD")
    request, queued = producer._enqueue_merge_candidate(
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
    assert queued["queued"] is True

    assert wrong_lane._merge_request_matches_active_lane(request) is False
    assert wrong_lane._consume_one_merge_candidate() is None
    deferred = wrong_lane.merge_queue.get(request.request_id)
    assert deferred is not None
    assert deferred.status == "pending"
    assert deferred.attempt == 1
    assert deferred.failure_count == 0

    result = producer._consume_one_merge_candidate()

    assert result is not None
    assert result["status"] in {"merged", "already_merged"}
    completed = producer.merge_queue.get(request.request_id)
    assert completed is not None
    assert completed.status == "completed"


def test_database_portal_projection_continues_into_shared_board(
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
    producer, _producer_paths, producer_binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=merge_queue_dir,
        attempt=producer_attempt,
        record=_database_projection_record(revision=2),
    )
    board = repo / "board.md"
    board.write_text(
        "## REF-040 Continue one exact database task merge\n\n"
        "- Status: todo\n\n"
        "## REF-041 Other task\n\n"
        "- Status: todo\n",
        encoding="utf-8",
    )
    consumer_state = tmp_path / "shared-board"
    consumer = PortalImplementationDaemon(
        todo_path=board,
        state_path=consumer_state / "state.json",
        strategy_path=consumer_state / "strategy.json",
        events_path=consumer_state / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## REF-",
        merge_target_branch=producer.resolved_merge_target_branch,
        merge_queue_dir=merge_queue_dir,
        worktree_pool_enabled=False,
    )
    task = producer._load_tasks()[0]
    commit = _git(repo, "rev-parse", "HEAD")
    request, _queued = producer._enqueue_merge_candidate(
        branch_name="implementation/ref-040-shared",
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

    assert result.get("already_merged") is True or result.get("merged") is True
    continuation = result["database_portal_merge_continuation"]
    assert continuation["task_id"] == "REF-040"
    assert continuation["task_cid"] == task_cid
    assert continuation["producer_binding_id"] == producer_binding["binding_id"]
    assert result.get("reason") != (
        "cross_board_manual_completion_authority_metadata_invalid"
    )


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
    assert train.portal_declared_outputs_match_target(quarantined) is True
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




def test_existing_declared_output_with_different_candidate_content_is_merged(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "switch", "-c", "implementation/replace-output")
    (repo / "base.txt").write_text("candidate output\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "replace declared output")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    target_before = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name="implementation/replace-output",
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "todo_path": str(
                tmp_path / "attempts" / "replace" / "task-projection.md"
            ),
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
    train = MergeTrain(repo, queue)

    assert train.portal_declared_outputs_match_target(request) is False
    result = train.run_once()

    assert result is not None
    assert result.get("status") == "merged"
    assert result.get("merged") is True
    assert result.get("reason") != "declared_outputs_already_on_target"
    assert result.get("mutation_short_circuited") is not True
    assert _git(repo, "rev-parse", "HEAD") != target_before
    assert (repo / "base.txt").read_text(encoding="utf-8") == (
        "candidate output\n"
    )
    completed = queue.get(request.request_id)
    assert completed is not None
    assert completed.status == "completed"


@pytest.mark.parametrize("mutation", ("mode", "deletion"))
def test_declared_output_mode_and_deletion_differences_are_merged(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo = _repo(tmp_path)
    branch = f"implementation/{mutation}-output"
    _git(repo, "switch", "-c", branch)
    if mutation == "mode":
        (repo / "base.txt").chmod(0o755)
        _git(repo, "add", "base.txt")
    else:
        (repo / "base.txt").unlink()
        _git(repo, "add", "-u", "base.txt")
    _git(repo, "commit", "-m", f"{mutation} declared output")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = queue.enqueue(
        branch_name=branch,
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "todo_path": str(
                tmp_path / "attempts" / mutation / "task-projection.md"
            ),
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
    train = MergeTrain(repo, queue)

    assert train.portal_declared_outputs_match_target(request) is False
    result = train.run_once()

    assert result is not None
    assert result.get("status") == "merged"
    assert result.get("reason") != "declared_outputs_already_on_target"
    if mutation == "mode":
        assert _git(repo, "ls-tree", "HEAD", "--", "base.txt").startswith(
            "100755 blob "
        )
    else:
        assert not (repo / "base.txt").exists()
    completed = queue.get(request.request_id)
    assert completed is not None
    assert completed.status == "completed"


def test_declared_output_comparison_is_config_independent_for_gitlinks(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    first_gitlink = _git(repo, "rev-parse", "HEAD")
    second_gitlink = _git(
        repo,
        "commit-tree",
        "HEAD^{tree}",
        "-p",
        "HEAD",
        "-m",
        "alternate gitlink object",
    )
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{first_gitlink},nested",
    )
    _git(repo, "commit", "-m", "target gitlink")
    _git(repo, "switch", "-c", "implementation/gitlink-output")
    _git(
        repo,
        "update-index",
        "--cacheinfo",
        f"160000,{second_gitlink},nested",
    )
    _git(repo, "commit", "-m", "candidate gitlink")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    _git(repo, "config", "diff.ignoreSubmodules", "all")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/gitlink-output",
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={"task": {"outputs": ["nested"]}},
    )
    train = MergeTrain(repo, queue)

    assert (
        train.portal_declared_outputs_match_commit_state(
            request,
            _git(repo, "rev-parse", "main"),
        )
        is False
    )


@pytest.mark.parametrize(
    "output",
    ("missing.txt", "../escape.txt", "./base.txt", "base\\path.txt"),
)
def test_declared_output_comparison_rejects_unproved_or_noncanonical_paths(
    tmp_path: Path,
    output: str,
) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="main",
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={"task": {"outputs": [output]}},
    )
    train = MergeTrain(repo, queue)

    assert (
        train.portal_declared_outputs_match_commit_state(request, candidate)
        is None
    )
    assert train.portal_declared_outputs_match_target(request) is False


def test_declared_output_and_ancestry_git_errors_remain_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    candidate = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="main",
        task_id="REF-040",
        canonical_task_id="task:cid:ref-040",
        commit_sha=candidate,
        metadata={"task": {"outputs": ["base.txt"]}},
    )
    train = MergeTrain(repo, queue)
    original_git = train._git

    def fail_tree_read(*args: str, **kwargs: object) -> object:
        if args and args[0] == "ls-tree":
            return subprocess.CompletedProcess(
                ["git", *args],
                128,
                stdout="",
                stderr="fixture tree failure",
            )
        return original_git(*args, **kwargs)

    monkeypatch.setattr(train, "_git", fail_tree_read)
    assert (
        train.portal_declared_outputs_match_commit_state(request, candidate)
        is None
    )

    monkeypatch.setattr(
        train,
        "_git",
        lambda *args, **_kwargs: subprocess.CompletedProcess(
            ["git", *args],
            128,
            stdout="",
            stderr="fixture ancestry failure",
        ),
    )
    assert train._is_ancestor_state(candidate, candidate) is None
    assert train._is_ancestor(candidate, candidate) is False
def test_database_portal_retry_continues_merge_into_current_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/ref-040")
    (repo / "base.txt").write_text(
        "database continuation candidate\n",
        encoding="utf-8",
    )
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "database continuation candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    task = producer._load_tasks()[0]
    request, queued = producer._enqueue_merge_candidate(
        branch_name="implementation/ref-040",
        implementation_commit=candidate,
        baseline_ref=baseline,
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
    producer._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "canonical_task_cid": task_cid,
            "attempt": 1,
            "returncode": 0,
            "attempt_consumed": True,
            "provider_dispatched": True,
            "branch": request.branch_name,
            "baseline_ref": baseline,
            "implementation_commit": candidate,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
            },
            "merge_result": dict(queued),
            "board_completion": {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            },
        },
    )
    [producer_source] = [
        event
        for event in producer._iter_merge_lifecycle_events()
        if event.get("type") == "implementation_finished"
    ]

    result = consumer._merge_train_callback(request)

    assert result.get("merged") is True
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
    consumer_events = [
        json.loads(line)
        for line in consumer_paths.events.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event.get("type") == "todo_status_updated"
        and event.get("task_id") == "REF-040"
        for event in consumer_events
    )
    [local_source] = [
        event
        for event in consumer_events
        if event.get("type") == "worktree_reconciliation_candidate_queued"
    ]
    [reconciled] = [
        event
        for event in consumer_events
        if event.get("type") == "merge_reconciled"
    ]
    assert consumer_events.index(local_source) < consumer_events.index(reconciled)
    assert not any(
        event.get("type") == "task_completed" for event in consumer_events
    )
    provenance = local_source["database_portal_merge_continuation_source"]
    assert (
        provenance["producer_completion_source_event_id"]
        == producer_source["event_id"]
    )
    assert provenance["producer_binding_id"] == producer_binding["binding_id"]
    assert provenance["consumer_binding_id"] == consumer_binding["binding_id"]
    assert provenance["producer_projection_path"] == str(
        producer_paths.task_projection.resolve()
    )
    assert provenance["producer_state_path"] == str(
        producer_paths.state.resolve()
    )
    assert provenance["producer_events_path"] == str(
        producer_paths.events.resolve()
    )
    assert reconciled["completion_source_event_id"] == local_source["event_id"]
    assert reconciled["database_portal_merge_continuation_source"] == provenance

    before_replay = consumer_paths.events.read_bytes()
    monkeypatch.setattr(
        consumer,
        "_completion_daemon_for_merge_request",
        lambda _metadata: pytest.fail(
            "sealed local replay reopened producer history"
        ),
    )
    replay = consumer._merge_train_callback(request)
    assert replay.get("merged") is True or replay.get("already_merged") is True
    assert replay["merge_reconciliation_receipt"]["replayed"] is True
    assert consumer_paths.events.read_bytes() == before_replay

    consumer.merge_queue.cancel(
        request.request_id,
        reason="direct_callback_already_integrated",
    )
    consumer.run_once()
    consumer_events = [
        json.loads(line)
        for line in consumer_paths.events.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event.get("type") == "task_completed"
        and event.get("task_id") == "REF-040"
        for event in consumer_events
    )
    [task_completed] = [
        event
        for event in consumer_events
        if event.get("type") == "task_completed"
        and event.get("task_id") == "REF-040"
    ]
    assert consumer_events.index(reconciled) < consumer_events.index(task_completed)
    producer_events = (
        [
            json.loads(line)
            for line in producer_paths.events.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        if producer_paths.events.exists()
        else []
    )
    assert not any(event.get("type") == "task_completed" for event in producer_events)


def test_delayed_schema_v3_callback_records_exact_reconciliation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    attempt = _database_projection_attempt(
        attempt_id="attempt:delayed-callback",
        claim_id="claim:delayed-callback",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=tmp_path / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(revision=2),
    )
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/delayed-callback")
    (repo / "base.txt").write_text("delayed candidate\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "delayed callback candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    task = daemon._load_tasks()[0]
    task_cid = daemon._canonical_ref(task)
    request, queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/delayed-callback",
        implementation_commit=candidate,
        baseline_ref=baseline,
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
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "canonical_task_cid": task_cid,
            "attempt": 1,
            "returncode": 0,
            "attempt_consumed": True,
            "provider_dispatched": False,
            "branch": request.branch_name,
            "baseline_ref": baseline,
            "implementation_commit": candidate,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
            },
            "merge_result": dict(queued),
            "board_completion": {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            },
        },
    )
    [source] = [
        event
        for event in daemon._iter_merge_lifecycle_events()
        if event.get("type") == "implementation_finished"
    ]

    result = daemon._merge_train_callback(request)

    assert result["merged"] is True
    assert result["target_commit"] == result["merge_commit"]
    events = daemon._iter_merge_lifecycle_events()
    [reconciled] = [
        event for event in events if event.get("type") == "merge_reconciled"
    ]
    assert events.index(source) < events.index(reconciled)
    assert not any(event.get("type") == "task_completed" for event in events)
    assert reconciled["resolved"] is True
    assert reconciled["task_id"] == task.task_id
    assert reconciled["canonical_task_cid"] == task_cid
    assert reconciled["attempt"] == 1
    assert reconciled["request_id"] == request.request_id
    assert reconciled["completion_source_event_id"] == source["event_id"]
    assert reconciled["baseline_ref"] == baseline
    assert reconciled["implementation_commit"] == candidate
    assert reconciled["landed_commit"] == candidate
    assert reconciled["merge_commit"] == result["merge_commit"]
    assert reconciled["target_commit"] == result["merge_commit"]
    assert reconciled["merge_result"]["merged"] is True
    assert reconciled["merge_result"]["queued"] is False
    assert reconciled["integration_commit_proof"]["passed"] is True
    evidence = reconciled["completion_receipt_evidence"]
    assert evidence["completion_task_cids"] == {task.task_id: task_cid}
    assert evidence["completion_receipts"][0]["canonical_task_cid"] == task_cid
    assert evidence["receipt_id"] == content_identity(
        {key: value for key, value in evidence.items() if key != "receipt_id"}
    )

    before_replay = paths.events.read_bytes()
    replay = daemon._merge_train_callback(request)

    assert (
        replay.get("merged") is True or replay.get("already_merged") is True
    ), replay
    assert replay["merge_reconciliation_receipt"]["replayed"] is True
    assert paths.events.read_bytes() == before_replay

    for tamper in ("extra_field", "foreign_completion_receipts"):
        forged = json.loads(json.dumps(reconciled))
        if tamper == "extra_field":
            forged["attacker_extension"] = True
        else:
            forged_evidence = forged["completion_receipt_evidence"]
            forged_evidence["completion_receipts"][0][
                "attacker_extension"
            ] = True
            forged_evidence["receipt_id"] = content_identity(
                {
                    key: value
                    for key, value in forged_evidence.items()
                    if key != "receipt_id"
                }
            )
        forged_events = [
            forged if event.get("event_id") == reconciled["event_id"] else event
            for event in events
        ]
        monkeypatch.setattr(
            daemon,
            "_iter_merge_lifecycle_events",
            lambda forged_events=forged_events: forged_events,
        )
        conflict = daemon._record_merge_queue_callback_reconciliation(
            request=request,
            task=task,
            metadata=request.metadata,
            implementation_commit=candidate,
            integration_commit=result["merge_commit"],
            integration_commit_proof=result["integration_commit_proof"],
            completion_task_cids=request.metadata["completion_task_cids"],
            completion_receipts=result["todo_update_result"][
                "completion_receipts"
            ],
            declared_output_invariant=result[
                "post_merge_declared_output_invariant"
            ],
        )
        assert conflict["recorded"] is False
        assert conflict["reason"] == (
            "merge_queue_reconciliation_receipt_conflict"
        )


def test_synchronous_schema_v3_callback_projects_source_before_completion(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    attempt = _database_projection_attempt(
        attempt_id="attempt:synchronous-callback",
        claim_id="claim:synchronous-callback",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=tmp_path / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(revision=2),
    )
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/synchronous-callback")
    (repo / "base.txt").write_text(
        "synchronous callback candidate\n",
        encoding="utf-8",
    )
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "synchronous callback candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    candidate_tree = _git(repo, "rev-parse", f"{candidate}^{{tree}}")
    _git(repo, "switch", "main")
    task = daemon._load_tasks()[0]
    task_cid = daemon._canonical_ref(task)
    request, queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/synchronous-callback",
        implementation_commit=candidate,
        baseline_ref=baseline,
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

    result = daemon._merge_train_callback(request)

    assert result["merged"] is True
    events = daemon._iter_merge_lifecycle_events()
    [enqueue_source] = [
        event
        for event in events
        if event.get("type") == "merge_candidate_enqueued"
    ]
    [projected_source] = [
        event
        for event in events
        if event.get("type")
        == "worktree_reconciliation_candidate_queued"
    ]
    [reconciled] = [
        event for event in events if event.get("type") == "merge_reconciled"
    ]
    [todo_updated] = [
        event for event in events if event.get("type") == "todo_status_updated"
    ]
    assert events.index(enqueue_source) < events.index(projected_source)
    assert events.index(projected_source) < events.index(reconciled)
    assert events.index(reconciled) < events.index(todo_updated)
    assert not any(
        event.get("type") in {"implementation_finished", "task_completed"}
        for event in events
    )
    provenance = projected_source["merge_queue_synchronous_source"]
    assert provenance["merge_candidate_enqueued_event_id"] == (
        enqueue_source["event_id"]
    )
    assert provenance["portal_attempt"] == 1
    assert provenance["baseline_ref"] == baseline
    assert provenance["implementation_commit"] == candidate
    assert provenance["validation_target_commit"] == candidate
    assert provenance["validation_target_tree"] == candidate_tree
    assert provenance["validation_repository_tree_id"] == (
        f"git-tree:{candidate_tree}"
    )
    assert reconciled["completion_source_event_id"] == (
        projected_source["event_id"]
    )
    assert "- Status: completed" in paths.task_projection.read_text(
        encoding="utf-8"
    )

    terminal_merge = dict(queued)
    terminal_merge.update(
        {
            "attempted": True,
            "merged": True,
            "queued": False,
            "merge_commit": result["merge_commit"],
            "target_commit": result["merge_commit"],
        }
    )
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "canonical_task_cid": task_cid,
            "attempt": 1,
            "returncode": 0,
            "attempt_consumed": True,
            "provider_dispatched": False,
            "branch": request.branch_name,
            "baseline_ref": baseline,
            "implementation_commit": candidate,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
            },
            "merge_result": terminal_merge,
            "board_completion": {
                "complete": True,
                "pending_merge": False,
                "reason": "merged_into_target",
            },
        },
    )
    replay = daemon._record_merge_queue_callback_reconciliation(
        request=request,
        task=task,
        metadata=request.metadata,
        implementation_commit=candidate,
        integration_commit=result["merge_commit"],
        integration_commit_proof=result["integration_commit_proof"],
        completion_task_cids=request.metadata["completion_task_cids"],
        completion_receipts=result["todo_update_result"][
            "completion_receipts"
        ],
        declared_output_invariant=result[
            "post_merge_declared_output_invariant"
        ],
    )
    assert replay["recorded"] is True
    assert replay["replayed"] is True


def test_missing_callback_source_never_publishes_completion(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    attempt = _database_projection_attempt(
        attempt_id="attempt:missing-callback-source",
        claim_id="claim:missing-callback-source",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=tmp_path / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(revision=2),
    )
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/missing-callback-source")
    (repo / "base.txt").write_text(
        "candidate whose source is unavailable\n",
        encoding="utf-8",
    )
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "missing callback source candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    task = daemon._load_tasks()[0]
    request, _queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/missing-callback-source",
        implementation_commit=candidate,
        baseline_ref=baseline,
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
    paths.events.unlink()

    result = daemon._merge_train_callback(request)

    assert result["merged"] is False
    assert result["completion_skipped"] is True
    assert result["reason"] == "merge_queue_reconciliation_source_pending"
    assert "todo_update_result" not in result
    assert "- Status: ready" in paths.task_projection.read_text(
        encoding="utf-8"
    )
    events = daemon._iter_merge_lifecycle_events()
    assert not any(
        event.get("type")
        in {
            "worktree_reconciliation_candidate_queued",
            "merge_reconciled",
            "todo_status_updated",
            "task_completed",
        }
        for event in events
    )


def test_reconciliation_failure_precedes_all_completion_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    attempt = _database_projection_attempt(
        attempt_id="attempt:reconciliation-failure",
        claim_id="claim:reconciliation-failure",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=tmp_path / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(revision=2),
    )
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/reconciliation-failure")
    (repo / "base.txt").write_text(
        "integrated without completion publication\n",
        encoding="utf-8",
    )
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "reconciliation failure candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    task = daemon._load_tasks()[0]
    task_cid = daemon._canonical_ref(task)
    request, queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/reconciliation-failure",
        implementation_commit=candidate,
        baseline_ref=baseline,
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
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "canonical_task_cid": task_cid,
            "attempt": 1,
            "returncode": 0,
            "attempt_consumed": True,
            "provider_dispatched": False,
            "branch": request.branch_name,
            "baseline_ref": baseline,
            "implementation_commit": candidate,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
            },
            "merge_result": dict(queued),
            "board_completion": {
                "complete": False,
                "pending_merge": True,
                "reason": "merge_queued_awaiting_integration",
            },
        },
    )
    observed_receipts: list[dict[str, object]] = []

    def reject_reconciliation(**kwargs: object) -> dict[str, object]:
        observed_receipts.extend(
            dict(receipt)
            for receipt in kwargs["completion_receipts"]  # type: ignore[index]
        )
        return {
            "recorded": False,
            "reason": "forced_reconciliation_failure",
        }

    monkeypatch.setattr(
        daemon,
        "_record_merge_queue_callback_reconciliation",
        reject_reconciliation,
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is False
    assert result["reason"] == "forced_reconciliation_failure"
    assert result["completion_skipped"] is True
    assert result["integration_occurred"] is True
    assert "todo_update_result" not in result
    assert _git(
        repo,
        "merge-base",
        "--is-ancestor",
        candidate,
        daemon.resolved_merge_target_branch,
    ) == ""
    assert "- Status: ready" in paths.task_projection.read_text(
        encoding="utf-8"
    )
    assert observed_receipts == [
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "member_completion_receipt@1"
            ),
            "task_id": task.task_id,
            "canonical_task_key": request.canonical_task_key,
            "canonical_task_cid": task_cid,
            "board_namespace": observed_receipts[0]["board_namespace"],
            "status": "succeeded",
        }
    ]
    events = daemon._iter_merge_lifecycle_events()
    assert not any(
        event.get("type")
        in {"merge_reconciled", "todo_status_updated", "task_completed"}
        for event in events
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_binding",
        "projection_authority",
        "task_cid_mismatch",
        "producer_events_path",
        "producer_state_path",
        "producer_events_symlink",
    ),
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
    if tamper == "producer_events_path":
        request.metadata["events_path"] = str(
            tmp_path / "foreign-events.jsonl"
        )
    elif tamper == "producer_state_path":
        request.metadata["state_path"] = str(
            tmp_path / "foreign-state.json"
        )
    elif tamper == "producer_events_symlink":
        foreign_events = tmp_path / "foreign-events.jsonl"
        foreign_events.write_bytes(producer_paths.events.read_bytes())
        producer_paths.events.unlink()
        producer_paths.events.symlink_to(foreign_events)
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
    target_branch = daemon.resolved_merge_target_branch
    assert target_branch == "implementation/tasks"
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
        _git(repo, "branch", "-f", target_branch, candidate)
        return {
            "merged": True,
            "returncode": 0,
            "merge_commit": _git(repo, "rev-parse", target_branch),
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
    assert (
        _git(repo, "merge-base", "--is-ancestor", candidate, target_branch)
        == ""
    )
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", candidate, "main"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        ).returncode
        == 1
    )


def test_same_projection_does_not_complete_nonancestor_stale_candidate(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    merge_queue_dir = tmp_path / "merge-queue"
    task_cid = "task:cid:ref-040"
    attempt = _database_projection_attempt(
        attempt_id="attempt:producer",
        claim_id="claim:producer",
        task_cid=task_cid,
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=merge_queue_dir,
        attempt=attempt,
        record=_database_projection_record(revision=2),
    )
    head_before = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/stale")
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    _git(repo, "add", "side.txt")
    _git(repo, "commit", "-m", "stale candidate")
    stale = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    task = daemon._load_tasks()[0]
    request, _queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/stale",
        implementation_commit=stale,
        baseline_ref=head_before,
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

    result = daemon._merge_train_callback(request)

    assert result.get("merged") is False
    assert result.get("already_merged") is False
    assert result.get("completion_skipped") is True
    assert result.get("reason") == (
        "merge_queue_reconciliation_completion_gate_invalid"
    )
    side_probe = subprocess.run(
        ["git", "cat-file", "-e", "HEAD:side.txt"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert side_probe.returncode != 0
    assert "- Status: ready" in paths.task_projection.read_text(
        encoding="utf-8"
    )


def test_false_completion_reopen_merges_and_seals_qualification_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    merge_queue_dir = tmp_path / "merge-queue"
    task_cid = "task:cid:ref-040"
    attempt = _database_projection_attempt(
        attempt_id="attempt:false-reopen",
        claim_id="claim:false-reopen",
        task_cid=task_cid,
        attempt_number=1,
    )
    daemon, paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=merge_queue_dir,
        attempt=attempt,
        record=_database_projection_record(revision=2),
    )
    target_branch = daemon.resolved_merge_target_branch
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/false-reopen")
    (repo / "base.txt").write_text("candidate output\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "candidate declared output")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    (repo / "target.txt").write_text("target advance\n", encoding="utf-8")
    _git(repo, "add", "target.txt")
    _git(repo, "commit", "-m", "advance target")
    _git(repo, "branch", "-f", target_branch, "HEAD")
    task = daemon._load_tasks()[0]
    request, _queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/false-reopen",
        implementation_commit=candidate,
        baseline_ref=baseline,
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
    claimed = daemon.merge_queue.claim_pending_request(
        request.request_id,
        consumer_id="merge-train:false-shortcut-fixture",
    )
    assert claimed is not None
    original_identity_check = daemon._declared_outputs_match_commits
    original_reconciliation = (
        daemon._record_merge_queue_callback_reconciliation
    )
    monkeypatch.setattr(
        daemon,
        "_declared_outputs_match_commits",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        daemon,
        "_record_merge_queue_callback_reconciliation",
        lambda **_kwargs: {
            "recorded": True,
            "legacy_false_completion_fixture": True,
        },
    )
    false_callback = daemon._merge_train_callback(claimed)
    monkeypatch.setattr(
        daemon,
        "_declared_outputs_match_commits",
        original_identity_check,
    )
    monkeypatch.setattr(
        daemon,
        "_record_merge_queue_callback_reconciliation",
        original_reconciliation,
    )
    assert false_callback["reason"] == "declared_outputs_already_on_target"
    assert false_callback["already_merged"] is True
    assert "- Status: completed" in paths.task_projection.read_text(
        encoding="utf-8"
    )
    false_target = _git(repo, "rev-parse", target_branch)
    daemon.merge_queue.complete(claimed)
    completed = daemon.merge_queue.get(request.request_id)
    assert completed is not None and completed.status == "completed"
    false_receipt = {
        "already_merged": True,
        "canonical_task_id": completed.canonical_identity,
        "commit_sha": candidate,
        "distributed_publication_admission": {
            "admitted": True,
            "distributed": False,
            "request_id": request.request_id,
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "distributed-lane-admission@1"
            ),
            "status": "local",
        },
        "finished_at": 2.0,
        "integrated": True,
        "merge_commit": false_target,
        "merged": False,
        "mutation_short_circuited": True,
        "reason": "declared_outputs_already_on_target",
        "request_id": request.request_id,
        "started_at": 1.0,
        "status": "already_merged",
        "target_branch": target_branch,
        "target_commit": false_target,
        "task_id": "REF-040",
    }
    reopened = daemon.merge_queue.reopen_false_positive_completion(
        completed,
        completion_receipt=false_receipt,
    )
    assert reopened is not None and reopened.status == "pending"

    validation_count = 0

    def validation(
        _workspace: Path,
        validation_task: object,
        log_path: Path,
        *,
        force_uncached: bool = False,
    ) -> dict[str, object]:
        nonlocal validation_count
        validation_count += 1
        assert validation_task.task_id == "REF-040"
        assert force_uncached is True
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("false reopen qualification passed\n")
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [
                {
                    "validation_result_digest": "sha256:" + "a" * 64,
                }
            ],
        }

    monkeypatch.setattr(daemon, "_run_validation_commands", validation)
    crashed_train = MergeTrain(
        repo,
        daemon.merge_queue,
        target_branch=target_branch,
        merge_callback=daemon._merge_train_callback,
    )
    crashed_claim = daemon.merge_queue.claim_pending_request(
        reopened.request_id,
        consumer_id=crashed_train.owner_id,
    )
    assert crashed_claim is not None
    lost_result = daemon._merge_train_callback(crashed_claim)
    assert lost_result["merged"] is True
    assert lost_result.get("already_merged") is not True
    assert lost_result["reason"] == "post_merge_declared_outputs_repaired"
    integration_commit = str(lost_result["merge_commit"])
    crashed_row = daemon.merge_queue.get(request.request_id)
    assert crashed_row is not None and crashed_row.status == "processing"

    target_advance_worktree = tmp_path / "target-advance"
    _git(
        repo,
        "worktree",
        "add",
        str(target_advance_worktree),
        target_branch,
    )
    (target_advance_worktree / "after-crash.txt").write_text(
        "descendant target advance\n",
        encoding="utf-8",
    )
    _git(target_advance_worktree, "add", "after-crash.txt")
    _git(
        target_advance_worktree,
        "commit",
        "-m",
        "advance target after crashed merge",
    )
    descendant_target = _git(target_advance_worktree, "rev-parse", "HEAD")
    _git(
        repo,
        "worktree",
        "remove",
        "--force",
        str(target_advance_worktree),
    )

    recovery_train = MergeTrain(
        repo,
        daemon.merge_queue,
        target_branch=target_branch,
        merge_callback=daemon._merge_train_callback,
    )

    @contextmanager
    def route_exact_recovery(train: MergeTrain) -> object:
        shortcut = train._portal_projection_invalid_metadata_already_on_target
        train._portal_projection_invalid_metadata_already_on_target = (
            lambda _request: False
        )
        try:
            yield
        finally:
            train._portal_projection_invalid_metadata_already_on_target = (
                shortcut
            )

    result = recovery_train.recover_one_integrated_quarantine(
        request_id=request.request_id,
        processor_context=route_exact_recovery,
        allow_post_merge_declared_output_recovery=True,
    )

    assert result is not None
    assert result["status"] == "already_merged", result
    assert "merge_result" in result, result
    merge_result = result["merge_result"]
    assert merge_result["reason"] == "post_merge_declared_outputs_repaired"
    qualification = merge_result["post_merge_declared_output_repair"]
    assert qualification["passed"] is True
    assert qualification["qualification_kind"] == (
        "false_positive_completion_reopen"
    )
    receipt = qualification["receipt"]
    receipt_id = receipt.pop("receipt_id")
    assert receipt_id == content_identity(receipt)
    receipt["receipt_id"] = receipt_id
    assert receipt["candidate_commit"] == candidate
    assert receipt["failed_integration_commit"] == false_target
    assert receipt["repair_parent_commit"] == false_target
    assert receipt["repair_commit"] == descendant_target
    assert qualification["integration_lineage"]["integration_commit"] == (
        integration_commit
    )
    assert receipt["entries"][0]["path"] == "base.txt"
    settled = daemon.merge_queue.get(request.request_id)
    assert settled is not None and settled.status == "completed"
    completion = settled.metadata["completion"]
    assert completion["reason"] == "post_merge_declared_outputs_repaired"
    assert completion["target_commit"] == receipt["repair_commit"]
    assert completion["repair_receipt"] == receipt
    assert validation_count == 2
    assert _git(repo, "show", f"{target_branch}:base.txt") == "candidate output"
    assert "- Status: completed" in paths.task_projection.read_text(
        encoding="utf-8"
    )


def test_false_completion_lineage_bounds_distance_not_repository_age(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/deep-history")
    (repo / "base.txt").write_text(
        "candidate output\n",
        encoding="utf-8",
    )
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "candidate declared output")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "foreign-previous", baseline)
    _git(repo, "commit", "--allow-empty", "-m", "foreign previous target")
    foreign_previous = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    _git(
        repo,
        "merge",
        "--no-ff",
        "foreign-previous",
        "-m",
        "retain foreign previous as a side-parent ancestor",
    )

    padding_commits: list[str] = []
    for index in range(513):
        _git(
            repo,
            "commit",
            "--allow-empty",
            "-m",
            f"older first-parent history {index}",
        )
        padding_commits.append(_git(repo, "rev-parse", "HEAD"))
    near_previous = padding_commits[-1]
    boundary_previous = padding_commits[1]
    beyond_bound_previous = padding_commits[0]

    _git(
        repo,
        "merge",
        "--no-ff",
        "implementation/deep-history",
        "-m",
        "integrate exact candidate",
    )
    target = _git(repo, "rev-parse", "HEAD")
    assert int(_git(repo, "rev-list", "--first-parent", "--count", target)) > 512

    attempt = _database_projection_attempt(
        attempt_id="attempt:deep-lineage",
        claim_id="claim:deep-lineage",
        task_cid="task:cid:ref-040",
        attempt_number=1,
    )
    daemon, _paths, _binding = _database_projection_daemon(
        repo=repo,
        attempt_root=repo / "attempts",
        merge_queue_dir=tmp_path / "merge-queue",
        attempt=attempt,
        record=_database_projection_record(revision=2),
    )
    task = daemon._load_tasks()[0]

    def prove(previous_target_commit: str) -> dict[str, object]:
        return daemon._false_positive_completion_integration_lineage(
            task,
            candidate_commit=candidate,
            baseline_ref=baseline,
            previous_target_commit=previous_target_commit,
            target_commit=target,
        )

    near = prove(near_previous)
    assert near["passed"] is True, near
    assert near["reason"] == "false_positive_completion_merge_lineage_proved"
    assert near["first_parent_distance"] == 1

    boundary = prove(boundary_previous)
    assert boundary["passed"] is True, boundary
    assert boundary["first_parent_distance"] == 512

    beyond_bound = prove(beyond_bound_previous)
    assert beyond_bound["passed"] is False
    assert beyond_bound["reason"] == "false_completion_lineage_history_unbounded"

    missing = prove(foreign_previous)
    assert missing["passed"] is False
    assert missing["reason"] == "false_completion_lineage_history_unbounded"


def test_false_completion_lineage_retry_requires_new_target_generation(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "implementation/lineage-retry")
    (repo / "base.txt").write_text(
        "candidate output\n",
        encoding="utf-8",
    )
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "candidate output")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    previous_target = _git(repo, "rev-parse", "HEAD")
    _git(
        repo,
        "merge",
        "--no-ff",
        "implementation/lineage-retry",
        "-m",
        "integrate candidate before lineage retry",
    )
    failed_target = _git(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "foreign-lineage-target", baseline)
    _git(repo, "switch", "foreign-lineage-target")
    _git(repo, "commit", "--allow-empty", "-m", "foreign lineage target")
    foreign_target = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")

    reason = "false_positive_completion_integration_lineage_unproven"
    queue = MergeQueue(
        tmp_path / "merge-queue",
        max_attempts=3,
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )

    def lineage(target: str, **updates: object) -> dict[str, object]:
        value: dict[str, object] = {
            "passed": False,
            "reason": "false_completion_lineage_history_unbounded",
            "candidate_commit": candidate,
            "baseline_commit": baseline,
            "previous_target_commit": previous_target,
            "target_commit": target,
            "ancestry": {
                "candidate_to_previous_target": 1,
                "baseline_to_previous_target": 0,
                "previous_target_to_target": 0,
                "candidate_to_target": 0,
            },
            "previous_declared_output_identity": False,
            "current_declared_output_identity": True,
        }
        value.update(updates)
        return value

    def exhausted_quarantine(
        suffix: str,
        *,
        proof_target: str,
        marker_schema: str = FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA,
        proof_updates: dict[str, object] | None = None,
    ) -> MergeRequest:
        task_id = f"LINEAGE-{suffix}"
        request = queue.enqueue(
            branch_name="implementation/lineage-retry",
            task_id=task_id,
            canonical_task_id=f"canonical:{suffix}",
            commit_sha=candidate,
            metadata={
                "baseline_ref": baseline,
                "task": {"outputs": ["base.txt"]},
            },
        )
        initial_claim = queue.claim_pending_request(
            request.request_id,
            consumer_id=f"merge-train:false-completion-{suffix}",
        )
        assert initial_claim is not None
        queue.complete(initial_claim)
        completed = queue.get(request.request_id)
        assert completed is not None and completed.status == "completed"
        false_completion_receipt = {
            "already_merged": True,
            "canonical_task_id": completed.canonical_identity,
            "commit_sha": candidate,
            "distributed_publication_admission": {
                "admitted": True,
                "distributed": False,
                "request_id": request.request_id,
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "distributed-lane-admission@1"
                ),
                "status": "local",
            },
            "finished_at": 2.0,
            "integrated": True,
            "merge_commit": previous_target,
            "merged": False,
            "mutation_short_circuited": True,
            "reason": "declared_outputs_already_on_target",
            "request_id": request.request_id,
            "started_at": 1.0,
            "status": "already_merged",
            "target_branch": "main",
            "target_commit": previous_target,
            "task_id": task_id,
        }
        reopened = queue.reopen_false_positive_completion(
            completed,
            completion_receipt=false_completion_receipt,
        )
        assert reopened is not None and reopened.status == "pending"
        for prior in range(2):
            claimed = queue.claim_pending_request(
                request.request_id,
                consumer_id=f"merge-train:prior-{suffix}-{prior}",
            )
            assert claimed is not None
            retried = queue.requeue(
                claimed,
                reason=f"prior bounded failure {prior}",
            )
            assert isinstance(retried, MergeRequest)
        claimed = queue.claim_pending_request(
            request.request_id,
            consumer_id=f"merge-train:terminal-{suffix}",
        )
        assert claimed is not None
        proof = lineage(proof_target, **(proof_updates or {}))
        queue.quarantine(
            claimed,
            reason=reason,
            metadata={
                "status": "quarantined",
                "reason": reason,
                "merge_result": {
                    "attempted": False,
                    "merged": False,
                    "returncode": 2,
                    "reason": reason,
                    "false_positive_completion_integration_lineage": proof,
                },
            },
        )
        stored = queue.get(request.request_id)
        assert stored is not None
        if marker_schema != FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA:
            with queue._connect() as connection:
                row = connection.execute(
                    "SELECT metadata_json FROM merge_requests "
                    "WHERE request_id=?",
                    (request.request_id,),
                ).fetchone()
                assert row is not None
                raw_metadata = json.loads(row["metadata_json"])
                raw_metadata["false_positive_completion_reopen"][
                    "schema"
                ] = marker_schema
                connection.execute(
                    "UPDATE merge_requests SET metadata_json=? "
                    "WHERE request_id=?",
                    (
                        json.dumps(
                            raw_metadata,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        request.request_id,
                    ),
                )
                connection.commit()
            stored = queue.get(request.request_id)
            assert stored is not None
        assert stored.status == "quarantined"
        assert stored.attempt == 3
        assert stored.failure_count == 3
        return stored

    callbacks: list[str] = []

    def fail_lineage_again(_request: MergeRequest) -> dict[str, object]:
        target = _git(repo, "rev-parse", "refs/heads/main")
        callbacks.append(target)
        return {
            "attempted": False,
            "merged": False,
            "returncode": 2,
            "reason": reason,
            "false_positive_completion_integration_lineage": lineage(target),
        }

    train = MergeTrain(
        repo,
        queue,
        target_branch="main",
        max_attempts=3,
        merge_callback=fail_lineage_again,
    )
    selected = exhausted_quarantine(
        "VALID",
        proof_target=failed_target,
    )
    assert DatabasePortalExecutionBridge._request_has_missing_output_recovery_lineage(
        selected
    )
    assert not train._quarantine_may_auto_recover(selected)
    assert not train._quarantine_may_auto_recover(
        selected,
        allow_post_merge_declared_output_recovery=True,
    )
    before_same_target = queue.get(selected.request_id)
    assert train.recover_one_integrated_quarantine(
        request_id=selected.request_id,
        allow_post_merge_declared_output_recovery=True,
    ) is None
    after_same_target = queue.get(selected.request_id)
    assert before_same_target == after_same_target
    assert callbacks == []

    _git(repo, "commit", "--allow-empty", "-m", "new retry generation")
    retry_target = _git(repo, "rev-parse", "HEAD")
    retry = train._false_positive_completion_lineage_retry(selected)
    assert retry == {
        "failed_target_commit": failed_target,
        "target_commit": retry_target,
    }
    first_result = train.recover_one_integrated_quarantine(
        request_id=selected.request_id,
        allow_post_merge_declared_output_recovery=True,
    )
    assert first_result is not None
    assert first_result["status"] == "quarantined"
    first_failure = queue.get(selected.request_id)
    assert first_failure is not None
    assert first_failure.status == "quarantined"
    assert first_failure.attempt == 3
    assert first_failure.failure_count == 4
    assert callbacks == [retry_target]
    assert len(first_failure.metadata["revivals"]) == 1
    assert failed_target in first_failure.metadata["revivals"][-1]["reason"]
    assert retry_target in first_failure.metadata["revivals"][-1]["reason"]
    assert (
        first_failure.metadata["quarantine"]["merge_result"][
            "false_positive_completion_integration_lineage"
        ]["target_commit"]
        == retry_target
    )

    same_generation = queue.get(selected.request_id)
    assert train.recover_one_integrated_quarantine(
        request_id=selected.request_id,
        allow_post_merge_declared_output_recovery=True,
    ) is None
    assert queue.get(selected.request_id) == same_generation
    assert callbacks == [retry_target]

    _git(repo, "commit", "--allow-empty", "-m", "second retry generation")
    second_target = _git(repo, "rev-parse", "HEAD")
    second_result = train.recover_one_integrated_quarantine(
        request_id=selected.request_id,
        allow_post_merge_declared_output_recovery=True,
    )
    assert second_result is not None
    assert second_result["status"] == "quarantined"
    second_failure = queue.get(selected.request_id)
    assert second_failure is not None
    assert second_failure.failure_count == 5
    assert len(second_failure.metadata["revivals"]) == 2
    assert callbacks == [retry_target, second_target]

    invalid_rows = (
        exhausted_quarantine("DIVERGENT", proof_target=foreign_target),
        exhausted_quarantine("UNKNOWN", proof_target="f" * 40),
        exhausted_quarantine("MALFORMED", proof_target="not-a-commit"),
        exhausted_quarantine(
            "INNER-REASON",
            proof_target=failed_target,
            proof_updates={"reason": "false_completion_lineage_commit_unavailable"},
        ),
        exhausted_quarantine(
            "OUTPUT-IDENTITY",
            proof_target=failed_target,
            proof_updates={"current_declared_output_identity": False},
        ),
    )
    for invalid in invalid_rows:
        assert DatabasePortalExecutionBridge._request_has_missing_output_recovery_lineage(
            invalid
        )
        assert not train._quarantine_may_auto_recover(
            invalid,
            allow_post_merge_declared_output_recovery=True,
        )
        before = queue.get(invalid.request_id)
        assert train.recover_one_integrated_quarantine(
            request_id=invalid.request_id,
            allow_post_merge_declared_output_recovery=True,
        ) is None
        assert queue.get(invalid.request_id) == before
    assert callbacks == [retry_target, second_target]

    malformed_marker = exhausted_quarantine(
        "MARKER",
        proof_target=failed_target,
        marker_schema="malformed-reopen-marker",
    )
    assert not DatabasePortalExecutionBridge._request_has_missing_output_recovery_lineage(
        malformed_marker
    )
    assert not train._quarantine_may_auto_recover(
        malformed_marker,
        allow_post_merge_declared_output_recovery=True,
    )
    before_marker = queue.get(malformed_marker.request_id)
    assert train.recover_one_integrated_quarantine(
        request_id=malformed_marker.request_id,
        allow_post_merge_declared_output_recovery=True,
    ) is None
    assert queue.get(malformed_marker.request_id) == before_marker
    assert callbacks == [retry_target, second_target]


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
    request = _quarantine_invalid_authority_projection(
        queue=queue,
        tmp_path=tmp_path,
        candidate=candidate,
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


def test_invalid_authority_metadata_quarantine_does_not_settle_on_path_existence(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "switch", "-c", "implementation/side")
    (repo / "base.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "change declared output")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    head_before = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = _quarantine_invalid_authority_projection(
        queue=queue,
        tmp_path=tmp_path,
        candidate=candidate,
    )

    train = MergeTrain(repo, queue)

    assert (
        train._portal_projection_invalid_metadata_already_on_target(request)
        is False
    )
    assert train.run_once() is None
    assert _git(repo, "rev-parse", "HEAD") == head_before
    assert (repo / "base.txt").read_text(encoding="utf-8") == "base\n"
    quarantined = queue.get(request.request_id)
    assert quarantined is not None
    assert quarantined.status == "quarantined"


def test_invalid_authority_metadata_quarantine_settles_for_candidate_ancestor(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "switch", "-c", "implementation/side")
    (repo / "base.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "change declared output")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    _git(repo, "merge", "--ff-only", "implementation/side")
    (repo / "base.txt").write_text("later target\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "advance declared output")
    head_before = _git(repo, "rev-parse", "HEAD")
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=checkout_repository_id(repo),
        target_branch="main",
        require_target_binding=True,
    )
    request = _quarantine_invalid_authority_projection(
        queue=queue,
        tmp_path=tmp_path,
        candidate=candidate,
    )

    train = MergeTrain(repo, queue)
    result = train.run_once()

    assert result is not None
    assert result.get("status") == "already_merged"
    assert result.get("reason") != "declared_outputs_already_on_target"
    assert _git(repo, "rev-parse", "HEAD") == head_before
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
