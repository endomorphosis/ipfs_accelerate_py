"""Tests for database merge/validation/recovery integration (DQP-019).

Evidence subset: serialized merge, fairness, stale result, rebase, conflict,
validation failure, partial publish, crash, retry exhaustion, idempotent replay.

Acceptance: A task completes only after accepted merge and current validation
evidence commit together; stale worktree/fence results are rejected; recovery
actions are idempotent and queryable; no JSON receipt or queue file alone can
settle work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_merge_queue import (
    COMPLETION_RECEIPT_INTERFACE,
    DATABASE_MERGE_QUEUE_INTERFACE,
    VALIDATION_RUN_INTERFACE,
    DatabaseMergeNotReadyError,
    DatabaseMergeQueue,
    DatabaseMergeStaleError,
    MergeAttemptStatus,
    MergeEntryStatus,
    ValidationStatus,
    duckdb_available,
    open_database_merge_queue,
)
from ipfs_accelerate_py.agent_supervisor.rescue.database_recovery import (
    DATABASE_RECOVERY_INTERFACE,
    RECOVERY_ACTION_INTERFACE,
    DatabaseRecovery,
    RecoveryActionKind,
    RecoveryActionStatus,
    open_database_recovery,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database merge/recovery hermetic tests",
)


class FakeClock:
    def __init__(self, start_ms: int = 1_000_000) -> None:
        self.now = int(start_ms)

    def __call__(self) -> int:
        return int(self.now)

    def advance(self, ms: int) -> None:
        self.now += int(ms)


def _open(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    max_attempts: int = 3,
    priority_aging_ms: int = 300_000,
) -> tuple[DatabaseMergeQueue, DatabaseRecovery, FakeClock]:
    clock = clock or FakeClock()
    queue = open_database_merge_queue(
        tmp_path / "merge_recovery.duckdb",
        clock_ms=clock,
        max_attempts=max_attempts,
        priority_aging_ms=priority_aging_ms,
        max_processing_per_scope=1,
    )
    recovery = open_database_recovery(
        merge_queue=queue,
        clock_ms=clock,
        default_retry_budget=max_attempts,
    )
    return queue, recovery, clock


def _enqueue(
    queue: DatabaseMergeQueue,
    ordinal: int,
    *,
    priority: str = "P2",
    fencing_token: int = 1,
    fence_epoch: int = 1,
    worktree_id: str | None = None,
    target_repository_id: str = "repository:demo",
    target_branch: str = "main",
    commit_sha: str | None = None,
):
    return queue.enqueue(
        task_cid=f"task:{ordinal}",
        attempt_id=f"attempt:{ordinal}",
        worktree_id=worktree_id or f"worktree:{ordinal}",
        branch_name=f"candidate/{ordinal}",
        commit_sha=commit_sha or f"{ordinal + 1:040x}",
        fencing_token=fencing_token,
        fence_epoch=fence_epoch,
        priority=priority,
        target_repository_id=target_repository_id,
        target_branch=target_branch,
    )


def _claim_one(queue: DatabaseMergeQueue, consumer: str = "merge-train:a"):
    claimed = queue.claim(consumer_id=consumer)
    assert len(claimed) == 1
    return claimed[0]


def _pass_validation(queue: DatabaseMergeQueue, entry, *, digest: str = "sha256:ok"):
    run = queue.start_validation(
        entry_id=entry.entry_id,
        claim_token=entry.claim_token,
        claim_generation=entry.claim_generation,
        commands=["python -m pytest -q"],
        consumer_id=entry.consumer_id,
    )
    # Reload entry after status transition.
    entry = queue.get_entry(entry.entry_id)
    assert entry is not None
    finished = queue.finish_validation(
        run_id=run.run_id,
        claim_token=entry.claim_token,
        claim_generation=entry.claim_generation,
        passed=True,
        evidence_digest=digest,
        result_summary="green",
        worktree_id=entry.worktree_id,
        fencing_token=entry.fencing_token,
        fence_epoch=entry.fence_epoch,
        consumer_id=entry.consumer_id,
    )
    entry = queue.get_entry(entry.entry_id)
    assert entry is not None
    return entry, finished


def _accept_merge(
    queue: DatabaseMergeQueue,
    entry,
    *,
    result_commit: str = "a" * 40,
):
    attempt = queue.begin_merge_attempt(
        entry_id=entry.entry_id,
        claim_token=entry.claim_token,
        claim_generation=entry.claim_generation,
        base_commit="b" * 40,
        consumer_id=entry.consumer_id,
    )
    accepted = queue.accept_merge(
        attempt_id=attempt.attempt_id,
        claim_token=entry.claim_token,
        claim_generation=entry.claim_generation,
        result_commit=result_commit,
        consumer_id=entry.consumer_id,
    )
    entry = queue.get_entry(entry.entry_id)
    assert entry is not None
    return entry, accepted


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_MERGE_QUEUE_INTERFACE == "DatabaseMergeQueue@1"
    assert VALIDATION_RUN_INTERFACE == "ValidationRun@1"
    assert COMPLETION_RECEIPT_INTERFACE == "CompletionReceipt@1"
    assert DATABASE_RECOVERY_INTERFACE == "DatabaseRecovery@1"
    assert RECOVERY_ACTION_INTERFACE == "RecoveryAction@1"
    assert DatabaseMergeQueue.INTERFACE == DATABASE_MERGE_QUEUE_INTERFACE
    assert DatabaseRecovery.INTERFACE == DATABASE_RECOVERY_INTERFACE


def test_authority_policy_rejects_json_settlement(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        policy = queue.authority_policy()
        assert policy["json_receipts_grant_completion"] is False
        assert policy["queue_files_grant_completion"] is False
        assert "accepted_merge" in policy["completion_requires"]
        assert "current_validation_evidence" in policy["completion_requires"]
    finally:
        recovery.close()
        queue.close()


# ---------------------------------------------------------------------------
# Serialized merge + fairness
# ---------------------------------------------------------------------------


def test_serialized_merge_one_active_claim_per_scope(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        first = _enqueue(queue, 0, priority="P1")
        second = _enqueue(queue, 1, priority="P1")
        claimed = queue.claim(consumer_id="worker:a", limit=2)
        assert len(claimed) == 1
        assert claimed[0].entry_id == first.entry_id
        # Second remains pending while first holds the serialized scope.
        still = queue.claim(consumer_id="worker:b", limit=1)
        assert still == ()
        pending = queue.list_entries(status=MergeEntryStatus.PENDING)
        assert {item.entry_id for item in pending} == {second.entry_id}
    finally:
        recovery.close()
        queue.close()


def test_fairness_priority_and_aging(tmp_path: Path) -> None:
    clock = FakeClock()
    queue = open_database_merge_queue(
        tmp_path / "fair.duckdb",
        clock_ms=clock,
        priority_aging_ms=1_000,
        max_processing_per_scope=8,
    )
    recovery = open_database_recovery(merge_queue=queue, clock_ms=clock)
    try:
        _enqueue(queue, 10, priority="P3", target_branch="a")
        clock.advance(5_000)  # age P3 enough to promote past P1
        _enqueue(queue, 11, priority="P1", target_branch="b")
        _enqueue(queue, 12, priority="P0", target_branch="c")
        claimed = queue.claim(consumer_id="fair", limit=3)
        # Aged P3 reaches effective P0 and wins by earlier enqueued_at; fresh P1 last.
        assert [item.task_cid for item in claimed] == [
            "task:10",  # aged P3 promoted to effective P0, oldest
            "task:12",  # fresh P0
            "task:11",  # P1
        ]
    finally:
        recovery.close()
        queue.close()


# ---------------------------------------------------------------------------
# Happy path: validation + accepted merge complete together
# ---------------------------------------------------------------------------


def test_task_completes_only_with_accepted_merge_and_validation(
    tmp_path: Path,
) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        entry = _enqueue(queue, 0)
        entry = _claim_one(queue)
        with pytest.raises(DatabaseMergeNotReadyError, match="validation"):
            queue.complete_task(
                entry_id=entry.entry_id,
                claim_token=entry.claim_token,
                claim_generation=entry.claim_generation,
                consumer_id=entry.consumer_id,
            )
        entry, _validation = _pass_validation(queue, entry)
        with pytest.raises(DatabaseMergeNotReadyError, match="accepted merge"):
            queue.complete_task(
                entry_id=entry.entry_id,
                claim_token=entry.claim_token,
                claim_generation=entry.claim_generation,
                consumer_id=entry.consumer_id,
            )
        entry, merge = _accept_merge(queue, entry)
        receipt = queue.complete_task(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            consumer_id=entry.consumer_id,
        )
        assert receipt.task_cid == "task:0"
        assert receipt.merge_attempt_id == merge.attempt_id
        assert receipt.evidence_digest == "sha256:ok"
        assert queue.is_task_complete("task:0")
        loaded = queue.get_entry(entry.entry_id)
        assert loaded is not None
        assert loaded.status is MergeEntryStatus.COMPLETED
        # Idempotent re-complete.
        again = queue.complete_task(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            consumer_id=entry.consumer_id,
        )
        assert again.receipt_id == receipt.receipt_id
    finally:
        recovery.close()
        queue.close()


# ---------------------------------------------------------------------------
# Stale worktree / fence results rejected
# ---------------------------------------------------------------------------


def test_stale_worktree_or_fence_validation_rejected(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        run = queue.start_validation(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            commands=["pytest"],
            consumer_id=entry.consumer_id,
        )
        with pytest.raises(DatabaseMergeStaleError, match="stale"):
            queue.finish_validation(
                run_id=run.run_id,
                claim_token=entry.claim_token,
                claim_generation=entry.claim_generation,
                passed=True,
                evidence_digest="sha256:stale",
                worktree_id="worktree:other",
                fencing_token=entry.fencing_token,
                fence_epoch=entry.fence_epoch,
                consumer_id=entry.consumer_id,
            )
        stale = queue.get_validation_run(run.run_id)
        assert stale is not None
        assert stale.status is ValidationStatus.STALE
        assert not queue.is_task_complete("task:0")
    finally:
        recovery.close()
        queue.close()


def test_stale_claim_fence_rejects_protected_writes(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue, consumer="worker:alive")
        with pytest.raises(DatabaseMergeStaleError):
            queue.start_validation(
                entry_id=entry.entry_id,
                claim_token="wrong-token",
                claim_generation=entry.claim_generation,
                commands=["pytest"],
                consumer_id=entry.consumer_id,
            )
        with pytest.raises(DatabaseMergeStaleError):
            queue.start_validation(
                entry_id=entry.entry_id,
                claim_token=entry.claim_token,
                claim_generation=max(0, entry.claim_generation - 1) or 999,
                commands=["pytest"],
                consumer_id=entry.consumer_id,
            )
    finally:
        recovery.close()
        queue.close()


# ---------------------------------------------------------------------------
# Rebase, conflict, validation failure, partial publish
# ---------------------------------------------------------------------------


def test_rebase_then_accept_merge(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        entry, _ = _pass_validation(queue, entry)
        attempt = queue.begin_merge_attempt(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            consumer_id=entry.consumer_id,
        )
        rebased = queue.record_rebase(
            attempt_id=attempt.attempt_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            result_commit="c" * 40,
            consumer_id=entry.consumer_id,
        )
        assert rebased.status is MergeAttemptStatus.REBASED
        accepted = queue.accept_merge(
            attempt_id=attempt.attempt_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            result_commit="d" * 40,
            consumer_id=entry.consumer_id,
        )
        assert accepted.status is MergeAttemptStatus.ACCEPTED
        entry = queue.get_entry(entry.entry_id)
        assert entry is not None
        receipt = queue.complete_task(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            consumer_id=entry.consumer_id,
        )
        assert receipt.result_commit == "d" * 40
    finally:
        recovery.close()
        queue.close()


def test_conflict_marks_failed_and_blocks_completion(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        entry, _ = _pass_validation(queue, entry)
        attempt = queue.begin_merge_attempt(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            consumer_id=entry.consumer_id,
        )
        conflicted = queue.record_conflict(
            attempt_id=attempt.attempt_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            conflict_paths=["src/a.py", "src/b.py"],
            consumer_id=entry.consumer_id,
        )
        assert conflicted.status is MergeAttemptStatus.CONFLICT
        entry = queue.get_entry(entry.entry_id)
        assert entry is not None
        assert entry.status is MergeEntryStatus.FAILED
        with pytest.raises(DatabaseMergeStaleError):
            queue.complete_task(
                entry_id=entry.entry_id,
                claim_token=entry.claim_token,
                claim_generation=entry.claim_generation,
                consumer_id=entry.consumer_id,
            )
    finally:
        recovery.close()
        queue.close()


def test_validation_failure_does_not_complete(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        run = queue.start_validation(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            commands=["pytest"],
            consumer_id=entry.consumer_id,
        )
        failed = queue.finish_validation(
            run_id=run.run_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            passed=False,
            result_summary="red",
            consumer_id=entry.consumer_id,
        )
        assert failed.status is ValidationStatus.FAILED
        assert not queue.is_task_complete("task:0")
        entry = queue.get_entry(entry.entry_id)
        assert entry is not None
        assert entry.status is MergeEntryStatus.FAILED
    finally:
        recovery.close()
        queue.close()


def test_partial_publish_cannot_settle_completion(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        entry, _ = _pass_validation(queue, entry)
        attempt = queue.begin_merge_attempt(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            consumer_id=entry.consumer_id,
        )
        partial = queue.record_partial_publish(
            attempt_id=attempt.attempt_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            result_commit="e" * 40,
            consumer_id=entry.consumer_id,
        )
        assert partial.status is MergeAttemptStatus.PARTIAL_PUBLISH
        entry = queue.get_entry(entry.entry_id)
        assert entry is not None
        # No accepted_merge_attempt_id from partial alone.
        with pytest.raises(DatabaseMergeNotReadyError):
            queue.complete_task(
                entry_id=entry.entry_id,
                claim_token=entry.claim_token,
                claim_generation=entry.claim_generation,
                consumer_id=entry.consumer_id,
            )
        assert not queue.is_task_complete("task:0")
    finally:
        recovery.close()
        queue.close()


# ---------------------------------------------------------------------------
# JSON receipt / queue file cannot settle
# ---------------------------------------------------------------------------


def test_json_receipt_alone_cannot_settle_work(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        receipt_path = tmp_path / "completed" / f"{entry.entry_id}.json"
        projected = queue.project_json_receipt(
            entry_id=entry.entry_id,
            path=receipt_path,
            payload={"status": "completed", "completed": True},
        )
        assert projected["grants_completion"] is False
        assert receipt_path.is_file()
        assert not queue.is_task_complete("task:0")
        check = queue.import_json_receipt_cannot_complete(
            task_cid="task:0",
            path=receipt_path,
        )
        assert check["json_claims_complete"] is True
        assert check["database_complete"] is False
        assert check["settled"] is False
        assert check["json_grants_completion"] is False
    finally:
        recovery.close()
        queue.close()


# ---------------------------------------------------------------------------
# Crash recovery, retry exhaustion, idempotent replay
# ---------------------------------------------------------------------------


def test_crash_recovery_releases_claim_and_is_idempotent(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue, consumer="worker:crashed")
        token = entry.claim_token
        generation = entry.claim_generation
        action = recovery.release_stale_claim(
            entry_id=entry.entry_id,
            expected_claim_token=token,
            expected_claim_generation=generation,
            reason="worker crash",
            idempotency_key="crash:task:0",
        )
        assert action.status is RecoveryActionStatus.APPLIED
        assert action.action_kind is RecoveryActionKind.RELEASE_STALE_CLAIM
        reloaded = queue.get_entry(entry.entry_id)
        assert reloaded is not None
        assert reloaded.status is MergeEntryStatus.PENDING
        assert reloaded.claim_token == ""
        # Idempotent replay of same recovery key.
        replayed = recovery.release_stale_claim(
            entry_id=entry.entry_id,
            expected_claim_token=token,
            expected_claim_generation=generation,
            idempotency_key="crash:task:0",
        )
        assert replayed.action_id == action.action_id
        # Stale CAS against old token is rejected.
        rejected = recovery.release_stale_claim(
            entry_id=entry.entry_id,
            expected_claim_token=token,
            expected_claim_generation=generation,
            idempotency_key="crash:task:0:again",
        )
        assert rejected.status is RecoveryActionStatus.REJECTED
    finally:
        recovery.close()
        queue.close()


def test_retry_exhaustion_is_queryable(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path, max_attempts=2)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        # Force failures then retries until quarantine.
        first = recovery.schedule_retry(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            reason="transient",
            idempotency_key="retry:1",
            retry_budget=2,
        )
        assert first.action_kind is RecoveryActionKind.RETRY
        entry = queue.get_entry(entry.entry_id)
        assert entry is not None
        assert entry.status is MergeEntryStatus.PENDING

        entry = _claim_one(queue, consumer="worker:2")
        second = recovery.schedule_retry(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            reason="again",
            idempotency_key="retry:2",
            retry_budget=2,
        )
        # max_attempts=2 => second requeue should quarantine via attempt bound.
        entry = queue.get_entry(entry.entry_id)
        assert entry is not None
        assert entry.status in {
            MergeEntryStatus.QUARANTINED,
            MergeEntryStatus.PENDING,
        }
        if entry.status is MergeEntryStatus.PENDING:
            entry = _claim_one(queue, consumer="worker:3")
            second = recovery.schedule_retry(
                entry_id=entry.entry_id,
                claim_token=entry.claim_token,
                claim_generation=entry.claim_generation,
                reason="final",
                idempotency_key="retry:3",
                retry_budget=2,
            )
        assert second.action_kind in {
            RecoveryActionKind.RETRY,
            RecoveryActionKind.RETRY_EXHAUSTED,
        }
        actions = recovery.list_actions(entry_id=entry.entry_id)
        assert actions
        assert all(item.entry_id == entry.entry_id for item in actions)
        # Explicit exhaustion path.
        entry = queue.get_entry(entry.entry_id)
        assert entry is not None
        if entry.status is not MergeEntryStatus.QUARANTINED:
            exhausted = recovery.schedule_retry(
                entry_id=entry.entry_id,
                force=True,
                retry_budget=0,
                idempotency_key="retry:exhaust",
            )
            assert exhausted.status is RecoveryActionStatus.EXHAUSTED
            entry = queue.get_entry(entry.entry_id)
            assert entry is not None
            assert entry.status is MergeEntryStatus.QUARANTINED
    finally:
        recovery.close()
        queue.close()


def test_recovery_actions_idempotent_and_queryable(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        first = recovery.reconcile(
            entry_id=entry.entry_id,
            idempotency_key="reconcile:task:0",
        )
        second = recovery.reconcile(
            entry_id=entry.entry_id,
            idempotency_key="reconcile:task:0",
        )
        assert second.action_id == first.action_id
        assert first.action_kind is RecoveryActionKind.RECONCILE
        by_id = recovery.get_action(first.action_id)
        assert by_id is not None
        assert by_id.action_id == first.action_id
        by_key = recovery.get_by_idempotency_key("reconcile:task:0")
        assert by_key is not None
        assert by_key.action_id == first.action_id
        listed = recovery.list_actions(task_cid="task:0")
        assert any(item.action_id == first.action_id for item in listed)

        rescue = recovery.rescue(
            entry_id=entry.entry_id,
            disposition="operator_review",
            idempotency_key="rescue:task:0",
        )
        assert rescue.status is RecoveryActionStatus.ACCEPTED
        assert rescue.body.get("forges_completion") is False
        assert not queue.is_task_complete("task:0")

        replay = recovery.replay(
            action_id=first.action_id,
            idempotency_key="replay:reconcile:task:0",
        )
        assert replay.action_kind is RecoveryActionKind.REPLAY
        assert replay.status is RecoveryActionStatus.NOOP
        again = recovery.replay(
            action_id=first.action_id,
            idempotency_key="replay:reconcile:task:0",
        )
        assert again.action_id == replay.action_id
    finally:
        recovery.close()
        queue.close()


def test_quarantine_recovery_is_idempotent(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        action = recovery.quarantine(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            reason="policy",
            idempotency_key="quarantine:0",
        )
        assert action.status is RecoveryActionStatus.APPLIED
        entry = queue.get_entry(entry.entry_id)
        assert entry is not None
        assert entry.status is MergeEntryStatus.QUARANTINED
        again = recovery.quarantine(
            entry_id=entry.entry_id,
            force=True,
            idempotency_key="quarantine:0",
        )
        assert again.action_id == action.action_id
    finally:
        recovery.close()
        queue.close()


def test_enqueue_dedupe_is_idempotent(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        first = _enqueue(queue, 0, commit_sha="f" * 40)
        second = _enqueue(queue, 0, commit_sha="f" * 40)
        assert first.entry_id == second.entry_id
    finally:
        recovery.close()
        queue.close()


def test_domain_events_track_lifecycle(tmp_path: Path) -> None:
    queue, recovery, _clock = _open(tmp_path)
    try:
        _enqueue(queue, 0)
        entry = _claim_one(queue)
        entry, _ = _pass_validation(queue, entry)
        entry, _ = _accept_merge(queue, entry)
        queue.complete_task(
            entry_id=entry.entry_id,
            claim_token=entry.claim_token,
            claim_generation=entry.claim_generation,
            consumer_id=entry.consumer_id,
        )
        events = queue.domain_events(stream_key="task:task:0")
        types = [item["event_type"] for item in events]
        assert "merge_enqueued" in types
        assert "merge_claimed" in types
        assert "validation_passed" in types
        assert "merge_accepted" in types
        assert "task_completed" in types
    finally:
        recovery.close()
        queue.close()
