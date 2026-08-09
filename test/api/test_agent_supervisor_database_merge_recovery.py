"""Tests for DatabaseMergeQueue + DatabaseRecovery (DQP-019).

Evidence subset: serialized merge, fairness, stale result, rebase, conflict,
validation failure, partial publish, crash, retry exhaustion, idempotent
replay.

Acceptance: A task completes only after accepted merge and current validation
evidence commit together; stale worktree/fence results are rejected; recovery
actions are idempotent and queryable; no JSON receipt or queue file alone can
settle work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_merge_queue import (
    DATABASE_MERGE_QUEUE_INTERFACE,
    VALIDATION_RUN_INTERFACE,
    DatabaseMergeQueue,
    DatabaseMergeQueueError,
    DatabaseMergeQueueNotReadyError,
    DatabaseMergeQueueStaleFenceError,
    EntryStatus,
    MergeAttemptStatus,
    MergeOutcome,
    ValidationStatus,
    duckdb_available as merge_duckdb_available,
    open_database_merge_queue,
)
from ipfs_accelerate_py.agent_supervisor.rescue.database_recovery import (
    DATABASE_RECOVERY_INTERFACE,
    RECOVERY_ACTION_INTERFACE,
    ActionKind,
    ActionStatus,
    DatabaseRecovery,
    DatabaseRecoveryError,
    DatabaseRecoveryExhaustedError,
    SubjectKind,
    SubjectStatus,
    duckdb_available as recovery_duckdb_available,
    open_database_recovery,
)

pytestmark = pytest.mark.skipif(
    not (merge_duckdb_available() and recovery_duckdb_available()),
    reason="DuckDB is required for DQP-019 hermetic tests",
)


class FakeClock:
    def __init__(self, start_ms: int = 1_000_000) -> None:
        self.now = int(start_ms)

    def __call__(self) -> int:
        return int(self.now)

    def advance(self, ms: int) -> None:
        self.now += int(ms)


def _open_queue(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    max_attempts: int = 3,
    max_processing: int = 1,
    priority_aging_ms: int = 0,
) -> tuple[DatabaseMergeQueue, FakeClock]:
    clock = clock or FakeClock()
    queue = open_database_merge_queue(
        tmp_path / "merge_queue.duckdb",
        clock_ms=clock,
        max_attempts=max_attempts,
        max_processing=max_processing,
        priority_aging_ms=priority_aging_ms,
    )
    return queue, clock


def _open_recovery(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    max_retries: int = 3,
) -> tuple[DatabaseRecovery, FakeClock]:
    clock = clock or FakeClock()
    recovery = open_database_recovery(
        tmp_path / "recovery.duckdb",
        clock_ms=clock,
        max_retries=max_retries,
    )
    return recovery, clock


def _enqueue(
    queue: DatabaseMergeQueue,
    ordinal: int,
    *,
    priority: str = "P2",
    repository_id: str = "repository:demo",
    target_branch: str = "main",
    fencing_token: int = 1,
    fence_epoch: int = 1,
    worktree_id: str | None = None,
    commit_sha: str | None = None,
):
    return queue.enqueue(
        repository_id=repository_id,
        target_branch=target_branch,
        source_branch=f"candidate/{ordinal}",
        task_cid=f"task:cid:{ordinal:03d}",
        worktree_id=worktree_id or f"worktree:{ordinal:03d}",
        commit_sha=commit_sha or f"{ordinal + 1:040x}",
        priority=priority,
        fencing_token=fencing_token,
        fence_epoch=fence_epoch,
    )


def _happy_path_to_accepted(
    queue: DatabaseMergeQueue,
    entry,
    *,
    consumer_id: str = "merge-train:test",
    evidence_digest: str = "sha256:validation-ok",
    result_commit_id: str = "deadbeef" * 5,
):
    if entry.status is EntryStatus.PENDING or not entry.claim_token:
        claimed = queue.claim_next(
            repository_id=entry.repository_id,
            target_branch=entry.target_branch,
            consumer_id=consumer_id,
        )
        assert len(claimed) == 1
        current = claimed[0]
    else:
        current = entry
    run = queue.start_validation(current, argv=["pytest", "-q"])
    run = queue.finish_validation(
        run,
        outcome="passed",
        evidence_digest=evidence_digest,
    )
    current = queue.get_entry(current.entry_id)
    assert current is not None
    attempt = queue.start_merge_attempt(current)
    attempt = queue.finish_merge_attempt(
        attempt,
        outcome=MergeOutcome.ACCEPTED,
        result_commit_id=result_commit_id,
    )
    current = queue.get_entry(current.entry_id)
    assert current is not None
    return current, run, attempt


# ---------------------------------------------------------------------------
# Interface identities / authority
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_MERGE_QUEUE_INTERFACE == "DatabaseMergeQueue@1"
    assert VALIDATION_RUN_INTERFACE == "ValidationRun@1"
    assert DATABASE_RECOVERY_INTERFACE == "DatabaseRecovery@1"
    assert RECOVERY_ACTION_INTERFACE == "RecoveryAction@1"
    assert DatabaseMergeQueue.INTERFACE == DATABASE_MERGE_QUEUE_INTERFACE
    assert DatabaseRecovery.INTERFACE == DATABASE_RECOVERY_INTERFACE


def test_authority_policy_rejects_json_settlement(tmp_path: Path) -> None:
    queue, _clock = _open_queue(tmp_path)
    recovery, _ = _open_recovery(tmp_path)
    try:
        policy = queue.authority_policy()
        assert policy["semantic_authority"] == "database"
        assert policy["settlement_authority"] == "database"
        assert policy["json_receipt_authority"] == "none"
        recovery_policy = recovery.authority_policy()
        assert recovery_policy["recovery_authority"] == "database"
        assert recovery_policy["json_receipt_authority"] == "none"
    finally:
        queue.close()
        recovery.close()


# ---------------------------------------------------------------------------
# Serialized merge / fairness
# ---------------------------------------------------------------------------


def test_serialized_merge_admits_one_active_claim_per_target(tmp_path: Path) -> None:
    queue, _clock = _open_queue(tmp_path, max_processing=1)
    try:
        first = _enqueue(queue, 0, priority="P1")
        second = _enqueue(queue, 1, priority="P0")
        claimed = queue.claim_next(
            repository_id=first.repository_id,
            target_branch=first.target_branch,
            consumer_id="worker:a",
            limit=2,
        )
        assert len(claimed) == 1
        assert claimed[0].task_cid == second.task_cid  # higher priority first
        blocked = queue.claim_next(
            repository_id=first.repository_id,
            target_branch=first.target_branch,
            consumer_id="worker:b",
            limit=1,
        )
        assert blocked == ()
        active = queue.list_entries(status=EntryStatus.CLAIMED)
        assert len(active) == 1
    finally:
        queue.close()


def test_fairness_orders_same_priority_by_enqueue_time(tmp_path: Path) -> None:
    clock = FakeClock()
    queue, clock = _open_queue(tmp_path, clock=clock, max_processing=3, priority_aging_ms=0)
    try:
        a = _enqueue(queue, 0, priority="P1")
        clock.advance(10)
        b = _enqueue(queue, 1, priority="P1")
        clock.advance(10)
        c = _enqueue(queue, 2, priority="P0")
        claimed = queue.claim_next(
            repository_id=a.repository_id,
            target_branch=a.target_branch,
            consumer_id="worker:fair",
            limit=3,
        )
        assert [item.task_cid for item in claimed] == [
            c.task_cid,
            a.task_cid,
            b.task_cid,
        ]
    finally:
        queue.close()


# ---------------------------------------------------------------------------
# Validation + merge + settlement
# ---------------------------------------------------------------------------


def test_settlement_requires_accepted_merge_and_passed_validation(
    tmp_path: Path,
) -> None:
    queue, _clock = _open_queue(tmp_path)
    try:
        entry = _enqueue(queue, 0)
        claimed = queue.claim_next(
            repository_id=entry.repository_id,
            target_branch=entry.target_branch,
            consumer_id="worker:settle",
        )[0]

        with pytest.raises(DatabaseMergeQueueNotReadyError):
            queue.settle(claimed)

        run = queue.start_validation(claimed, argv=["python", "-m", "pytest", "-q"])
        with pytest.raises(DatabaseMergeQueueNotReadyError):
            queue.start_merge_attempt(claimed)

        run = queue.finish_validation(
            run, outcome="passed", evidence_digest="sha256:ok"
        )
        assert run.status is ValidationStatus.PASSED
        claimed = queue.get_entry(claimed.entry_id)
        assert claimed is not None

        attempt = queue.start_merge_attempt(claimed)
        with pytest.raises(DatabaseMergeQueueNotReadyError):
            queue.settle(claimed)

        attempt = queue.finish_merge_attempt(
            attempt,
            outcome="accepted",
            result_commit_id="cafebabe" * 5,
        )
        assert attempt.status is MergeAttemptStatus.ACCEPTED
        claimed = queue.get_entry(claimed.entry_id)
        assert claimed is not None
        assert claimed.status is EntryStatus.ACCEPTED

        receipt = queue.settle(claimed)
        assert receipt.validation_run_id == run.run_id
        assert receipt.merge_attempt_id == attempt.merge_attempt_id
        assert receipt.evidence_digest == "sha256:ok"
        assert receipt.result_commit_id == "cafebabe" * 5
        settled = queue.get_entry(claimed.entry_id)
        assert settled is not None
        assert settled.status is EntryStatus.SETTLED
        assert settled.settlement_id == receipt.settlement_id
        # Idempotent settlement replay
        replay = queue.settle(claimed)
        assert replay.settlement_id == receipt.settlement_id
    finally:
        queue.close()


def test_json_receipt_cannot_settle_work(tmp_path: Path) -> None:
    queue, _clock = _open_queue(tmp_path)
    try:
        entry = _enqueue(queue, 0)
        current, _run, _attempt = _happy_path_to_accepted(queue, entry)
        with pytest.raises(DatabaseMergeQueueError, match="JSON receipt"):
            queue.settle(
                current,
                body={"json_receipt_path": "/tmp/fake-receipt.json"},
            )
        with pytest.raises(DatabaseMergeQueueError, match="JSON receipt"):
            queue.settle(
                current,
                body={"queue_file": "/tmp/queue/completed/entry.json"},
            )
        # Without JSON authority fields, settlement still works from DB rows.
        receipt = queue.settle(current)
        assert receipt.settlement_id
        assert receipt.to_dict()["json_receipt_authority"] == "none"
    finally:
        queue.close()


# ---------------------------------------------------------------------------
# Stale fence / worktree rejection
# ---------------------------------------------------------------------------


def test_stale_worktree_and_fence_results_are_rejected(tmp_path: Path) -> None:
    queue, _clock = _open_queue(tmp_path)
    try:
        entry = _enqueue(queue, 0, fencing_token=7, fence_epoch=3)
        claimed = queue.claim_next(
            repository_id=entry.repository_id,
            target_branch=entry.target_branch,
            consumer_id="worker:owner",
        )[0]
        run = queue.start_validation(claimed, argv=["pytest"])
        run = queue.finish_validation(
            run, outcome="passed", evidence_digest="sha256:ok"
        )
        claimed = queue.get_entry(claimed.entry_id)
        assert claimed is not None

        # Reconstruct a stale view with wrong worktree / fence / claim token.
        from dataclasses import replace

        impostor = replace(
            claimed,
            worktree_id="worktree:stale",
        )
        with pytest.raises(DatabaseMergeQueueStaleFenceError, match="worktree"):
            queue.start_merge_attempt(impostor)

        impostor = replace(
            claimed,
            fencing_token=claimed.fencing_token + 1,
        )
        with pytest.raises(DatabaseMergeQueueStaleFenceError, match="fencing"):
            queue.start_merge_attempt(impostor)

        impostor = replace(claimed, claim_token="stale-token")
        with pytest.raises(DatabaseMergeQueueStaleFenceError, match="claim token"):
            queue.start_merge_attempt(impostor)

        # Current owner still succeeds.
        attempt = queue.start_merge_attempt(claimed)
        assert attempt.worktree_id == claimed.worktree_id
        assert queue.owns_claim(claimed) is True
        assert queue.owns_claim(claimed, claim_token="stale") is False
    finally:
        queue.close()


# ---------------------------------------------------------------------------
# Validation failure / conflict / rebase / partial publish
# ---------------------------------------------------------------------------


def test_validation_failure_blocks_merge_and_settlement(tmp_path: Path) -> None:
    queue, _clock = _open_queue(tmp_path)
    try:
        entry = _enqueue(queue, 0)
        claimed = queue.claim_next(
            repository_id=entry.repository_id,
            target_branch=entry.target_branch,
            consumer_id="worker:val",
        )[0]
        run = queue.start_validation(claimed, argv=["pytest", "-q"])
        run = queue.finish_validation(
            run, outcome="failed", evidence_digest="sha256:fail"
        )
        assert run.status is ValidationStatus.FAILED
        failed = queue.get_entry(claimed.entry_id)
        assert failed is not None
        assert failed.status is EntryStatus.FAILED
        with pytest.raises(DatabaseMergeQueueStaleFenceError):
            # claim still present on failed entry only if status allows; finish
            # validation moved status to failed which is not claim-active for merge.
            queue.start_merge_attempt(claimed)
    finally:
        queue.close()


def test_conflict_rebase_and_partial_publish_leave_unsettled(tmp_path: Path) -> None:
    queue, _clock = _open_queue(tmp_path, max_processing=3)
    try:
        outcomes = (
            (MergeOutcome.CONFLICT, EntryStatus.CONFLICT, "merge_conflict"),
            (MergeOutcome.REBASE_REQUIRED, EntryStatus.FAILED, "rebase_required"),
            (MergeOutcome.PARTIAL_PUBLISH, EntryStatus.FAILED, "partial_publish"),
        )
        for index, (outcome, expected_status, reason) in enumerate(outcomes):
            entry = _enqueue(queue, index, priority="P0")
            claimed = queue.claim_next(
                repository_id=entry.repository_id,
                target_branch=entry.target_branch,
                consumer_id=f"worker:{index}",
            )[0]
            run = queue.start_validation(claimed, argv=["pytest"])
            queue.finish_validation(
                run, outcome="passed", evidence_digest=f"sha256:ok-{index}"
            )
            claimed = queue.get_entry(claimed.entry_id)
            assert claimed is not None
            attempt = queue.start_merge_attempt(claimed)
            finished = queue.finish_merge_attempt(attempt, outcome=outcome)
            assert finished.status is MergeAttemptStatus(outcome.value)
            current = queue.get_entry(claimed.entry_id)
            assert current is not None
            assert current.status is expected_status
            assert current.failure_reason == reason
            assert current.settlement_id == ""
            with pytest.raises(
                (DatabaseMergeQueueNotReadyError, DatabaseMergeQueueStaleFenceError)
            ):
                queue.settle(current)
    finally:
        queue.close()


# ---------------------------------------------------------------------------
# Crash recovery / retry exhaustion
# ---------------------------------------------------------------------------


def test_crash_recovery_releases_claim_and_rejects_stale_worker(
    tmp_path: Path,
) -> None:
    queue, _clock = _open_queue(tmp_path, max_attempts=3)
    try:
        entry = _enqueue(queue, 0)
        claimed = queue.claim_next(
            repository_id=entry.repository_id,
            target_branch=entry.target_branch,
            consumer_id="worker:crashed",
        )[0]
        run = queue.start_validation(claimed, argv=["pytest"])
        assert run.status is ValidationStatus.RUNNING

        recovered = queue.recover_stale_claim(claimed.entry_id, reason="process_crash")
        assert recovered.status is EntryStatus.PENDING
        assert recovered.claim_token == ""
        interrupted = queue.get_validation_run(run.run_id)
        assert interrupted is not None
        assert interrupted.status is ValidationStatus.INTERRUPTED

        with pytest.raises(DatabaseMergeQueueStaleFenceError):
            queue.finish_validation(
                run, outcome="passed", evidence_digest="sha256:stale"
            )

        # Replacement worker can reclaim and complete.
        replacement = queue.claim_next(
            repository_id=entry.repository_id,
            target_branch=entry.target_branch,
            consumer_id="worker:replacement",
        )[0]
        assert replacement.claim_generation > claimed.claim_generation
        current, _run, _attempt = _happy_path_to_accepted(
            queue,
            replacement,
            consumer_id="worker:replacement",
            evidence_digest="sha256:replacement-ok",
            result_commit_id="beadface" * 5,
        )
        assert current.status is EntryStatus.ACCEPTED
        receipt = queue.settle(current)
        assert receipt.task_cid == entry.task_cid
    finally:
        queue.close()


def test_retry_exhaustion_quarantines_entry(tmp_path: Path) -> None:
    queue, _clock = _open_queue(tmp_path, max_attempts=2)
    try:
        entry = _enqueue(queue, 0)
        for attempt_index in range(2):
            claimed = queue.claim_next(
                repository_id=entry.repository_id,
                target_branch=entry.target_branch,
                consumer_id=f"worker:{attempt_index}",
            )
            assert claimed
            current = claimed[0]
            run = queue.start_validation(current, argv=["pytest"])
            queue.finish_validation(
                run, outcome="failed", evidence_digest=f"sha256:fail-{attempt_index}"
            )
            current = queue.get_entry(current.entry_id)
            assert current is not None
            assert current.status is EntryStatus.FAILED
            assert current.claim_token
            current = queue.requeue(current, reason="validation_failed")
            if current.status is EntryStatus.QUARANTINED:
                break
        final = queue.get_entry(entry.entry_id)
        assert final is not None
        assert final.status is EntryStatus.QUARANTINED
        assert (
            queue.claim_next(
                repository_id=entry.repository_id,
                target_branch=entry.target_branch,
                consumer_id="worker:late",
            )
            == ()
        )
    finally:
        queue.close()


# ---------------------------------------------------------------------------
# Recovery actions: idempotent replay / reconciliation / rescue
# ---------------------------------------------------------------------------


def test_recovery_actions_are_idempotent_and_queryable(tmp_path: Path) -> None:
    recovery, _clock = _open_recovery(tmp_path, max_retries=3)
    try:
        subject = recovery.register_subject(
            subject_kind=SubjectKind.MERGE_ENTRY,
            subject_ref="entry-1",
            task_cid="task:cid:001",
            entry_id="entry:1",
            worktree_id="worktree:1",
            fencing_token=4,
            fence_epoch=2,
        )
        assert subject.status is SubjectStatus.OPEN

        first = recovery.decide_and_apply(
            subject_id=subject.subject_id,
            action_kind=ActionKind.RECONCILE,
            reason="restart",
            idempotency_key="idem:reconcile:1",
            body={"cursor": 12},
            result={"replayed_event_count": 3, "ok": True},
        )
        assert first.status in {ActionStatus.APPLIED, ActionStatus.REPLAYED}
        assert first.result_digest

        second = recovery.decide_and_apply(
            subject_id=subject.subject_id,
            action_kind=ActionKind.RECONCILE,
            reason="restart",
            idempotency_key="idem:reconcile:1",
            body={"cursor": 12},
            result={"replayed_event_count": 3, "ok": True},
        )
        assert second.action_id == first.action_id
        assert second.status is ActionStatus.REPLAYED
        assert second.result_digest == first.result_digest

        listed = recovery.list_actions(subject_id=subject.subject_id)
        assert len(listed) == 1
        assert listed[0].action_id == first.action_id
        by_key = recovery.get_action_by_idempotency_key("idem:reconcile:1")
        assert by_key is not None
        assert by_key.action_id == first.action_id

        receipts = recovery.list_reconciliation_receipts(subject_id=subject.subject_id)
        assert len(receipts) == 1
        assert receipts[0].replayed_event_count == 3
        events = recovery.events(subject_id=subject.subject_id)
        types = {item["event_type"] for item in events}
        assert "action_decided" in types
        assert "action_applied" in types or "action_replayed" in types
    finally:
        recovery.close()


def test_recovery_retry_exhaustion_and_rescue(tmp_path: Path) -> None:
    recovery, _clock = _open_recovery(tmp_path, max_retries=2)
    try:
        subject = recovery.register_subject(
            subject_kind=SubjectKind.TASK,
            subject_ref="task-retry",
            task_cid="task:cid:retry",
            max_retries=2,
        )
        first = recovery.decide_and_apply(
            subject_id=subject.subject_id,
            action_kind=ActionKind.RETRY,
            reason="validation_failed",
            idempotency_key="idem:retry:1",
            result={"attempt": 1},
        )
        assert first.status is ActionStatus.APPLIED
        subject = recovery.get_subject(subject.subject_id)
        assert subject is not None
        assert subject.retry_count == 1

        second = recovery.decide_and_apply(
            subject_id=subject.subject_id,
            action_kind=ActionKind.RETRY,
            reason="validation_failed",
            idempotency_key="idem:retry:2",
            result={"attempt": 2},
        )
        assert second.status is ActionStatus.EXHAUSTED
        subject = recovery.get_subject(subject.subject_id)
        assert subject is not None
        assert subject.status is SubjectStatus.QUARANTINED

        with pytest.raises(DatabaseRecoveryExhaustedError):
            recovery.decide_action(
                subject_id=subject.subject_id,
                action_kind=ActionKind.RETRY,
                reason="again",
                idempotency_key="idem:retry:3",
            )

        rescue = recovery.decide_and_apply(
            subject_id=subject.subject_id,
            action_kind=ActionKind.RESCUE,
            reason="operator_rescue",
            idempotency_key="idem:rescue:1",
            result={"rescued": True},
        )
        assert rescue.status is ActionStatus.APPLIED
        subject = recovery.get_subject(subject.subject_id)
        assert subject is not None
        assert subject.status is SubjectStatus.RESCUED
    finally:
        recovery.close()


def test_recovery_rejects_json_receipt_authority(tmp_path: Path) -> None:
    recovery, _clock = _open_recovery(tmp_path)
    try:
        subject = recovery.register_subject(
            subject_kind=SubjectKind.SESSION,
            subject_ref="session-1",
        )
        with pytest.raises(DatabaseRecoveryError, match="JSON receipt"):
            recovery.decide_action(
                subject_id=subject.subject_id,
                action_kind=ActionKind.REPLAY,
                body={"json_receipt_path": "/tmp/receipt.json"},
            )
        with pytest.raises(DatabaseRecoveryError, match="JSON receipt"):
            action = recovery.decide_action(
                subject_id=subject.subject_id,
                action_kind=ActionKind.FENCE_STALE_CLAIM,
                reason="stale",
                idempotency_key="idem:fence:1",
            )
            recovery.apply_action(
                action,
                result={"queue_file": "/tmp/queue.json"},
            )
    finally:
        recovery.close()


def test_merge_and_recovery_share_task_worktree_fence_coordinates(
    tmp_path: Path,
) -> None:
    queue, clock = _open_queue(tmp_path)
    recovery, _ = _open_recovery(tmp_path, clock=clock)
    try:
        entry = _enqueue(
            queue,
            0,
            fencing_token=11,
            fence_epoch=5,
            worktree_id="worktree:shared",
        )
        claimed = queue.claim_next(
            repository_id=entry.repository_id,
            target_branch=entry.target_branch,
            consumer_id="worker:shared",
        )[0]
        subject = recovery.register_subject(
            subject_kind=SubjectKind.MERGE_ENTRY,
            subject_ref=claimed.entry_id,
            task_cid=claimed.task_cid,
            entry_id=claimed.entry_id,
            worktree_id=claimed.worktree_id,
            fencing_token=claimed.fencing_token,
            fence_epoch=claimed.fence_epoch,
        )
        action = recovery.decide_and_apply(
            subject_id=subject.subject_id,
            action_kind=ActionKind.INTERRUPT_MERGE,
            reason="partial_publish",
            idempotency_key=f"idem:interrupt:{claimed.entry_id}",
            result={"interrupted": True},
        )
        assert action.task_cid == claimed.task_cid
        assert action.worktree_id == claimed.worktree_id
        assert action.fencing_token == claimed.fencing_token
        assert action.fence_epoch == claimed.fence_epoch
        assert action.entry_id == claimed.entry_id

        recovered = queue.recover_stale_claim(claimed.entry_id, reason="partial_publish")
        assert recovered.status is EntryStatus.PENDING
        actions = recovery.list_actions(task_cid=claimed.task_cid)
        assert any(item.action_kind is ActionKind.INTERRUPT_MERGE for item in actions)
    finally:
        queue.close()
        recovery.close()
