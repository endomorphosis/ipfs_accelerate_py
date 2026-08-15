from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge import merge_queue as merge_queue_module
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    _LEGACY_JSON_IMPORT_MARKER,
    MergeQueue,
    MergeQueueFenceError,
    MergeQueueFullError,
    MergeQueueIntegrityError,
    MergeRequest,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
)


def _enqueue(
    queue: MergeQueue,
    ordinal: int,
    *,
    priority: str = "P2",
    worktree_bytes: int = 0,
):
    metadata = {"worktree_bytes": worktree_bytes} if worktree_bytes else {}
    return queue.enqueue(
        branch_name=f"candidate/{ordinal}",
        task_id=f"TASK-{ordinal}",
        canonical_task_id=f"canonical-task-{ordinal}",
        commit_sha=f"{ordinal + 1:040x}",
        priority=priority,
        metadata=metadata,
    )


def _legacy_request(
    ordinal: int,
    *,
    request_id: str | None = None,
    canonical_task_id: str | None = None,
    commit_sha: str | None = None,
) -> MergeRequest:
    return MergeRequest(
        request_id=request_id or f"legacy-request-{ordinal}",
        branch_name=f"legacy/{ordinal}",
        task_id=f"LEGACY-{ordinal}",
        priority="P2",
        lane_id="legacy",
        enqueued_at=float(ordinal),
        canonical_task_id=canonical_task_id or f"legacy-canonical-{ordinal}",
        commit_sha=commit_sha or f"{ordinal + 1:040x}",
    )


def _write_legacy_receipt(path: Path, request: MergeRequest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(request.to_dict(), sort_keys=True),
        encoding="utf-8",
    )


def _clear_legacy_import_marker(queue: MergeQueue) -> None:
    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM agent_supervisor_store_metadata WHERE key=?",
            (_LEGACY_JSON_IMPORT_MARKER,),
        )
        connection.commit()


def _set_legacy_import_marker(queue: MergeQueue, value: str) -> None:
    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE agent_supervisor_store_metadata
            SET value=?
            WHERE key=?
            """,
            (value, _LEGACY_JSON_IMPORT_MARKER),
        )
        connection.commit()


def _has_legacy_import_marker(queue: MergeQueue) -> bool:
    with queue._connect() as connection:
        rows = connection.execute(
            "SELECT key FROM agent_supervisor_store_metadata"
        ).fetchall()
    return any(
        str(row["key"]) == _LEGACY_JSON_IMPORT_MARKER
        for row in rows
    )


def test_legacy_json_receipts_are_imported_and_marked_once(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue"
    request = _legacy_request(0)
    _write_legacy_receipt(
        queue_path / "completed" / f"{request.request_id}.json",
        request,
    )

    queue = MergeQueue(queue_path)

    stored = queue.get(request.request_id)
    assert stored is not None
    assert stored.status == "completed"
    assert _has_legacy_import_marker(queue) is True


def test_completed_legacy_import_does_not_rescan_new_projection_files(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    late_receipt = _legacy_request(1)
    _write_legacy_receipt(
        queue.pending_dir / f"{late_receipt.request_id}.json",
        late_receipt,
    )

    restarted = MergeQueue(queue_path)

    assert restarted.get(late_receipt.request_id) is None
    assert _has_legacy_import_marker(restarted) is True


def test_legacy_import_full_scan_skips_existing_authoritative_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    existing = _enqueue(queue, 0)
    _clear_legacy_import_marker(queue)

    def fail_if_inserted(*_args, **_kwargs) -> None:
        raise AssertionError("an existing stage projection must not be reinserted")

    monkeypatch.setattr(MergeQueue, "_insert", fail_if_inserted)
    restarted = MergeQueue(queue_path)

    assert restarted.get(existing.request_id) is not None
    assert _has_legacy_import_marker(restarted) is True


@pytest.mark.parametrize(
    ("authoritative_commit", "receipt_commit"),
    (
        ("", "a" * 40),
        ("a" * 40, ""),
    ),
    ids=("database-empty", "receipt-empty"),
)
def test_legacy_import_preserves_primary_identity_when_one_dedupe_is_empty(
    tmp_path: Path,
    authoritative_commit: str,
    receipt_commit: str,
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    existing = queue.enqueue(
        branch_name="candidate/legacy-primary",
        task_id="LEGACY-PRIMARY",
        canonical_task_id="canonical-legacy-primary",
        commit_sha=authoritative_commit,
    )
    _clear_legacy_import_marker(queue)
    _write_legacy_receipt(
        queue.pending_dir / f"{existing.request_id}.json",
        replace(existing, commit_sha=receipt_commit),
    )

    restarted = MergeQueue(queue_path)

    durable = restarted.get(existing.request_id)
    assert durable is not None
    assert durable.dedupe_key == existing.dedupe_key
    assert _has_legacy_import_marker(restarted) is True


def test_legacy_import_fails_closed_on_unknown_marker_value(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    _set_legacy_import_marker(queue, "partial")

    with pytest.raises(
        MergeQueueIntegrityError,
        match="import marker is invalid",
    ):
        MergeQueue(queue_path)


def test_legacy_import_fails_closed_on_request_id_identity_conflict(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    existing = _enqueue(queue, 0)
    _clear_legacy_import_marker(queue)
    conflicting = _legacy_request(
        2,
        request_id=existing.request_id,
    )
    _write_legacy_receipt(
        queue.processing_dir / f"{conflicting.request_id}.json",
        conflicting,
    )

    with pytest.raises(
        MergeQueueIntegrityError,
        match="authoritative dedupe identity",
    ):
        MergeQueue(queue_path)

    assert _has_legacy_import_marker(queue) is False
    durable = queue.get(existing.request_id)
    assert durable is not None
    assert durable.dedupe_key == existing.dedupe_key


def test_legacy_import_conflict_rolls_back_rows_and_completion_marker(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    existing = _enqueue(queue, 0)
    _clear_legacy_import_marker(queue)
    missing = _legacy_request(3)
    _write_legacy_receipt(
        queue.pending_dir / f"{missing.request_id}.json",
        missing,
    )
    conflicting = _legacy_request(
        4,
        request_id="different-request-id",
        canonical_task_id=existing.canonical_task_id,
        commit_sha=existing.commit_sha,
    )
    _write_legacy_receipt(
        queue.processing_dir / f"{conflicting.request_id}.json",
        conflicting,
    )

    with pytest.raises(
        MergeQueueIntegrityError,
        match="authoritative request identity",
    ):
        MergeQueue(queue_path)

    assert queue.get(missing.request_id) is None
    assert _has_legacy_import_marker(queue) is False
    assert queue.get(existing.request_id) is not None


def test_enqueue_dedupe_lookup_uses_an_unfiltered_authoritative_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    existing = _enqueue(queue, 0)
    original_execute = DuckDBConnection.execute
    observed_scans: list[str] = []

    def reject_filtered_dedupe_sql(
        connection: DuckDBConnection,
        sql: str,
        parameters=None,
    ):
        normalized = " ".join(str(sql).upper().split())
        assert "WHERE DEDUPE_KEY" not in normalized
        if normalized == "SELECT * FROM MERGE_REQUESTS":
            observed_scans.append(normalized)
        return original_execute(connection, sql, parameters)

    monkeypatch.setattr(
        DuckDBConnection,
        "execute",
        reject_filtered_dedupe_sql,
    )
    duplicate = queue.enqueue(
        branch_name="candidate/duplicate",
        task_id="TASK-DUPLICATE",
        canonical_task_id=existing.canonical_task_id,
        commit_sha=existing.commit_sha,
    )

    assert duplicate.request_id == existing.request_id
    assert observed_scans == ["SELECT * FROM MERGE_REQUESTS"]


def test_enqueue_exception_recovery_reuses_the_authoritative_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    existing = _enqueue(queue, 0)
    original_find = queue._find_by_dedupe_key
    calls: list[str] = []

    def miss_once(
        connection: DuckDBConnection,
        dedupe_key: str,
    ):
        calls.append(dedupe_key)
        if len(calls) == 1:
            return None
        return original_find(connection, dedupe_key)

    original_execute = DuckDBConnection.execute

    def reject_filtered_dedupe_sql(
        connection: DuckDBConnection,
        sql: str,
        parameters=None,
    ):
        normalized = " ".join(str(sql).upper().split())
        assert "WHERE DEDUPE_KEY" not in normalized
        return original_execute(connection, sql, parameters)

    monkeypatch.setattr(queue, "_find_by_dedupe_key", miss_once)
    monkeypatch.setattr(
        DuckDBConnection,
        "execute",
        reject_filtered_dedupe_sql,
    )
    duplicate = queue.enqueue(
        branch_name="candidate/recovered-duplicate",
        task_id="TASK-RECOVERED-DUPLICATE",
        canonical_task_id=existing.canonical_task_id,
        commit_sha=existing.commit_sha,
    )

    assert duplicate.request_id == existing.request_id
    assert calls == [existing.dedupe_key, existing.dedupe_key]


def test_authoritative_dedupe_scan_fails_closed_on_multiple_matches(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    existing = _enqueue(queue, 0)
    duplicate = replace(
        existing,
        request_id="duplicate-logical-dedupe",
    )

    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP INDEX merge_requests_dedupe")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        queue._insert(connection, duplicate, ignore=False)
        connection.commit()
        with pytest.raises(
            MergeQueueIntegrityError,
            match="multiple requests for dedupe_key",
        ):
            queue._find_by_dedupe_key(connection, existing.dedupe_key)


def test_batch_claims_have_a_deterministic_total_order_and_unique_fences(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(
        tmp_path / "queue",
        clock=lambda: 100.0,
        max_processing=8,
        priority_aging_seconds=0,
    )
    low = _enqueue(queue, 0, priority="P3")
    high_b = _enqueue(queue, 1, priority="P0")
    high_a = _enqueue(queue, 2, priority="P0")
    medium = _enqueue(queue, 3, priority="P1")

    claimed = queue.dequeue_many(8, consumer_id="merge-train:deterministic")

    same_priority = sorted((high_a.request_id, high_b.request_id))
    assert [request.request_id for request in claimed] == [
        *same_priority,
        medium.request_id,
        low.request_id,
    ]
    assert all(request.consumer_id == "merge-train:deterministic" for request in claimed)
    assert all(request.claim_token for request in claimed)
    assert all(request.claim_generation == 1 for request in claimed)
    assert len({request.claim_token for request in claimed}) == len(claimed)


@pytest.mark.parametrize(
    "stale_request",
    (
        lambda claimed: replace(claimed, consumer_id="merge-train:impostor"),
        lambda claimed: replace(claimed, claim_token="stale-token"),
        lambda claimed: replace(
            claimed, claim_generation=max(0, claimed.claim_generation - 1)
        ),
    ),
    ids=("wrong-owner", "wrong-token", "stale-generation"),
)
def test_completion_requires_the_exact_current_claim_fence(
    tmp_path: Path, stale_request
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    pending = _enqueue(queue, 0)
    claimed = queue.dequeue(consumer_id="merge-train:owner")
    assert claimed is not None

    with pytest.raises(MergeQueueFenceError):
        queue.complete(stale_request(claimed))

    stored = queue.get(pending.request_id)
    assert stored is not None
    assert stored.status == "processing"
    assert stored.consumer_id == claimed.consumer_id
    assert stored.claim_token == claimed.claim_token
    queue.complete(claimed, metadata={"validated": True})
    assert queue.get(pending.request_id).status == "completed"  # type: ignore[union-attr]


def test_recovered_claim_increments_generation_and_fences_crashed_worker(
    tmp_path: Path,
) -> None:
    now = [10.0]
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        clock=lambda: now[0],
        max_age_seconds=5,
        max_attempts=3,
    )
    pending = _enqueue(queue, 0)
    crashed_claim = queue.dequeue(consumer_id="worker:crashed")
    assert crashed_claim is not None

    now[0] = 20.0
    restarted = MergeQueue(
        queue_path,
        clock=lambda: now[0],
        max_age_seconds=5,
        max_attempts=3,
    )
    replacement = restarted.dequeue(consumer_id="worker:replacement")
    assert replacement is not None
    assert replacement.request_id == pending.request_id
    assert replacement.claim_generation > crashed_claim.claim_generation
    assert replacement.claim_token != crashed_claim.claim_token

    with pytest.raises(MergeQueueFenceError):
        restarted.complete(crashed_claim)
    assert restarted.get(pending.request_id).status == "processing"  # type: ignore[union-attr]

    restarted.complete(replacement)
    durable = MergeQueue(queue_path).get(pending.request_id)
    assert durable is not None
    assert durable.status == "completed"
    assert durable.claim_generation == replacement.claim_generation + 1
    assert durable.claim_token == ""


def test_capacity_merge_debt_and_worktree_disk_admission_are_bounded(
    tmp_path: Path,
) -> None:
    observed_worktree_bytes = [0]
    queue = MergeQueue(
        tmp_path / "queue",
        max_queue_size=3,
        max_processing=2,
        max_worktree_bytes=10,
        worktree_usage=lambda: observed_worktree_bytes[0],
    )
    requests = [_enqueue(queue, ordinal, worktree_bytes=6) for ordinal in range(3)]
    with pytest.raises(MergeQueueFullError):
        _enqueue(queue, 3, worktree_bytes=1)

    first_batch = queue.dequeue_many(3, consumer_id="merge-train:first")
    assert len(first_batch) == 1
    assert queue.dequeue_many(1, consumer_id="merge-train:blocked") == ()
    observed_worktree_bytes[0] = 10
    status = queue.status()
    assert status["merge_debt"] == 1
    assert status["max_processing"] == 2
    assert status["reserved_worktree_bytes"] == 6
    assert status["max_worktree_bytes"] == 10
    assert status["disk_backpressure"] is True
    assert status["backpressure"] is True

    queue.complete(first_batch[0])
    observed_worktree_bytes[0] = 0
    second = queue.dequeue(consumer_id="merge-train:second")
    assert second is not None
    assert second.request_id in {
        requests[1].request_id,
        requests[2].request_id,
    }
    assert queue.status()["reserved_worktree_bytes"] == 6


def test_merge_debt_stops_additional_claims_until_a_slot_is_released(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue", max_processing=2)
    for ordinal in range(4):
        _enqueue(queue, ordinal)

    claimed = queue.dequeue_many(4, consumer_id="merge-train:batch")

    assert len(claimed) == 2
    assert queue.dequeue(consumer_id="merge-train:other") is None
    status = queue.status()
    assert status["merge_debt"] == status["max_processing"] == 2
    assert status["backpressure"] is True

    queue.complete(claimed[0])
    replacement = queue.dequeue(consumer_id="merge-train:other")
    assert replacement is not None
    assert replacement.request_id not in {
        request.request_id for request in claimed
    }
    assert queue.status()["merge_debt"] == 2


def test_deferred_claim_persists_cooldown_without_consuming_retry(
    tmp_path: Path,
) -> None:
    now = [100.0]
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        clock=lambda: now[0],
        max_attempts=2,
    )
    request = queue.enqueue(
        branch_name="implementation/wait-for-live-lock",
        task_id="WAIT-FOR-LOCK",
        commit_sha="d" * 40,
    )
    claimed = queue.dequeue(consumer_id="merge-train:first")
    assert claimed is not None

    deferred = queue.defer(
        claimed,
        reason="lock_exists",
        delay_seconds=30,
        metadata={"lock_owner_pid": 4321},
    )

    assert deferred is not None
    assert deferred.status == "pending"
    assert deferred.attempt == 1
    assert deferred.failure_count == 0
    assert deferred.retry_not_before == 130.0
    assert deferred.metadata["deferrals"][-1]["metadata"] == {
        "lock_owner_pid": 4321
    }
    with pytest.raises(MergeQueueFenceError):
        queue.complete(claimed)

    restarted = MergeQueue(
        queue_path,
        clock=lambda: now[0],
        max_attempts=2,
    )
    assert restarted.dequeue(consumer_id="merge-train:too-early") is None
    now[0] = 130.0
    reclaimed = restarted.dequeue(consumer_id="merge-train:after-cooldown")
    assert reclaimed is not None
    assert reclaimed.request_id == request.request_id
    assert reclaimed.attempt == 1
    assert reclaimed.failure_count == 0
    assert reclaimed.retry_not_before == 0.0


@pytest.mark.parametrize("delay", [float("nan"), float("inf"), 3601.0])
def test_defer_rejects_unbounded_cooldown(
    tmp_path: Path,
    delay: float,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="implementation/unbounded-cooldown",
        task_id="UNBOUNDED-COOLDOWN",
        commit_sha="e" * 40,
    )
    claimed = queue.dequeue(consumer_id="merge-train:cooldown-validation")
    assert claimed is not None

    with pytest.raises(ValueError, match="deferral delay"):
        queue.defer(
            claimed,
            reason="lock_exists",
            delay_seconds=delay,
        )

    stored = queue.get(request.request_id)
    assert stored is not None
    assert stored.status == "processing"
    assert stored.claim_token == claimed.claim_token


def test_failed_validation_is_quarantined_with_a_durable_receipt(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    pending = _enqueue(queue, 0)
    claimed = queue.dequeue(consumer_id="merge-train:validator")
    assert claimed is not None

    receipt_path = queue.fail(
        claimed,
        reason="post-merge validation failed",
        metadata={"validation_receipt_id": "sha256:failed"},
    )

    assert receipt_path is not None
    assert receipt_path.parent == queue.quarantine_dir
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["request_id"] == pending.request_id
    assert payload["status"] == "quarantined"
    assert payload["failure_reason"] == "post-merge validation failed"
    assert payload["receipt_type"] == "merge_quarantine"
    assert payload["metadata"]["quarantine"] == {
        "validation_receipt_id": "sha256:failed"
    }

    restarted = MergeQueue(queue_path)
    stored = restarted.get(pending.request_id)
    assert stored is not None
    assert stored.status == "quarantined"
    assert restarted.dequeue(consumer_id="merge-train:restart") is None
    assert restarted.status()["quarantined"] == 1
    duplicate = restarted.enqueue(
        branch_name="candidate/duplicate",
        task_id="TASK-ALIAS",
        canonical_task_id=pending.canonical_task_id,
        commit_sha=pending.commit_sha,
    )
    assert duplicate.request_id == pending.request_id


def test_cancelled_work_is_fenced_and_survives_restart(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    pending = _enqueue(queue, 0)
    claimed = queue.dequeue(consumer_id="merge-train:obsolete-base")
    assert claimed is not None

    cancelled = queue.cancel(
        claimed,
        reason="base advanced while preflight was running",
        metadata={"replacement_base": "b" * 40},
    )

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    with pytest.raises(MergeQueueFenceError):
        queue.complete(claimed)
    restarted = MergeQueue(queue_path)
    durable = restarted.get(pending.request_id)
    assert durable is not None
    assert durable.status == "cancelled"
    assert durable.failure_reason == "base advanced while preflight was running"
    assert restarted.status()["cancelled"] == 1
    assert restarted.dequeue(consumer_id="merge-train:restart") is None


def test_bound_main_consumer_cannot_claim_benchmark_or_legacy_request(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "shared-queue"
    repository_id = f"repository:sha256:{'a' * 64}"
    legacy = MergeQueue(queue_path).enqueue(
        branch_name="implementation/legacy",
        task_id="LEGACY-001",
        canonical_task_id="canonical-legacy",
        commit_sha="1" * 40,
        priority="P0",
    )
    benchmark_queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="benchmark/semantic-roundtrip",
        require_target_binding=True,
    )
    benchmark = benchmark_queue.enqueue(
        branch_name="implementation/benchmark",
        task_id="SRT-014",
        canonical_task_id="canonical-srt-014",
        commit_sha="2" * 40,
        priority="P0",
    )
    main_queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="main",
        require_target_binding=True,
    )
    main = main_queue.enqueue(
        branch_name="implementation/main",
        task_id="MAIN-001",
        canonical_task_id="canonical-main",
        commit_sha="3" * 40,
        priority="P1",
    )

    claimed_by_main = main_queue.dequeue_many(
        3,
        consumer_id="merge-train:main",
    )

    assert [request.request_id for request in claimed_by_main] == [
        main.request_id
    ]
    assert main_queue.pending_count() == 0
    assert main_queue.processing_count() == 1
    assert main_queue.active_canonical_task_ids() == {"canonical-main"}
    assert main_queue.status()["target_branch"] == "main"
    assert benchmark_queue.get(benchmark.request_id).status == "pending"  # type: ignore[union-attr]
    assert benchmark_queue.get(benchmark.request_id).consumer_id == ""  # type: ignore[union-attr]
    assert benchmark_queue.get(legacy.request_id).status == "pending"  # type: ignore[union-attr]
    with pytest.raises(MergeQueueFenceError, match="target differs"):
        main_queue.cancel(benchmark.request_id)
    claimed_by_benchmark = benchmark_queue.dequeue(
        consumer_id="merge-train:benchmark"
    )
    assert claimed_by_benchmark is not None
    assert claimed_by_benchmark.request_id == benchmark.request_id
    assert main_queue.owns_claim(claimed_by_benchmark) is False
    with pytest.raises(MergeQueueFenceError, match="target differs"):
        main_queue.complete(claimed_by_benchmark)

    assert main_queue.recover_abandoned_train_claims() == 1
    main_after_recovery = main_queue.get(main.request_id)
    assert main_after_recovery is not None
    assert main_after_recovery.status == "pending"
    assert main_after_recovery.attempt == 2
    benchmark_after_recovery = benchmark_queue.get(benchmark.request_id)
    assert benchmark_after_recovery is not None
    assert benchmark_after_recovery.status == "processing"
    assert benchmark_after_recovery.attempt == 1
    assert benchmark_after_recovery.consumer_id == "merge-train:benchmark"


def test_case_distinct_git_targets_have_distinct_deduplication_keys(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "shared-queue"
    repository_id = f"repository:sha256:{'b' * 64}"
    upper_queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="Feature",
        require_target_binding=True,
    )
    lower_queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="feature",
        require_target_binding=True,
    )
    enqueue_kwargs = {
        "branch_name": "implementation/case-sensitive",
        "task_id": "CASE-001",
        "canonical_task_id": "canonical-case-001",
        "commit_sha": "4" * 40,
    }

    upper = upper_queue.enqueue(**enqueue_kwargs)
    lower = lower_queue.enqueue(**enqueue_kwargs)

    assert upper.request_id != lower.request_id
    assert upper.target_branch == "Feature"
    assert lower.target_branch == "feature"
    assert upper_queue.pending_count() == 1
    assert lower_queue.pending_count() == 1


def test_constructor_does_not_reinsert_existing_stage_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    pending = _enqueue(queue, 0)
    inserts: list[str] = []
    original = MergeQueue._insert

    def _tracking_insert(self, connection, request, *, ignore):
        inserts.append(request.request_id)
        return original(self, connection, request, ignore=ignore)

    monkeypatch.setattr(MergeQueue, "_insert", _tracking_insert)
    restarted = MergeQueue(queue_path)
    stored = restarted.get(pending.request_id)
    assert stored is not None
    assert stored.status == "pending"
    assert inserts == []


def test_ignore_insert_skips_duplicate_primary_key_without_or_ignore(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    pending = _enqueue(queue, 0)
    duplicate = queue.enqueue(
        branch_name="candidate/duplicate",
        task_id="TASK-ALIAS",
        canonical_task_id=pending.canonical_task_id,
        commit_sha=pending.commit_sha,
    )
    assert duplicate.request_id == pending.request_id
    assert queue.status()["pending"] == 1


def test_bloated_store_rebuild_preserves_live_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    pending = _enqueue(queue, 0)
    monkeypatch.setattr(merge_queue_module, "MERGE_QUEUE_BLOAT_REBUILD_BYTES", 1)
    rebuilt = MergeQueue(queue_path)
    stored = rebuilt.get(pending.request_id)
    assert stored is not None
    assert stored.task_id == pending.task_id
    assert stored.commit_sha == pending.commit_sha
    assert stored.status == "pending"
    assert rebuilt.database_path.stat().st_size < 8 * 1024 * 1024
