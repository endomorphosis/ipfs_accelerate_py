from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MergeQueue,
    MergeQueueAuthorityRotationReceiptError,
    MergeQueueFenceError,
    MergeQueueFullError,
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


def test_dequeue_slice_cannot_claim_queued_sibling_or_stale_revision(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue", max_processing=2)
    sibling = queue.enqueue(
        branch_name="candidate/dcr-012",
        task_id="DCR-012",
        canonical_task_id="cid-dcr-012",
        commit_sha="1" * 40,
        priority="P0",
    )
    selected = queue.enqueue(
        branch_name="candidate/dcr-011",
        task_id="DCR-011",
        canonical_task_id="cid-dcr-011",
        canonical_task_key="key-dcr-011",
        commit_sha="2" * 40,
        priority="P3",
    )

    claimed = queue.dequeue(
        consumer_id="dcr-011-one-shot",
        allowed_canonical_task_cids=("cid-dcr-011",),
    )

    assert claimed is not None
    assert claimed.request_id == selected.request_id
    assert queue.get(sibling.request_id).status == "pending"  # type: ignore[union-attr]
    assert (
        queue.dequeue(
            consumer_id="stale-dcr-011-one-shot",
            allowed_task_ids=("DCR-011",),
            allowed_canonical_task_cids=("cid-stale-dcr-011",),
        )
        is None
    )
    assert queue.get(sibling.request_id).status == "pending"  # type: ignore[union-attr]


def test_manual_authority_row_requires_exact_producer_request_id(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue", max_processing=2)
    authority = queue.enqueue(
        branch_name="candidate/dcr-011",
        task_id="DCR-011",
        canonical_task_id="cid-dcr-011",
        commit_sha="a" * 40,
        priority="P0",
        metadata={
            "validation_proof": {
                "manual_completion_authority_full_evidence_id": (
                    "baguqeera" + "a" * 48
                )
            }
        },
    )
    ordinary = queue.enqueue(
        branch_name="candidate/ordinary",
        task_id="ORDINARY-001",
        canonical_task_id="cid-ordinary-001",
        commit_sha="b" * 40,
        priority="P3",
    )

    # A peer lane's ordinary pre-task dequeue skips the higher-priority
    # process-local capability instead of claiming and quarantining it.
    peer_claim = queue.dequeue(consumer_id="peer-lane")
    assert peer_claim is not None
    assert peer_claim.request_id == ordinary.request_id
    assert queue.get(authority.request_id).status == "pending"  # type: ignore[union-attr]

    # Task/CID identity is intentionally insufficient; only the producer's
    # exact post-enqueue request capability may claim the row.
    assert (
        queue.dequeue(
            consumer_id="peer-same-slice",
            allowed_task_ids=("DCR-011",),
            allowed_canonical_task_cids=("cid-dcr-011",),
        )
        is None
    )
    producer_claim = queue.dequeue(
        consumer_id="producer-lane",
        allowed_task_ids=("DCR-011",),
        allowed_canonical_task_cids=("cid-dcr-011",),
        allowed_request_ids=(authority.request_id,),
    )
    assert producer_claim is not None
    assert producer_claim.request_id == authority.request_id


def test_fresh_revalidation_rotates_only_exact_pending_authority_row(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    commit = "c" * 40

    def metadata(
        evidence_id: str,
        *,
        baseline_ref: str = "d" * 40,
        authority_context_id: str = "context-current",
        worktree_path: str = "/tmp/old-candidate",
    ) -> dict[str, object]:
        return {
            "baseline_ref": baseline_ref,
            "completion_task_cids": {"DCR-011": "cid-dcr-011"},
            "manual_completion_authority_context_id": authority_context_id,
            "manual_completion_authority_revocation_generation": (
                1 if authority_context_id == "context-current" else 2
            ),
            "worktree_path": worktree_path,
            "manual_completion_authority_rotation_binding_id": (
                "rotation-binding-current"
            ),
            "validation_proof": {
                "manual_completion_authority_full_evidence_id": evidence_id
            }
        }

    stale = queue.enqueue(
        branch_name="candidate/dcr-011",
        task_id="DCR-011",
        canonical_task_id="cid-dcr-011",
        canonical_task_key="key-dcr-011",
        commit_sha=commit,
        metadata=metadata("stale-evidence"),
    )
    assert (
        queue.rotate_pending_manual_authority_evidence(
            stale.request_id,
            commit_sha=commit,
            branch_name="candidate/dcr-011",
            task_id="DCR-011",
            canonical_task_id="cid-dcr-011",
            canonical_task_key="key-dcr-011",
            expected_previous_evidence_id="wrong-stale-evidence",
            metadata=metadata("fresh-evidence"),
            lane_id="fresh-producer",
            attempt=2,
        )
        is None
    )
    assert (
        queue.rotate_pending_manual_authority_evidence(
            stale.request_id,
            commit_sha=commit,
            branch_name="candidate/dcr-011",
            task_id="DCR-011",
            canonical_task_id="cid-stale-dcr-011",
            canonical_task_key="key-dcr-011",
            expected_previous_evidence_id="stale-evidence",
            metadata=metadata("fresh-evidence"),
            lane_id="fresh-producer",
            attempt=2,
        )
        is None
    )
    assert (
        queue.rotate_pending_manual_authority_evidence(
            stale.request_id,
            commit_sha=commit,
            branch_name="candidate/dcr-011",
            task_id="DCR-011",
            canonical_task_id="cid-dcr-011",
            canonical_task_key="key-dcr-011",
            expected_previous_evidence_id="stale-evidence",
            metadata=metadata(
                "fresh-evidence",
                baseline_ref="e" * 40,
            ),
            lane_id="fresh-producer",
            attempt=2,
        )
        is None
    )
    rotated = queue.rotate_pending_manual_authority_evidence(
        stale.request_id,
        commit_sha=commit,
        branch_name="candidate/dcr-011",
        task_id="DCR-011",
        canonical_task_id="cid-dcr-011",
        canonical_task_key="key-dcr-011",
        expected_previous_evidence_id="stale-evidence",
        metadata=metadata(
            "fresh-evidence",
            authority_context_id="context-after-restart",
            worktree_path="/tmp/recovered-candidate",
        ),
        lane_id="fresh-producer",
        attempt=2,
    )
    assert rotated is not None
    assert rotated.request_id == stale.request_id
    assert rotated.lane_id == "fresh-producer"
    assert rotated.attempt == 2
    assert rotated.metadata["validation_proof"][  # type: ignore[index]
        "manual_completion_authority_full_evidence_id"
    ] == "fresh-evidence"
    assert (
        rotated.metadata["manual_completion_authority_context_id"]
        == "context-after-restart"
    )
    assert rotated.metadata["worktree_path"] == "/tmp/recovered-candidate"
    assert queue.dequeue(consumer_id="peer-lane") is None
    claimed = queue.dequeue(
        consumer_id="fresh-producer",
        allowed_request_ids=(stale.request_id,),
    )
    assert claimed is not None
    assert claimed.request_id == stale.request_id
    assert (
        queue.rotate_pending_manual_authority_evidence(
            stale.request_id,
            commit_sha=commit,
            branch_name="candidate/dcr-011",
            task_id="DCR-011",
            canonical_task_id="cid-dcr-011",
            canonical_task_key="key-dcr-011",
            expected_previous_evidence_id="fresh-evidence",
            metadata=metadata("newer-evidence"),
            lane_id="impostor",
            attempt=3,
        )
        is None
    )


def test_authority_rotation_receipt_failure_is_typed_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    immutable = {
        "baseline_ref": "d" * 40,
        "completion_task_cids": {"DCR-011": "cid-dcr-011"},
        "manual_completion_authority_rotation_binding_id": "binding-exact",
    }

    def metadata(evidence_id: str) -> dict[str, object]:
        return {
            **immutable,
            "validation_proof": {
                "manual_completion_authority_full_evidence_id": evidence_id
            },
        }

    stale = queue.enqueue(
        branch_name="candidate/dcr-011",
        task_id="DCR-011",
        canonical_task_id="cid-dcr-011",
        canonical_task_key="key-dcr-011",
        commit_sha="c" * 40,
        metadata=metadata("stale-evidence"),
    )
    original_write = queue._write_stage_receipt
    monkeypatch.setattr(
        queue,
        "_write_stage_receipt",
        lambda _request: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    with pytest.raises(MergeQueueAuthorityRotationReceiptError) as raised:
        queue.rotate_pending_manual_authority_evidence(
            stale.request_id,
            commit_sha="c" * 40,
            branch_name="candidate/dcr-011",
            task_id="DCR-011",
            canonical_task_id="cid-dcr-011",
            canonical_task_key="key-dcr-011",
            expected_previous_evidence_id="stale-evidence",
            metadata=metadata("fresh-evidence"),
            lane_id="fresh-producer",
            attempt=2,
        )
    assert raised.value.request_id == stale.request_id
    durable = queue.get(stale.request_id)
    assert durable is not None and durable.status == "pending"
    assert durable.metadata["validation_proof"][  # type: ignore[index]
        "manual_completion_authority_full_evidence_id"
    ] == "fresh-evidence"

    monkeypatch.setattr(queue, "_write_stage_receipt", original_write)
    recovered = queue.rotate_pending_manual_authority_evidence(
        stale.request_id,
        commit_sha="c" * 40,
        branch_name="candidate/dcr-011",
        task_id="DCR-011",
        canonical_task_id="cid-dcr-011",
        canonical_task_key="key-dcr-011",
        expected_previous_evidence_id="fresh-evidence",
        metadata=metadata("newer-evidence"),
        lane_id="recovered-producer",
        attempt=3,
    )
    assert recovered is not None
    assert recovered.metadata["validation_proof"][  # type: ignore[index]
        "manual_completion_authority_full_evidence_id"
    ] == "newer-evidence"


def test_crashed_final_attempt_authority_claim_requires_fresh_rotation(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue", max_attempts=1)

    def metadata(
        evidence_id: str,
        *,
        context: str,
    ) -> dict[str, object]:
        return {
            "baseline_ref": "d" * 40,
            "completion_task_cids": {"DCR-011": "cid-dcr-011"},
            "manual_completion_authority_context_id": context,
            "manual_completion_authority_rotation_binding_id": (
                "binding-exact"
            ),
            "validation_proof": {
                "manual_completion_authority_full_evidence_id": evidence_id
            },
        }

    pending = queue.enqueue(
        branch_name="candidate/dcr-011",
        task_id="DCR-011",
        canonical_task_id="cid-dcr-011",
        canonical_task_key="key-dcr-011",
        commit_sha="c" * 40,
        metadata=metadata("stale-evidence", context="old-context"),
    )
    claimed = queue.dequeue(
        consumer_id="merge-train:crashed-producer",
        allowed_request_ids=(pending.request_id,),
    )
    assert claimed is not None and claimed.attempt == 1

    assert queue.recover_abandoned_train_claims() == 1
    recovered = queue.get(pending.request_id)
    assert recovered is not None
    assert recovered.status == "pending"
    assert recovered.attempt == 1
    assert recovered.failure_count == 1
    assert recovered.failure_reason == (
        "manual_completion_authority_revalidation_required"
    )
    assert queue.dequeue(consumer_id="peer-generic") is None

    rotated = queue.rotate_pending_manual_authority_evidence(
        pending.request_id,
        commit_sha="c" * 40,
        branch_name="candidate/dcr-011",
        task_id="DCR-011",
        canonical_task_id="cid-dcr-011",
        canonical_task_key="key-dcr-011",
        expected_previous_evidence_id="stale-evidence",
        metadata=metadata("fresh-evidence", context="fresh-context"),
        lane_id="merge-train:fresh-producer",
        attempt=1,
    )
    assert rotated is not None
    renewed_claim = queue.dequeue(
        consumer_id="merge-train:fresh-producer",
        allowed_request_ids=(pending.request_id,),
    )
    assert renewed_claim is not None
    assert renewed_claim.request_id == pending.request_id


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
