from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MergeQueue,
    MergeQueueFenceError,
    MergeQueueFullError,
    MergeQueueIntegrityError,
    POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
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


def _post_merge_denial_record(
    *,
    repository_id: str,
    target_branch: str,
) -> dict[str, object]:
    finding_material = {
        "source_ordinal": 1,
        "code": "missing-fence",
        "severity": "high",
        "summary": "Bind the correction to the exact reviewed candidate.",
    }
    material: dict[str, object] = {
        "schema": POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA,
        "target_repository_id": repository_id,
        "target_branch": target_branch,
        "task_id": "UIR-002",
        "canonical_task_key": "task/v1/example",
        "canonical_task_cid": "baguqeeraexample",
        "board_namespace": "uiir-v1",
        "task_binding_id": "baguqeerataskbinding",
        "review_attempt": 1,
        "implementation_attempt": 1,
        "target_implementation_attempt": 2,
        "implementation_commit": "1" * 40,
        "merge_commit": "2" * 40,
        "repository_tree_id": f"git-tree:{'3' * 40}",
        "review_receipt_id": "baguqeerareceipt",
        "review_request_id": "baguqeerarequest",
        "review_response_id": "baguqeeraresponse",
        "diff_binding_id": "baguqeeradiff",
        "implementer_provenance_id": "baguqeeraprovenance",
        "correction_origin_stream_id": "event-log:sha256:origin",
        "correction_authorized": True,
        "decision": "changes_required",
        "source_finding_count": 1,
        "included_finding_count": 1,
        "truncated": False,
        "findings": [
            {
                **finding_material,
                "finding_id": content_identity(finding_material),
            }
        ],
        "repository_write_authorized": False,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    terminal_material = {
        "target_repository_id": repository_id,
        "target_branch": target_branch,
        "task_id": material["task_id"],
        "canonical_task_key": material["canonical_task_key"],
        "canonical_task_cid": material["canonical_task_cid"],
        "task_binding_id": material["task_binding_id"],
        "implementation_commit": material["implementation_commit"],
    }
    material["terminal_key_id"] = content_identity(terminal_material)
    return {
        **material,
        "denial_id": content_identity(material),
    }


def _evolved_post_merge_denial_record(
    record: dict[str, object],
    *,
    marker: str,
    correction_authorized: bool,
) -> dict[str, object]:
    evolved = dict(record)
    digit = "4" if marker == "a" else "5"
    evolved.update(
        {
            "review_attempt": 2,
            "merge_commit": digit * 40,
            "repository_tree_id": f"git-tree:{digit * 40}",
            "review_receipt_id": f"baguqeerareceipt{marker}",
            "review_request_id": f"baguqeerarequest{marker}",
            "review_response_id": f"baguqeeraresponse{marker}",
            "diff_binding_id": f"baguqeeradiff{marker}",
            "implementer_provenance_id": (
                f"baguqeeraprovenance{marker}"
            ),
            "correction_origin_stream_id": (
                f"event-log:sha256:origin-{marker}"
            ),
            "correction_authorized": correction_authorized,
        }
    )
    evolved.pop("denial_id")
    evolved["denial_id"] = content_identity(evolved)
    return evolved


def test_post_merge_denial_registry_is_permanent_idempotent_and_restart_safe(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'a' * 64}"
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    record = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )

    assert queue.record_post_merge_review_denial(record) == record
    assert queue.record_post_merge_review_denial(record) == record
    restarted = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )

    assert restarted.verified_post_merge_review_denials() == (record,)
    with restarted._connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM post_merge_review_denials"
        ).fetchone()
    assert count is not None and int(count["count"]) == 1


def test_post_merge_denial_registry_coalesces_evolved_target_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'b' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    record = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(record)
    terminal_only = _evolved_post_merge_denial_record(
        record,
        marker="a",
        correction_authorized=False,
    )

    assert queue.record_post_merge_review_denial(terminal_only) == record

    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """UPDATE post_merge_review_denials
               SET record_json='{"tampered":true}'
               WHERE terminal_key_id=?""",
            (record["terminal_key_id"],),
        )
        connection.commit()
    with pytest.raises(MergeQueueIntegrityError, match="schema fields"):
        queue.verified_post_merge_review_denials()


@pytest.mark.parametrize("origin_first", (False, True))
def test_post_merge_denial_registry_authorized_origin_wins_in_both_orders(
    tmp_path: Path,
    origin_first: bool,
) -> None:
    repository_id = f"repository:sha256:{'d' * 64}"
    queue = MergeQueue(
        tmp_path / f"queue-{int(origin_first)}",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    origin = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    consumer = _evolved_post_merge_denial_record(
        origin,
        marker="a",
        correction_authorized=False,
    )

    for candidate in (
        (origin, consumer)
        if origin_first
        else (consumer, origin)
    ):
        queue.record_post_merge_review_denial(candidate)

    assert queue.verified_post_merge_review_denials() == (origin,)


def test_post_merge_denial_registry_evolved_authorized_origins_converge(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'e' * 64}"
    seed = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    first = _evolved_post_merge_denial_record(
        seed,
        marker="a",
        correction_authorized=True,
    )
    second = _evolved_post_merge_denial_record(
        seed,
        marker="b",
        correction_authorized=True,
    )
    results: list[dict[str, object]] = []
    for index, order in enumerate(((first, second), (second, first))):
        queue = MergeQueue(
            tmp_path / f"authorized-{index}",
            target_repository_id=repository_id,
            target_branch="agent/uiir",
            require_target_binding=True,
        )
        for candidate in order:
            queue.record_post_merge_review_denial(candidate)
        results.append(queue.verified_post_merge_review_denials()[0])

    assert results[0] == results[1]
    assert results[0]["correction_authorized"] is True
    assert results[0] in (first, second)


def test_post_merge_denial_correction_authority_promotes_monotonically(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'c' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    authorized = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    terminal_only = dict(authorized)
    terminal_only["correction_authorized"] = False
    terminal_only.pop("denial_id")
    terminal_only["denial_id"] = content_identity(terminal_only)

    assert queue.record_post_merge_review_denial(terminal_only) == terminal_only
    assert queue.record_post_merge_review_denial(authorized) == authorized
    assert queue.record_post_merge_review_denial(terminal_only) == authorized
    assert queue.verified_post_merge_review_denials() == (authorized,)


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
