from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError

import pytest

from ipfs_accelerate_py.agent_supervisor.provider_batch_scheduler import (
    PARTIAL_CANCELLATION_REQUIREMENT_ID,
    ProviderBatchCapacity,
    ProviderBatchEvidenceReceipt,
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
    ProviderBatchStatus,
)


def _request(request_id: str, payload: object, **overrides: object) -> ProviderBatchRequest:
    values: dict[str, object] = {
        "request_id": request_id,
        "payload": payload,
        "provider_id": "provider-a",
        "route": "proof",
        "model": "model-a",
        "operation": "generate",
        "context_limit": 8_192,
        "policy": {"network": False},
        "generation_settings": {"temperature": 0},
        "token_budget": 500,
        "timeout_ms": 2_000,
        "provenance": {"goal_id": "ASI-G060", "request": request_id},
    }
    values.update(overrides)
    return ProviderBatchRequest(**values)  # type: ignore[arg-type]


def _config(**overrides: object) -> ProviderBatchSchedulerConfig:
    values: dict[str, object] = {
        "max_batch_size": 8,
        "batch_window_ms": 20,
        "max_parallel_batches": 2,
        "provider_limits": {"provider-a": 1},
        "admission_retry_ms": 1,
    }
    values.update(overrides)
    return ProviderBatchSchedulerConfig(**values)  # type: ignore[arg-type]


def test_only_fully_compatible_requests_share_provider_call() -> None:
    calls: list[tuple[str, ...]] = []

    def dispatch(requests: object) -> list[str]:
        members = tuple(requests)  # type: ignore[arg-type]
        calls.append(tuple(item.request_id for item in members))
        return [str(item.payload).upper() for item in members]

    with ProviderBatchScheduler(dispatch, config=_config()) as scheduler:
        futures = [
            scheduler.submit(_request("first", "a")),
            scheduler.submit(_request("second", "b")),
            scheduler.submit(
                _request(
                    "different-policy",
                    "c",
                    policy={"network": True},
                )
            ),
        ]
        results = [future.result(timeout=2) for future in futures]

    assert [result.output for result in results] == ["A", "B", "C"]
    assert sorted(calls) == [("different-policy",), ("first", "second")]
    assert results[0].batch_id == results[1].batch_id
    assert results[2].batch_id != results[0].batch_id


def test_cancelled_batch_member_does_not_cancel_sibling_and_emits_evidence() -> None:
    entered = threading.Event()
    release = threading.Event()

    def dispatch(requests: object) -> list[str]:
        members = tuple(requests)  # type: ignore[arg-type]
        entered.set()
        assert release.wait(2)
        # The scheduler deliberately does not forward one member's token as a
        # batch-wide provider cancellation signal.
        assert all(item.cancellation_token is None for item in members)
        return [f"accepted:{item.request_id}" for item in members]

    scheduler = ProviderBatchScheduler(dispatch, config=_config(batch_window_ms=5))
    cancelled = scheduler.submit(_request("cancel-me", "a", token_budget=100))
    sibling = scheduler.submit(_request("keep-me", "b", token_budget=900))
    assert entered.wait(2)

    assert scheduler.cancel("cancel-me") is True
    release.set()
    kept = sibling.result(timeout=2)
    assert kept.status is ProviderBatchStatus.SUCCEEDED
    assert kept.output == "accepted:keep-me"
    with pytest.raises(CancelledError):
        cancelled.result()
    assert scheduler.flush(2)

    receipts = scheduler.partial_cancellation_evidence()
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.verify_integrity()
    assert receipt.proved_requirement_ids == (
        PARTIAL_CANCELLATION_REQUIREMENT_ID,
    )
    assert {member.request_id: member.status for member in receipt.members} == {
        "cancel-me": ProviderBatchStatus.CANCELLED,
        "keep-me": ProviderBatchStatus.SUCCEEDED,
    }
    assert {member.token_budget for member in receipt.members} == {100, 900}
    assert kept.receipt_id == receipt.evidence_id

    # Identical caller-authored data is diagnostic, not producer evidence.
    forged = ProviderBatchEvidenceReceipt(
        batch_id=receipt.batch_id,
        provider_id=receipt.provider_id,
        compatibility_digest=receipt.compatibility_digest,
        started_at_ms=receipt.started_at_ms,
        completed_at_ms=receipt.completed_at_ms,
        members=receipt.members,
    )
    assert forged.verify_integrity()
    assert forged.proved_requirement_ids == ()
    scheduler.shutdown()


def test_receipt_digest_rejects_tampering() -> None:
    with ProviderBatchScheduler(
        lambda requests: [item.payload for item in requests],
        config=_config(batch_window_ms=0),
    ) as scheduler:
        result = scheduler.execute(_request("one", {"answer": 42}), wait_timeout=2)
        receipt = scheduler.evidence_receipts()[0]

    assert result.successful
    payload = receipt.to_dict()
    with pytest.raises(ValueError, match="digest mismatch"):
        ProviderBatchEvidenceReceipt(
            batch_id=receipt.batch_id,
            provider_id=receipt.provider_id,
            compatibility_digest=receipt.compatibility_digest,
            started_at_ms=receipt.started_at_ms,
            completed_at_ms=receipt.completed_at_ms,
            members=receipt.members,
            content_digest="0" * 64,
        )
    assert payload["content_digest"] == receipt.content_digest


def test_singleflight_collapses_identical_work_but_preserves_member_identity() -> None:
    calls = 0

    def dispatch(requests: object) -> list[object]:
        nonlocal calls
        calls += 1
        members = tuple(requests)  # type: ignore[arg-type]
        return [{"payload": item.payload} for item in members]

    with ProviderBatchScheduler(dispatch, config=_config()) as scheduler:
        first = scheduler.submit(_request("first", "same", provenance={"lane": 1}))
        second = scheduler.submit(_request("second", "same", provenance={"lane": 2}))
        first_result = first.result(timeout=2)
        second_result = second.result(timeout=2)
        metrics = scheduler.metrics()

    assert calls == 1
    assert first_result.execution_id == second_result.execution_id
    assert first_result.request_id == "first"
    assert second_result.request_id == "second"
    assert first_result.provenance != second_result.provenance
    assert first_result.singleflight_shared
    assert second_result.singleflight_shared
    assert metrics.singleflight_hits == 1
    assert metrics.provider_calls_avoided >= 1


def test_member_failure_isolated_and_provider_capacity_checked_before_dispatch() -> None:
    available = False
    capacity_samples = 0
    calls = 0

    def capacity(provider_id: str) -> ProviderBatchCapacity:
        nonlocal capacity_samples
        capacity_samples += 1
        return ProviderBatchCapacity(
            provider_id=provider_id,
            healthy=True,
            max_batch_size=2,
            max_concurrent_batches=1,
            available_concurrent_batches=int(available),
        )

    def dispatch(requests: object) -> list[object]:
        nonlocal calls
        calls += 1
        members = tuple(requests)  # type: ignore[arg-type]
        return [
            ValueError("bad member") if item.request_id == "bad" else item.payload
            for item in members
        ]

    with ProviderBatchScheduler(
        dispatch,
        config=_config(batch_window_ms=2),
        capacity_supplier=capacity,
    ) as scheduler:
        bad = scheduler.submit(_request("bad", "bad"))
        good = scheduler.submit(_request("good", "good"))
        time.sleep(0.02)
        assert calls == 0
        available = True
        assert bad.result(timeout=2).status is ProviderBatchStatus.FAILED
        assert good.result(timeout=2).status is ProviderBatchStatus.SUCCEEDED
        metrics = scheduler.metrics()

    assert calls == 1
    assert capacity_samples >= 2
    assert metrics.admission_deferrals > 0
    assert metrics.max_observed_batch_size == 2
    assert metrics.provider_calls_by_id == {"provider-a": 1}
    assert metrics.average_members_per_call_millionths == 2_000_000


def test_dispatch_failure_degrades_to_independent_deterministic_fallback() -> None:
    def fail(_requests: object) -> object:
        raise RuntimeError("provider unavailable")

    with ProviderBatchScheduler(
        fail,
        fallback=lambda request: f"fallback:{request.request_id}",
        config=_config(batch_window_ms=2),
    ) as scheduler:
        results = scheduler.execute_many(
            [_request("one", "a"), _request("two", "b")],
            wait_timeout=2,
        )
        metrics = scheduler.metrics()

    assert [result.status for result in results] == [
        ProviderBatchStatus.FALLBACK,
        ProviderBatchStatus.FALLBACK,
    ]
    assert [result.output for result in results] == ["fallback:one", "fallback:two"]
    assert metrics.fallback_requests == 2
