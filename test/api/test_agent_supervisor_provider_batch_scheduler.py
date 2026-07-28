from __future__ import annotations

import json
import threading
import time
from concurrent.futures import CancelledError

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    PARTIAL_CANCELLATION_REQUIREMENT_ID,
    ProviderBatchAdmissionGrant,
    ProviderBatchCapacity,
    ProviderBatchEvidenceReceipt,
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
    ProviderBatchStatus,
    ResourceSchedulerBatchAdmission,
)
from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import (
    StructuredPlanRouterConfig,
    generate_structured_plan_branches,
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
        assert len(members) == 2
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


def test_member_timeout_is_independent_from_running_batch_sibling() -> None:
    entered = threading.Event()
    release = threading.Event()

    def dispatch(requests: object) -> list[str]:
        members = tuple(requests)  # type: ignore[arg-type]
        assert len(members) == 2
        entered.set()
        assert release.wait(2)
        return [str(item.payload) for item in members]

    scheduler = ProviderBatchScheduler(
        dispatch,
        config=_config(batch_window_ms=10, admission_retry_ms=1),
    )
    short = scheduler.submit(
        _request(
            "short-timeout",
            "short",
            timeout_ms=50,
            token_budget=100,
            provenance={"deadline": "short"},
        )
    )
    long = scheduler.submit(
        _request(
            "long-timeout",
            "long",
            timeout_ms=1_000,
            token_budget=900,
            provenance={"deadline": "long"},
        )
    )
    try:
        assert entered.wait(2)
        short_result = short.result(timeout=1)
        assert short_result.status is ProviderBatchStatus.TIMED_OUT
        assert short_result.token_budget == 100
        assert short_result.timeout_ms == 50
        assert short_result.provenance == {"deadline": "short"}

        release.set()
        long_result = long.result(timeout=2)
        assert long_result.status is ProviderBatchStatus.SUCCEEDED
        assert long_result.output == "long"
        assert long_result.token_budget == 900
        assert long_result.timeout_ms == 1_000
        assert long_result.provenance == {"deadline": "long"}
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_pending=True)


def test_provider_concurrency_limit_is_never_exceeded() -> None:
    release = threading.Event()
    first_entered = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def dispatch(requests: object) -> list[str]:
        nonlocal active, maximum_active
        members = tuple(requests)  # type: ignore[arg-type]
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            first_entered.set()
        try:
            assert release.wait(2)
            return [item.request_id for item in members]
        finally:
            with lock:
                active -= 1

    scheduler = ProviderBatchScheduler(
        dispatch,
        config=_config(
            max_batch_size=1,
            batch_window_ms=0,
            max_parallel_batches=4,
            provider_limits={"provider-a": 1},
        ),
    )
    futures = [
        scheduler.submit(_request(f"limited-{index}", index))
        for index in range(3)
    ]
    try:
        assert first_entered.wait(2)
        time.sleep(0.02)
        assert maximum_active == 1
        assert scheduler.metrics().active_batches == 1
        release.set()
        assert all(item.result(timeout=2).successful for item in futures)
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_pending=True)

    assert maximum_active == 1
    assert scheduler.metrics().provider_calls == 3


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


def test_incompatible_queues_are_served_round_robin_without_starvation() -> None:
    """A deep route queue must not starve another compatible class."""

    capacity_available = threading.Event()
    calls: list[str] = []

    def capacity(provider_id: str) -> ProviderBatchCapacity:
        return ProviderBatchCapacity(
            provider_id=provider_id,
            max_batch_size=1,
            max_concurrent_batches=1,
            available_concurrent_batches=int(capacity_available.is_set()),
        )

    def dispatch(requests: object) -> list[str]:
        members = tuple(requests)  # type: ignore[arg-type]
        assert len(members) == 1
        calls.append(members[0].route)
        return [members[0].request_id]

    scheduler = ProviderBatchScheduler(
        dispatch,
        config=_config(
            max_batch_size=1,
            batch_window_ms=0,
            max_parallel_batches=1,
        ),
        capacity_supplier=capacity,
    )
    try:
        futures = [
            scheduler.submit(_request("a-1", "a-1", route="route-a")),
            scheduler.submit(_request("a-2", "a-2", route="route-a")),
            scheduler.submit(_request("a-3", "a-3", route="route-a")),
            scheduler.submit(_request("b-1", "b-1", route="route-b")),
        ]
        # No request can leave the queue until every fairness class is present.
        capacity_available.set()
        assert [item.result(timeout=2).successful for item in futures] == [
            True,
            True,
            True,
            True,
        ]
    finally:
        capacity_available.set()
        scheduler.shutdown(wait=True, cancel_pending=True)

    assert calls == ["route-a", "route-b", "route-a", "route-a"]


def test_queue_metrics_exclude_active_members_and_do_not_double_count_cancellation() -> None:
    entered = threading.Event()
    release = threading.Event()

    def dispatch(requests: object) -> list[str]:
        members = tuple(requests)  # type: ignore[arg-type]
        entered.set()
        assert release.wait(2)
        return [item.request_id for item in members]

    scheduler = ProviderBatchScheduler(
        dispatch,
        config=_config(batch_window_ms=5, max_parallel_batches=1),
    )
    cancelled = scheduler.submit(_request("cancelled", "a"))
    sibling = scheduler.submit(_request("sibling", "b"))
    try:
        assert entered.wait(2)
        assert scheduler.cancel("cancelled")

        active = scheduler.metrics()
        assert active.active_batches == 1
        assert active.queued_requests == 0

        release.set()
        assert sibling.result(timeout=2).successful
        with pytest.raises(CancelledError):
            cancelled.result()
        assert scheduler.flush(2)
        completed = scheduler.metrics()
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_pending=True)

    assert completed.completed_requests == 2
    assert completed.cancelled_requests == 1
    assert completed.provider_calls == 1
    assert completed.average_members_per_call_millionths == 2_000_000


@pytest.mark.parametrize("dispatch_fails", [False, True])
def test_admission_grant_precedes_dispatch_and_is_always_released(
    dispatch_fails: bool,
) -> None:
    events: list[str] = []
    releases = 0
    grant_active = False

    def release_grant() -> None:
        nonlocal releases, grant_active
        assert grant_active
        grant_active = False
        releases += 1
        events.append("release")

    def admission(
        _key: object,
        _requests: object,
        _capacity: object,
    ) -> ProviderBatchAdmissionGrant:
        nonlocal grant_active
        assert not grant_active
        assert events == []
        grant_active = True
        events.append("admit")
        return ProviderBatchAdmissionGrant(admitted=True, release=release_grant)

    def dispatch(requests: object) -> list[str]:
        members = tuple(requests)  # type: ignore[arg-type]
        assert grant_active
        assert releases == 0
        events.append("dispatch")
        if dispatch_fails:
            raise RuntimeError("provider failed after model admission")
        return [item.request_id for item in members]

    with ProviderBatchScheduler(
        dispatch,
        config=_config(batch_window_ms=0),
        admission=admission,
    ) as scheduler:
        result = scheduler.execute(_request("admitted", "work"), wait_timeout=2)

    assert result.status is (
        ProviderBatchStatus.FAILED
        if dispatch_fails
        else ProviderBatchStatus.SUCCEEDED
    )
    assert events == ["admit", "dispatch", "release"]
    assert releases == 1
    assert grant_active is False


def test_release_failure_is_observable_without_stopping_shared_scheduler() -> None:
    releases = 0
    second_released = threading.Event()

    def release() -> None:
        nonlocal releases
        releases += 1
        if releases == 1:
            raise RuntimeError("lease service failed after provider completion")
        second_released.set()

    def admission(
        _key: object,
        _requests: object,
        _capacity: object,
    ) -> ProviderBatchAdmissionGrant:
        return ProviderBatchAdmissionGrant(admitted=True, release=release)

    with ProviderBatchScheduler(
        lambda requests: [item.payload for item in requests],
        config=_config(max_batch_size=1, batch_window_ms=0),
        admission=admission,
    ) as scheduler:
        first = scheduler.execute(_request("release-failed", "first"), wait_timeout=2)
        second = scheduler.execute(_request("release-recovered", "second"), wait_timeout=2)
        assert second_released.wait(2)
        metrics = scheduler.metrics()

    assert first.output == "first"
    assert second.output == "second"
    assert releases == 2
    assert metrics.provider_calls == 2
    assert metrics.admission_errors == 1


def test_capacity_supplier_failure_defers_work_without_killing_coordinator() -> None:
    samples = 0
    calls = 0

    def capacity(provider_id: str) -> ProviderBatchCapacity:
        nonlocal samples
        samples += 1
        if samples == 1:
            raise RuntimeError("transient GPU telemetry failure")
        return ProviderBatchCapacity(
            provider_id=provider_id,
            max_concurrent_batches=1,
            available_concurrent_batches=1,
        )

    def dispatch(requests: object) -> list[str]:
        nonlocal calls
        calls += 1
        return [item.request_id for item in requests]  # type: ignore[union-attr]

    scheduler = ProviderBatchScheduler(
        dispatch,
        config=_config(batch_window_ms=0),
        capacity_supplier=capacity,
    )
    try:
        result = scheduler.execute(_request("recover", "work"), wait_timeout=2)
        metrics = scheduler.metrics()
    finally:
        scheduler.shutdown(wait=True, cancel_pending=True)

    assert result.status is ProviderBatchStatus.SUCCEEDED
    assert samples >= 2
    assert calls == 1
    assert metrics.admission_deferrals >= 1


def test_resource_scheduler_adapter_reserves_aggregate_batch_before_dispatch() -> None:
    events: list[object] = []
    lease = object()

    class FakeResourceScheduler:
        def acquire(self, requirement, **kwargs):  # type: ignore[no-untyped-def]
            events.append(("acquire", requirement, kwargs))
            decision = type(
                "Decision",
                (),
                {"admitted": True, "reason": ""},
            )()
            return decision, lease

        def release(self, released, *, reason):  # type: ignore[no-untyped-def]
            events.append(("release", released, reason))
            return True

    admission = ResourceSchedulerBatchAdmission(
        FakeResourceScheduler(),
        host_supplier={"available_memory_bytes": 10_000_000},
        provider_supplier=lambda provider_id: {
            provider_id: {
                "healthy": True,
                "max_concurrency": 1,
                "active_requests": 0,
            }
        },
        gpu_memory_bytes=4_096,
    )

    def dispatch(requests):  # type: ignore[no-untyped-def]
        events.append(("dispatch", tuple(item.request_id for item in requests)))
        return [item.payload for item in requests]

    with ProviderBatchScheduler(
        dispatch,
        config=_config(batch_window_ms=10),
        admission=admission,
    ) as scheduler:
        results = scheduler.execute_many(
            [
                _request("aggregate-a", "a", token_budget=100),
                _request("aggregate-b", "b", token_budget=300),
            ],
            wait_timeout=2,
        )

    assert all(item.successful for item in results)
    assert [item[0] for item in events] == ["acquire", "dispatch", "release"]
    requirement = events[0][1]
    assert requirement.stage == "inference"
    assert requirement.provider_id == "provider-a"
    assert requirement.context_tokens == 8_192
    assert requirement.token_budget == 400
    assert requirement.quota_units == 1
    assert requirement.gpu_memory_bytes == 4_096
    assert events[-1][1] is lease


def test_late_singleflight_subscriber_is_never_lost_at_completion_boundary() -> None:
    receipt_started = threading.Event()
    release_receipt = threading.Event()

    class PausedReceiptScheduler(ProviderBatchScheduler):
        def _build_receipt(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            receipt_started.set()
            assert release_receipt.wait(2)
            return super()._build_receipt(*args, **kwargs)

    scheduler = PausedReceiptScheduler(
        lambda requests: [item.payload for item in requests],
        config=_config(batch_window_ms=0),
    )
    try:
        first = scheduler.submit(_request("race-first", "identical"))
        assert receipt_started.wait(2)
        second = scheduler.submit(_request("race-second", "identical"))
        release_receipt.set()
        assert first.result(timeout=2).output == "identical"
        assert second.result(timeout=2).output == "identical"
        assert scheduler.flush(2)
    finally:
        release_receipt.set()
        scheduler.shutdown(wait=True, cancel_pending=True)

    # The second subscriber arrived after the first execution was sealed.  It
    # is a fresh provider call, not a false hit left unresolved forever.
    assert scheduler.metrics().provider_calls == 2


def test_structured_planning_route_uses_shared_singleflight_and_provenance(
    tmp_path,
) -> None:
    admit = threading.Event()
    both_submitted = threading.Event()
    submit_lock = threading.Lock()
    physical_calls = 0

    class ObservedScheduler(ProviderBatchScheduler):
        submissions = 0

        def submit(self, request):  # type: ignore[no-untyped-def]
            future = super().submit(request)
            with submit_lock:
                self.submissions += 1
                if self.submissions == 2:
                    both_submitted.set()
            return future

    def capacity(provider_id: str) -> ProviderBatchCapacity:
        return ProviderBatchCapacity(
            provider_id=provider_id,
            max_concurrent_batches=1,
            available_concurrent_batches=int(admit.is_set()),
        )

    response = json.dumps(
        {
            "branches": [
                {
                    "branch_id": "shared-plan",
                    "summary": "Implement the shared route",
                    "predicted_files": ["planner.py"],
                    "predicted_symbols": ["plan"],
                    "dependencies": [],
                    "validation_commands": ["pytest -q"],
                    "validation_proof": ["pytest exits with status 0"],
                    "estimated_cost": 1.0,
                    "risk": 0.1,
                    "expected_objective_delta": 0.8,
                    "source": "llm_router",
                }
            ]
        }
    )

    def dispatch(requests):  # type: ignore[no-untyped-def]
        nonlocal physical_calls
        physical_calls += 1
        return [response for _item in requests]

    scheduler = ObservedScheduler(
        dispatch,
        config=_config(batch_window_ms=0),
        capacity_supplier=capacity,
    )
    config = StructuredPlanRouterConfig(
        repo_root=tmp_path,
        provider="provider-a",
        model="model-a",
        branch_count=1,
        provider_batch_scheduler=scheduler,
    )
    results = [None, None]

    def plan(index: int) -> None:
        results[index] = generate_structured_plan_branches(
            {
                "task_id": "same-task",
                "title": "Same planning work",
                "outputs": ["planner.py"],
            },
            config=config,
        )

    threads = [threading.Thread(target=plan, args=(index,)) for index in range(2)]
    try:
        for thread in threads:
            thread.start()
        assert both_submitted.wait(2)
        admit.set()
        for thread in threads:
            thread.join(2)
        assert all(not thread.is_alive() for thread in threads)
    finally:
        admit.set()
        scheduler.shutdown(wait=True, cancel_pending=True)

    assert physical_calls == 1
    assert all(result is not None and not result.used_fallback for result in results)
    batches = [result.batch_result for result in results if result is not None]
    assert len(batches) == 2
    assert all(item is not None and item.singleflight_shared for item in batches)
    assert len({item.execution_id for item in batches if item is not None}) == 1
    assert len({item.request_id for item in batches if item is not None}) == 2


# ---------------------------------------------------------------------------
# ASI-167: endpoint usage projection into batch admission
# ---------------------------------------------------------------------------


def _load_declared_batch_module():
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "provider_batch_scheduler.py"
    )
    name = "ipfs_accelerate_py.agent_supervisor._declared_pbs_for_batch_tests"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_asi167_batch_requirement_and_symbols_installed() -> None:
    mod = _load_declared_batch_module()
    from ipfs_accelerate_py.agent_supervisor.runtime import provider_batch_scheduler as runtime_pbs

    assert mod.ENDPOINT_USAGE_BATCH_ADMISSION_REQUIREMENT_ID.startswith("requirement:")
    assert mod.PHYSICAL_BATCH_RESERVE_ONCE_REQUIREMENT_ID.startswith("requirement:")
    assert hasattr(runtime_pbs, "reserve_physical_batch")
    assert hasattr(runtime_pbs, "UsageAwareProviderBatchScheduler")


def test_asi167_physical_batch_member_cancel_does_not_charge_sibling() -> None:
    mod = _load_declared_batch_module()
    requests = [
        _request("left", "a", token_budget=100),
        _request("right", "b", token_budget=250),
    ]
    snapshot = {
        "scope_id": "scope:batch",
        "usage_revision": "rev",
        "state": "available",
        "headroom": [
            {
                "dimension": "total_tokens",
                "available": {"kind": "finite", "value": 10_000},
                "ceiling": {"kind": "finite", "value": 10_000},
                "reserved": {"kind": "finite", "value": 0},
                "state": "available",
            },
            {
                "dimension": "concurrent_requests",
                "available": {"kind": "finite", "value": 2},
                "ceiling": {"kind": "finite", "value": 2},
                "reserved": {"kind": "finite", "value": 0},
                "state": "available",
            },
        ],
        "reservations": [],
        "reason_codes": [],
    }
    reservation, grant = mod.reserve_physical_batch(
        requests,
        provider_id="provider-a",
        snapshot=snapshot,
        mode=mod.UsageAdmissionMode.ENFORCE,
        shared_overhead_tokens=40,
        base_capacity=ProviderBatchCapacity(
            provider_id="provider-a",
            healthy=True,
            max_batch_size=8,
            max_concurrent_batches=2,
            available_concurrent_batches=2,
            token_budget_remaining=10_000,
        ),
    )
    assert grant.admitted is True and reservation is not None
    assert reservation.settle_shared_overhead_once() == 40
    assert reservation.settle_shared_overhead_once() == 0
    reservation.cancel_member("left")
    assert reservation.member_attributions["left"].charged is False
    assert reservation.member_attributions["right"].charged is True
    assert reservation.total_charged_tokens() == 250 + 40


def test_asi167_usage_aware_batch_scheduler_enforce_capacity_supplier() -> None:
    mod = _load_declared_batch_module()
    calls: list[tuple[str, ...]] = []

    def dispatch(requests: object) -> list[str]:
        members = tuple(requests)  # type: ignore[arg-type]
        calls.append(tuple(item.request_id for item in members))
        return [f"ok:{item.request_id}" for item in members]

    def snapshot_supplier(_provider_id: str) -> dict[str, object]:
        return {
            "scope_id": "scope:live",
            "usage_revision": "rev",
            "state": "available",
            "headroom": [
                {
                    "dimension": "concurrent_requests",
                    "available": {"kind": "finite", "value": 1},
                    "ceiling": {"kind": "finite", "value": 1},
                    "reserved": {"kind": "finite", "value": 0},
                    "state": "available",
                },
                {
                    "dimension": "total_tokens",
                    "available": {"kind": "finite", "value": 50_000},
                    "ceiling": {"kind": "finite", "value": 50_000},
                    "reserved": {"kind": "finite", "value": 0},
                    "state": "available",
                },
            ],
            "reservations": [],
            "reason_codes": [],
        }

    with mod.UsageAwareProviderBatchScheduler(
        dispatch,
        config=_config(batch_window_ms=10, max_parallel_batches=1),
        usage_mode=mod.UsageAdmissionMode.ENFORCE,
        usage_snapshot_supplier=snapshot_supplier,
        shared_overhead_tokens=10,
    ) as scheduler:
        futures = [
            scheduler.submit(_request("u1", "a", token_budget=100)),
            scheduler.submit(_request("u2", "b", token_budget=100)),
        ]
        results = [future.result(timeout=3) for future in futures]
    assert all(result.status is ProviderBatchStatus.SUCCEEDED for result in results)
    assert calls  # at least one physical dispatch
