"""Physical provider accounting against the real endpoint CAS coordinator."""

from __future__ import annotations

import threading
from concurrent.futures import CancelledError
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchAdmissionGrant,
    ProviderBatchCapacity,
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
    ProviderBatchStatus,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_usage import (
    PhysicalBatchUsageAdmission,
    PhysicalBatchUsageError,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.ledger import UsageLedger
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitWindow,
    Quantity,
    ReservationState,
    UsageDimension,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore


SCOPE = stable_id("scope", "physical-batch-tests")


class CountingCoordinator(UsageCoordinator):
    def __init__(self, store: InMemoryUsageLedgerStore) -> None:
        super().__init__(store)
        self.reserve_calls = 0
        self.mark_calls = 0
        self.commit_calls = 0

    def reserve(self, *args: Any, **kwargs: Any) -> Any:
        self.reserve_calls += 1
        return super().reserve(*args, **kwargs)

    def mark_dispatched(self, *args: Any, **kwargs: Any) -> Any:
        self.mark_calls += 1
        return super().mark_dispatched(*args, **kwargs)

    def commit(self, *args: Any, **kwargs: Any) -> Any:
        self.commit_calls += 1
        return super().commit(*args, **kwargs)


def _coordinator(*, concurrent: int = 1, tokens: int = 10_000, clock=None):
    clock = clock or FakeClock()
    store = InMemoryUsageLedgerStore(clock=clock)
    coordinator = CountingCoordinator(store)
    limits = []
    for dimension, ceiling in (
        (UsageDimension.CONCURRENT_REQUESTS, concurrent),
        (UsageDimension.REQUESTS, 100),
        (UsageDimension.BATCH_ITEMS, 1_000),
        (UsageDimension.INPUT_TOKENS, tokens),
        (UsageDimension.OUTPUT_TOKENS, tokens),
        (UsageDimension.TOTAL_TOKENS, tokens),
    ):
        limits.append(UsageLimit(
            scope_id=SCOPE,
            dimension=dimension,
            ceiling=Quantity.finite(ceiling),
            window=LimitWindow(kind=(
                WindowKind.CONCURRENT if dimension is UsageDimension.CONCURRENT_REQUESTS else WindowKind.LIFETIME
            )),
        ))
    coordinator.configure_limits(SCOPE, limits)
    return coordinator, store, clock


def _request(request_id: str, payload: str = "payload", **values: Any) -> ProviderBatchRequest:
    fields = dict(
        request_id=request_id,
        payload=payload,
        provider_id="test-provider",
        model="test-model",
        context_limit=32,
        token_budget=10,
        timeout_ms=3_000,
        usage_scope_id=SCOPE,
    )
    fields.update(values)
    return ProviderBatchRequest(**fields)


def _adapter(coordinator: UsageCoordinator, **values: Any) -> PhysicalBatchUsageAdmission:
    fields = dict(
        scope_supplier=lambda _provider: SCOPE,
        max_batch_tokens=10_000,
        reservation_ttl_ms=60_000,
    )
    fields.update(values)
    return PhysicalBatchUsageAdmission(coordinator, **fields)


def _config(**values: Any) -> ProviderBatchSchedulerConfig:
    fields = dict(
        max_batch_size=8,
        batch_window_ms=15,
        max_parallel_batches=2,
        provider_limits={"test-provider": 2},
        admission_retry_ms=5,
    )
    fields.update(values)
    return ProviderBatchSchedulerConfig(**fields)


def _grant(adapter: PhysicalBatchUsageAdmission, *requests: ProviderBatchRequest):
    return adapter(
        requests[0].batch_key,
        requests,
        ProviderBatchCapacity(provider_id="test-provider", healthy=True),
    )


def _records(store: InMemoryUsageLedgerStore):
    return list(UsageLedger(store.read()).iter_reservation_records(SCOPE))


def _amount(vector: UsageVector, dimension: UsageDimension) -> int:
    entry = vector.get(dimension)
    return int(entry.amount) if entry is not None else 0


@pytest.mark.parametrize("field,value", [
    ("max_batch_tokens", True), ("max_batch_tokens", 0),
    ("reservation_ttl_ms", 0), ("shared_overhead_tokens", -1),
])
def test_explicit_budgets_reject_invalid_scalars(field, value):
    coordinator, _store, _clock = _coordinator()
    with pytest.raises(ValueError):
        _adapter(coordinator, **{field: value})


def test_snapshot_or_duck_typed_coordinator_cannot_authorize():
    with pytest.raises(TypeError, match="real UsageCoordinator"):
        _adapter({"state": "available"})


def test_actual_reservation_aggregates_separate_token_dimensions_and_overhead_once():
    coordinator, store, _clock = _coordinator()
    requests = (_request("left", "a"), _request("right", "b"))
    grant = _grant(_adapter(coordinator, shared_overhead_tokens=4), *requests)
    assert grant.admitted and not grant.allow_fallback
    record = _records(store)[0]
    assert record["reserved_amounts"] == {
        "requests": 1, "batch_items": 2, "concurrent_requests": 1,
        "input_tokens": 68, "output_tokens": 20, "total_tokens": 88,
    }
    grant.before_dispatch(requests)
    grant.after_dispatch(requests, ["a", "b"], None)
    grant.release_once()
    grant.release_once()
    record = _records(store)[0]
    assert record["state"] == ReservationState.COMMITTED.value
    assert record["charged_amounts"]["total_tokens"] == 88
    assert "concurrent_requests" not in record["charged_amounts"]
    assert (coordinator.reserve_calls, coordinator.mark_calls, coordinator.commit_calls) == (1, 1, 1)
    assert grant.usage_receipt["settlement_basis"] == "estimated_ceiling"
    assert "payload" not in repr(grant.usage_receipt)


def test_advancing_wall_clock_does_not_make_every_reservation_stale():
    class AdvancingClock(FakeClock):
        def now(self):
            return self.advance(milliseconds=1)

    coordinator, store, _clock = _coordinator(clock=AdvancingClock())
    # Time-bearing snapshot revisions differ even without a ledger mutation.
    assert coordinator.snapshot(SCOPE).usage_revision != coordinator.snapshot(SCOPE).usage_revision
    request = _request("advancing-clock")
    grant = _grant(_adapter(coordinator), request)
    assert grant.admitted
    grant.before_dispatch((request,))
    grant.after_dispatch((request,), ["ok"], None)
    grant.release_once()
    assert _records(store)[0]["state"] == "committed"
    assert coordinator.reserve_calls == 1


def test_predispatch_pruning_settles_only_live_members_and_one_overhead():
    coordinator, store, _clock = _coordinator()
    left, right = _request("left", "a"), _request("right", "b")
    grant = _grant(_adapter(coordinator, shared_overhead_tokens=4), left, right)
    grant.before_dispatch((right,))
    grant.after_dispatch((right,), ["b"], None)
    grant.release_once()
    assert _records(store)[0]["charged_amounts"]["total_tokens"] == 46
    assert grant.usage_receipt["dispatched_member_count"] == 1


def test_pruned_member_is_not_charged_when_dispatched_callback_fails():
    coordinator, store, _clock = _coordinator()
    left, right = _request("left", "a"), _request("right", "b")
    grant = _grant(_adapter(coordinator, shared_overhead_tokens=4), left, right)
    grant.before_dispatch((right,))
    grant.after_dispatch((right,), None, RuntimeError("unknown provider outcome"))
    grant.release_once()
    assert _records(store)[0]["charged_amounts"]["total_tokens"] == 46
    assert grant.usage_receipt["settlement_basis"] == "conservative_live_ceiling"


def test_undispatched_release_refunds_entire_hold():
    coordinator, store, _clock = _coordinator()
    grant = _grant(_adapter(coordinator), _request("unused"))
    grant.release_once()
    record = _records(store)[0]
    assert record["state"] == "released" and record["charged_amounts"] == {}
    assert coordinator.mark_calls == coordinator.commit_calls == 0


@pytest.mark.parametrize("changes", [
    {"usage_scope_id": ""}, {"usage_scope_id": "scope:other"},
    {"context_limit": 0}, {"token_budget": 0}, {"context_limit": 20_000},
    {"timeout_ms": 0}, {"timeout_ms": 55_000},
])
def test_scope_and_budget_refusals_are_terminal_without_reservation(changes):
    coordinator, store, _clock = _coordinator()
    grant = _grant(_adapter(coordinator), _request("refused", **changes))
    assert not grant.admitted and not grant.retryable
    assert coordinator.reserve_calls == 0 and _records(store) == []


def test_scope_is_part_of_singleflight_and_batch_identity():
    first = _request("a")
    second = _request("b", usage_scope_id="scope:other")
    assert first.batch_key != second.batch_key
    assert first.execution_fingerprint != second.execution_fingerprint


def test_host_refusal_never_reserves_usage():
    coordinator, store, _clock = _coordinator()
    adapter = _adapter(coordinator, host_admission=lambda *_: ProviderBatchAdmissionGrant(False))
    assert not _grant(adapter, _request("host-pressure")).admitted
    assert coordinator.reserve_calls == 0 and _records(store) == []


def test_usage_refusal_unwinds_host_then_recovers_with_fresh_attempt():
    coordinator, store, _clock = _coordinator()
    held = coordinator.reserve(
        SCOPE, UsageVector.of(concurrent_requests=1), request_id="outside",
        idempotency_key="outside", owner_id="outside",
    )
    released = []
    now = [0]
    adapter = _adapter(coordinator, clock_ms=lambda: now[0], host_admission=lambda *_: ProviderBatchAdmissionGrant(
        True, release=lambda: released.append("host"),
    ))
    request = _request("waiting")
    first = _grant(adapter, request)
    assert not first.admitted and first.retryable and released == ["host"]
    coordinator.cancel(held.reservation_id)
    now[0] += 100
    second = _grant(adapter, request)
    assert second.admitted
    second.release_once()
    assert released == ["host", "host"]
    assert all(record["state"] == "released" for record in _records(store))


def test_quota_pressure_backoff_bounds_ledger_writes_and_host_admission_then_recovers():
    coordinator, store, _clock = _coordinator()
    outside = coordinator.reserve(
        SCOPE, UsageVector.of(concurrent_requests=1), request_id="outside",
        idempotency_key="outside", owner_id="outside",
    )
    now = [0]
    host_attempts = []

    def host(*_args):
        host_attempts.append(now[0])
        return ProviderBatchAdmissionGrant(True)

    adapter = _adapter(coordinator, host_admission=host, clock_ms=lambda: now[0])
    request = _request("pressured")
    assert not _grant(adapter, request).admitted
    revision = store.read()["revision"]
    for _ in range(100):
        deferred = _grant(adapter, request)
        assert not deferred.admitted and deferred.reason == "usage_reservation_backoff"
    assert store.read()["revision"] == revision and host_attempts == [0]
    # Subsequent failures double the delay, capped at one second.
    for expected_time in (100, 300, 700, 1500, 2500):
        now[0] = expected_time - 1
        assert _grant(adapter, request).reason == "usage_reservation_backoff"
        now[0] = expected_time
        assert _grant(adapter, request).reason.startswith("usage_reservation_denied")
    coordinator.cancel(outside.reservation_id)
    now[0] = 3500
    recovered = _grant(adapter, request)
    assert recovered.admitted
    recovered.release_once()
    # A successful grant clears cooldown, allowing the next batch immediately.
    following = _grant(adapter, request)
    assert following.admitted
    following.release_once()


def test_two_schedulers_contend_for_one_real_endpoint_slot():
    coordinator, store, _clock = _coordinator()
    first_started, finish_first, second_started = threading.Event(), threading.Event(), threading.Event()
    active = peak = 0
    lock = threading.Lock()

    def dispatch(requests):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if requests[0].request_id == "first":
                first_started.set()
                assert finish_first.wait(3)
            else:
                second_started.set()
            return ["ok" for _ in requests]
        finally:
            with lock:
                active -= 1

    first = ProviderBatchScheduler(dispatch, config=_config(), admission=_adapter(coordinator))
    second = ProviderBatchScheduler(dispatch, config=_config(), admission=_adapter(coordinator))
    try:
        first_future = first.submit(_request("first", "a"))
        assert first_started.wait(3)
        second_future = second.submit(_request("second", "b"))
        assert not second_started.wait(0.05)
        finish_first.set()
        assert first_future.result(timeout=3).status is ProviderBatchStatus.SUCCEEDED
        assert second_future.result(timeout=3).status is ProviderBatchStatus.SUCCEEDED
        assert peak == 1
        assert sum(record["charged_amounts"].get("requests", 0) for record in _records(store)) == 2
    finally:
        finish_first.set()
        first.shutdown(cancel_pending=True)
        second.shutdown(cancel_pending=True)


def test_late_singleflight_member_and_cancelled_sibling_share_one_charge():
    coordinator, store, _clock = _coordinator()
    started, finish = threading.Event(), threading.Event()

    def dispatch(requests):
        started.set()
        assert finish.wait(3)
        return ["ok" for _ in requests]

    scheduler = ProviderBatchScheduler(dispatch, config=_config(), admission=_adapter(coordinator))
    try:
        first = scheduler.submit(_request("first"))
        assert started.wait(3)
        sibling = scheduler.submit(_request("sibling"))
        assert scheduler.cancel("first")
        assert _records(store)[0]["state"] == "held"
        finish.set()
        with pytest.raises(CancelledError):
            first.result()
        result = sibling.result(timeout=3)
        assert result.status is ProviderBatchStatus.SUCCEEDED and result.singleflight_shared
        assert result.physical_usage["reservation_id"] == _records(store)[0]["reservation_id"]
        assert _records(store)[0]["charged_amounts"]["total_tokens"] == 42
        assert (coordinator.reserve_calls, coordinator.mark_calls, coordinator.commit_calls) == (1, 1, 1)
    finally:
        finish.set()
        scheduler.shutdown(cancel_pending=True)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_dispatch_mark_failure_prevents_provider_call_and_cleans_conservatively(monkeypatch, ambiguous):
    coordinator, store, _clock = _coordinator()
    original = coordinator.mark_dispatched

    def fail(reservation_id):
        if ambiguous:
            original(reservation_id)
        raise RuntimeError("mark store unavailable")

    monkeypatch.setattr(coordinator, "mark_dispatched", fail)
    calls, fallbacks = [], []
    with ProviderBatchScheduler(
        lambda requests: calls.append(requests), config=_config(), admission=_adapter(coordinator),
        fallback=lambda request: fallbacks.append(request),
    ) as scheduler:
        result = scheduler.execute(_request("mark-failure"), wait_timeout=3)
    assert result.status is ProviderBatchStatus.FAILED and calls == fallbacks == []
    record = _records(store)[0]
    assert record["state"] == ("committed" if ambiguous else "released")
    assert record["charged_amounts"].get("total_tokens", 0) == (42 if ambiguous else 0)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_settlement_failure_preserves_hold_or_confirmed_charge_and_releases_host(monkeypatch, ambiguous):
    coordinator, store, _clock = _coordinator()
    original = coordinator.commit

    def fail(*args, **kwargs):
        if ambiguous:
            original(*args, **kwargs)
        raise RuntimeError("commit response lost")

    monkeypatch.setattr(coordinator, "commit", fail)
    released, fallbacks = [], []
    adapter = _adapter(coordinator, host_admission=lambda *_: ProviderBatchAdmissionGrant(
        True, release=lambda: released.append("host"),
    ))
    with ProviderBatchScheduler(
        lambda requests: ["ok" for _ in requests], config=_config(), admission=adapter,
        fallback=lambda request: fallbacks.append(request),
    ) as scheduler:
        result = scheduler.execute(_request("settle-failure"), wait_timeout=3)
    assert result.status is ProviderBatchStatus.FAILED
    assert result.physical_usage["state"] == "settlement_pending"
    assert released == ["host"] and fallbacks == []
    record = _records(store)[0]
    assert record["state"] == ("committed" if ambiguous else "held")
    assert record["dispatched"] is True


def test_provider_exception_conservatively_charges_without_unaccounted_fallback():
    coordinator, store, _clock = _coordinator()
    fallbacks = []

    def dispatch(_requests):
        raise RuntimeError("provider outcome unknown")

    with ProviderBatchScheduler(
        dispatch, config=_config(), admission=_adapter(coordinator),
        fallback=lambda request: fallbacks.append(request),
    ) as scheduler:
        result = scheduler.execute(_request("provider-failure"), wait_timeout=3)
    assert result.status is ProviderBatchStatus.FAILED and fallbacks == []
    record = _records(store)[0]
    assert record["state"] == "committed" and record["charged_amounts"]["total_tokens"] == 42
    assert "concurrent_requests" not in record["charged_amounts"]
    assert result.physical_usage["settlement_basis"] == "conservative_live_ceiling"


@pytest.mark.parametrize("foreign", [False, True])
def test_missing_or_foreign_dispatch_acknowledgement_cannot_authorize(monkeypatch, foreign):
    coordinator, store, _clock = _coordinator(concurrent=2)
    outside = coordinator.reserve(
        SCOPE, UsageVector.of(concurrent_requests=1), request_id="outside",
        idempotency_key="outside", owner_id="outside",
    )
    monkeypatch.setattr(coordinator, "mark_dispatched", lambda _rid: outside.reservation if foreign else None)
    request = _request("bad-ack")
    grant = _grant(_adapter(coordinator), request)
    with pytest.raises(PhysicalBatchUsageError, match="mark failed"):
        grant.before_dispatch((request,))
    grant.release_once()
    ours = next(record for record in _records(store) if record["reservation_id"] == grant.usage_receipt["reservation_id"])
    assert ours["state"] == "released" and ours["charged_amounts"] == {}
    coordinator.cancel(outside.reservation_id)


def test_missing_settlement_acknowledgement_keeps_reconciliation_pending(monkeypatch):
    coordinator, store, _clock = _coordinator()
    original = coordinator.commit

    def commit_without_ack(*args, **kwargs):
        original(*args, **kwargs)
        return None

    monkeypatch.setattr(coordinator, "commit", commit_without_ack)
    request = _request("no-settlement-ack")
    grant = _grant(_adapter(coordinator), request)
    grant.before_dispatch((request,))
    with pytest.raises(PhysicalBatchUsageError, match="reconciliation"):
        grant.after_dispatch((request,), ["ok"], None)
    grant.release_once()
    assert grant.usage_receipt["state"] == "settlement_pending"
    assert _records(store)[0]["state"] == "committed"
    assert _records(store)[0]["charged_amounts"]["total_tokens"] == 42


def test_measured_usage_is_distinct_from_input_output_reservation_ceilings():
    coordinator, store, _clock = _coordinator()
    adapter = _adapter(coordinator, usage_extractor=lambda _requests, raw: UsageVector.of(**raw["usage"]))
    request = _request("measured")
    grant = _grant(adapter, request)
    grant.before_dispatch((request,))
    grant.after_dispatch((request,), {"usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}}, None)
    grant.release_once()
    assert _records(store)[0]["charged_amounts"]["total_tokens"] == 13
    assert grant.usage_receipt["settlement_basis"] == "provider_reported"
    assert _amount(UsageVector.from_dict(grant.usage_receipt["requested"]), UsageDimension.TOTAL_TOKENS) == 42


@pytest.mark.parametrize("vector", [
    UsageVector.of(total_tokens=5),
    UsageVector.of(input_tokens=2, output_tokens=2, total_tokens=5),
])
def test_incomplete_or_inconsistent_measured_usage_retains_dispatched_hold(vector):
    coordinator, store, _clock = _coordinator()
    request = _request("bad-observation")
    grant = _grant(_adapter(coordinator, usage_extractor=lambda *_: vector), request)
    grant.before_dispatch((request,))
    with pytest.raises(PhysicalBatchUsageError):
        grant.after_dispatch((request,), {}, None)
    grant.release_once()
    assert _records(store)[0]["state"] == "held"


def test_existing_host_lifecycle_hooks_are_composed_and_released():
    coordinator, _store, _clock = _coordinator()
    events = []
    host = ProviderBatchAdmissionGrant(
        True, before_dispatch=lambda _requests: events.append("before"),
        after_dispatch=lambda *_args: events.append("after"),
        release=lambda: events.append("release"),
    )
    request = _request("host-hooks")
    grant = _grant(_adapter(coordinator, host_admission=lambda *_: host), request)
    grant.before_dispatch((request,))
    grant.after_dispatch((request,), ["ok"], None)
    grant.release_once()
    assert events == ["before", "after", "release"]


def test_expired_reservation_cannot_mark_dispatched():
    coordinator, store, clock = _coordinator()
    request = _request("expired", timeout_ms=1)
    grant = _grant(_adapter(coordinator, reservation_ttl_ms=10, settlement_margin_ms=1), request)
    clock.advance(milliseconds=11)
    with pytest.raises(PhysicalBatchUsageError, match="mark failed"):
        grant.before_dispatch((request,))
    with pytest.raises(PhysicalBatchUsageError, match="reconciliation"):
        grant.release_once()
    assert coordinator.reclaim_expired(SCOPE)["count"] == 1
    assert _records(store)[0]["charged_amounts"] == {}


def test_expired_dispatched_hold_reclaims_conservatively_after_failed_settlement():
    coordinator, store, clock = _coordinator()
    request = _request("expired-running", timeout_ms=1)
    grant = _grant(_adapter(coordinator, reservation_ttl_ms=10, settlement_margin_ms=1), request)
    grant.before_dispatch((request,))
    clock.advance(milliseconds=11)
    with pytest.raises(PhysicalBatchUsageError, match="reconciliation"):
        grant.after_dispatch((request,), ["late"], None)
    grant.release_once()
    assert coordinator.reclaim_expired(SCOPE)["count"] == 1
    record = _records(store)[0]
    assert record["state"] == "expired" and record["charged_amounts"]["total_tokens"] == 42
    assert "concurrent_requests" not in record["charged_amounts"]
