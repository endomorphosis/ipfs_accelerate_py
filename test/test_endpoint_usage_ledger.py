"""Tests for the atomic durable usage ledger and reservation coordinator."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.endpoint_usage.coordinator import (
    ReserveDecision,
    UsageCoordinator,
)
from ipfs_accelerate_py.endpoint_usage.ledger import (
    ATOMIC_USAGE_LEDGER_REQUIREMENT_ID,
    CapacityDenied,
    StaleSnapshot,
    UsageLedger,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    Provenance,
    Quantity,
    ReservationState,
    UsageDimension,
    UsageErrorCode,
    UsageEstimate,
    UsageEventKind,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import (
    AdmissionAuthorityError,
    CapacityPartition,
    CompareAndSetConflict,
    CorruptionError,
    DurableUsageLedgerStore,
    FakeClock,
    IPFSAuditMirror,
    InMemoryUsageLedgerStore,
    MigrationError,
    PartitionedUsageLedgerStore,
    SchemaDriftError,
    SplitWriterError,
    StaleFenceError,
    StoreExhaustedError,
    empty_ledger_document,
    migrate_document,
    read_only_recovery_view,
    validate_document,
)
from ipfs_accelerate_py.endpoint_usage.identity import (
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    ProtocolKind,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _scope(**overrides) -> EndpointUsageScope:
    provider_id = overrides.pop("provider_id", stable_id("provider", "ledger-test"))
    defaults = {
        "provider_id": provider_id,
        "protocol": ProtocolKind.HTTPS,
        "operation": "text.chat",
        "deployment_id": stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.test/v1"
        ),
        "credential_pseudonym": credential_configuration_pseudonym(
            "env:LEDGER_TEST_KEY", key_id="ledger-default"
        ),
    }
    defaults.update(overrides)
    return EndpointUsageScope(**defaults)


def _limit(
    scope_id: str,
    dimension: UsageDimension,
    ceiling: int,
    *,
    window: LimitWindow | None = None,
    enforcement: LimitEnforcement = LimitEnforcement.HARD,
    currency: str | None = None,
) -> UsageLimit:
    if window is None:
        window = LimitWindow(kind=WindowKind.FIXED, length_ms=60_000)
    kwargs = {
        "scope_id": scope_id,
        "dimension": dimension,
        "ceiling": Quantity.finite(ceiling),
        "window": window,
        "remaining": Quantity.finite(ceiling),
        "used": Quantity.finite(0),
        "enforcement": enforcement,
        "provenance": Provenance(source=LimitSource.CONFIGURED),
    }
    if currency is not None:
        kwargs["currency"] = currency
    return UsageLimit(**kwargs)


def _coord(
    *,
    clock: FakeClock | None = None,
    writer_id: str = "writer-a",
    fence: int = 1,
    store=None,
    partition: CapacityPartition | None = None,
) -> tuple[UsageCoordinator, FakeClock, InMemoryUsageLedgerStore]:
    clock = clock or FakeClock()
    if store is None:
        store = InMemoryUsageLedgerStore(clock=clock, writer_id=writer_id, fence=fence)
    coord = UsageCoordinator(
        store, writer_id=writer_id, fence=fence, partition=partition
    )
    return coord, clock, store


# ---------------------------------------------------------------------------
# Requirement / basic store
# ---------------------------------------------------------------------------


def test_requirement_id_is_stable():
    assert ATOMIC_USAGE_LEDGER_REQUIREMENT_ID == "requirement:atomic-usage-ledger.v1"
    assert UsageCoordinator.requirement_id == ATOMIC_USAGE_LEDGER_REQUIREMENT_ID
    assert UsageLedger.requirement_id == ATOMIC_USAGE_LEDGER_REQUIREMENT_ID


def test_fake_clock_is_deterministic():
    clock = FakeClock(datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc))
    assert clock.to_rfc3339().startswith("2024-06-01T12:00:00")
    clock.advance(milliseconds=1500)
    assert clock.now().second == 1
    with pytest.raises(ValueError):
        clock.advance(milliseconds=-1)
    clock.set(datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert clock.now().year == 2025


def test_in_memory_cas_conflict():
    store = InMemoryUsageLedgerStore(writer_id="w", fence=1)
    doc = store.read()
    doc2 = store.read()
    doc["metadata"] = {"a": 1}
    store.compare_and_set(0, doc, writer_id="w", fence=1)
    doc2["metadata"] = {"b": 2}
    with pytest.raises(CompareAndSetConflict):
        store.compare_and_set(0, doc2, writer_id="w", fence=1)


def test_stale_fence_and_split_writer_fail_closed():
    store = InMemoryUsageLedgerStore(writer_id="owner", fence=5)
    doc = store.read()
    doc["metadata"] = {"x": 1}
    with pytest.raises(StaleFenceError):
        store.compare_and_set(0, doc, writer_id="owner", fence=4)
    with pytest.raises(SplitWriterError):
        store.compare_and_set(0, doc, writer_id="intruder", fence=5)
    # Takeover with higher fence is allowed.
    committed = store.compare_and_set(0, doc, writer_id="intruder", fence=6)
    assert committed["writer_id"] == "intruder"
    assert committed["fence"] == 6


def test_store_exhaustion_fail_closed():
    store = InMemoryUsageLedgerStore(
        writer_id="w", fence=1, max_events=2, max_document_bytes=10_000
    )
    doc = store.read()
    doc["events"] = [{"sequence": i, "kind": "x"} for i in range(3)]
    # validate_document may not care about event shape deeply for capacity check
    # but events need to be serializable — bypass by using empty-ish events after
    # stuffing metadata to force byte limit instead.
    store2 = InMemoryUsageLedgerStore(
        writer_id="w", fence=1, max_document_bytes=200
    )
    doc = store2.read()
    doc["metadata"] = {"pad": "x" * 500}
    with pytest.raises(StoreExhaustedError):
        store2.compare_and_set(0, doc, writer_id="w", fence=1)


def test_schema_drift_and_corruption_fail_closed():
    with pytest.raises(SchemaDriftError):
        validate_document({"schema": "other", "schema_version": "1.0"})
    with pytest.raises(SchemaDriftError):
        validate_document(
            {
                "schema": "ipfs_accelerate_py.endpoint_usage.ledger-store@1",
                "schema_version": "9.9",
            }
        )
    with pytest.raises(CorruptionError):
        validate_document("not-an-object")  # type: ignore[arg-type]


def test_migration_fail_closed():
    doc = empty_ledger_document()
    migrated = migrate_document(doc)
    assert migrated["schema_version"] == "1.0"
    with pytest.raises(MigrationError):
        migrate_document(doc, target_schema_version="99.0")
    with pytest.raises(MigrationError):
        migrate_document({"schema": "unknown@1", "schema_version": "1.0"})


def test_ipfs_mirror_cannot_authorize_admission():
    mirror = IPFSAuditMirror()
    assert mirror.authorizes_admission is False
    cid = mirror.put_event({"kind": "reservation", "n": 1})
    assert mirror.get(cid)["n"] == 1
    with pytest.raises(AdmissionAuthorityError):
        mirror.authorize_admission()
    with pytest.raises(AdmissionAuthorityError):
        UsageCoordinator(mirror)  # type: ignore[arg-type]


def test_partition_scales_ceiling_conservatively():
    part = CapacityPartition(node_id="node-1", numerator=1, denominator=4)
    assert part.scale_ceiling(10) == 2
    assert part.scale_ceiling(3) == 0


# ---------------------------------------------------------------------------
# Core reservation lifecycle
# ---------------------------------------------------------------------------


def test_atomic_reserve_commit_and_headroom():
    coord, clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(
        scope_id,
        [
            _limit(scope_id, UsageDimension.REQUESTS, 2),
            _limit(scope_id, UsageDimension.INPUT_TOKENS, 1000),
        ],
    )
    estimate = UsageEstimate(
        scope_id=scope_id,
        operation="text.chat",
        requested=UsageVector.of(requests=1, input_tokens=100),
    )
    decision = coord.reserve(
        scope_id,
        UsageVector.of(requests=1, input_tokens=100),
        request_id="req-1",
        attempt_id="1",
        idempotency_key="idem-1",
        owner_id="owner-1",
        estimate=estimate,
        ttl_ms=30_000,
    )
    assert decision.granted is True
    assert decision.reservation is not None
    assert decision.reservation.state is ReservationState.HELD
    rid = decision.reservation_id
    assert rid and rid.startswith("ures_")

    snap = coord.snapshot(scope_id)
    # One request held.
    req_head = next(h for h in snap.headroom if h.dimension is UsageDimension.REQUESTS)
    assert req_head.available.value == 1

    coord.mark_dispatched(rid)
    result = coord.commit(rid, UsageVector.of(requests=1, input_tokens=80))
    assert result.state is ReservationState.COMMITTED
    assert result.charged.get(UsageDimension.INPUT_TOKENS).amount.value == 80

    snap2 = coord.snapshot(scope_id)
    req_head2 = next(h for h in snap2.headroom if h.dimension is UsageDimension.REQUESTS)
    assert req_head2.available.value == 1  # 2 ceiling - 1 committed


def test_reserve_denied_when_limit_exhausted():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 1)])
    d1 = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="r1",
        attempt_id="1",
        idempotency_key="i1",
        owner_id="o1",
    )
    assert d1.granted
    d2 = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="r2",
        attempt_id="1",
        idempotency_key="i2",
        owner_id="o2",
    )
    assert d2.granted is False
    assert d2.error_code == UsageErrorCode.LIMIT_EXHAUSTED.value
    assert d2.reservation is not None
    assert d2.reservation.state is ReservationState.REJECTED


def test_caller_budget_enforced_in_same_cas():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.INPUT_TOKENS, 10_000)])
    decision = coord.reserve(
        scope_id,
        UsageVector.of(input_tokens=500),
        request_id="r1",
        attempt_id="1",
        idempotency_key="i1",
        owner_id="o1",
        caller_budget=UsageVector.of(input_tokens=100),
    )
    assert decision.granted is False
    assert "caller_budget_exceeded" in decision.reason_codes


def test_replay_returns_same_decision_new_attempt_distinct():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 10)])
    first = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="req-x",
        attempt_id="1",
        idempotency_key="same-key",
        owner_id="owner",
    )
    replay = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="req-x",
        attempt_id="1",
        idempotency_key="same-key",
        owner_id="owner",
    )
    assert replay.replayed is True
    assert replay.granted is True
    assert replay.reservation_id == first.reservation_id

    attempt2 = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="req-x",
        attempt_id="2",
        idempotency_key="same-key",
        owner_id="owner",
    )
    assert attempt2.replayed is False
    assert attempt2.granted is True
    assert attempt2.reservation_id != first.reservation_id


def test_reservation_id_binds_request_attempt_scope_owner_estimate_ttl():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 5)])
    estimate = UsageEstimate(
        scope_id=scope_id,
        operation="text.chat",
        requested=UsageVector.of(requests=1),
    )
    a = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="bind-req",
        attempt_id="1",
        idempotency_key="bind-idem",
        owner_id="owner-a",
        estimate=estimate,
        ttl_ms=10_000,
    )
    b = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="bind-req",
        attempt_id="1",
        idempotency_key="bind-idem",
        owner_id="owner-b",  # different owner → different framed identity on first create path
        estimate=estimate,
        ttl_ms=10_000,
    )
    # Different owners with same idempotency index key (scope/request/attempt/idem)
    # still replay the first decision — owner is bound into reservation_id of the
    # first grant, and replay returns that same decision.
    assert b.replayed is True
    assert b.reservation_id == a.reservation_id
    assert a.reservation is not None
    assert a.reservation.owner_id == "owner-a"
    assert a.reservation.request_id.endswith("attempt=1")
    assert a.reservation.estimate_id == estimate.estimate_id
    assert a.reservation.expires_at is not None


def test_cancel_before_dispatch_releases_all():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 1)])
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="c1",
        attempt_id="1",
        idempotency_key="c1",
        owner_id="o",
    )
    result = coord.cancel(d.reservation_id)
    assert result.state is ReservationState.RELEASED
    assert result.charged.entries == ()
    # Capacity free again.
    d2 = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="c2",
        attempt_id="1",
        idempotency_key="c2",
        owner_id="o",
    )
    assert d2.granted is True


def test_cancel_after_dispatch_conservatively_settles():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(
        scope_id,
        [
            _limit(scope_id, UsageDimension.REQUESTS, 5),
            _limit(scope_id, UsageDimension.INPUT_TOKENS, 500),
            _limit(
                scope_id,
                UsageDimension.CONCURRENT_REQUESTS,
                2,
                window=LimitWindow(kind=WindowKind.CONCURRENT),
            ),
        ],
    )
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1, input_tokens=100, concurrent_requests=1),
        request_id="d1",
        attempt_id="1",
        idempotency_key="d1",
        owner_id="o",
    )
    coord.mark_dispatched(d.reservation_id)
    result = coord.cancel(d.reservation_id, reason="cancelled")
    assert result.state is ReservationState.COMMITTED
    # Provider-chargeable settled; concurrent released.
    assert result.charged.get(UsageDimension.REQUESTS).amount.value == 1
    assert result.charged.get(UsageDimension.INPUT_TOKENS).amount.value == 100
    assert result.charged.get(UsageDimension.CONCURRENT_REQUESTS) is None


def test_timeout_after_dispatch_settles_conservatively():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 3)])
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="t1",
        attempt_id="1",
        idempotency_key="t1",
        owner_id="o",
    )
    result = coord.timeout(d.reservation_id, after_dispatch=True)
    assert result.state is ReservationState.COMMITTED
    assert result.charged.get(UsageDimension.REQUESTS).amount.value == 1


def test_stream_settles_monotonically():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.OUTPUT_TOKENS, 1000)])
    d = coord.reserve(
        scope_id,
        UsageVector.of(output_tokens=100),
        request_id="s1",
        attempt_id="1",
        idempotency_key="s1",
        owner_id="o",
    )
    s1 = coord.settle_stream(d.reservation_id, UsageVector.of(output_tokens=10))
    assert s1.charged.get(UsageDimension.OUTPUT_TOKENS).amount.value == 10
    s2 = coord.settle_stream(d.reservation_id, UsageVector.of(output_tokens=40))
    assert s2.charged.get(UsageDimension.OUTPUT_TOKENS).amount.value == 40
    from ipfs_accelerate_py.endpoint_usage.ledger import LedgerError

    with pytest.raises(LedgerError):
        coord.settle_stream(d.reservation_id, UsageVector.of(output_tokens=20))
    final = coord.commit(d.reservation_id, UsageVector.of(output_tokens=55))
    assert final.charged.get(UsageDimension.OUTPUT_TOKENS).amount.value == 55


def test_batch_charges_overhead_and_members_exactly_once():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 100)])
    first = coord.settle_batch(
        batch_id="batch-1",
        scope_id=scope_id,
        shared_overhead=UsageVector.of(requests=1),
        members={
            "m1": UsageVector.of(requests=1),
            "m2": UsageVector.of(requests=1),
        },
        request_id="batch-req",
        owner_id="owner",
        idempotency_key="batch-idem",
    )
    assert first["overhead_charged"] is True
    assert set(first["members_charged"]) == {"m1", "m2"}
    # Replay batch settlement — members already charged.
    second = coord.settle_batch(
        batch_id="batch-1",
        scope_id=scope_id,
        shared_overhead=UsageVector.of(requests=1),
        members={
            "m1": UsageVector.of(requests=1),
            "m2": UsageVector.of(requests=1),
        },
        request_id="batch-req",
        owner_id="owner",
        idempotency_key="batch-idem",
    )
    # No new member charges.
    assert set(second["members_charged"]) == {"m1", "m2"}
    snap = coord.snapshot(scope_id)
    req = next(h for h in snap.headroom if h.dimension is UsageDimension.REQUESTS)
    # 1 overhead + 2 members = 3
    assert req.available.value == 97


def test_correction_appends_and_references_prior_event():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 10)])
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="corr-1",
        attempt_id="1",
        idempotency_key="corr-1",
        owner_id="o",
    )
    coord.mark_dispatched(d.reservation_id)
    committed = coord.commit(d.reservation_id, UsageVector.of(requests=1))
    correction = coord.correct(
        scope_id,
        supersedes_event_id=committed.event_id,
        units=UsageVector.of(requests=1),
        reservation_id=d.reservation_id,
        reason="provider_correction",
    )
    assert correction.kind is UsageEventKind.CORRECTION
    assert correction.supersedes_event_id == committed.event_id
    # Source commit event still present.
    events = _store.read()["events"]
    assert any(e["event_id"] == committed.event_id for e in events)
    assert any(e["event_id"] == correction.event_id for e in events)


def test_expired_owner_cannot_mutate_reclaim_does_not_double_release():
    coord, clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 2)])
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="exp-1",
        attempt_id="1",
        idempotency_key="exp-1",
        owner_id="o",
        ttl_ms=1_000,
    )
    clock.advance(milliseconds=2_000)
    from ipfs_accelerate_py.endpoint_usage.ledger import LedgerError

    with pytest.raises(LedgerError):
        coord.mark_dispatched(d.reservation_id)
    reclaimed = coord.reclaim_expired(scope_id)
    assert d.reservation_id in reclaimed["reclaimed"]
    # Second reclaim is a no-op (no double release).
    reclaimed2 = coord.reclaim_expired(scope_id)
    assert reclaimed2["count"] == 0
    # Capacity free for a new reserve.
    d2 = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="exp-2",
        attempt_id="1",
        idempotency_key="exp-2",
        owner_id="o",
    )
    assert d2.granted is True


def test_dispatched_expiry_conservatively_charges():
    coord, clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(
        scope_id,
        [
            _limit(scope_id, UsageDimension.REQUESTS, 5),
            _limit(scope_id, UsageDimension.INPUT_TOKENS, 500),
        ],
    )
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1, input_tokens=50),
        request_id="dexp",
        attempt_id="1",
        idempotency_key="dexp",
        owner_id="o",
        ttl_ms=500,
    )
    coord.mark_dispatched(d.reservation_id)
    clock.advance(milliseconds=1_000)
    reclaimed = coord.reclaim_expired(scope_id)
    assert reclaimed["count"] == 1
    record = UsageLedger(_store.read()).get_reservation_record(d.reservation_id)
    assert record["state"] == ReservationState.EXPIRED.value
    assert record["charged_amounts"]["requests"] == 1
    assert record["charged_amounts"]["input_tokens"] == 50


def test_reset_is_deterministic():
    coord, clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 1)])
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="rst",
        attempt_id="1",
        idempotency_key="rst",
        owner_id="o",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id)
    # Exhausted.
    denied = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="rst2",
        attempt_id="1",
        idempotency_key="rst2",
        owner_id="o",
    )
    assert denied.granted is False
    event = coord.reset(scope_id, reason="admin_reset")
    assert event.kind is UsageEventKind.RELEASE
    # After reset, capacity available again.
    ok = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="rst3",
        attempt_id="1",
        idempotency_key="rst3",
        owner_id="o",
    )
    assert ok.granted is True


def test_clock_jump_forward_is_deterministic_for_windows():
    clock = FakeClock(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    coord, clock, _store = _coord(clock=clock)
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(
        scope_id,
        [
            _limit(
                scope_id,
                UsageDimension.REQUESTS,
                1,
                window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
            )
        ],
    )
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="clk",
        attempt_id="1",
        idempotency_key="clk",
        owner_id="o",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id)
    denied = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="clk2",
        attempt_id="1",
        idempotency_key="clk2",
        owner_id="o",
    )
    assert denied.granted is False
    # Jump past window.
    clock.advance(milliseconds=60_001)
    ok = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="clk3",
        attempt_id="1",
        idempotency_key="clk3",
        owner_id="o",
    )
    assert ok.granted is True


def test_compaction_preserves_replay_and_occupancy():
    coord, _clock, store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 10)])
    ids = []
    for i in range(5):
        d = coord.reserve(
            scope_id,
            UsageVector.of(requests=1),
            request_id="comp-%d" % i,
            attempt_id="1",
            idempotency_key="comp-%d" % i,
            owner_id="o",
        )
        coord.mark_dispatched(d.reservation_id)
        coord.commit(d.reservation_id)
        ids.append(d.reservation_id)
    before = coord.snapshot(scope_id)
    receipt = coord.compact(retain_events=1)
    assert receipt["compacted"] >= 1
    after = coord.snapshot(scope_id)
    # Occupancy preserved via checkpoint materialization.
    before_h = next(h for h in before.headroom if h.dimension is UsageDimension.REQUESTS)
    after_h = next(h for h in after.headroom if h.dimension is UsageDimension.REQUESTS)
    assert before_h.available.value == after_h.available.value
    # Replay still works for a prior idempotency key.
    replay = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="comp-0",
        attempt_id="1",
        idempotency_key="comp-0",
        owner_id="o",
    )
    assert replay.replayed is True
    assert replay.reservation_id == ids[0]
    # Document still validates.
    validate_document(store.read())


def test_durable_store_round_trip(tmp_path: Path):
    path = tmp_path / "ledger.json"
    clock = FakeClock()
    store = DurableUsageLedgerStore(path, clock=clock, writer_id="disk", fence=1)
    coord = UsageCoordinator(store, writer_id="disk", fence=1)
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 3)])
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="disk-1",
        attempt_id="1",
        idempotency_key="disk-1",
        owner_id="o",
    )
    assert d.granted
    cp = coord.checkpoint()
    assert cp["revision"] >= 1
    store.close()

    store2 = DurableUsageLedgerStore(path, clock=clock, writer_id="disk", fence=1)
    coord2 = UsageCoordinator(store2, writer_id="disk", fence=1)
    snap = coord2.snapshot(scope_id)
    assert any(r.reservation_id == d.reservation_id for r in snap.reservations)
    view = coord2.recovery_view()
    assert view["read_only"] is True
    assert view["reservation_count"] >= 1


def test_stale_usage_revision_fails_closed():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 5)])
    snap = coord.snapshot(scope_id)
    coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="stale-1",
        attempt_id="1",
        idempotency_key="stale-1",
        owner_id="o",
    )
    with pytest.raises(StaleSnapshot):
        coord.reserve(
            scope_id,
            UsageVector.of(requests=1),
            request_id="stale-2",
            attempt_id="1",
            idempotency_key="stale-2",
            owner_id="o",
            expected_usage_revision=snap.usage_revision,
        )


def test_overlapping_windows_checked_together():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(
        scope_id,
        [
            _limit(
                scope_id,
                UsageDimension.REQUESTS,
                10,
                window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
            ),
            _limit(
                scope_id,
                UsageDimension.REQUESTS,
                2,
                window=LimitWindow(kind=WindowKind.SLIDING, length_ms=10_000),
            ),
        ],
    )
    # Sliding ceiling of 2 is the tighter constraint.
    assert coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="ov-1",
        attempt_id="1",
        idempotency_key="ov-1",
        owner_id="o",
    ).granted
    assert coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="ov-2",
        attempt_id="1",
        idempotency_key="ov-2",
        owner_id="o",
    ).granted
    denied = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="ov-3",
        attempt_id="1",
        idempotency_key="ov-3",
        owner_id="o",
    )
    assert denied.granted is False


def test_concurrent_window_releases_on_commit():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(
        scope_id,
        [
            _limit(
                scope_id,
                UsageDimension.CONCURRENT_REQUESTS,
                1,
                window=LimitWindow(kind=WindowKind.CONCURRENT),
            )
        ],
    )
    d1 = coord.reserve(
        scope_id,
        UsageVector.of(concurrent_requests=1),
        request_id="con-1",
        attempt_id="1",
        idempotency_key="con-1",
        owner_id="o",
    )
    assert d1.granted
    d2 = coord.reserve(
        scope_id,
        UsageVector.of(concurrent_requests=1),
        request_id="con-2",
        attempt_id="1",
        idempotency_key="con-2",
        owner_id="o",
    )
    assert d2.granted is False
    coord.mark_dispatched(d1.reservation_id)
    # Commit with release of concurrent (charged_amounts empty for concurrent
    # via commit actual that omits concurrent, release_unused True).
    coord.commit(d1.reservation_id, UsageVector())
    d3 = coord.reserve(
        scope_id,
        UsageVector.of(concurrent_requests=1),
        request_id="con-3",
        attempt_id="1",
        idempotency_key="con-3",
        owner_id="o",
    )
    assert d3.granted is True


def test_partitioned_store_records_partition_and_scales_limits():
    clock = FakeClock()
    inner = InMemoryUsageLedgerStore(clock=clock, writer_id="n1", fence=1)
    part = CapacityPartition(node_id="n1", numerator=1, denominator=2)
    store = PartitionedUsageLedgerStore(inner, part)
    coord = UsageCoordinator(store, writer_id="n1", fence=1, partition=part)
    scope = _scope()
    scope_id = scope.scope_id
    # Global ceiling 4 → partition share 2.
    coord.configure_limits(
        scope_id,
        [_limit(scope_id, UsageDimension.REQUESTS, 4)],
        apply_partition=True,
    )
    assert coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="p1",
        attempt_id="1",
        idempotency_key="p1",
        owner_id="o",
    ).granted
    assert coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="p2",
        attempt_id="1",
        idempotency_key="p2",
        owner_id="o",
    ).granted
    denied = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="p3",
        attempt_id="1",
        idempotency_key="p3",
        owner_id="o",
    )
    assert denied.granted is False
    assert store.read()["partition"]["node_id"] == "n1"


def test_refund_reduces_charged_usage():
    coord, _clock, store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 2)])
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="ref",
        attempt_id="1",
        idempotency_key="ref",
        owner_id="o",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id)
    event = coord.refund(
        scope_id,
        UsageVector.of(requests=1),
        reservation_id=d.reservation_id,
    )
    assert event.kind is UsageEventKind.REFUND
    record = UsageLedger(store.read()).get_reservation_record(d.reservation_id)
    assert record["charged_amounts"].get("requests", 0) == 0


def test_coordinator_stale_fence_on_read_path():
    clock = FakeClock()
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="w1", fence=1)
    coord = UsageCoordinator(store, writer_id="w1", fence=1)
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 5)])
    # Takeover bumps fence.
    store.bump_fence(writer_id="w2")
    with pytest.raises(StaleFenceError):
        coord.reserve(
            scope_id,
            UsageVector.of(requests=1),
            request_id="f1",
            attempt_id="1",
            idempotency_key="f1",
            owner_id="o",
        )


def test_concurrent_reserves_do_not_overshoot():
    clock = FakeClock()
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="w", fence=1)
    scope = _scope()
    scope_id = scope.scope_id
    # Pre-configure limits single-threaded.
    bootstrap = UsageCoordinator(store, writer_id="w", fence=1)
    bootstrap.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 5)])

    results: list[ReserveDecision] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        c = UsageCoordinator(store, writer_id="w", fence=1)
        d = c.reserve(
            scope_id,
            UsageVector.of(requests=1),
            request_id="race-%d" % n,
            attempt_id="1",
            idempotency_key="race-%d" % n,
            owner_id="o-%d" % n,
        )
        with lock:
            results.append(d)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    granted = [r for r in results if r.granted]
    denied = [r for r in results if not r.granted]
    assert len(granted) == 5
    assert len(denied) == 7


def test_read_only_recovery_view():
    doc = empty_ledger_document(writer_id="w", fence=2)
    view = read_only_recovery_view(doc)
    assert view["read_only"] is True
    assert view["fence"] == 2


def test_unknown_hard_ceiling_fails_closed():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    limit = UsageLimit(
        scope_id=scope_id,
        dimension=UsageDimension.REQUESTS,
        ceiling=Quantity.unknown(),
        window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.UNKNOWN),
    )
    coord.configure_limits(scope_id, [limit])
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="unk",
        attempt_id="1",
        idempotency_key="unk",
        owner_id="o",
    )
    assert d.granted is False


def test_billing_window_reset_at():
    clock = FakeClock(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    coord, clock, _store = _coord(clock=clock)
    scope = _scope()
    scope_id = scope.scope_id
    reset_at = "2024-01-01T00:01:00.000000Z"
    coord.configure_limits(
        scope_id,
        [
            _limit(
                scope_id,
                UsageDimension.REQUESTS,
                1,
                window=LimitWindow(kind=WindowKind.BILLING, reset_at=reset_at),
            )
        ],
    )
    d = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="bill",
        attempt_id="1",
        idempotency_key="bill",
        owner_id="o",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id)
    assert (
        coord.reserve(
            scope_id,
            UsageVector.of(requests=1),
            request_id="bill2",
            attempt_id="1",
            idempotency_key="bill2",
            owner_id="o",
        ).granted
        is False
    )
    clock.set(datetime(2024, 1, 1, 0, 1, 1, tzinfo=timezone.utc))
    assert (
        coord.reserve(
            scope_id,
            UsageVector.of(requests=1),
            request_id="bill3",
            attempt_id="1",
            idempotency_key="bill3",
            owner_id="o",
        ).granted
        is True
    )


def test_denied_replay_is_stable():
    coord, _clock, _store = _coord()
    scope = _scope()
    scope_id = scope.scope_id
    coord.configure_limits(scope_id, [_limit(scope_id, UsageDimension.REQUESTS, 0)])
    first = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="den",
        attempt_id="1",
        idempotency_key="den",
        owner_id="o",
    )
    assert first.granted is False
    second = coord.reserve(
        scope_id,
        UsageVector.of(requests=1),
        request_id="den",
        attempt_id="1",
        idempotency_key="den",
        owner_id="o",
    )
    assert second.replayed is True
    assert second.granted is False
    assert second.reservation_id == first.reservation_id
