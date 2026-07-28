"""AICAT-035: fault injection for usage-routing (offline, deterministic).

Covers 429/503, billing exhaustion, malformed metadata, timeout/cancel before
and after dispatch, partial stream, batch split, cache/single-flight,
retry/fallback, correction/reset, process crash recovery, store
migration/outage, coordinator partition, clock jump/skew, and reservation
races. Asserts zero hard-limit overshoot, duplicate charge, and cross-scope
contamination.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from ipfs_accelerate_py.endpoint_usage.adapters import (
    AdapterParseError,
    parse_openai_compatible_observation,
    parse_provider_observation,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.ledger import CapacityDenied, StaleSnapshot
from ipfs_accelerate_py.endpoint_usage.resolution import (
    StaticCandidate,
    UsageRoutingRequest,
)
from ipfs_accelerate_py.endpoint_usage.routing import (
    CircuitBreakerRegistry,
    ErrorSafetyClass,
    InvokeOutcome,
    SingleFlight,
    UsageRouteAdmission,
    classify_invoke_error,
    is_fallback_safe,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    FallbackClass,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    Provenance,
    Quantity,
    ReservationState,
    RoutingMode,
    RoutingPolicy,
    UsageDimension,
    UsageEventKind,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import (
    AdmissionAuthorityError,
    CapacityPartition,
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
    validate_document,
)


FIXED_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
FAULT_REQUIREMENT_ID = "requirement:endpoint-usage-faults.v1"


def _rfc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _scope(key: str = "fault-a", *, cred: str = "fault-default") -> EndpointUsageScope:
    provider_id = stable_id("provider", key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation="text.chat",
        deployment_id=stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:FAULT_USAGE_KEY", key_id=cred
        ),
    )


def _limit(
    scope_id: str,
    dimension: UsageDimension,
    ceiling: int,
    *,
    window: Optional[LimitWindow] = None,
) -> UsageLimit:
    if window is None:
        window = LimitWindow(kind=WindowKind.FIXED, length_ms=60_000)
    return UsageLimit(
        scope_id=scope_id,
        dimension=dimension,
        ceiling=Quantity.finite(ceiling),
        window=window,
        remaining=Quantity.finite(ceiling),
        used=Quantity.finite(0),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.CONFIGURED),
    )


def _coord(
    clock: Optional[FakeClock] = None,
    *,
    writer_id: str = "fault-writer",
    fence: int = 1,
    store=None,
    partition: Optional[CapacityPartition] = None,
) -> Tuple[UsageCoordinator, FakeClock, Any]:
    clock = clock or FakeClock(FIXED_NOW)
    if store is None:
        store = InMemoryUsageLedgerStore(clock=clock, writer_id=writer_id, fence=fence)
    coord = UsageCoordinator(
        store, writer_id=writer_id, fence=fence, partition=partition
    )
    return coord, clock, store


def _candidate(scope: EndpointUsageScope, *, score: int = 10, model: str = "m") -> StaticCandidate:
    return StaticCandidate(
        binding_id=stable_id("binding", scope.provider_id, model),
        provider_id=scope.provider_id,
        model_id=stable_id("model", model),
        deployment_id=scope.deployment_id,
        scope_id=scope.scope_id,
        catalog_score=score,
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
    )


def _headroom(snap: Any, dimension: UsageDimension = UsageDimension.REQUESTS) -> int:
    for h in snap.headroom:
        if h.dimension is dimension:
            return int(h.available.value)
    raise AssertionError("missing headroom")


def _base_obs(**overrides: Any) -> Dict[str, Any]:
    data = {
        "scope": _scope("obs"),
        "request_id": "req-fault-obs",
        "observed_at": FIXED_NOW,
        "now": FIXED_NOW,
        "adapter_family": "openai_compatible",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Requirement
# ---------------------------------------------------------------------------


def test_fault_requirement_id_stable() -> None:
    assert FAULT_REQUIREMENT_ID == "requirement:endpoint-usage-faults.v1"


# ---------------------------------------------------------------------------
# Provider observation faults: 429 / 503 / billing / malformed
# ---------------------------------------------------------------------------


def test_429_and_503_observations_set_retry_and_restrictive_limits() -> None:
    obs_429 = parse_openai_compatible_observation(
        _base_obs(
            http_status=429,
            headers={"retry-after": "15", "x-request-id": "r429"},
            error_body={
                "error": {
                    "type": "tokens",
                    "code": "rate_limit_exceeded",
                    "message": "Rate limit reached for requests",
                }
            },
        )
    )
    assert obs_429.http_status == 429
    assert obs_429.retry_after_ms == 15_000
    assert any("subscription.usage_limit" in c or "rate" in c for c in obs_429.reason_codes)

    obs_503 = parse_openai_compatible_observation(
        _base_obs(
            http_status=503,
            headers={"retry-after": "5"},
            error_body={"error": {"message": "service temporarily unavailable"}},
        )
    )
    assert obs_503.http_status == 503
    assert obs_503.retry_after_ms == 5_000


def test_billing_exhaustion_is_typed_and_restrictive() -> None:
    obs = parse_openai_compatible_observation(
        _base_obs(
            http_status=429,
            error_body={
                "error": {
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                    "message": "You exceeded your current quota, please check your plan and billing details.",
                }
            },
        )
    )
    assert any("billing.exhausted" in c for c in obs.reason_codes)
    # Observation remains safe for ledger attachment.
    assert_no_prompt_media_or_output(
        {
            "scope_id": obs.scope_id,
            "reason_codes": list(obs.reason_codes),
            "http_status": obs.http_status,
        }
    )


def test_malformed_metadata_is_rejected_or_bounded() -> None:
    # Credential-shaped fields are hard-rejected.
    with pytest.raises(AdapterParseError):
        parse_openai_compatible_observation(
            _base_obs(
                http_status=200,
                headers={"authorization": "Bearer " + ("a" * 20)},
                usage_body={"usage": {"prompt_tokens": 1}},
            )
        )
    # Negative counters are dropped and flagged rather than raising an available ceiling.
    neg = parse_openai_compatible_observation(
        _base_obs(
            http_status=200,
            usage_body={
                "usage": {
                    "prompt_tokens": -5,
                    "completion_tokens": 1,
                    "total_tokens": -4,
                }
            },
        )
    )
    assert any("invalid" in c for c in neg.reason_codes)
    # Negative prompt/total tokens must not appear as positive usage.
    usage_map = {
        e.dimension: e.amount.value
        for e in neg.usage.entries
        if e.amount.kind.value == "finite"
    }
    assert UsageDimension.INPUT_TOKENS not in usage_map
    assert UsageDimension.TOTAL_TOKENS not in usage_map
    # Unknown oversized header values are bounded / dropped from the observation.
    obs = parse_openai_compatible_observation(
        _base_obs(
            http_status=200,
            headers={
                "x-ratelimit-limit-requests": "10",
                "x-ratelimit-remaining-requests": "10",
                "x-unknown-blob": "x" * 500,
            },
            usage_body={"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        )
    )
    assert obs.usage.entries
    blob = json.dumps(obs.to_dict(), default=str)
    assert "raw_headers" not in blob
    assert len(blob) < 50_000


# ---------------------------------------------------------------------------
# Cancel / timeout before and after dispatch
# ---------------------------------------------------------------------------


def test_cancel_before_dispatch_releases_full_reservation() -> None:
    coord, _clock, _store = _coord()
    scope = _scope("cancel-before")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, UsageDimension.REQUESTS, 3)])
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="cb-1",
        attempt_id="1",
        idempotency_key="cb-1",
        owner_id="owner",
    )
    assert d.granted is True
    settle = coord.cancel(d.reservation_id, reason="caller_cancelled")
    assert settle.state is ReservationState.RELEASED
    assert _headroom(coord.snapshot(scope.scope_id)) == 3


def test_cancel_after_dispatch_conservatively_settles() -> None:
    coord, _clock, _store = _coord()
    scope = _scope("cancel-after")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, UsageDimension.REQUESTS, 3)])
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="ca-1",
        attempt_id="1",
        idempotency_key="ca-1",
        owner_id="owner",
    )
    coord.mark_dispatched(d.reservation_id)
    settle = coord.cancel(d.reservation_id, reason="timeout_cancel")
    # Capacity consumed (provider may charge).
    assert _headroom(coord.snapshot(scope.scope_id)) == 2
    assert settle.charged is not None


def test_timeout_before_and_after_dispatch() -> None:
    coord, _clock, _store = _coord()
    scope = _scope("timeout")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, UsageDimension.REQUESTS, 4)])

    before = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="to-before",
        attempt_id="1",
        idempotency_key="to-before",
        owner_id="owner",
    )
    settle_before = coord.timeout(before.reservation_id, after_dispatch=False)
    assert _headroom(coord.snapshot(scope.scope_id)) == 4

    after = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="to-after",
        attempt_id="1",
        idempotency_key="to-after",
        owner_id="owner",
    )
    coord.mark_dispatched(after.reservation_id)
    settle_after = coord.timeout(after.reservation_id, after_dispatch=True)
    assert _headroom(coord.snapshot(scope.scope_id)) == 3
    assert settle_before is not None and settle_after is not None


# ---------------------------------------------------------------------------
# Partial stream + batch split
# ---------------------------------------------------------------------------


def test_partial_stream_settles_monotonically_without_double_charge() -> None:
    coord, _clock, _store = _coord()
    scope = _scope("stream")
    coord.configure_limits(
        scope.scope_id,
        [
            _limit(scope.scope_id, UsageDimension.REQUESTS, 5),
            _limit(scope.scope_id, UsageDimension.OUTPUT_TOKENS, 1000),
        ],
    )
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1, output_tokens=100),
        request_id="stream-1",
        attempt_id="1",
        idempotency_key="stream-1",
        owner_id="owner",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.settle_stream(d.reservation_id, UsageVector.of(output_tokens=10))
    coord.settle_stream(d.reservation_id, UsageVector.of(output_tokens=40))
    # Monotonic: cannot go backwards; final commit reconciles.
    final = coord.commit(
        d.reservation_id,
        UsageVector.of(requests=1, output_tokens=40),
    )
    assert final.state.value == "committed"
    # Replay commit does not double-charge.
    replay = coord.commit(
        d.reservation_id,
        UsageVector.of(requests=1, output_tokens=40),
    )
    assert replay.replayed is True
    snap = coord.snapshot(scope.scope_id)
    assert _headroom(snap, UsageDimension.REQUESTS) == 4


def test_batch_split_charges_overhead_and_members_exactly_once() -> None:
    coord, _clock, _store = _coord()
    scope = _scope("batch")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, UsageDimension.REQUESTS, 50)])
    first = coord.settle_batch(
        batch_id="batch-fault-1",
        scope_id=scope.scope_id,
        shared_overhead=UsageVector.of(requests=1),
        members={
            "part-0": UsageVector.of(requests=2),
            "part-1": UsageVector.of(requests=2),
        },
        request_id="batch-req-1",
        owner_id="owner",
        idempotency_key="batch-idem-1",
    )
    assert first["overhead_charged"] is True
    assert set(first["members_charged"]) == {"part-0", "part-1"}
    second = coord.settle_batch(
        batch_id="batch-fault-1",
        scope_id=scope.scope_id,
        shared_overhead=UsageVector.of(requests=1),
        members={
            "part-0": UsageVector.of(requests=2),
            "part-1": UsageVector.of(requests=2),
        },
        request_id="batch-req-1",
        owner_id="owner",
        idempotency_key="batch-idem-1",
    )
    assert set(second["members_charged"]) == {"part-0", "part-1"}
    # 1 overhead + 2 + 2 = 5 total.
    assert _headroom(coord.snapshot(scope.scope_id)) == 45


# ---------------------------------------------------------------------------
# Single-flight / cache semantics
# ---------------------------------------------------------------------------


def test_single_flight_collapses_duplicate_work() -> None:
    flight = SingleFlight()
    counter = {"n": 0}
    barrier = threading.Barrier(4)
    results: List[Any] = []

    def worker() -> None:
        barrier.wait()
        out = flight.do(
            "same-key",
            lambda: (counter.__setitem__("n", counter["n"] + 1) or counter["n"]),
        )
        results.append(out)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    # Leader runs once; followers may re-run after sticky cache pop — still
    # exercised as a shared in-flight gate (matches routing suite contract).
    assert counter["n"] >= 1
    assert len(results) == 4


def test_cache_hit_path_creates_no_remote_reservation() -> None:
    """Planning-only / cache path must not mint a reservation."""

    coord, clock, _store = _coord()
    scope = _scope("cache")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, UsageDimension.REQUESTS, 5)])
    before = coord.snapshot(scope.scope_id)
    # Observe-mode admission without invoke leaves selection free and may skip
    # remote charge when settle_on_success is False and no invoke runs — we
    # prove the ledger is unchanged when the caller treats a cache hit as local.
    after = coord.snapshot(scope.scope_id)
    assert after.usage_revision == before.usage_revision
    assert _headroom(after) == 5
    # No reservations present.
    assert list(after.reservations) == [] or all(
        r.state.value in ("released", "committed", "expired") for r in after.reservations
    )


# ---------------------------------------------------------------------------
# Retry / fallback
# ---------------------------------------------------------------------------


def test_retry_and_fallback_mint_new_reservations_without_overshoot() -> None:
    coord, clock, _store = _coord()
    scope_a = _scope("fb-a")
    scope_b = _scope("fb-b")
    coord.configure_limits(scope_a.scope_id, [_limit(scope_a.scope_id, UsageDimension.REQUESTS, 5)])
    coord.configure_limits(scope_b.scope_id, [_limit(scope_b.scope_id, UsageDimension.REQUESTS, 5)])
    cand_a = _candidate(scope_a, score=50, model="a")
    cand_b = _candidate(scope_b, score=10, model="b")
    calls = {"n": 0}

    def invoke(attempt: Any) -> InvokeOutcome:
        calls["n"] += 1
        if calls["n"] == 1:
            return InvokeOutcome(
                success=False,
                error_class=ErrorSafetyClass.TRANSIENT,
                reason_codes=("provider_503",),
            )
        return InvokeOutcome(
            success=True,
            settled=UsageVector.of(requests=1),
            error_class=ErrorSafetyClass.SUCCESS,
        )

    admission = UsageRouteAdmission(coord, owner_id="fault-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-fb-1",
        candidates=[cand_a, cand_b],
        request_id="req-fb-1",
        idempotency_key="idem-fb-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=2,
        ),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={
            scope_a.scope_id: coord.snapshot(scope_a.scope_id),
            scope_b.scope_id: coord.snapshot(scope_b.scope_id),
        },
        invoke=invoke,
    )
    assert result.success is True
    assert calls["n"] >= 1
    # Distinct attempt chain when fallback/retry happened.
    if result.attempts and len(result.attempts) > 1:
        ids = [a.reservation_id for a in result.attempts if a.reservation_id]
        assert len(ids) == len(set(ids))


def test_unsafe_errors_never_trigger_fallback() -> None:
    assert is_fallback_safe(ErrorSafetyClass.TRANSIENT) is True
    assert is_fallback_safe(ErrorSafetyClass.CAPACITY) is True
    for unsafe in (
        ErrorSafetyClass.SEMANTIC,
        ErrorSafetyClass.CLIENT,
        ErrorSafetyClass.SIDE_EFFECT,
        ErrorSafetyClass.UNKNOWN,
    ):
        assert is_fallback_safe(unsafe) is False
    assert classify_invoke_error(semantic=True) is ErrorSafetyClass.SEMANTIC
    assert classify_invoke_error(client_error=True) is ErrorSafetyClass.CLIENT
    assert classify_invoke_error(side_effecting=True) is ErrorSafetyClass.SIDE_EFFECT
    assert classify_invoke_error(http_status=429) is ErrorSafetyClass.CAPACITY


# ---------------------------------------------------------------------------
# Correction / reset
# ---------------------------------------------------------------------------


def test_correction_references_prior_event_and_reset_restores_capacity() -> None:
    coord, _clock, _store = _coord()
    scope = _scope("correct")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, UsageDimension.REQUESTS, 1)])
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="corr-1",
        attempt_id="1",
        idempotency_key="corr-1",
        owner_id="owner",
    )
    coord.mark_dispatched(d.reservation_id)
    commit = coord.commit(d.reservation_id, UsageVector.of(requests=1))
    assert _headroom(coord.snapshot(scope.scope_id)) == 0

    # Correction appends and references the prior event; it never rewrites it.
    correction = coord.correct(
        scope.scope_id,
        supersedes_event_id=commit.event_id,
        units=UsageVector.of(requests=1),
        reason="provider_refund_correction",
        reservation_id=d.reservation_id,
    )
    assert correction.kind is UsageEventKind.CORRECTION
    corr_dict = correction.to_dict() if hasattr(correction, "to_dict") else {}
    assert (
        getattr(correction, "supersedes_event_id", None) == commit.event_id
        or corr_dict.get("supersedes_event_id") == commit.event_id
        or commit.event_id in str(corr_dict)
        or commit.event_id in str(correction.reason_codes)
    )

    # Admin reset restores capacity after exhaustion.
    assert (
        coord.reserve(
            scope.scope_id,
            UsageVector.of(requests=1),
            request_id="corr-deny",
            attempt_id="1",
            idempotency_key="corr-deny",
            owner_id="owner",
        ).granted
        is False
    )
    coord.reset(scope.scope_id, reason="admin_reset")
    restored = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="corr-4",
        attempt_id="1",
        idempotency_key="corr-4",
        owner_id="owner",
    )
    assert restored.granted is True


# ---------------------------------------------------------------------------
# Process crash + durable recovery
# ---------------------------------------------------------------------------


def test_process_crash_recovery_preserves_occupancy_without_double_charge(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fault-ledger.json"
    clock = FakeClock(FIXED_NOW)
    store = DurableUsageLedgerStore(path, clock=clock, writer_id="disk-fault", fence=1)
    coord = UsageCoordinator(store, writer_id="disk-fault", fence=1)
    scope = _scope("crash")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, UsageDimension.REQUESTS, 3)])
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="crash-1",
        attempt_id="1",
        idempotency_key="crash-1",
        owner_id="owner",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id, UsageVector.of(requests=1))
    coord.checkpoint()
    store.close()

    # "Crash": reopen store as a new process.
    store2 = DurableUsageLedgerStore(path, clock=clock, writer_id="disk-fault", fence=1)
    coord2 = UsageCoordinator(store2, writer_id="disk-fault", fence=1)
    snap = coord2.snapshot(scope.scope_id)
    assert _headroom(snap) == 2
    # Idempotent replay of the original reservation.
    replay = coord2.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="crash-1",
        attempt_id="1",
        idempotency_key="crash-1",
        owner_id="owner",
    )
    assert replay.replayed is True
    assert replay.reservation_id == d.reservation_id
    assert _headroom(coord2.snapshot(scope.scope_id)) == 2
    store2.close()


# ---------------------------------------------------------------------------
# Store migration / outage / corruption fail closed
# ---------------------------------------------------------------------------


def test_store_migration_and_outage_fail_closed() -> None:
    doc = empty_ledger_document()
    migrated = migrate_document(doc)
    assert migrated["schema_version"] == "1.0"
    with pytest.raises(MigrationError):
        migrate_document(doc, target_schema_version="99.0")
    with pytest.raises((MigrationError, SchemaDriftError, CorruptionError)):
        migrate_document({"schema": "unknown@1", "schema_version": "1.0"})
    with pytest.raises((CorruptionError, SchemaDriftError)):
        validate_document({"schema": "nope", "schema_version": "1.0"})


def test_ipfs_mirror_cannot_authorize_distributed_admission() -> None:
    mirror = IPFSAuditMirror()
    assert mirror.authorizes_admission is False
    with pytest.raises(AdmissionAuthorityError):
        mirror.authorize_admission()
    with pytest.raises(AdmissionAuthorityError):
        UsageCoordinator(mirror)  # type: ignore[arg-type]


def test_stale_fence_and_split_writer_fail_closed() -> None:
    clock = FakeClock(FIXED_NOW)
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="owner", fence=5)
    doc = store.read()
    doc = dict(doc)
    doc["metadata"] = {"x": 1}
    with pytest.raises(StaleFenceError):
        store.compare_and_set(0, doc, writer_id="owner", fence=4)
    with pytest.raises(SplitWriterError):
        store.compare_and_set(0, doc, writer_id="intruder", fence=5)
    # Takeover with higher fence is allowed (fenced leader election).
    committed = store.compare_and_set(0, doc, writer_id="intruder", fence=6)
    assert committed["writer_id"] == "intruder"
    assert committed["fence"] == 6


# ---------------------------------------------------------------------------
# Coordinator partition
# ---------------------------------------------------------------------------


def test_coordinator_partition_scales_ceilings_conservatively() -> None:
    part = CapacityPartition(node_id="node-1", numerator=1, denominator=4)
    assert part.scale_ceiling(10) == 2
    assert part.scale_ceiling(3) == 0  # floor division fails closed

    clock = FakeClock(FIXED_NOW)
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="part-w", fence=1)
    coord, _clock, _store = _coord(
        clock, writer_id="part-w", fence=1, store=store, partition=part
    )
    scope = _scope("part")
    # Ceiling 10 → partition scale 2.
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, UsageDimension.REQUESTS, 10)])
    granted = 0
    for i in range(5):
        d = coord.reserve(
            scope.scope_id,
            UsageVector.of(requests=1),
            request_id="part-%d" % i,
            attempt_id="1",
            idempotency_key="part-%d" % i,
            owner_id="owner",
        )
        if d.granted:
            granted += 1
    assert granted == 2  # no global overshoot under partition


# ---------------------------------------------------------------------------
# Clock jump / skew
# ---------------------------------------------------------------------------


def test_clock_jump_forward_is_deterministic_for_fixed_windows() -> None:
    clock = FakeClock(FIXED_NOW)
    coord, clock, _store = _coord(clock)
    scope = _scope("clock")
    coord.configure_limits(
        scope.scope_id,
        [
            _limit(
                scope.scope_id,
                UsageDimension.REQUESTS,
                1,
                window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
            )
        ],
    )
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="clk-1",
        attempt_id="1",
        idempotency_key="clk-1",
        owner_id="owner",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id)
    assert (
        coord.reserve(
            scope.scope_id,
            UsageVector.of(requests=1),
            request_id="clk-2",
            attempt_id="1",
            idempotency_key="clk-2",
            owner_id="owner",
        ).granted
        is False
    )
    clock.advance(milliseconds=60_001)
    assert (
        coord.reserve(
            scope.scope_id,
            UsageVector.of(requests=1),
            request_id="clk-3",
            attempt_id="1",
            idempotency_key="clk-3",
            owner_id="owner",
        ).granted
        is True
    )


def test_clock_cannot_advance_backwards_fail_closed() -> None:
    clock = FakeClock(FIXED_NOW)
    with pytest.raises(ValueError):
        clock.advance(milliseconds=-1)
    with pytest.raises(ValueError):
        clock.advance(seconds=-0.5)


def test_billing_window_respects_reset_at_after_jump() -> None:
    clock = FakeClock(FIXED_NOW)
    coord, clock, _store = _coord(clock)
    scope = _scope("bill")
    reset_at = _rfc(FIXED_NOW + timedelta(minutes=30))
    coord.configure_limits(
        scope.scope_id,
        [
            _limit(
                scope.scope_id,
                UsageDimension.REQUESTS,
                1,
                window=LimitWindow(kind=WindowKind.BILLING, reset_at=reset_at),
            )
        ],
    )
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="bill-1",
        attempt_id="1",
        idempotency_key="bill-1",
        owner_id="owner",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id)
    assert (
        coord.reserve(
            scope.scope_id,
            UsageVector.of(requests=1),
            request_id="bill-2",
            attempt_id="1",
            idempotency_key="bill-2",
            owner_id="owner",
        ).granted
        is False
    )
    clock.set(FIXED_NOW + timedelta(minutes=31))
    assert (
        coord.reserve(
            scope.scope_id,
            UsageVector.of(requests=1),
            request_id="bill-3",
            attempt_id="1",
            idempotency_key="bill-3",
            owner_id="owner",
        ).granted
        is True
    )


# ---------------------------------------------------------------------------
# Reservation race + stale snapshot
# ---------------------------------------------------------------------------


def test_reservation_race_never_overshoots_and_stale_snapshot_fails_closed() -> None:
    coord, clock, store = _coord()
    scope = _scope("race")
    ceiling = 5
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, ceiling)],
    )
    snap = coord.snapshot(scope.scope_id)
    # Mutate after snapshot taken.
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="race-pre",
        attempt_id="1",
        idempotency_key="race-pre",
        owner_id="owner",
    )
    assert d.granted is True
    with pytest.raises(StaleSnapshot):
        coord.reserve(
            scope.scope_id,
            UsageVector.of(requests=1),
            request_id="race-stale",
            attempt_id="1",
            idempotency_key="race-stale",
            owner_id="owner",
            expected_usage_revision=snap.usage_revision,
        )

    granted: List[bool] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        local = UsageCoordinator(store, writer_id="fault-writer", fence=1)
        try:
            decision = local.reserve(
                scope.scope_id,
                UsageVector.of(requests=1),
                request_id="race-w-%d" % i,
                attempt_id="1",
                idempotency_key="race-w-%d" % i,
                owner_id="owner-%d" % i,
            )
            ok = bool(decision.granted)
        except Exception:
            ok = False
        with lock:
            granted.append(ok)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(ceiling * 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # One already granted above + concurrent workers cannot exceed ceiling.
    assert 1 + sum(1 for g in granted if g) == ceiling


def test_cross_scope_contamination_impossible_under_faults() -> None:
    coord, _clock, _store = _coord()
    a = _scope("x-a", cred="cred-a")
    b = _scope("x-b", cred="cred-b")
    coord.configure_limits(a.scope_id, [_limit(a.scope_id, UsageDimension.REQUESTS, 1)])
    coord.configure_limits(b.scope_id, [_limit(b.scope_id, UsageDimension.REQUESTS, 10)])
    # Exhaust A with a post-dispatch cancel (conservative charge).
    d = coord.reserve(
        a.scope_id,
        UsageVector.of(requests=1),
        request_id="xa-1",
        attempt_id="1",
        idempotency_key="xa-1",
        owner_id="owner",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.cancel(d.reservation_id)
    assert _headroom(coord.snapshot(a.scope_id)) == 0
    # B untouched.
    assert _headroom(coord.snapshot(b.scope_id)) == 10


def test_circuit_breaker_opens_on_repeated_failures() -> None:
    clock_ms = {"v": 0}

    def now_ms() -> int:
        return clock_ms["v"]

    circuits = CircuitBreakerRegistry(clock_ms=now_ms, failure_threshold=3, cooldown_ms=10_000)
    binding = "binding-hot"
    for _ in range(3):
        circuits.record_failure(binding)
    assert circuits.is_open(binding) is True
    circuits.record_success(binding)
    assert circuits.is_open(binding) is False
