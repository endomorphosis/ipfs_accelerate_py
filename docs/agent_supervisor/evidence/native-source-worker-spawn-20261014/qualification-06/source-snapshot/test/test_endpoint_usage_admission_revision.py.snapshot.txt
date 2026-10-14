"""Stored-state admission pins remain stable across observational clock drift.

All owners are actual isolated ledger stores. No provider or external service is
used; clocks and the two deliberately contended CAS boundaries are controlled.
"""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.endpoint_usage.admission import (
    UsageAdmissionSnapshot,
    admission_revision_for_document,
    validate_admission_revision,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import snapshot_identity, stable_id
from ipfs_accelerate_py.endpoint_usage.ledger import LedgerError, StaleSnapshot, UsageLedger
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitWindow,
    Quantity,
    UsageDimension,
    UsageEstimate,
    UsageEventKind,
    UsageLimit,
    UsageSnapshot,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import (
    CapacityPartition,
    CompareAndSetConflict,
    DurableUsageLedgerStore,
    FakeClock,
    InMemoryUsageLedgerStore,
    StaleFenceError,
    SystemClock,
)


def _limit(scope, ceiling=10, *, dimension=UsageDimension.REQUESTS, window=None):
    return UsageLimit(
        scope_id=scope,
        dimension=dimension,
        ceiling=Quantity.finite(ceiling),
        remaining=Quantity.finite(ceiling),
        used=Quantity.finite(0),
        window=window or LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
    )


def _reserve(owner, key, *, pin=None, requested=None, **kwargs):
    options = {
        "request_id": key,
        "idempotency_key": key,
        "owner_id": "test-owner",
        "expected_admission_revision": pin,
    }
    options.update(kwargs)
    return owner.coord.reserve(
        owner.scope, requested or UsageVector.of(requests=1), **options
    )


@pytest.fixture(params=["memory", "durable"])
def owner_factory(request, tmp_path):
    stores = []

    def create(*, clock=None):
        clock = clock or FakeClock()
        path = tmp_path / ("ledger-%s.json" % len(stores))
        arguments = {"clock": clock, "writer_id": "test-writer", "fence": 1}
        if request.param == "durable":
            store = DurableUsageLedgerStore(path, **arguments)
        else:
            store = InMemoryUsageLedgerStore(**arguments)
        stores.append(store)
        coord = UsageCoordinator(store, writer_id="test-writer", fence=1)
        scope = stable_id("scope", "admission-revision-test")
        coord.configure_limits(scope, [_limit(scope)])
        return SimpleNamespace(
            coord=coord, store=store, clock=clock, scope=scope, path=path,
            durable=request.param == "durable",
        )

    yield create
    for store in reversed(stores):
        store.close()


@pytest.fixture
def owner(owner_factory):
    return owner_factory()


class _AdvancingSystemClock:
    """Use UTC wall time, guaranteeing a distinct instant on every read."""

    def __init__(self):
        self.clock = SystemClock()
        self.previous = self.clock.now()
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            self.previous = max(self.clock.now(), self.previous + timedelta(milliseconds=1))
            return self.previous


def test_advancing_wall_clock_does_not_invalidate_admission_pin(owner_factory):
    owner = owner_factory(clock=_AdvancingSystemClock())
    before = owner.store.read()
    first = owner.coord.admission_snapshot(owner.scope)
    second = owner.coord.admission_snapshot(owner.scope)
    assert first.snapshot.observed_at != second.snapshot.observed_at
    assert first.snapshot.usage_revision != second.snapshot.usage_revision
    assert first.admission_revision == second.admission_revision
    assert first.ledger_revision == second.ledger_revision == before["revision"]
    assert owner.store.read() == before
    decision = _reserve(owner, "clock-advances", pin=first.admission_revision)
    assert decision.granted


def test_legacy_snapshot_identity_and_serialization_are_unchanged(owner):
    legacy = owner.coord.snapshot(owner.scope)
    expected = snapshot_identity({
        "scope_id": legacy.scope_id,
        "observed_at": legacy.observed_at,
        "state": legacy.state.value,
        "limits": [item.limit_id for item in legacy.limits],
        "reservations": [item.reservation_id for item in legacy.reservations],
        "reason_codes": list(legacy.reason_codes),
    })
    assert legacy.usage_revision == expected
    assert UsageSnapshot.from_dict(legacy.to_dict()).to_dict() == legacy.to_dict()
    assert "admission_revision" not in legacy.to_dict()
    assert owner.coord.admission_snapshot(owner.scope).snapshot == legacy
    owner.clock.advance(milliseconds=1)
    with pytest.raises(StaleSnapshot):
        _reserve(owner, "legacy-stale", expected_usage_revision=legacy.usage_revision)


def test_committed_occupancy_changes_pin_despite_legacy_identity_alias(owner):
    before = owner.coord.admission_snapshot(owner.scope)
    decision = _reserve(owner, "charged")
    owner.coord.commit(decision.reservation_id, UsageVector.of(requests=1))
    after = owner.coord.admission_snapshot(owner.scope)
    assert before.snapshot.reservations == after.snapshot.reservations == ()
    assert before.snapshot.state == after.snapshot.state
    assert before.snapshot.usage_revision == after.snapshot.usage_revision
    assert before.snapshot.headroom != after.snapshot.headroom
    assert before.admission_revision != after.admission_revision
    document = owner.store.read()
    with pytest.raises(StaleSnapshot):
        _reserve(owner, "stale-charge", pin=before.admission_revision)
    assert owner.store.read() == document


def _mutate_document(owner, mutate):
    document = owner.store.read()
    ledger = UsageLedger(document)
    mutate(ledger)
    owner.store.compare_and_set(
        document["revision"], ledger.document, writer_id="test-writer", fence=1
    )


@pytest.mark.parametrize("mutation", [
    "caller_budget", "same_id_limit", "cooldown", "disabled", "reset",
    "compaction", "other_scope",
])
def test_committed_mutations_invalidate_whole_ledger_pin(owner, mutation):
    before = owner.coord.admission_snapshot(owner.scope)
    if mutation == "caller_budget":
        owner.coord.configure_caller_budget(owner.scope, UsageVector.of(requests=0))
    elif mutation == "same_id_limit":
        replacement = _limit(owner.scope, ceiling=9)
        assert replacement.limit_id == before.snapshot.limits[0].limit_id
        owner.coord.configure_limits(owner.scope, [replacement])
    elif mutation == "cooldown":
        owner.coord.append_observation(
            owner.scope, kind=UsageEventKind.OBSERVATION_FAILURE,
            cooldown_until="2024-01-01T00:01:00Z",
        )
    elif mutation == "disabled":
        _mutate_document(owner, lambda ledger: ledger.disable_scope(owner.scope, reason="maintenance"))
    elif mutation == "reset":
        owner.coord.reset(owner.scope)
    elif mutation == "compaction":
        owner.coord.compact()
    else:
        scope = stable_id("scope", "other-scope")
        owner.coord.configure_limits(scope, [_limit(scope)])
    after = owner.coord.admission_snapshot(owner.scope)
    assert before.admission_revision != after.admission_revision
    document = owner.store.read()
    with pytest.raises(StaleSnapshot):
        _reserve(owner, "mutation-stale", pin=before.admission_revision)
    assert owner.store.read() == document
    if mutation == "caller_budget":
        decision = _reserve(owner, "current-budget", pin=after.admission_revision)
        assert not decision.granted
        assert "caller_budget_exceeded" in decision.reason_codes


def test_checkpoint_receipt_does_not_change_stored_admission_state(owner):
    before = owner.coord.admission_snapshot(owner.scope)
    document = owner.store.read()
    owner.clock.advance(seconds=1)
    receipt = owner.coord.checkpoint()
    assert receipt["revision"] == document["revision"]
    assert owner.store.read() == document
    assert owner.coord.admission_snapshot(owner.scope).admission_revision == before.admission_revision


def test_refund_changes_pin_even_when_legacy_snapshot_identity_aliases(owner):
    charged = _reserve(owner, "refundable")
    owner.coord.commit(charged.reservation_id, UsageVector.of(requests=1))
    before = owner.coord.admission_snapshot(owner.scope)
    owner.coord.refund(owner.scope, UsageVector.of(requests=1), reservation_id=charged.reservation_id)
    after = owner.coord.admission_snapshot(owner.scope)
    assert after.snapshot.usage_revision == before.snapshot.usage_revision
    assert after.snapshot.headroom[0].available.value > before.snapshot.headroom[0].available.value
    assert after.admission_revision != before.admission_revision
    with pytest.raises(StaleSnapshot):
        _reserve(owner, "refunded-stale", pin=before.admission_revision)


def test_scope_fence_and_partition_are_bound(owner):
    before = owner.coord.admission_snapshot(owner.scope)
    other = owner.coord.admission_snapshot(stable_id("scope", "other-scope"))
    assert before.admission_revision != other.admission_revision
    with pytest.raises(StaleSnapshot):
        _reserve(owner, "wrong-scope", pin=other.admission_revision)
    partitioned = UsageCoordinator(
        owner.store, writer_id="test-writer", fence=1,
        partition=CapacityPartition(node_id="test-node", numerator=1, denominator=2),
    )
    partition_pin = partitioned.admission_snapshot(owner.scope)
    assert partition_pin.admission_revision != before.admission_revision
    with pytest.raises(StaleSnapshot):
        partitioned.reserve(
            owner.scope, UsageVector.of(requests=1), request_id="partition",
            idempotency_key="partition", owner_id="test-owner",
            expected_admission_revision=before.admission_revision,
        )
    fence = owner.store.bump_fence(writer_id="replacement-writer")
    replacement = UsageCoordinator(owner.store, writer_id="replacement-writer", fence=fence)
    current = replacement.admission_snapshot(owner.scope)
    assert current.fence == current.ledger_fence == fence
    assert current.admission_revision != before.admission_revision
    with pytest.raises(StaleFenceError):
        _reserve(owner, "stale-fence", pin=before.admission_revision)
    with pytest.raises(StaleSnapshot):
        replacement.reserve(
            owner.scope, UsageVector.of(requests=1), request_id="new-fence",
            idempotency_key="new-fence", owner_id="test-owner",
            expected_admission_revision=before.admission_revision,
        )


def test_durable_reopen_preserves_pin(owner):
    if not owner.durable:
        pytest.skip("durable reopen qualification")
    before = owner.coord.admission_snapshot(owner.scope)
    reopened = DurableUsageLedgerStore(
        owner.path, clock=owner.clock, writer_id="test-writer", fence=1
    )
    try:
        coord = UsageCoordinator(reopened, writer_id="test-writer", fence=1)
        owner.clock.advance(seconds=1)
        assert coord.admission_snapshot(owner.scope).admission_revision == before.admission_revision
        decision = coord.reserve(
            owner.scope, UsageVector.of(requests=1), request_id="reopened",
            idempotency_key="reopened", owner_id="test-owner",
            expected_admission_revision=before.admission_revision,
        )
        assert decision.granted
        assert len(owner.store.read()["reservations"]) == 1
    finally:
        reopened.close()


def test_transient_cas_conflict_retries_the_same_unchanged_pin(owner, monkeypatch):
    pin = owner.coord.admission_snapshot(owner.scope).admission_revision
    original = owner.store.compare_and_set
    calls = []

    def conflict_once(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            raise CompareAndSetConflict("injected transient conflict without mutation")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner.store, "compare_and_set", conflict_once)
    decision = _reserve(owner, "transient", pin=pin)
    assert decision.granted
    assert len(calls) == 2 and calls[0] == calls[1]
    assert len(owner.store.read()["reservations"]) == 1


def test_concurrent_same_pin_has_one_winner_then_refreshes_safely(owner, monkeypatch):
    owner.coord.configure_limits(owner.scope, [_limit(owner.scope, ceiling=2)])
    pin = owner.coord.admission_snapshot(owner.scope).admission_revision
    barrier = threading.Barrier(2, timeout=5)
    original = owner.store.compare_and_set
    seen = set()
    lock = threading.Lock()
    conflicts = []

    def contended(*args, **kwargs):
        with lock:
            first = threading.get_ident() not in seen
            seen.add(threading.get_ident())
        if first:
            barrier.wait()
        try:
            return original(*args, **kwargs)
        except CompareAndSetConflict:
            conflicts.append(True)
            raise

    monkeypatch.setattr(owner.store, "compare_and_set", contended)

    def reserve(key):
        try:
            return _reserve(owner, key, pin=pin)
        except StaleSnapshot as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(reserve, "racer-%s" % i) for i in range(2)]
        results = [future.result(timeout=10) for future in futures]
    monkeypatch.setattr(owner.store, "compare_and_set", original)
    assert len(conflicts) == 1
    assert sum(isinstance(result, StaleSnapshot) for result in results) == 1
    assert sum(getattr(result, "granted", False) for result in results) == 1
    current = owner.coord.admission_snapshot(owner.scope)
    assert _reserve(owner, "refreshed", pin=current.admission_revision).granted
    current = owner.coord.admission_snapshot(owner.scope)
    assert not _reserve(owner, "full", pin=current.admission_revision).granted
    assert owner.coord.snapshot(owner.scope).headroom[0].reserved.value == 2


@pytest.mark.parametrize("kind", [
    WindowKind.FIXED, WindowKind.SLIDING, WindowKind.BILLING, WindowKind.TOKEN_BUCKET,
])
def test_window_rollover_rechecks_capacity_without_changing_stored_pin(owner, kind):
    if kind is WindowKind.BILLING:
        window = LimitWindow(kind=kind, reset_at="2024-01-01T00:00:01Z")
    elif kind is WindowKind.TOKEN_BUCKET:
        window = LimitWindow(kind=kind, length_ms=1000, refill_per_second=1, burst=1)
    else:
        window = LimitWindow(kind=kind, length_ms=1000)
    owner.coord.configure_limits(owner.scope, [_limit(owner.scope, ceiling=1, window=window)])
    charged = _reserve(owner, "old-window")
    owner.coord.commit(charged.reservation_id, UsageVector.of(requests=1))
    pin = owner.coord.admission_snapshot(owner.scope)
    assert pin.snapshot.headroom[0].available.value == 0
    owner.clock.advance(milliseconds=1001)
    after = owner.coord.admission_snapshot(owner.scope)
    assert after.admission_revision == pin.admission_revision
    assert after.snapshot.headroom[0].available.value == 1
    assert _reserve(owner, "new-window", pin=pin.admission_revision).granted
    current = owner.coord.admission_snapshot(owner.scope)
    assert not _reserve(owner, "new-window-full", pin=current.admission_revision).granted


def test_backward_clock_rechecks_capacity_instead_of_trusting_snapshot(owner):
    window = LimitWindow(kind=WindowKind.FIXED, length_ms=1000)
    owner.coord.configure_limits(owner.scope, [_limit(owner.scope, ceiling=1, window=window)])
    start = owner.clock.now()
    charged = _reserve(owner, "charged-before-jump")
    owner.coord.commit(charged.reservation_id, UsageVector.of(requests=1))
    owner.clock.advance(seconds=2)
    pin = owner.coord.admission_snapshot(owner.scope)
    assert pin.snapshot.headroom[0].available.value == 1
    owner.clock.set(start)
    assert owner.coord.admission_snapshot(owner.scope).admission_revision == pin.admission_revision
    decision = _reserve(owner, "backward-jump", pin=pin.admission_revision)
    assert not decision.granted
    assert "limit_exhausted" in decision.reason_codes


def test_expired_concurrent_hold_is_rechecked_without_a_ledger_write(owner):
    owner.coord.configure_limits(owner.scope, [_limit(
        owner.scope, ceiling=1, dimension=UsageDimension.CONCURRENT_REQUESTS,
        window=LimitWindow(kind=WindowKind.CONCURRENT),
    )])
    units = UsageVector.of(concurrent_requests=1)
    assert _reserve(owner, "expiring", requested=units, ttl_ms=10).granted
    pin = owner.coord.admission_snapshot(owner.scope)
    assert pin.snapshot.headroom[0].available.value == 0
    owner.clock.advance(milliseconds=11)
    after = owner.coord.admission_snapshot(owner.scope)
    assert after.admission_revision == pin.admission_revision
    assert after.snapshot.headroom[0].available.value == 1
    assert _reserve(owner, "replacement", requested=units, pin=pin.admission_revision).granted
    current = owner.coord.admission_snapshot(owner.scope)
    assert not _reserve(owner, "overlap", requested=units, pin=current.admission_revision).granted


def test_elapsed_cooldown_keeps_pin_but_current_time_allows_admission(owner):
    owner.coord.append_observation(
        owner.scope, kind=UsageEventKind.OBSERVATION_FAILURE,
        cooldown_until="2024-01-01T00:00:01Z",
    )
    pin = owner.coord.admission_snapshot(owner.scope)
    owner.clock.advance(milliseconds=1001)
    assert owner.coord.admission_snapshot(owner.scope).admission_revision == pin.admission_revision
    assert _reserve(owner, "cooldown-ended", pin=pin.admission_revision).granted


def test_exact_replay_accepts_original_self_stale_pin_without_second_hold(owner):
    pin = owner.coord.admission_snapshot(owner.scope).admission_revision
    original = _reserve(owner, "replay", pin=pin)
    assert original.granted
    assert owner.coord.admission_snapshot(owner.scope).admission_revision != pin
    owner.clock.advance(milliseconds=1)
    replay = _reserve(owner, "replay", pin=pin)
    assert replay.replayed and replay.granted
    assert replay.reservation_id == original.reservation_id
    assert len(owner.store.read()["reservations"]) == 1
    assert owner.coord.snapshot(owner.scope).headroom[0].reserved.value == 1


@pytest.mark.parametrize("changed", [
    "requested", "budget", "owner", "lease", "ttl", "fence", "estimate", "pin", "legacy",
])
def test_replay_with_changed_contract_is_rejected_without_mutation(owner, changed):
    pin = owner.coord.admission_snapshot(owner.scope).admission_revision
    original = _reserve(owner, "contract", pin=pin)
    assert original.granted
    options = {"pin": pin}
    if changed == "requested":
        options["requested"] = UsageVector.of(requests=2)
    elif changed == "budget":
        options["caller_budget"] = UsageVector.of(requests=1)
    elif changed == "owner":
        options["owner_id"] = "different-owner"
    elif changed == "lease":
        options["lease_id"] = "different-lease"
    elif changed == "ttl":
        options["ttl_ms"] = 1000
    elif changed == "fence":
        options["fence"] = 2
    elif changed == "estimate":
        options["estimate"] = UsageEstimate(
            scope_id=owner.scope, operation="text.chat", requested=UsageVector.of(requests=1)
        )
    elif changed == "pin":
        options["pin"] = owner.coord.admission_snapshot(owner.scope).admission_revision
    else:
        options["pin"] = None
    before = owner.store.read()
    with pytest.raises(LedgerError) as error:
        _reserve(owner, "contract", **options)
    assert "admission_replay_contract_mismatch" in error.value.reason_codes
    assert owner.store.read() == before


def test_denied_pinned_replay_stays_denied_after_capacity_is_changed(owner):
    owner.coord.configure_caller_budget(owner.scope, UsageVector.of(requests=0))
    pin = owner.coord.admission_snapshot(owner.scope).admission_revision
    original = _reserve(owner, "denied", pin=pin)
    assert not original.granted
    owner.coord.configure_caller_budget(owner.scope, UsageVector.of(requests=10))
    replay = _reserve(owner, "denied", pin=pin)
    assert replay.replayed and not replay.granted
    assert replay.reservation_id == original.reservation_id
    document = owner.store.read()
    assert len(document["idempotency"]) == 1
    assert document["reservations"] == {}


def test_old_protocol_cannot_be_replayed_through_a_new_pin(owner):
    original = _reserve(owner, "legacy-contract")
    assert original.granted
    pin = owner.coord.admission_snapshot(owner.scope).admission_revision
    before = owner.store.read()
    with pytest.raises(LedgerError) as error:
        _reserve(owner, "legacy-contract", pin=pin)
    assert "admission_replay_contract_mismatch" in error.value.reason_codes
    assert owner.store.read() == before


def test_replay_cannot_change_the_coordinator_partition(owner):
    pin = owner.coord.admission_snapshot(owner.scope).admission_revision
    assert _reserve(owner, "partition-contract", pin=pin).granted
    partitioned = UsageCoordinator(
        owner.store, writer_id="test-writer", fence=1,
        partition=CapacityPartition(node_id="test-node", numerator=1, denominator=2),
    )
    before = owner.store.read()
    with pytest.raises(LedgerError) as error:
        partitioned.reserve(
            owner.scope, UsageVector.of(requests=1), request_id="partition-contract",
            idempotency_key="partition-contract", owner_id="test-owner",
            expected_admission_revision=pin,
        )
    assert "admission_replay_contract_mismatch" in error.value.reason_codes
    assert owner.store.read() == before


def test_wrapper_roundtrip_and_document_binding(owner):
    view = owner.coord.admission_snapshot(owner.scope)
    payload = view.to_dict()
    assert payload["schema"] == "ipfs_accelerate_py.endpoint_usage.admission_snapshot@1"
    assert payload["schema_version"] == 1
    assert UsageAdmissionSnapshot.from_dict(payload).to_dict() == payload
    assert view.snapshot == view.usage_snapshot
    assert view.ledger_state_id == view.document_cid
    assert view.ledger_fence == view.fence
    assert validate_admission_revision(view.admission_revision) == view.admission_revision
    assert admission_revision_for_document(owner.store.read(), owner.scope) == view.admission_revision
    reordered = dict(reversed(list(owner.store.read().items())))
    assert admission_revision_for_document(reordered, owner.scope) == view.admission_revision
    altered = owner.store.read()
    altered["caller_budgets"][owner.scope] = {"requests": 0}
    assert altered["revision"] == view.ledger_revision
    assert admission_revision_for_document(altered, owner.scope) != view.admission_revision


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong"), ("schema_version", True), ("ledger_revision", True),
    ("ledger_revision", -1), ("fence", -1), ("document_cid", "invalid"),
    ("admission_revision", "uadm_" + "0" * 64), ("unknown_field", 1),
])
def test_wrapper_rejects_malformed_or_forged_payload(owner, field, value):
    payload = copy.deepcopy(owner.coord.admission_snapshot(owner.scope).to_dict())
    payload[field] = value
    with pytest.raises(ValueError):
        UsageAdmissionSnapshot.from_dict(payload)


def test_wrapper_rejects_missing_fields_and_foreign_observation(owner):
    payload = owner.coord.admission_snapshot(owner.scope).to_dict()
    missing = dict(payload)
    missing.pop("ledger_revision")
    with pytest.raises(ValueError):
        UsageAdmissionSnapshot.from_dict(missing)
    foreign = owner.coord.snapshot(stable_id("scope", "foreign-scope"))
    payload["usage_snapshot"] = foreign.to_dict()
    with pytest.raises(ValueError):
        UsageAdmissionSnapshot.from_dict(payload)


@pytest.mark.parametrize("pin", ["", "usnap_" + "0" * 64, "uadm_short", 1, True])
def test_invalid_pin_is_rejected_before_store_access(owner, monkeypatch, pin):
    def forbidden():
        raise AssertionError("invalid admission pin accessed the ledger")

    monkeypatch.setattr(owner.store, "read", forbidden)
    with pytest.raises(ValueError):
        _reserve(owner, "invalid", pin=pin)


def test_dual_pins_are_rejected_before_store_access(owner, monkeypatch):
    pin = owner.coord.admission_snapshot(owner.scope)

    def forbidden():
        raise AssertionError("dual pins accessed the ledger")

    monkeypatch.setattr(owner.store, "read", forbidden)
    with pytest.raises(ValueError):
        _reserve(
            owner, "dual", pin=pin.admission_revision,
            expected_usage_revision=pin.snapshot.usage_revision,
        )
