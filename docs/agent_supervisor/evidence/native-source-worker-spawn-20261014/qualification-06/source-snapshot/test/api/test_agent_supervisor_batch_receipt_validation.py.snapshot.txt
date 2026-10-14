"""Physical references require native member issuance and actual ledger settlement."""

import copy
import threading
from concurrent.futures import CancelledError
from dataclasses import replace
from types import SimpleNamespace

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
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitEnforcement,
    LimitWindow,
    Quantity,
    UsageDimension,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore


SCOPE = stable_id("scope", "native-physical-receipts")


def _request(request_id="member", payload="immutable input", **overrides):
    values = dict(
        request_id=request_id, payload=payload, provider_id="authored-provider",
        route="authored-route", model="authored-model", operation="text.generate",
        context_limit=32, token_budget=10, timeout_ms=3_000, usage_scope_id=SCOPE,
        provenance={"consumer": "authored"},
    )
    values.update(overrides)
    return ProviderBatchRequest(**values)


@pytest.fixture
def owner_factory():
    resources = []

    def create(*, dispatch=None, receipt_history=256, usage_extractor=None,
               require_hard_limits=False, host_admission=None):
        clock = FakeClock()
        store = InMemoryUsageLedgerStore(clock=clock, writer_id="receipt-writer", fence=3)
        coordinator = UsageCoordinator(store, writer_id="receipt-writer", fence=3)
        coordinator.configure_limits(SCOPE, [UsageLimit(
            scope_id=SCOPE, dimension=dimension, ceiling=Quantity.finite(10_000),
            window=LimitWindow(kind=WindowKind.CONCURRENT if dimension is UsageDimension.CONCURRENT_REQUESTS
                               else WindowKind.LIFETIME),
        ) for dimension in (
            UsageDimension.REQUESTS, UsageDimension.CONCURRENT_REQUESTS, UsageDimension.BATCH_ITEMS,
            UsageDimension.INPUT_TOKENS, UsageDimension.OUTPUT_TOKENS, UsageDimension.TOTAL_TOKENS,
        )])
        admission = PhysicalBatchUsageAdmission(
            coordinator, scope_supplier=lambda provider: SCOPE if provider == "authored-provider" else "",
            max_batch_tokens=10_000, reservation_ttl_ms=20_000, usage_extractor=usage_extractor,
            require_hard_limits=require_hard_limits, host_admission=host_admission,
        )
        scheduler = ProviderBatchScheduler(
            dispatch or (lambda requests: ["authored:" + request.payload for request in requests]),
            config=ProviderBatchSchedulerConfig(
                batch_window_ms=1, admission_retry_ms=1, max_parallel_batches=1,
                receipt_history=receipt_history,
            ),
            admission=admission,
        )
        owner = SimpleNamespace(
            clock=clock, store=store, coordinator=coordinator, admission=admission, scheduler=scheduler,
        )
        resources.append(owner)
        return owner

    yield create
    for owner in reversed(resources):
        owner.scheduler.shutdown(wait=True, cancel_pending=True)
        owner.store.close()


@pytest.fixture
def owner(owner_factory):
    return owner_factory()


def _execute(owner, request=None):
    request = request or _request()
    result = owner.scheduler.execute(request, wait_timeout=3)
    assert owner.scheduler.flush(timeout=3)
    return request, result


def test_native_receipt_and_physical_ledger_validation_are_read_only(owner, monkeypatch):
    request, result = _execute(owner)
    before = owner.store.read()
    reads = []
    original = owner.store.read

    def counted():
        reads.append(True)
        return original()

    monkeypatch.setattr(owner.store, "read", counted)
    assert owner.scheduler.admission is owner.admission
    native = owner.scheduler.validate_result_receipt(request, result)
    assert reads == []
    assert native is owner.scheduler.evidence_receipts()[0]
    assert native.members[0].execution_fingerprint == request.execution_fingerprint
    for _ in range(2):
        validated = owner.admission.validate_receipt(result.physical_usage, request=request)
        assert validated == result.physical_usage and validated is not result.physical_usage
        validated["reserved_request_digests"].clear()
    assert len(reads) == 2 and original() == before
    assert len(before["reservations"]) == 1
    with pytest.raises(AttributeError):
        owner.scheduler.admission = None


@pytest.mark.parametrize("field,value", [
    ("output", "substituted output"), ("error", "substituted error"),
    ("request_id", "foreign"), ("provider_id", "foreign-provider"),
    ("batch_id", "foreign-batch"), ("execution_id", "foreign-execution"),
    ("receipt_id", "sha256:" + "0" * 64), ("status", ProviderBatchStatus.FAILED),
    ("token_budget", 11), ("timeout_ms", 4_000), ("singleflight_shared", True),
    ("provenance", {"consumer": "forged"}), ("physical_usage", {}),
])
def test_native_result_substitution_is_rejected(owner, field, value):
    request, result = _execute(owner)
    with pytest.raises(ValueError):
        owner.scheduler.validate_result_receipt(request, replace(result, **{field: value}))


@pytest.mark.parametrize("field,value", [
    ("payload", "different input with the same request ID"), ("request_id", "foreign-member"),
    ("model", "other-model"), ("route", "other-route"), ("token_budget", 11),
    ("provenance", {"consumer": "forged"}), ("payload", {"mutable": "input"}),
])
def test_request_identity_and_immutable_input_are_bound(owner, field, value):
    request, result = _execute(owner)
    with pytest.raises(ValueError):
        owner.scheduler.validate_result_receipt(replace(request, **{field: value}), result)


def test_foreign_scheduler_and_serialized_evidence_cannot_establish_issuance(owner_factory):
    first, second = owner_factory(), owner_factory()
    request, result = _execute(first)
    with pytest.raises(ValueError):
        second.scheduler.validate_result_receipt(request, result)
    with pytest.raises(ValueError):
        first.scheduler.validate_result_receipt(request, result.to_dict())
    forged = replace(first.scheduler.evidence_receipts()[0], _producer_seal=None)
    first.scheduler._receipts.clear()
    first.scheduler._receipts.append(forged)
    with pytest.raises(ValueError):
        first.scheduler.validate_result_receipt(request, result)


def test_evicted_native_evidence_requires_reconciliation(owner_factory):
    owner = owner_factory(receipt_history=1)
    first_request, first_result = _execute(owner)
    second_request, second_result = _execute(owner, _request("next", "other input"))
    with pytest.raises(ValueError):
        owner.scheduler.validate_result_receipt(first_request, first_result)
    assert owner.scheduler.validate_result_receipt(second_request, second_result)


def test_inline_future_callback_can_validate_native_and_physical_evidence(owner_factory):
    started, release, observed = threading.Event(), threading.Event(), threading.Event()

    def dispatch(requests):
        started.set()
        assert release.wait(3)
        return ["output" for _ in requests]

    owner = owner_factory(dispatch=dispatch)
    request = _request()
    failures = []
    future = owner.scheduler.submit(request)
    try:
        assert started.wait(3)

        def validate(done):
            try:
                result = done.result()
                owner.scheduler.validate_result_receipt(request, result)
                owner.admission.validate_receipt(result.physical_usage, request=request)
            except Exception as exc:
                failures.append(exc)
            finally:
                observed.set()

        future.add_done_callback(validate)
        release.set()
        assert future.result(timeout=3).successful
        assert observed.wait(3) and failures == []
    finally:
        release.set()


def test_late_singleflight_subscriber_references_canceled_representative_charge(owner_factory):
    started, release = threading.Event(), threading.Event()

    def dispatch(requests):
        started.set()
        assert release.wait(3)
        return ["output" for _ in requests]

    owner = owner_factory(dispatch=dispatch)
    first_request, sibling_request = _request("representative"), _request("sibling")
    first = owner.scheduler.submit(first_request)
    try:
        assert started.wait(3)
        sibling = owner.scheduler.submit(sibling_request)
        assert owner.scheduler.cancel(first_request.request_id)
        release.set()
        with pytest.raises(CancelledError):
            first.result(timeout=3)
        result = sibling.result(timeout=3)
        assert result.singleflight_shared and result.successful
        native = owner.scheduler.validate_result_receipt(sibling_request, result)
        physical = owner.admission.validate_receipt(result.physical_usage, request=sibling_request)
        assert len(native.members) == 2
        assert physical["dispatched_member_count"] == 1
        assert physical["dispatched_execution_fingerprints"] == [sibling_request.execution_fingerprint]
        assert len(owner.store.read()["reservations"]) == 1
    finally:
        release.set()


@pytest.mark.parametrize("field,value", [
    ("scope_digest", "0" * 64), ("batch_key_digest", "0" * 64),
    ("physical_call_id", "physical:foreign"), ("event_id", "event:foreign"),
    ("reservation_id", "reservation:foreign"), ("state", "held"),
    ("settlement_basis", "unconfirmed"), ("reserved_member_count", True),
    ("dispatched_execution_fingerprints", ["0" * 64]), ("shared_overhead_tokens", 2),
    ("settled", UsageVector.of(requests=99).to_dict()), ("unexpected", "forged"),
])
def test_physical_receipt_drift_is_rejected_against_actual_owner(owner, field, value):
    request, result = _execute(owner)
    receipt = copy.deepcopy(result.physical_usage)
    receipt[field] = value
    with pytest.raises(PhysicalBatchUsageError):
        owner.admission.validate_receipt(receipt, request=request)


def test_physical_membership_rejects_changed_payload_and_scope(owner):
    request, result = _execute(owner)
    for changed in (
        replace(request, payload="unrelated input"),
        replace(request, usage_scope_id=stable_id("scope", "foreign")),
    ):
        with pytest.raises(PhysicalBatchUsageError):
            owner.admission.validate_receipt(result.physical_usage, request=changed)


@pytest.mark.parametrize("mutation", ["refund", "correction", "reset", "fence", "missing_record"])
def test_authoritative_settlement_changes_invalidate_old_projection(owner, mutation):
    request, result = _execute(owner)
    physical = result.physical_usage
    reservation_id = physical["reservation_id"]
    if mutation == "refund":
        owner.coordinator.refund(SCOPE, UsageVector.of(input_tokens=1), reservation_id=reservation_id)
    elif mutation == "correction":
        owner.coordinator.correct(
            SCOPE, supersedes_event_id=physical["event_id"],
            units=UsageVector.from_dict(physical["settled"]), reservation_id=reservation_id,
        )
    elif mutation == "reset":
        owner.coordinator.reset(SCOPE)
    elif mutation == "fence":
        owner.store.bump_fence(writer_id="next-writer")
    else:
        document = owner.store.read()
        document["reservations"].pop(reservation_id)
        owner.store.compare_and_set(document["revision"], document, writer_id="receipt-writer", fence=3)
    assert owner.scheduler.validate_result_receipt(request, result)
    with pytest.raises(PhysicalBatchUsageError):
        owner.admission.validate_receipt(physical, request=request)


def test_compacted_terminal_record_preserves_reference_validation(owner):
    request, result = _execute(owner)
    owner.coordinator.compact()
    assert owner.store.read()["events"] == []
    assert owner.admission.validate_receipt(result.physical_usage, request=request)["state"] == "committed"


def test_unavailable_ledger_never_creates_a_validation_receipt(owner, monkeypatch):
    request, result = _execute(owner)

    def unavailable():
        raise OSError("authored ledger read failure")

    monkeypatch.setattr(owner.store, "read", unavailable)
    with pytest.raises(PhysicalBatchUsageError):
        owner.admission.validate_receipt(result.physical_usage, request=request)


def test_failed_callback_charge_is_a_reference_without_success(owner_factory):
    def fail(_requests):
        raise RuntimeError("authored callback failure")

    owner = owner_factory(dispatch=fail)
    request, result = _execute(owner)
    assert result.status is ProviderBatchStatus.FAILED and result.output is None
    assert owner.scheduler.validate_result_receipt(request, result)
    physical = owner.admission.validate_receipt(result.physical_usage, request=request)
    assert physical["settlement_basis"] == "conservative_live_ceiling"


def test_provider_reported_units_can_be_lower_than_declared_ceiling(owner_factory):
    owner = owner_factory(usage_extractor=lambda _requests, _raw: UsageVector.of(
        input_tokens=2, output_tokens=1, total_tokens=3,
    ))
    request, result = _execute(owner)
    physical = owner.admission.validate_receipt(result.physical_usage, request=request)
    assert physical["settlement_basis"] == "provider_reported"
    assert int(UsageVector.from_dict(physical["settled"]).get(UsageDimension.TOTAL_TOKENS).amount) == 3


@pytest.mark.parametrize("invalid", [0, 1, "true", None])
def test_required_hard_limits_option_is_strict_boolean(owner, invalid):
    with pytest.raises(TypeError, match="require_hard_limits"):
        PhysicalBatchUsageAdmission(
            owner.coordinator, scope_supplier=lambda _provider: SCOPE,
            max_batch_tokens=10_000, reservation_ttl_ms=20_000,
            require_hard_limits=invalid,
        )


@pytest.mark.parametrize("policy", [
    "missing_output", "requests_only", "budget_only", "soft_output",
    "diagnostic_output", "unlimited_output", "unknown_output",
])
def test_required_hard_limits_refuse_weak_configuration_and_release_host(owner_factory, policy):
    released, calls = [], []

    def host(*_args):
        return ProviderBatchAdmissionGrant(True, release=lambda: released.append(True))

    owner = owner_factory(
        require_hard_limits=True, host_admission=host,
        dispatch=lambda requests: calls.append(tuple(requests)) or ["output"] * len(requests),
    )
    limits = list(owner.coordinator.admission_snapshot(SCOPE).snapshot.limits)
    if policy == "budget_only":
        limits = []
        owner.coordinator.configure_caller_budget(SCOPE, UsageVector.of(
            requests=100, concurrent_requests=100, batch_items=100,
            input_tokens=10_000, output_tokens=10_000, total_tokens=20_000,
        ))
    elif policy == "requests_only":
        limits = [limit for limit in limits if limit.dimension is UsageDimension.REQUESTS]
    elif policy == "missing_output":
        limits = [limit for limit in limits if limit.dimension is not UsageDimension.OUTPUT_TOKENS]
    else:
        options = {
            "soft_output": {"enforcement": LimitEnforcement.SOFT},
            "diagnostic_output": {"enforcement": LimitEnforcement.DIAGNOSTIC},
            "unlimited_output": {"ceiling": Quantity.unlimited()},
            "unknown_output": {"ceiling": Quantity.unknown()},
        }[policy]
        limits = [replace(limit, limit_id=None, **options) if limit.dimension is UsageDimension.OUTPUT_TOKENS
                  else limit for limit in limits]
    owner.coordinator.configure_limits(SCOPE, limits)
    request = _request()
    before = owner.store.read()
    grant = owner.admission(request.batch_key, (request,), ProviderBatchCapacity(request.provider_id))
    assert not grant.admitted and not grant.retryable
    assert grant.reason == "usage_hard_limits_required"
    assert released == [True] and calls == [] and owner.store.read() == before


@pytest.mark.parametrize("drift", ["scope", "fence", "unfenced"])
def test_required_hard_limits_reject_invalid_paired_owner_before_reserve(owner_factory, monkeypatch, drift):
    releases = []
    owner = owner_factory(
        require_hard_limits=True,
        host_admission=lambda *_args: ProviderBatchAdmissionGrant(
            True, release=lambda: releases.append(True),
        ),
    )
    if drift == "scope":
        foreign = owner.coordinator.admission_snapshot(stable_id("scope", "foreign"))
        monkeypatch.setattr(owner.coordinator, "admission_snapshot", lambda _scope: foreign)
    elif drift == "fence":
        owner.store.bump_fence(writer_id="next-writer")
    else:
        monkeypatch.setattr(owner.coordinator, "_fence", None)
    request = _request()
    before = owner.store.read()
    grant = owner.admission(request.batch_key, (request,), ProviderBatchCapacity(request.provider_id))
    assert not grant.admitted and not grant.retryable
    assert grant.reason == "usage_admission_owner_mismatch"
    assert releases == [True] and owner.store.read() == before


def test_required_hard_limits_pin_prevents_dispatch_after_limit_removal(owner_factory, monkeypatch):
    calls, releases, attempts = [], [], []
    owner = owner_factory(
        require_hard_limits=True,
        dispatch=lambda requests: calls.append(True) or ["output"] * len(requests),
        host_admission=lambda *_args: ProviderBatchAdmissionGrant(
            True, release=lambda: releases.append(True),
        ),
    )
    original = owner.coordinator.reserve

    def mutate_before_reserve(*args, **kwargs):
        attempts.append(kwargs)
        assert kwargs.get("expected_admission_revision")
        limits = owner.coordinator.admission_snapshot(SCOPE).snapshot.limits
        owner.coordinator.configure_limits(SCOPE, [
            limit for limit in limits if limit.dimension is not UsageDimension.OUTPUT_TOKENS
        ])
        return original(*args, **kwargs)

    monkeypatch.setattr(owner.coordinator, "reserve", mutate_before_reserve)
    _request_used, result = _execute(owner)
    assert result.status is ProviderBatchStatus.FAILED and calls == []
    assert "usage_hard_limits_required" in result.error
    assert len(attempts) == 1 and len(releases) == 2
    assert owner.store.read()["reservations"] == {}


def test_required_hard_limits_retry_fresh_pin_before_one_actual_dispatch(owner_factory, monkeypatch):
    calls, attempts = [], []
    owner = owner_factory(
        require_hard_limits=True,
        dispatch=lambda requests: calls.append(True) or ["output"] * len(requests),
    )
    original = owner.coordinator.reserve

    def mutate_once(*args, **kwargs):
        attempts.append(kwargs["expected_admission_revision"])
        if len(attempts) == 1:
            owner.coordinator.configure_caller_budget(SCOPE, UsageVector.of(requests=10_000))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner.coordinator, "reserve", mutate_once)
    request, result = _execute(owner)
    assert result.successful and calls == [True]
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    assert len(owner.store.read()["reservations"]) == 1
    assert owner.admission.validate_receipt(result.physical_usage, request=request)


def test_legacy_admission_does_not_read_paired_view_or_pass_pin(owner, monkeypatch):
    original = owner.coordinator.reserve
    seen = []

    def forbidden(_scope):
        raise AssertionError("legacy adapter must not request a paired view")

    def capture(*args, **kwargs):
        seen.append(kwargs)
        assert "expected_admission_revision" not in kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(owner.coordinator, "admission_snapshot", forbidden)
    monkeypatch.setattr(owner.coordinator, "reserve", capture)
    _request_used, result = _execute(owner)
    assert result.successful and len(seen) == 1
