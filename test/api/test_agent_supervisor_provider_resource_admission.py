"""Provider admission contracts exercised with local, bounded stub callbacks."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchAdmissionGrant,
    ProviderBatchCapacity,
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
    ProviderBatchStatus,
    ResourceSchedulerBatchAdmission,
)


def _request(name="member", payload="work", **overrides):
    values = dict(request_id=name, payload=payload, provider_id="stub", route="planning",
                  model="stub-model", operation="generate", token_budget=100,
                  context_limit=2048, timeout_ms=2000)
    values.update(overrides)
    return ProviderBatchRequest(**values)


def _config(**overrides):
    values = dict(batch_window_ms=10, admission_retry_ms=1, max_batch_size=4,
                  max_parallel_batches=1, require_sealed_inputs=False)
    values.update(overrides)
    return ProviderBatchSchedulerConfig(**values)


class _Resource:
    def __init__(self, decide=None):
        self.requirements = []
        self.releases = []
        self.released = threading.Event()
        self.decide = decide

    def acquire(self, requirement, **kwargs):
        self.requirements.append(requirement)
        if self.decide is not None:
            return self.decide(requirement)
        return SimpleNamespace(admitted=True, reason="", retryable=True), object()

    def release(self, lease, *, reason):
        self.releases.append((lease, reason))
        self.released.set()
        return True


def _grant(adapter, requests):
    return adapter(requests[0].batch_key, requests, ProviderBatchCapacity(provider_id="stub"))


@pytest.mark.parametrize("claims", [
    {"input_sealed": False}, {"trace_identity": "other"},
    {"expected_trace_identity": "other"}, {"tokenizer_identity": "other"},
    {"expected_tokenizer_identity": "other"}, {"checkpoint_key": "other"},
    {"exclusive_authorities": ["checkpoint"]}, {"network_allowlist": ["example.test"]},
    {"requires_network": True}, {"requires_gpu": True}, {"requires_prover": True},
    {"mutates_checkpoint": True}, {"mutates_tokenizer": True},
    {"stage": "training"}, {"resource_profile": "R0"},
    {"claimed_resource_kinds": ["gpu"]}, {"memory_bytes": 1},
    {"gpu_memory_bytes": 1}, {"disk_bytes": 1}, {"process_slots": 2},
    {"required_capabilities": ["cpu"]}, {"timeout_ms": 10},
    {"resource_class": "cpu-small"},
])
def test_resource_claim_changes_prevent_batching_and_singleflight(claims):
    ordinary = _request("first", provenance={"task": "one"})
    changed = _request("second", provenance={"task": "two", **claims})
    assert ordinary.batch_key != changed.batch_key
    assert ordinary.execution_fingerprint != changed.execution_fingerprint


def test_unrelated_provenance_keeps_singleflight_compatible():
    first = _request("one", provenance={"task_id": "a"})
    second = _request("two", provenance={"task_id": "b"})
    assert first.batch_key == second.batch_key
    assert first.execution_fingerprint == second.execution_fingerprint


@pytest.mark.parametrize("reverse", [False, True])
def test_adapter_rejects_mixed_claims_without_last_writer_override(reverse):
    resource = _Resource()
    requests = [_request("unsafe", provenance={"input_sealed": False}),
                _request("safe", provenance={"input_sealed": True})]
    if reverse:
        requests.reverse()
    grant = _grant(ResourceSchedulerBatchAdmission(resource), requests)
    assert not grant.admitted and not grant.retryable
    assert grant.reason == "incompatible_resource_claims"
    assert resource.requirements == []


@pytest.mark.parametrize("claims", [
    {"input_sealed": "false"}, {"requires_network": 1}, {"requires_gpu": None},
    {"requires_prover": "false"}, {"mutates_tokenizer": "no"},
    {"mutates_checkpoint": []}, {"memory_bytes": True}, {"memory_bytes": "4096"},
    {"gpu_memory_bytes": -1}, {"disk_bytes": 1.5}, {"process_slots": 0},
    {"timeout_ms": "10"}, {"exclusive_authorities": "checkpoint"},
    {"required_capabilities": [1]}, {"network_allowlist": "example.test"},
    {"claimed_resource_kinds": [""]}, {"trace_identity": 1}, {"stage": {}},
])
def test_invalid_claims_fail_once_without_provider_or_resource_acquire(claims):
    resource = _Resource()
    calls = []
    with ProviderBatchScheduler(
        lambda requests: calls.append(requests), config=_config(batch_window_ms=0),
        admission=ResourceSchedulerBatchAdmission(resource),
    ) as scheduler:
        result = scheduler.execute(_request(provenance=claims), wait_timeout=2)
        metrics = scheduler.metrics()
    assert result.status is ProviderBatchStatus.FAILED
    assert "invalid_resource_claims" in result.error
    assert metrics.admission_deferrals == 0
    assert calls == [] and resource.requirements == []


def test_aggregate_reservation_preserves_configured_minima_and_member_claims():
    resource = _Resource()
    claims = dict(input_sealed=True, trace_identity="trace:1", expected_trace_identity="trace:1",
                  tokenizer_identity="tok:1", expected_tokenizer_identity="tok:1",
                  checkpoint_key="read:1", memory_bytes=16, gpu_memory_bytes=256,
                  disk_bytes=32, process_slots=2, timeout_ms=1000,
                  required_capabilities=["llm:json"], requires_network=True,
                  network_allowlist=["example.test"], claimed_resource_kinds=["network"])
    adapter = ResourceSchedulerBatchAdmission(
        resource, memory_bytes=128, gpu_memory_bytes=64,
        required_capabilities=("cpu",), network_allowlist=("example.test", "other.test"),
    )
    grant = _grant(adapter, [_request("a", provenance=claims, token_budget=11),
                              _request("b", provenance=claims, token_budget=17)])
    assert grant.admitted
    requirement, = resource.requirements
    assert requirement.token_budget == 28 and requirement.quota_units == 1
    assert requirement.memory_bytes == 128 and requirement.gpu_memory_bytes == 256
    assert requirement.process_slots == 2 and requirement.disk_bytes == 32
    assert requirement.required_capabilities == ("cpu", "llm:json")
    assert requirement.trace_identity == requirement.expected_trace_identity == "trace:1"
    assert requirement.tokenizer_identity == requirement.expected_tokenizer_identity == "tok:1"
    assert requirement.checkpoint_key == "read:1" and requirement.timeout_ms == 1000
    assert requirement.network_allowlist == ("example.test",)
    assert set(requirement.claimed_resource_kinds) == {"provider", "token", "network"}
    grant.release_once()
    grant.release_once()
    assert len(resource.releases) == 1


@pytest.mark.parametrize("claims,reason", [
    ({"resource_class": "cpu-small"}, "resource_class_mismatch"),
    ({"network_allowlist": ["untrusted.test"]}, "network_allowlist_mismatch"),
])
def test_caller_cannot_replace_adapter_policy(claims, reason):
    resource = _Resource()
    adapter = ResourceSchedulerBatchAdmission(resource, network_allowlist=("trusted.test",))
    grant = _grant(adapter, [_request(provenance=claims)])
    assert not grant.admitted and not grant.retryable and grant.reason == reason
    assert not resource.requirements


def test_permanent_owner_refusal_returns_member_failure_without_fallback():
    resource = _Resource(lambda requirement: (
        SimpleNamespace(admitted=False, reason="stale_trace", retryable=False), None,
    ))
    calls = []
    with ProviderBatchScheduler(
        lambda requests: calls.append("dispatch"), fallback=lambda request: calls.append("fallback"),
        config=_config(batch_window_ms=0), admission=ResourceSchedulerBatchAdmission(resource),
    ) as scheduler:
        result = scheduler.execute(_request(), wait_timeout=2)
    assert result.status is ProviderBatchStatus.FAILED
    assert "stale_trace" in result.error and calls == []
    assert len(resource.requirements) == 1 and not resource.releases


def test_live_pressure_refusal_retries_and_releases_after_recovery():
    attempts = 0

    def decide(requirement):
        nonlocal attempts
        attempts += 1
        return (SimpleNamespace(admitted=attempts > 1, reason="memory_pressure" if attempts == 1 else "",
                                retryable=True), object() if attempts > 1 else None)

    resource = _Resource(decide)
    with ProviderBatchScheduler(
        lambda requests: [item.payload for item in requests], config=_config(batch_window_ms=0),
        admission=ResourceSchedulerBatchAdmission(resource),
    ) as scheduler:
        result = scheduler.execute(_request(), wait_timeout=2)
        assert resource.released.wait(2)
        assert scheduler.metrics().admission_deferrals >= 1
    assert result.successful and attempts == 2 and len(resource.releases) == 1


@pytest.mark.parametrize("claims", [
    {"exclusive_authorities": ["checkpoint"]}, {"mutates_tokenizer": True},
    {"mutates_checkpoint": True}, {"stage": "checkpoint"},
    {"stage": "vocab"}, {"stage": "ckpt"}, {"stage": "corpus"},
    {"stage": "split"}, {"stage": "promote"},
])
def test_exclusive_claims_dispatch_one_physical_member_at_a_time(claims):
    resource = _Resource()
    calls = []

    def dispatch(requests):
        calls.append(tuple(item.request_id for item in requests))
        return [item.payload for item in requests]

    with ProviderBatchScheduler(dispatch, config=_config(), admission=ResourceSchedulerBatchAdmission(resource)) as scheduler:
        results = scheduler.execute_many([_request("a", "one", provenance=claims),
                                          _request("b", "two", provenance=claims)], wait_timeout=2)
    assert all(item.successful for item in results)
    assert calls == [("a",), ("b",)]


@pytest.mark.parametrize("stage", ["vocab", "ckpt", "corpus", "split", "promote"])
def test_direct_adapter_refuses_multiple_exclusive_stage_members(stage):
    resource = _Resource()
    requests = [_request("a", "one", provenance={"stage": stage}),
                _request("b", "two", provenance={"stage": stage})]
    grant = _grant(ResourceSchedulerBatchAdmission(resource), requests)
    assert not grant.admitted and not grant.retryable
    assert grant.reason == "exclusive_batch_members"
    assert resource.requirements == []


def test_unsafe_late_subscriber_cannot_join_admitted_safe_execution():
    entered, finish = threading.Event(), threading.Event()
    calls = []

    def dispatch(requests):
        calls.append(tuple(item.request_id for item in requests))
        entered.set()
        assert finish.wait(2)
        return ["ok" for _ in requests]

    def decide(requirement):
        return (SimpleNamespace(admitted=requirement.input_sealed,
                                reason="" if requirement.input_sealed else "unsealed_data", retryable=False),
                object() if requirement.input_sealed else None)

    resource = _Resource(decide)
    scheduler = ProviderBatchScheduler(dispatch, config=_config(batch_window_ms=0),
                                       admission=ResourceSchedulerBatchAdmission(resource))
    try:
        safe = scheduler.submit(_request("safe", provenance={"input_sealed": True}))
        assert entered.wait(2)
        unsafe = scheduler.submit(_request("unsafe", provenance={"input_sealed": False}))
        finish.set()
        assert safe.result(timeout=2).successful
        assert unsafe.result(timeout=2).status is ProviderBatchStatus.FAILED
        assert calls == [("safe",)] and scheduler.metrics().singleflight_hits == 0
    finally:
        finish.set()
        scheduler.shutdown(cancel_pending=True)


@pytest.mark.parametrize("expire", [False, True])
def test_cancellation_or_expiry_during_admission_prevents_dispatch_and_releases(expire):
    clock, cancelled, calls = [0], threading.Event(), []

    def decide(requirement):
        if expire:
            clock[0] = 11
        else:
            cancelled.set()
        return SimpleNamespace(admitted=True, reason="", retryable=True), object()

    resource = _Resource(decide)
    with ProviderBatchScheduler(
        lambda requests: calls.append(requests), config=_config(batch_window_ms=0),
        admission=ResourceSchedulerBatchAdmission(resource), clock_ms=lambda: clock[0],
    ) as scheduler:
        result = scheduler.execute(_request(cancellation_token=cancelled, timeout_ms=10), wait_timeout=2)
        assert resource.released.wait(2)
    assert result.status is (ProviderBatchStatus.TIMED_OUT if expire else ProviderBatchStatus.CANCELLED)
    assert calls == [] and len(resource.releases) == 1


def test_inflight_cancellation_holds_grant_until_callback_returns():
    entered, finish, cancelled = threading.Event(), threading.Event(), threading.Event()
    resource = _Resource()

    def dispatch(requests):
        entered.set()
        assert finish.wait(2)
        return ["ok" for _ in requests]

    scheduler = ProviderBatchScheduler(dispatch, config=_config(batch_window_ms=0),
                                       admission=ResourceSchedulerBatchAdmission(resource))
    try:
        future = scheduler.submit(_request(cancellation_token=cancelled))
        assert entered.wait(2)
        cancelled.set()
        assert future.result(timeout=2).status is ProviderBatchStatus.CANCELLED
        assert not resource.releases
        finish.set()
        assert resource.released.wait(2) and len(resource.releases) == 1
    finally:
        finish.set()
        scheduler.shutdown(cancel_pending=True)


def test_provider_exception_releases_resource_grant_once():
    resource = _Resource()

    def dispatch(requests):
        raise RuntimeError("authored failure")

    with ProviderBatchScheduler(dispatch, config=_config(batch_window_ms=0),
                               admission=ResourceSchedulerBatchAdmission(resource)) as scheduler:
        result = scheduler.execute(_request(), wait_timeout=2)
        assert resource.released.wait(2)
    assert result.status is ProviderBatchStatus.FAILED and len(resource.releases) == 1


def test_queued_claims_are_detached_from_caller_mutation():
    entered, finish = threading.Event(), threading.Event()
    resource = _Resource()

    def dispatch(requests):
        if requests[0].request_id == "blocker":
            entered.set()
            assert finish.wait(2)
        return ["ok" for _ in requests]

    scheduler = ProviderBatchScheduler(dispatch, config=_config(batch_window_ms=0),
                                       admission=ResourceSchedulerBatchAdmission(resource))
    try:
        blocker = scheduler.submit(_request("blocker", "first"))
        assert entered.wait(2)
        request = _request("queued", "second", provenance={"input_sealed": True, "required_capabilities": ["cpu"]})
        queued = scheduler.submit(request)
        request.provenance["input_sealed"] = False
        request.provenance["required_capabilities"].append("untrusted")
        finish.set()
        assert blocker.result(timeout=2).successful and queued.result(timeout=2).successful
        assert resource.requirements[1].input_sealed
        assert resource.requirements[1].required_capabilities == ("cpu",)
    finally:
        finish.set()
        scheduler.shutdown(cancel_pending=True)


def test_decision_conversion_preserves_permanent_refusal():
    decision = SimpleNamespace(admitted=False, reason="stale_trace", retryable=False)
    assert not ProviderBatchAdmissionGrant.from_value(decision).retryable
    assert not ProviderBatchAdmissionGrant.from_value((decision, None)).retryable
