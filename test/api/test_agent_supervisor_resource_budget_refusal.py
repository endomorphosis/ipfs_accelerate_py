"""Finite lease ceilings distinguish immutable excess from live contention."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProofResourceClass,
    ProviderCapacity,
    ResourceLeaseBudget,
    ResourcePolicy,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


@pytest.fixture(autouse=True)
def _local_receipts_only():
    with public_replay_without_metadata():
        yield


def _host(**overrides):
    values = dict(
        observed_at_ms=1000, cpu_percent=10, memory_percent=10, disk_percent=10,
        memory_available_bytes=1_000_000, disk_available_bytes=1_000_000,
        gpu_memory_total_bytes=1_000_000, gpu_memory_available_bytes=1_000_000,
        worker_limit=8, available_worker_capacity=8, capabilities=("cpu", "prover"),
        resource_classes=tuple(item.value for item in ProofResourceClass),
    )
    values.update(overrides)
    return HostResourceSnapshot(**values)


def _providers():
    return [ProviderCapacity(
        provider_id="stub", healthy=True, latency_ms=5, max_concurrency=8,
        active_requests=0, quota_remaining=1000, token_budget_remaining=1_000_000,
        context_window_tokens=1_000_000,
    )]


def _budget(**overrides):
    values = dict(max_parallel=8, max_processes=8, max_cpu_proof_concurrency=8,
                  max_model_concurrency=8, max_artifact_concurrency=8)
    values.update(overrides)
    return ResourceLeaseBudget(**values)


def _scheduler():
    return ResourceScheduler(ResourcePolicy(
        max_lanes=8, max_cpu_proof_concurrency=8, max_model_concurrency=8,
        max_artifact_concurrency=8, max_gpu_concurrency=8, max_prover_concurrency=8,
    ))


def _model(lane_id="candidate", **overrides):
    values = dict(lane_id=lane_id, stage="inference",
                  resource_class=ProofResourceClass.MODEL_DRAFT.value,
                  requires_provider=True, provider_id="stub", token_budget=50,
                  quota_units=1, context_tokens=100, memory_bytes=100, disk_bytes=100)
    values.update(overrides)
    return LaneResourceRequirements(**values)


def _acquire(scheduler, requirement, budget):
    decision, lease = scheduler.acquire(requirement, budget=budget, host=_host(), providers=_providers())
    assert decision.admitted, decision.reasons
    assert lease is not None
    return lease


def _assert_empty_reservations(decision):
    assert not decision.admitted and decision.effective_slots == 0
    reservations = {key: value for key, value in decision.to_dict().items() if key.startswith("reserved_")}
    assert reservations and not any(reservations.values())


@pytest.mark.parametrize("budget_values,claim_values,reason", [
    ({"context_tokens": 128}, {"context_tokens": 129}, "lease_context_budget"),
    ({"memory_bytes": 128}, {"memory_bytes": 129}, "lease_memory_budget"),
    ({"gpu_memory_bytes": 128}, {"gpu_memory_bytes": 129}, "lease_gpu_memory_budget"),
    ({"disk_bytes": 128}, {"disk_bytes": 129}, "lease_disk_budget"),
    ({"model_token_limit": 100}, {"token_budget": 101}, "lease_token_budget"),
    ({"provider_quota": 1}, {"quota_units": 2}, "lease_provider_quota"),
    ({"max_processes": 1}, {"process_slots": 2}, "lease_process_capacity"),
    ({"max_parallel": 1}, {"process_slots": 2}, "lease_process_capacity"),
    ({"max_model_concurrency": 1}, {"process_slots": 2}, "lease_model_concurrency"),
    ({"max_artifact_concurrency": 1}, {"process_slots": 2, "resource_class": "io-artifact"},
     "lease_artifact_concurrency"),
    ({"max_cpu_proof_concurrency": 1}, {"process_slots": 2, "resource_class": "cpu-proof-solver"},
     "lease_cpu_proof_concurrency"),
])
@pytest.mark.parametrize("operation", ["acquire", "transition"])
def test_request_exceeding_immutable_budget_is_terminal(budget_values, claim_values, reason, operation):
    scheduler, budget = _scheduler(), _budget(**budget_values)
    original = _acquire(scheduler, _model(), budget) if operation == "transition" else None
    before = tuple(lease.to_dict() for lease in scheduler.active_leases)
    try:
        candidate = _model(**claim_values)
        if original is None:
            decision, rejected = scheduler.acquire(candidate, budget=budget, host=_host(), providers=_providers())
        else:
            decision, rejected = scheduler.transition(original, candidate, host=_host(), providers=_providers())
        assert rejected is None and decision.retryable is False
        assert reason in decision.reasons
        _assert_empty_reservations(decision)
        assert tuple(lease.to_dict() for lease in scheduler.active_leases) == before
    finally:
        if original is not None:
            assert scheduler.release(original)
    assert scheduler.active_leases == ()


@pytest.mark.parametrize("budget_values,claim_values,reason", [
    ({"memory_bytes": 150}, {}, "lease_memory_budget"),
    ({"gpu_memory_bytes": 150}, {"gpu_memory_bytes": 100}, "lease_gpu_memory_budget"),
    ({"disk_bytes": 150}, {}, "lease_disk_budget"),
    ({"model_token_limit": 75}, {}, "lease_token_budget"),
    ({"provider_quota": 1}, {}, "lease_provider_quota"),
    ({"max_processes": 1}, {}, "lease_process_capacity"),
    ({"max_parallel": 1}, {}, "lease_process_capacity"),
    ({"max_model_concurrency": 1}, {}, "lease_model_concurrency"),
    ({"max_artifact_concurrency": 1}, {"resource_class": "io-artifact"}, "lease_artifact_concurrency"),
    ({"max_cpu_proof_concurrency": 1}, {"resource_class": "cpu-proof-solver"}, "lease_cpu_proof_concurrency"),
])
def test_budget_contention_retries_after_active_lease_release(budget_values, claim_values, reason):
    scheduler, budget = _scheduler(), _budget(**budget_values)
    first = _acquire(scheduler, _model("active", **claim_values), budget)
    try:
        candidate = _model(**claim_values)
        decision, rejected = scheduler.acquire(candidate, budget=budget, host=_host(), providers=_providers())
        assert rejected is None and decision.retryable is True
        assert reason in decision.reasons
        _assert_empty_reservations(decision)
        assert scheduler.active_leases == (first,)
    finally:
        assert scheduler.release(first)
    recovered = _acquire(scheduler, candidate, budget)
    assert scheduler.release(recovered) and scheduler.active_leases == ()


def test_context_ceiling_applies_per_request_without_summing_active_contexts():
    scheduler, budget = _scheduler(), _budget(context_tokens=128)
    first = _acquire(scheduler, _model("first"), budget)
    second = None
    try:
        second = _acquire(scheduler, _model("second"), budget)
        assert first.child_limits.context_tokens == second.child_limits.context_tokens == 100
    finally:
        if second is not None:
            assert scheduler.release(second)
        assert scheduler.release(first)


@pytest.mark.parametrize("claims", [
    {"requires_gpu": True}, {"claimed_resource_kinds": ("gpu",)}, {"stage": "training"},
])
def test_fully_occupied_gpu_is_live_pressure_even_when_total_telemetry_is_present(claims):
    scheduler = _scheduler()
    requirement = _model(**claims)
    assert requirement.gpu_memory_bytes == 0
    decision, lease = scheduler.acquire(
        requirement, host=_host(gpu_memory_available_bytes=0), providers=_providers(),
    )
    assert lease is None and decision.retryable is True
    assert "host_gpu_memory_headroom" in decision.reasons
    assert "gpu_telemetry_missing" not in decision.reasons
    _assert_empty_reservations(decision)
    recovered = _acquire(scheduler, requirement, _budget())
    assert scheduler.release(recovered)
