"""Resource-owner admission invariants; no provider or child process execution."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProofResourceClass,
    ProviderCapacity,
    ResourceLeaseBudget,
    ResourcePolicy,
    ResourceScheduler,
    campaign_resource_profile_spec,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
    public_replay_without_metadata,
)


@pytest.fixture(autouse=True)
def _local_receipts_only():
    with public_replay_without_metadata():
        yield


def _host():
    return HostResourceSnapshot(
        observed_at_ms=1000, cpu_percent=10, memory_percent=10, disk_percent=10,
        memory_available_bytes=512 * 1024**3, disk_available_bytes=1024 * 1024**3,
        worker_limit=8, available_worker_capacity=8,
        capabilities=("cpu", "network", "https", "prover"),
        resource_classes=tuple(item.value for item in ProofResourceClass),
    )


def _provider(**overrides):
    values = dict(provider_id="stub", healthy=True, latency_ms=5,
                  max_concurrency=8, active_requests=0, quota_remaining=1000,
                  token_budget_remaining=1_000_000, context_window_tokens=1_000_000)
    values.update(overrides)
    return ProviderCapacity(**values)


def _budget(**overrides):
    values = dict(max_parallel=8, max_processes=8, max_cpu_proof_concurrency=8,
                  max_model_concurrency=8, max_artifact_concurrency=8)
    values.update(overrides)
    return ResourceLeaseBudget(**values)


def _scheduler(**overrides):
    values = dict(max_lanes=8, max_cpu_proof_concurrency=8,
                  max_model_concurrency=8, max_artifact_concurrency=8)
    values.update(overrides)
    return ResourceScheduler(ResourcePolicy(**values))


def _model(lane_id, **overrides):
    values = dict(lane_id=lane_id, resource_class=ProofResourceClass.MODEL_DRAFT.value,
                  requires_provider=True, provider_id="stub", token_budget=50,
                  quota_units=1, context_tokens=100, memory_bytes=100, disk_bytes=100)
    values.update(overrides)
    return LaneResourceRequirements(**values)


def _assert_no_reservations(decision):
    assert not decision.admitted
    assert decision.effective_slots == 0
    reserved = {name: value for name, value in decision.to_dict().items()
                if name.startswith("reserved_")}
    assert reserved
    assert all(value == 0 for value in reserved.values()), reserved


def _acquire(scheduler, requirement, budget):
    decision, lease = scheduler.acquire(requirement, budget=budget,
                                        host=_host(), providers=[_provider()])
    assert decision.admitted, decision.reasons
    assert lease is not None
    return lease


@pytest.mark.parametrize("trusted,requested,reason", [
    ((), ("https://caller.invalid",), "network_denied"),
    (("https://trusted.invalid",), ("https://caller.invalid",), "network_not_allowlisted"),
    (("https://trusted.invalid",), (), "network_denied"),
])
def test_request_cannot_authorize_its_own_network_allowlist(trusted, requested, reason):
    scheduler = _scheduler(network_allowlist=trusted)
    requirement = LaneResourceRequirements(
        lane_id="network", resource_class="cpu-validation", requires_network=True,
        network_allowlist=requested,
    )
    decision, lease = scheduler.acquire(requirement, host=_host())
    assert lease is None
    assert reason in decision.reasons
    assert decision.retryable is False
    _assert_no_reservations(decision)
    assert scheduler.active_leases == ()


def test_trusted_network_policy_admits_only_requested_subset():
    scheduler = _scheduler(network_allowlist=("https://trusted.invalid", "https://other.invalid"))
    requirement = LaneResourceRequirements(
        lane_id="network", resource_class="cpu-validation", requires_network=True,
        network_allowlist=("https://trusted.invalid",),
    )
    lease = _acquire(scheduler, requirement, _budget())
    assert lease.requirement.network_allowlist == ("https://trusted.invalid",)
    assert scheduler.release(lease)


def test_reusing_lane_identity_cannot_bypass_an_exclusive_claim():
    scheduler = _scheduler()
    requirement = LaneResourceRequirements(
        lane_id="exclusive", resource_class="cpu-validation",
        exclusive_authorities=("checkpoint",),
    )
    first = _acquire(scheduler, requirement, _budget())
    before = first.to_dict()
    decision, second = scheduler.acquire(requirement, host=_host())
    assert second is None
    assert {"lane_already_active", "exclusive_authority_collision"} <= set(decision.reasons)
    _assert_no_reservations(decision)
    assert scheduler.active_leases == (first,)
    assert first.to_dict() == before
    assert scheduler.release(first)
    recovered = _acquire(scheduler, requirement, _budget())
    assert recovered.lease_id != first.lease_id
    assert scheduler.release(recovered)


@pytest.mark.parametrize("claims,reason", [
    ({"trace_identity": "old", "expected_trace_identity": "current"}, "stale_trace"),
    ({"tokenizer_identity": "old", "expected_tokenizer_identity": "current"},
     "incompatible_tokenizer_mutation"),
])
@pytest.mark.parametrize("operation", ["acquire", "transition"])
def test_immutable_stale_claims_are_terminal_and_preserve_active_leases(claims, reason, operation):
    scheduler = _scheduler()
    budget = _budget()
    original = None
    if operation == "transition":
        original = _acquire(scheduler, _model("candidate"), budget)
    before = tuple(lease.to_dict() for lease in scheduler.active_leases)
    candidate = _model("candidate", **claims)
    if original is None:
        decision, rejected = scheduler.acquire(candidate, budget=budget, host=_host(), providers=[_provider()])
    else:
        decision, rejected = scheduler.transition(original, candidate, host=_host(), providers=[_provider()])
    assert rejected is None
    assert reason in decision.reasons
    assert decision.retryable is False
    _assert_no_reservations(decision)
    assert tuple(lease.to_dict() for lease in scheduler.active_leases) == before
    corrected = _model("candidate")
    if original is None:
        corrected_lease = _acquire(scheduler, corrected, budget)
    else:
        recovered, corrected_lease = scheduler.transition(original, corrected, host=_host(), providers=[_provider()])
        assert recovered.admitted
        assert corrected_lease.lease_id == original.lease_id
    assert scheduler.release(corrected_lease)


@pytest.mark.parametrize("budget_change,candidate_change,reason", [
    ({"model_token_limit": 150}, {"token_budget": 120}, "lease_token_budget"),
    ({"provider_quota": 3}, {"quota_units": 3}, "lease_provider_quota"),
    ({"memory_bytes": 350}, {"memory_bytes": 300}, "lease_memory_budget"),
    ({"disk_bytes": 350}, {"disk_bytes": 300}, "lease_disk_budget"),
])
@pytest.mark.parametrize("operation", ["acquire", "transition"])
def test_aggregate_budget_refusal_is_atomic_and_clears_rejected_reservations(
    budget_change, candidate_change, reason, operation,
):
    scheduler = _scheduler()
    budget = _budget(**budget_change)
    peer = _acquire(scheduler, _model("peer"), budget)
    original = _acquire(scheduler, _model("candidate"), budget) if operation == "transition" else None
    before = tuple(lease.to_dict() for lease in scheduler.active_leases)
    candidate = _model("candidate", **candidate_change)
    if original is None:
        decision, rejected = scheduler.acquire(candidate, budget=budget, host=_host(), providers=[_provider()])
    else:
        decision, rejected = scheduler.transition(original, candidate, host=_host(), providers=[_provider()])
    assert rejected is None
    assert reason in decision.reasons
    _assert_no_reservations(decision)
    assert tuple(lease.to_dict() for lease in scheduler.active_leases) == before
    assert scheduler.release(peer)
    if original is None:
        corrected = _acquire(scheduler, candidate, budget)
    else:
        recovered, corrected = scheduler.transition(original, candidate, host=_host(), providers=[_provider()])
        assert recovered.admitted, recovered.reasons
        assert corrected.lease_id == original.lease_id
        assert corrected.acquired_at_ms == original.acquired_at_ms
    assert scheduler.active_leases == (corrected,)
    assert scheduler.release(corrected)
    assert scheduler.active_leases == ()


@pytest.mark.parametrize("resource_class,budget_field,reason", [
    (ProofResourceClass.MODEL_DRAFT.value, "max_model_concurrency", "lease_model_concurrency"),
    (ProofResourceClass.SOLVER.value, "max_cpu_proof_concurrency", "lease_cpu_proof_concurrency"),
    (ProofResourceClass.ARTIFACT.value, "max_artifact_concurrency", "lease_artifact_concurrency"),
])
@pytest.mark.parametrize("operation", ["acquire", "transition"])
def test_lease_pool_ceiling_survives_stage_changes(resource_class, budget_field, reason, operation):
    scheduler = _scheduler()
    budget = _budget(**{budget_field: 1})
    peer_requirement = _model("peer", resource_class=resource_class)
    peer = _acquire(scheduler, peer_requirement, budget)
    original = None
    if operation == "transition":
        original = _acquire(scheduler, LaneResourceRequirements(
            lane_id="candidate", resource_class=ProofResourceClass.VALIDATION.value,
        ), budget)
    before = tuple(lease.to_dict() for lease in scheduler.active_leases)
    candidate = replace(peer_requirement, lane_id="candidate")
    if original is None:
        decision, rejected = scheduler.acquire(candidate, budget=budget, host=_host(), providers=[_provider()])
    else:
        decision, rejected = scheduler.transition(original, candidate, host=_host(), providers=[_provider()])
    assert rejected is None
    assert reason in decision.reasons
    _assert_no_reservations(decision)
    assert tuple(lease.to_dict() for lease in scheduler.active_leases) == before
    assert scheduler.release(peer)
    if original is None:
        recovered = _acquire(scheduler, candidate, budget)
    else:
        decision, recovered = scheduler.transition(original, candidate, host=_host(), providers=[_provider()])
        assert decision.admitted
        assert recovered.lease_id == original.lease_id
    assert scheduler.release(recovered)


@pytest.mark.parametrize("operation", ["acquire", "transition"])
def test_lease_latency_bound_cannot_be_widened_by_a_new_stage(operation):
    scheduler = _scheduler()
    budget = _budget(maximum_provider_latency_ms=10)
    original = _acquire(scheduler, _model("candidate"), budget) if operation == "transition" else None
    before = tuple(lease.to_dict() for lease in scheduler.active_leases)
    candidate = _model("candidate", max_provider_latency_ms=1000)
    if original is None:
        decision, rejected = scheduler.acquire(candidate, budget=budget, host=_host(), providers=[_provider(latency_ms=11)])
    else:
        decision, rejected = scheduler.transition(original, candidate, host=_host(), providers=[_provider(latency_ms=11)])
    assert rejected is None
    assert "provider_latency" in decision.reasons
    _assert_no_reservations(decision)
    assert tuple(lease.to_dict() for lease in scheduler.active_leases) == before
    if original is None:
        recovered = _acquire(scheduler, candidate, budget)
    else:
        decision, recovered = scheduler.transition(original, candidate, host=_host(), providers=[_provider()])
        assert decision.admitted
        assert recovered.lease_id == original.lease_id
    assert recovered.requirement.max_provider_latency_ms == 10
    assert recovered.child_limits.maximum_provider_latency_ms == 10
    assert scheduler.release(recovered)


@pytest.mark.parametrize("profile_name", ["RP-CPU-S", "RP-CPU-M", "RP-PROVER", "RP-IO-PINNED"])
def test_profile_bound_leases_pass_narrowed_memory_disk_and_time_to_children(profile_name):
    profile = campaign_resource_profile_spec(profile_name)
    requirement = profile.to_requirement(
        "profile", memory_bytes=profile.memory_bytes // 4,
        disk_bytes=profile.disk_bytes // 4, timeout_ms=profile.wall_time_ms // 4,
    )
    budget = _budget(memory_bytes=profile.memory_bytes * 2,
                     disk_bytes=profile.disk_bytes * 2, wall_time_ms=profile.wall_time_ms * 2)
    scheduler = _scheduler(network_allowlist=profile.network_allowlist)
    lease = _acquire(scheduler, requirement, budget)
    child = lease.child_limits
    assert child.memory_bytes == requirement.memory_bytes <= profile.memory_bytes
    assert child.disk_bytes == requirement.disk_bytes <= profile.disk_bytes
    assert child.wall_time_ms == requirement.timeout_ms <= profile.wall_time_ms
    assert child.max_processes <= requirement.process_slots <= profile.cpu_count
    assert scheduler.release(lease)


@pytest.mark.parametrize("budget_value,declared,expected", [
    (1000, 500, 500), (500, 1000, 500), (0, 500, 500), (500, 0, 500), (0, 0, 0),
])
def test_child_limits_intersect_finite_campaign_and_parent_limits(budget_value, declared, expected):
    budget = _budget(memory_bytes=budget_value, disk_bytes=budget_value, wall_time_ms=budget_value)
    requirement = LaneResourceRequirements(
        lane_id="child", memory_bytes=declared, disk_bytes=declared, timeout_ms=declared,
    )
    child = budget.child_limits(requirement)
    assert child.memory_bytes == expected
    assert child.disk_bytes == expected
    assert child.wall_time_ms == expected
