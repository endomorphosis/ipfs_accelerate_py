"""Derived resource kinds must obey the same policy as explicit flags."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import resource_scheduler as module


@pytest.fixture(autouse=True)
def _no_metadata_mirroring(monkeypatch):
    monkeypatch.setattr(module, "_mirror_resource_admission", lambda decision: decision)


def _host(**overrides):
    values = dict(observed_at_ms=1000, cpu_percent=10, memory_percent=10, disk_percent=10,
                  memory_available_bytes=10_000_000, disk_available_bytes=10_000_000,
                  worker_limit=8, available_worker_capacity=8,
                  capabilities=("cpu", "network", "prover"),
                  resource_classes=("cpu-small", "cpu-medium", "network", "gpu", "prover"))
    values.update(overrides)
    return module.HostResourceSnapshot(**values)


def _lane(kind, source, lane_id="candidate", **overrides):
    values = dict(lane_id=lane_id, stage="analysis", resource_class="cpu-small")
    if source == "class":
        values["resource_class"] = kind
    elif source == "claim":
        values["claimed_resource_kinds"] = (kind,)
    elif source == "stage":
        values["stage"] = {"gpu": "training", "network": "network", "prover": "prover"}[kind]
    values.update(overrides)
    return module.LaneResourceRequirements(**values)


def _scheduler(**overrides):
    return module.ResourceScheduler(module.ResourcePolicy(
        max_lanes=8, require_provider_telemetry=False, **overrides,
    ))


@pytest.mark.parametrize("source", ["class", "claim", "stage"])
def test_derived_network_kind_cannot_bypass_trusted_allowlist(source):
    requirement = _lane("network", source, network_allowlist=("https://untrusted.test",))
    assert requirement.requires_network is False
    decision = _scheduler(network_allowlist=("https://trusted.test",)).evaluate(requirement, host=_host())
    assert not decision.admitted and not decision.retryable
    assert "network_not_allowlisted" in decision.reasons


@pytest.mark.parametrize("source", ["class", "claim", "stage"])
def test_derived_network_kind_requires_network_capability(source):
    trusted = ("https://trusted.test",)
    requirement = _lane("network", source, network_allowlist=trusted)
    decision = _scheduler(network_allowlist=trusted).evaluate(
        requirement, host=_host(capabilities=("cpu", "prover")),
    )
    assert not decision.admitted and "network_capability_missing" in decision.reasons


@pytest.mark.parametrize("source", ["class", "claim", "stage"])
def test_derived_gpu_kind_requires_live_gpu_telemetry(source):
    requirement = _lane("gpu", source)
    assert requirement.requires_gpu is False and requirement.gpu_memory_bytes == 0
    decision = _scheduler().evaluate(requirement, host=_host(gpu_memory_total_bytes=0))
    assert not decision.admitted and decision.retryable
    assert "gpu_telemetry_missing" in decision.reasons


@pytest.mark.parametrize("source", ["class", "claim", "stage"])
def test_derived_prover_kind_requires_prover_capability(source):
    requirement = _lane("prover", source)
    assert requirement.requires_prover is False
    decision = _scheduler().evaluate(
        requirement, host=_host(capabilities=("cpu",), resource_classes=("cpu-small", "cpu-medium")),
    )
    assert not decision.admitted and "prover_not_admitted" in decision.reasons


@pytest.mark.parametrize("kind", ["network", "gpu", "prover"])
@pytest.mark.parametrize("source", ["class", "claim", "stage"])
def test_derived_resource_kinds_count_toward_concurrency(kind, source):
    trusted = ("https://trusted.test",)
    extra = {"network_allowlist": trusted} if kind == "network" else {}
    active = _lane(kind, "claim", lane_id="active", **extra)
    candidate = _lane(kind, source, **extra)
    scheduler = _scheduler(network_allowlist=trusted, max_gpu_concurrency=1,
                           max_network_concurrency=1, max_prover_concurrency=1)
    decision = scheduler.evaluate(
        candidate, host=_host(gpu_memory_total_bytes=10_000_000, gpu_memory_available_bytes=10_000_000),
        active_requirements=(active,),
    )
    assert not decision.admitted and decision.retryable
    assert f"{kind}_concurrency" in decision.reasons
