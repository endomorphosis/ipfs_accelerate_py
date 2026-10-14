"""Immutable admission claims retain constraints and cannot enlarge profiles."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    AdmissionDecision,
    CampaignResourceProfileSpec,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ResourcePolicy,
    ResourceScheduler,
    campaign_resource_profile_spec,
    lane_requirements_for_resource_profile,
    stage_admission_profile,
)


def test_campaign_claim_json_roundtrip_preserves_explicit_constraints() -> None:
    claim = LaneResourceRequirements(
        lane_id="bounded-request", stage="evaluation", timeout_ms=200,
        input_sealed=False, trace_identity="trace:old", expected_trace_identity="trace:new",
        checkpoint_key="checkpoint:a", tokenizer_identity="tok:a", expected_tokenizer_identity="tok:b",
        exclusive_authorities=("compiler-contract:a",), network_allowlist=("https://approved.invalid/object",),
        requires_network=True, requires_gpu=True, requires_prover=True,
        mutates_tokenizer=True, mutates_checkpoint=True,
        claimed_resource_kinds=("cpu", "network"),
    )
    restored = LaneResourceRequirements.from_mapping(json.loads(json.dumps(claim.to_dict())))
    assert restored == claim
    assert restored.input_sealed is False
    assert restored.expected_trace_identity != restored.trace_identity
    assert "compiler-contract:a" in restored.exclusive_authority_keys


@pytest.mark.parametrize("field,value", [
    ("input_sealed", "false"), ("input_sealed", None), ("input_sealed", 1),
    ("requires_network", "false"), ("requires_gpu", 1), ("requires_prover", None),
    ("mutates_tokenizer", "true"), ("mutates_checkpoint", 0),
    ("timeout_ms", True), ("timeout_ms", 1.5), ("timeout_ms", "10"), ("timeout_ms", -1),
    ("trace_identity", 1), ("expected_trace_identity", None), ("checkpoint_key", "bad\x00key"),
    ("tokenizer_identity", ["tok"]), ("expected_tokenizer_identity", {}),
    ("exclusive_authorities", "checkpoint"), ("exclusive_authorities", [1]),
    ("network_allowlist", "https://approved.invalid"), ("network_allowlist", [""]),
    ("claimed_resource_kinds", ["unregistered"]), ("claimed_resource_kinds", ["cpu"] * 65),
    ("resource_profile", None),
])
def test_new_claim_fields_reject_malformed_values_on_both_entrypoints(field, value) -> None:
    with pytest.raises(ValueError):
        LaneResourceRequirements(**{field: value})
    with pytest.raises(ValueError):
        LaneResourceRequirements.from_mapping({field: value})


@pytest.mark.parametrize("field,value", [
    ("memory_bytes", 4 * 1024**3 + 1), ("disk_bytes", 4 * 1024**3 + 1),
    ("process_slots", 3), ("timeout_ms", 30 * 60 * 1000 + 1), ("timeout_ms", 0),
    ("requires_gpu", True), ("gpu_memory_bytes", 1), ("requires_provider", True),
    ("provider_id", "remote"), ("token_budget", 1), ("requires_prover", True),
    ("requires_network", True), ("network_allowlist", ("https://unapproved.invalid",)),
    ("claimed_resource_kinds", ("gpu",)), ("resource_class", "gpu"),
    ("resource_class", "cpu-proof-solver"), ("memory_bytes", True), ("memory_bytes", False),
    ("disk_bytes", False), ("gpu_memory_bytes", False),
])
def test_cpu_profile_cannot_be_expanded_through_helper_or_mapping(field, value) -> None:
    with pytest.raises(ValueError):
        lane_requirements_for_resource_profile("bounded", "RP-CPU-S", **{field: value})
    raw = lane_requirements_for_resource_profile("bounded", "RP-CPU-S").to_dict()
    raw[field] = value
    with pytest.raises(ValueError):
        LaneResourceRequirements.from_mapping(raw)


@pytest.mark.parametrize("overrides", [
    {"resource_profile": ""}, {"resource_profile": "RP-MIXED"},
    {"unrecognized_limit": 1}, {"requires_provider": "false"},
])
def test_profile_helper_rejects_discarded_or_weakened_contracts(overrides) -> None:
    with pytest.raises(ValueError):
        lane_requirements_for_resource_profile("bounded", "RP-CPU-S", **overrides)


def test_profile_quantity_reductions_roundtrip_with_optional_provider() -> None:
    claim = lane_requirements_for_resource_profile(
        "training", "RP-GPU", stage="training", memory_bytes=1024,
        disk_bytes=0, gpu_memory_bytes=512, timeout_ms=500, input_sealed=True,
    )
    assert not claim.provider_required
    assert claim.resource_pool == "gpu"
    assert claim.requires_gpu
    assert LaneResourceRequirements.from_mapping(claim.to_dict()) == claim


def test_profile_zero_estimates_become_finite_reserved_ceiling_claims() -> None:
    profile = campaign_resource_profile_spec("RP-GPU")
    claim = profile.to_requirement("training", memory_bytes=0, disk_bytes=0, gpu_memory_bytes=0)
    assert (claim.memory_bytes, claim.disk_bytes, claim.gpu_memory_bytes) == (
        profile.memory_bytes, profile.disk_bytes, profile.gpu_memory_bytes,
    )


def test_mixed_profile_cpu_work_does_not_request_optional_gpu_network_or_prover() -> None:
    claim = lane_requirements_for_resource_profile(
        "cpu-only", "RP-MIXED", stage="analysis", memory_bytes=1024, disk_bytes=1024,
    )
    assert claim.gpu_memory_bytes == 0
    assert not claim.requires_gpu and not claim.requires_network and not claim.requires_prover
    assert not claim.network_allowlist
    assert not set(claim.resource_kinds).intersection({"gpu", "network", "prover"})
    assert LaneResourceRequirements.from_mapping(claim.to_dict()) == claim
    owner = ResourceScheduler(ResourcePolicy(max_lanes=1))
    decision = owner.evaluate(claim, host=HostResourceSnapshot(
        cpu_percent=10, memory_percent=10, disk_percent=10,
        memory_available_bytes=4096, disk_available_bytes=4096,
        worker_limit=1, available_worker_capacity=1,
        capabilities=("cpu",), resource_classes=("cpu-medium",),
    ))
    assert decision.admitted


@pytest.mark.parametrize("controls", [
    {"requires_gpu": True}, {"stage": "training"}, {"resource_class": "gpu"},
    {"claimed_resource_kinds": ("gpu",)},
])
def test_mixed_profile_explicit_gpu_demand_fills_finite_vram_ceiling(controls) -> None:
    claim = lane_requirements_for_resource_profile("gpu-opt-in", "RP-MIXED", gpu_memory_bytes=0, **controls)
    assert claim.gpu_memory_bytes == 16 * 1024**3


def test_mixed_profile_positive_vram_stays_bounded_and_network_opt_in_is_explicit() -> None:
    claim = lane_requirements_for_resource_profile("gpu-opt-in", "RP-MIXED", gpu_memory_bytes=1024)
    assert claim.gpu_memory_bytes == 1024
    with pytest.raises(ValueError, match="exceeds resource profile"):
        lane_requirements_for_resource_profile("oversized", "RP-MIXED", gpu_memory_bytes=16 * 1024**3 + 1)
    network = lane_requirements_for_resource_profile("network-opt-in", "RP-MIXED", requires_network=True)
    assert network.network_allowlist == campaign_resource_profile_spec("RP-MIXED").network_allowlist


@pytest.mark.parametrize("stage", ["training", "prover", "publication"])
def test_cpu_only_profile_cannot_hide_resource_needs_in_stage(stage) -> None:
    with pytest.raises(ValueError, match="exceeds resource profile"):
        lane_requirements_for_resource_profile("disallowed", "RP-CPU-S", stage=stage)


def test_profile_network_and_gpu_telemetry_requirements_cannot_expand_or_disappear() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        lane_requirements_for_resource_profile(
            "pin", "RP-IO-PINNED", network_allowlist=("https://unapproved.invalid",),
        )
    with pytest.raises(ValueError, match="telemetry"):
        lane_requirements_for_resource_profile("training", "RP-GPU", requires_gpu=False)


@pytest.mark.parametrize("name,cpus,memory,disk,minutes", [
    ("RP-CPU-S", 2, 4, 4, 30), ("RP-CPU-M", 4, 12, 20, 90),
    ("RP-IO-PINNED", 4, 12, 100, 240), ("RP-PROVER", 6, 16, 40, 120),
    ("RP-GPU", 8, 32, 250, 480), ("RP-MIXED", 8, 32, 250, 480),
])
def test_named_profile_ceilings_match_reviewed_board(name, cpus, memory, disk, minutes) -> None:
    spec = campaign_resource_profile_spec(name)
    assert (spec.cpu_count, spec.memory_bytes, spec.disk_bytes, spec.wall_time_ms) == (
        cpus, memory * 1024**3, disk * 1024**3, minutes * 60 * 1000,
    )


@pytest.mark.parametrize("field,value", [
    ("max_gpu_concurrency", "1"), ("max_network_concurrency", True),
    ("max_prover_concurrency", -1), ("allow_unsealed_inputs", "false"),
    ("network_allowlist", "https://unapproved.invalid"),
])
def test_policy_rejects_malformed_new_trusted_fields(field, value) -> None:
    with pytest.raises(ValueError):
        ResourcePolicy(**{field: value})
    with pytest.raises(ValueError):
        ResourcePolicy.from_mapping({field: value})


def test_policy_roundtrip_and_retryable_decision() -> None:
    policy = ResourcePolicy(max_gpu_concurrency=2, max_network_concurrency=3,
                            max_prover_concurrency=4, network_allowlist=("https://trusted.invalid",))
    assert ResourcePolicy.from_mapping(json.loads(json.dumps(policy.to_dict()))) == policy
    decision = AdmissionDecision("bad", False, reasons=("stale_trace",), retryable=False)
    assert decision.to_dict()["retryable"] is False
    with pytest.raises(ValueError):
        AdmissionDecision("bad", False, retryable="false")


def test_campaign_stage_aliases_and_training_requirements_are_available() -> None:
    training = stage_admission_profile("train")
    assert training.stage == "training"
    assert training.requires_gpu and training.requires_sealed_inputs
    assert stage_admission_profile("publish").requires_network


def test_profile_spec_rejects_ambiguous_boolean_and_list_values() -> None:
    with pytest.raises(ValueError):
        CampaignResourceProfileSpec("custom", 1, 0, 0, allows_network="false")
    with pytest.raises(ValueError):
        CampaignResourceProfileSpec("custom", 1, 0, 0, network_allowlist="url")
