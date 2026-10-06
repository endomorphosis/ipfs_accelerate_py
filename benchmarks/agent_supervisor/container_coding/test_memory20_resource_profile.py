"""Explicit 20 GiB experiment, exact transport, unchanged admission gates.

Docker boundaries and telemetry are authored fixtures. The context ingress,
resource scheduler and leases are real; this does not measure model memory or
claim a live benchmark score.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as profiles
from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualification
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_qualification import docker_boundary  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_transport import selected  # noqa: F401
from test.api.semantic_state.test_source384_context_lifetime import context_case  # noqa: F401


PROFILE = profiles.PLANNER180_20GIB_SOURCE384_PROFILE


def _observation():
    return dict(schema="terminal-source384-cgroup-observation@1", cgroup_path="/sys/fs/cgroup",
        cpu_max="500000 100000", memory_max=str(20480 * 1024**2), detected_cpu_slots=5,
        detected_total_memory_mb=20480, available_memory_mb=13300)


def test_memory20_is_explicit_and_preserves_every_existing_profile_and_work_budget():
    assert PROFILE == "source384-5cpu-20gib-planner180@1"
    assert profiles.resource_environment(PROFILE) == dict(override_cpus=5, override_memory_mb=20480,
        cpu_enforcement_policy="limit", memory_enforcement_policy="limit")
    assert profiles.execution_budget(PROFILE) == profiles.execution_budget(profiles.PLANNER180_SOURCE384_PROFILE)
    assert profiles.planner_timeout_seconds(PROFILE) == 180
    assert profiles.native_start_timeout_ms(PROFILE, remaining_work_seconds=840) == 120000
    assert profiles.native_start_timeout_ms(PROFILE, remaining_work_seconds=5) == 5000
    assert profiles.admission_environment(PROFILE) == profiles.admission_environment(profiles.PLANNER180_SOURCE384_PROFILE)
    for selected_profile, memory in ((profiles.SOURCE384_PROFILE, 12288),
            (profiles.EXTENDED_SOURCE384_PROFILE, 16384), (profiles.PLANNER180_SOURCE384_PROFILE, 16384)):
        assert profiles.resource_environment(selected_profile)["override_memory_mb"] == memory
    with pytest.raises(ValueError, match="conflicts"):
        profiles.apply_resource_profile({"environment": {"override_memory_mb": 16384}}, PROFILE)


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_memory20_compares_only_with_identical_resources_across_arms(tmp_path, arm):
    from harbor.models.job.config import JobConfig
    from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as supervisor
    from benchmarks.agent_supervisor.container_coding import native_codex_baseline as baseline
    from benchmarks.agent_supervisor.container_coding.benchmark_controls import compare_controls
    from benchmarks.agent_supervisor.container_coding.test_benchmark_controls import observation

    def normalize(value):
        return JobConfig.model_validate(value, extra="forbid").model_dump(mode="json")

    native = normalize(baseline.config_for(tmp_path / "data", tmp_path / "native", resource_profile=PROFILE))
    other = normalize(supervisor.config_for(tmp_path / "data", tmp_path / "other", tmp_path / "archive",
        arm, resource_profile=PROFILE))
    prior = normalize(baseline.config_for(tmp_path / "data", tmp_path / "prior",
        resource_profile=profiles.PLANNER180_SOURCE384_PROFILE))
    assert compare_controls(observation(native), observation(other))["matches"] is True
    assert compare_controls(observation(native), observation(prior))["matches"] is False


@pytest.mark.parametrize("field,value", [("memory_max", str(16384 * 1024**2)),
    ("detected_total_memory_mb", 16384), ("detected_total_memory_mb", True),
    ("cpu_max", "400000 100000"), ("memory_max", "max"),
    ("available_memory_mb", 20481), ("available_memory_mb", True)])
def test_memory20_cgroup_must_match_the_selected_enforced_limit(field, value):
    record = _observation()
    assert qualification.validate_resource_observation(record, PROFILE) == record
    record[field] = value
    with pytest.raises(ValueError):
        qualification.validate_resource_observation(record, PROFILE)


def test_memory20_preflight_keeps_six_gib_request_and_increases_twenty_percent_headroom(tmp_path):
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(
        return_code=0, stdout=json.dumps(_observation()), stderr="")))
    assert asyncio.run(qualification.observe_resources(environment, output=tmp_path,
        profile=PROFILE)) == _observation()
    estimate = json.loads((tmp_path / "admission-estimate.json").read_bytes())
    assert estimate["requested_parent_memory_mb"] == 6144
    assert estimate["derived_default_headroom_mb"] == 4096
    assert estimate["minimum_available_memory_mb"] == 10240
    assert estimate["exact_admission_decision"] is False
    assert estimate["actual_native_admission_required"] is True
    assert json.loads((tmp_path / "resources.json").read_bytes()) == _observation()


def test_memory20_reaches_real_docker_constructor_and_each_preflight(docker_boundary):
    arguments, calls, environment, _ = docker_boundary
    qualification.observe_resources.return_value = _observation()
    result = asyncio.run(deployment.qualify_original_container(**arguments, install_codex=False,
        resource_profile=PROFILE, source384_context=True))
    assert result["qualified"] is True
    assert {key: calls[0][key] for key in profiles.resource_environment(PROFILE)} == profiles.resource_environment(PROFILE)
    assert qualification.observe_resources.await_count == 2
    assert all(call.kwargs["profile"] == PROFILE for call in qualification.observe_resources.await_args_list)
    assert qualification.qualify_context.await_args.kwargs["profile"] == PROFILE
    environment.stop.assert_awaited_once_with(delete=True)


@pytest.mark.parametrize("total,available,late_pressure,expected", [
    (16384, 9204, False, "refused"),
    (20480, 13300, False, "validated"),
    (20480, 10239, False, "refused"),
    (20480, 13300, True, "refused"),
])
def test_precoding_context_ingress_preserves_real_root_and_child_pressure_gates(
        context_case, monkeypatch, total, available, late_pressure, expected):
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as implementation
    from ipfs_datasets_py.logic.software_contracts import codebase_resources, codebase_source_units_384 as units
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources

    case = context_case
    readings = [ProofHostResources(5, total, available)]
    config = resources.ResourceSchedulerConfig.for_proof_host(
        state_path=case.output / "pressure-scheduler.json", proof_resource_profile="local-benchmark@1",
        proof_resource_sampler=lambda: readings[0], lane_reservations={}, total_child_process_slots=4,
        proof_memory_stall_percent=10., proof_recovery_enabled=True, proof_recovery_grants=1,
        auto_renew_leases=False, proof_backoff_seconds=.005, poll_interval_seconds=.002)
    scheduler = resources.GlobalResourceScheduler(config)
    assert config.proof_memory_headroom_mb == total - int(total * .8)
    monkeypatch.setattr(codebase_resources, "get_global_resource_scheduler", lambda: scheduler)
    # Bound test waiting only; all admission decisions and lease checks are native.
    monkeypatch.setattr(codebase_resources, "codebase_admission_timeout",
        lambda *args, remaining_seconds, **kwargs: min(1., remaining_seconds))
    validator = units.validate_shared_parent_units
    observed = []

    def validate(index, repository, report, **kwargs):
        parent = kwargs["parent_lease"]
        assert parent.memory_mb == 6144
        assert kwargs["memory_mb"] == 4096
        if late_pressure:
            readings[0] = ProofHostResources(5, total, 10239)
        with codebase_resources.acquire_codebase_resources(parent_lease=parent,
                timeout_seconds=1., memory_mb=4096) as child:
            observed.append((parent.memory_mb, child.memory_mb))
            validator(index, repository, report, **kwargs)

    monkeypatch.setattr(units, "validate_shared_parent_units", validate)
    task = implementation.PortalTask(task_id="RESOURCE-020", title="Validate current context",
        status="ready", completion="manual", priority="P1", track="context", outputs=["module.py"],
        validation=[], acceptance="Current captured source only", metadata={}, canonical_task_cid="task:memory20")
    metadata = {"semantic context artifact": "semantic.json", "semantic context sha256": "b" * 64}
    payload = dict(schema=bundles.SOURCE384_SCHEMA, completion_authority=False, tasks=[dict(
        task_id=task.task_id, task_cid=task.canonical_task_cid, metadata=metadata,
        source384_context=deepcopy(case.receipt))])
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    (case.repository / "context.json").write_bytes(raw)
    daemon = object.__new__(implementation.TodoImplementationDaemon)
    daemon.repo_root = case.repository
    daemon._task_context_nomination_bundle = dict(artifact="context.json", sha256=hashlib.sha256(raw).hexdigest())
    token = implementation._source384_callback_deadline.set(time.monotonic() + 30)
    try:
        if expected == "validated":
            result = daemon._task_metadata_snapshot(task, include_context=True)
            assert all(result[key] == value for key, value in metadata.items())
            assert observed == [(6144, 4096)]
        else:
            with pytest.raises(implementation.Source384ResourceDeferred) as caught:
                daemon._task_metadata_snapshot(task, include_context=True)
            assert caught.value.reason == "source384_resource_admission_deferred"
            native = caught.value.__cause__
            assert type(native) is resources.LeaseTimeoutError
            assert native.admission_observation["primary_gate"]["reason"] == "proof_memory_headroom"
            sample = native.admission_observation["last_sample"]
            assert sample["thresholds"]["memory_headroom_mb"] == config.proof_memory_headroom_mb
            assert sample["additional_request_memory_mb"] == (0 if late_pressure else 6144)
            assert sample["reserved_root_memory_mb"] == (6144 if late_pressure else 0)
            assert observed == []
    finally:
        implementation._source384_callback_deadline.reset(token)
    assert task.metadata == {}
    snapshot = scheduler.snapshot()
    assert snapshot["active_lease_count"] == snapshot["waiting_request_count"] == 0
