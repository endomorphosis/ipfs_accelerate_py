"""A refused native runtime constructor must dispose only unlaunched custody."""
from __future__ import annotations

import json
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints import isolated_benchmark_runtime as isolated
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import open_database_coordinator


@pytest.mark.parametrize("failure_stage", ["adapter", "orchestrator", "service"])
def test_refused_isolated_constructor_releases_actual_unlaunched_run_lease(tmp_path, monkeypatch, failure_stage):
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")
    coordinators = []
    def capture_coordinator(path):
        coordinator = open_database_coordinator(path)
        coordinators.append(coordinator)
        return coordinator
    monkeypatch.setattr(isolated, "open_database_coordinator", capture_coordinator)
    refusal = RuntimeError("authored initialization refusal")
    acquired = []
    def refuse(*args, **kwargs):
        acquired.extend(coordinators[0].list_active_leases())
        raise refusal
    target = {
        "adapter": "NativeSupervisorHealthAdapter",
        "orchestrator": "LifecycleOrchestrator",
        "service": "SupervisorControlService",
    }[failure_stage]
    monkeypatch.setattr(isolated, target, refuse)
    directory = tmp_path / "runtime"
    try:
        with pytest.raises(RuntimeError, match="authored initialization refusal") as caught:
            isolated.IsolatedBenchmarkRuntime.create(directory)
        assert caught.value is refusal
        assert len(coordinators) == 1
        assert not coordinators[0].is_open
        assert len(acquired) == 1
        with open_database_coordinator(directory / "state/coordination.duckdb") as observed:
            assert observed.get_lease(acquired[0].lease_id).state.value == "released"
        receipt = json.loads((directory / "state/construction-cleanup.json").read_text())
        assert receipt["run_lease_released"] is True
        assert receipt["native_STOP_proved"] is False
        assert receipt["completion_authority"] is False
        assert not (directory / "state/supervisor-process.log").exists()
    finally:
        for coordinator in coordinators:
            coordinator.close()


def test_refused_isolated_constructor_retains_custody_if_initialization_launched_a_child(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")
    captured = []
    class CapturingRuntime(isolated.IsolatedBenchmarkRuntime):
        def __init__(self):
            captured.append(self)
    refusal = RuntimeError("authored refusal after child creation")
    def launch_then_refuse(**kwargs):
        runtime = captured[0]
        runtime.process._popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
        raise refusal
    monkeypatch.setattr(isolated, "SupervisorControlService", launch_then_refuse)
    try:
        with pytest.raises(RuntimeError, match="constructor cleanup unproven"):
            CapturingRuntime.create(tmp_path / "runtime")
        runtime = captured[0]
        assert len(runtime._children) == 1
        assert runtime._children[0].poll() is None
        assert runtime.coordinator.is_open
        assert runtime.coordinator.get_lease(runtime.lease.lease_id).state.value == "accepted"
        assert runtime._construction_cleanup_failed
        assert not (runtime.state / "construction-cleanup.json").exists()
    finally:
        for runtime in captured:
            for child in getattr(runtime, "_children", ()):
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
            if getattr(runtime, "coordinator", None) is not None:
                runtime.coordinator.release(runtime.lease, expected_fencing_token=runtime.lease.fencing_token,
                                            expected_fence_epoch=runtime.lease.fence_epoch)
                runtime.coordinator.close()
