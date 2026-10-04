from __future__ import annotations

import time
import threading
import json
from dataclasses import replace

import pytest

from test.integration.test_admitted_benchmark_runtime import admitted
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import ControlBounds, Operation, get_operation_catalog
from ipfs_accelerate_py.agent_supervisor.control.control_plane import SupervisorControlService
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap import STARTUP_WAIT_ENV, _startup_wait_seconds


def test_default_start_deadline_includes_fresh_prelaunch_work(admitted, monkeypatch):
    """Real launch custody must count validation work before child creation."""
    runtime, _owner, _prepared = admitted
    original = runtime.process._popen

    def delayed_launch(*args, **kwargs):
        time.sleep(20.1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.process, "_popen", delayed_launch)
    started_at = time.monotonic()
    result = runtime.start()
    assert time.monotonic() - started_at >= 20.1
    assert not result.succeeded
    assert result.error.message == "startup did not prove sustained health before its deadline"
    assert not runtime.process.snapshot(runtime.profile).members
    assert all(child.poll() is not None for child in runtime._children)


@pytest.mark.parametrize("admitted", ["extended-startup"], indirect=True)
def test_signed_extended_start_allows_fresh_checks_without_borrowing_stop_budget(admitted, monkeypatch):
    """Actual native health after slow checks, including a >30 s bootstrap wait."""
    runtime, _owner, _prepared = admitted
    original = runtime._verify

    def slow_fresh_verification(*args, **kwargs):
        if threading.current_thread() is runtime._bootstrap_thread:
            time.sleep(31.0)
        elif runtime._startup_trace[-1]["phase"] == "launch_validation":
            time.sleep(20.1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_verify", slow_fresh_verification)
    assert runtime.manifest["start_timeout_ms"] == 120_000
    assert dict(runtime.manifest["environment"])[STARTUP_WAIT_ENV] == "120000"
    started = runtime.start()
    assert started.succeeded, started.error
    assert runtime.bootstrap_receipts and not runtime.bootstrap_errors
    diagnostic = runtime.startup_diagnostics()
    phases = {row["phase"]: row for row in diagnostic["observations"]}
    assert phases["launch_validation"]["seconds"] >= 20.1
    assert phases["bootstrap_validation"]["seconds"] >= 31.0
    assert phases["bootstrap_validation"]["status"] == "completed"
    assert diagnostic["start_timeout_ms"] == 120_000
    assert diagnostic["stop_timeout_ms"] == 20_000
    assert diagnostic["bootstrap_wait_seconds"] == 120
    assert runtime.request(Operation.STOP).bounds.timeout_ms == 20_000
    stopped = runtime.stop()
    assert stopped.succeeded
    assert not runtime.process.snapshot(runtime.profile).members
    assert all(child.poll() is not None for child in runtime._children)
    (runtime.directory / "startup-control-observation.json").write_text(json.dumps({
        "schema": "authored-native-startup-delay-control@1",
        "authored_launch_delay_seconds": 20.1, "authored_bootstrap_delay_seconds": 31.0,
        "startup": diagnostic, "start_status": started.status.value,
        "stop_status": stopped.status.value, "remaining_processes": 0,
        "all_child_processes_reaped": True, "original_benchmark_subcause_proved": False,
    }, indent=2) + "\n")


@pytest.mark.parametrize("admitted", ["extended-startup"], indirect=True)
def test_local_start_override_preserves_catalog_other_bounds_and_signed_custody(admitted, monkeypatch):
    runtime, _owner, _prepared = admitted
    request = runtime.request(Operation.START)
    canonical = get_operation_catalog()
    assert canonical.by_name["start"].bounds.timeout_ms == 30_000
    assert runtime.service._catalog.content_id == canonical.content_id
    assert runtime.service._catalog.by_name["stop"].bounds.timeout_ms == 30_000
    default = SupervisorControlService(repository_allowlist=(runtime.repository,), state_allowlist=(runtime.state,))
    with pytest.raises(ValueError, match="timeout_ms"):
        default._check_bounds(request)
    runtime.service._check_bounds(request)
    oversized = replace(request, bounds=replace(request.bounds, max_items=request.bounds.max_items+1))
    with pytest.raises(ValueError, match="max_items"):
        runtime.service._check_bounds(oversized)
    stop = runtime.request(Operation.STOP)
    with pytest.raises(ValueError, match="timeout_ms"):
        runtime.service._check_bounds(replace(stop, bounds=ControlBounds(timeout_ms=120_000)))
    monkeypatch.setattr(runtime, "start_timeout_ms", 119_000)
    with pytest.raises(ValueError, match="signed launch"):
        runtime.start()
    assert runtime._children == []


def test_unsigned_start_override_and_ambient_bootstrap_wait_are_rejected(admitted, monkeypatch):
    runtime, _owner, _prepared = admitted
    assert "start_timeout_ms" not in runtime.manifest
    assert dict(runtime.profile.environment)[STARTUP_WAIT_ENV] == "30000"
    monkeypatch.setattr(runtime, "start_timeout_ms", 120_000)
    with pytest.raises(ValueError, match="signed launch"):
        runtime.start()
    assert runtime._children == []


@pytest.mark.parametrize("admitted", ["extended-startup"], indirect=True)
@pytest.mark.parametrize("field", ["service", "bootstrap_environment", "manifest_and_property"])
def test_signed_startup_budget_rejects_independent_mutations(admitted, monkeypatch, field):
    runtime, _owner, _prepared = admitted
    if field == "service":
        monkeypatch.setattr(runtime.service, "_local_start_timeout_ms", 119_000)
    elif field == "bootstrap_environment":
        monkeypatch.setattr(runtime, "profile", replace(runtime.profile, profile_id="", environment=tuple(
            (key, "119000") if key == STARTUP_WAIT_ENV else (key, value)
            for key, value in runtime.profile.environment)))
    else:
        monkeypatch.setattr(runtime, "manifest", {**runtime.manifest, "start_timeout_ms": 119_000})
        monkeypatch.setattr(runtime, "start_timeout_ms", 119_000)
        monkeypatch.setattr(runtime.service, "_local_start_timeout_ms", 119_000)
    with pytest.raises(ValueError, match="signed launch|launch environment|launch grant"):
        runtime.start()
    assert runtime._children == [] and not runtime.bootstrap_receipts


@pytest.mark.parametrize("value", [True, 1999, 120001, 120000.0])
def test_local_service_start_override_is_bounded(tmp_path, value):
    with pytest.raises(ValueError, match="local START timeout"):
        SupervisorControlService(repository_allowlist=(tmp_path,), state_allowlist=(tmp_path,), local_start_timeout_ms=value)


@pytest.mark.parametrize("value", [None, "2000", "30000", "120000", "0", "120001", "2.0", " 30000", "-1"])
def test_bootstrap_wait_defaults_and_strict_bound(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(STARTUP_WAIT_ENV, raising=False)
        assert _startup_wait_seconds() == 30
    else:
        monkeypatch.setenv(STARTUP_WAIT_ENV, value)
        if value in {"2000", "30000", "120000"}:
            assert _startup_wait_seconds() == int(value)/1000
        else:
            with pytest.raises(ValueError, match="native bootstrap wait"):
                _startup_wait_seconds()


def test_explicit_start_override_requires_local_benchmark_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    with pytest.raises(ValueError, match="local benchmark profile"):
        AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=None, server=None, source=None, start_timeout_ms=120_000)
