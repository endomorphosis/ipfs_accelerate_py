"""Failure observations preserve the task result and expose no frame values."""
from dataclasses import replace
import json
import linecache
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import proof_resource_safety as resources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError


def _expected_resources():
    return dict(cpu_slots=5, total_memory_mb=12288, available_memory_mb=1024,
        memory_stall_percent=12.5, cpu_stall_percent=3.0, io_stall_percent=4.0)


@pytest.fixture
def resource_sample(monkeypatch):
    sample = resources.ProofHostResources(5, 12288, 1024, 12.5, 3.0, 4.0)
    monkeypatch.setattr(resources, "collect_proof_host_resources", lambda: sample)
    return sample


def test_metadata_excludes_frame_locals_source_and_exception_chain(resource_sample, monkeypatch):
    def forbidden_source_read(*args, **kwargs):
        raise AssertionError("traceback diagnostics must not read source")
    monkeypatch.setattr(linecache, "getline", forbidden_source_read)
    monkeypatch.setattr(linecache, "getlines", forbidden_source_read)
    def fail():
        private_local = "PRIVATE_FRAME_VALUE"
        try:
            raise ValueError("PRIVATE_CHAIN_MESSAGE")
        except ValueError as cause:
            raise LeaseTimeoutError("PRIVATE_PRIMARY_MESSAGE") from cause
    try:
        fail()
    except LeaseTimeoutError as error:
        result = driver._failure_diagnostics(error, phase="initial_context")
    assert result["error_phase"] == "initial_context"
    assert result["failure_resources"] == _expected_resources()
    frames = result["error_traceback"]["frames"]
    assert frames[-1]["function"] == "fail"
    assert all(set(frame) == {"file", "function", "line"} for frame in frames)
    assert all(type(frame["line"]) is int and frame["line"] > 0 for frame in frames)
    assert "PRIVATE_" not in json.dumps(result)


@pytest.mark.parametrize("depth,truncated", [(80, False), (300, True)])
def test_traceback_walk_and_retained_frames_are_bounded(resource_sample, depth, truncated):
    def recurse(remaining):
        if remaining:
            recurse(remaining - 1)
        else:
            raise RuntimeError("private")
    try:
        recurse(depth)
    except RuntimeError as error:
        result = driver._failure_diagnostics(error, phase="planning")["error_traceback"]
    assert len(result["frames"]) == 20
    assert result["frames_walked"] <= 256
    assert result["frames_omitted"] == result["frames_walked"] - 20
    assert result["walk_truncated"] is truncated


def test_generated_filename_and_function_names_are_bounded(resource_sample):
    namespace = {}
    exec(compile("def " + "f" * 300 + "():\n    raise RuntimeError('private')\n",
                 "/" + "x" * 900, "exec"), namespace)
    try:
        namespace["f" * 300]()
    except RuntimeError as error:
        frames = driver._failure_diagnostics(error, phase="planning")["error_traceback"]["frames"]
    assert len(frames[-1]["file"]) == 512
    assert len(frames[-1]["function"]) == 128
    assert "private" not in json.dumps(frames)


def test_resource_failure_retains_traceback_without_diagnostic_message(monkeypatch):
    def unavailable():
        raise OSError("PRIVATE_RESOURCE_MESSAGE")
    monkeypatch.setattr(resources, "collect_proof_host_resources", unavailable)
    try:
        raise LeaseTimeoutError("primary")
    except LeaseTimeoutError as error:
        result = driver._failure_diagnostics(error, phase="initial_context")
    assert result["failure_resource_error"] == "OSError"
    assert result["error_traceback"]["frames"]
    assert "failure_resources" not in result
    assert "PRIVATE_" not in json.dumps(result)


def test_traceback_failure_does_not_suppress_resource_sample(resource_sample):
    class BrokenTraceback(RuntimeError):
        def __getattribute__(self, name):
            if name == "__traceback__":
                raise ValueError("PRIVATE_TRACE_MESSAGE")
            return super().__getattribute__(name)
    result = driver._failure_diagnostics(BrokenTraceback("primary"), phase="prepare")
    assert result["failure_traceback_error"] == "ValueError"
    assert result["failure_resources"] == _expected_resources()
    assert "PRIVATE_" not in json.dumps(result)


def test_actual_canonical_resource_probe_reports_only_finite_resource_values():
    result = driver._failure_diagnostics(RuntimeError("authored control"), phase="prepare")
    sample = result["failure_resources"]
    assert set(sample) == {"cpu_slots", "total_memory_mb", "available_memory_mb",
                           "memory_stall_percent", "cpu_stall_percent", "io_stall_percent"}
    resources.ProofHostResources(**sample)
    assert result["error_traceback"]["frames"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("stage", ["prepare", "initial_context", "planning"])
@pytest.mark.parametrize("diagnostic_failure", [None, "resources", "helper"])
def test_driver_failure_retains_primary_result_phase_and_cleanup(
        tmp_path, monkeypatch, resource_sample, stage, diagnostic_failure):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda mask: None)
    signals, cleanup = [], []
    monkeypatch.setattr(driver.signal, "signal", lambda *args: signals.append(args))
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: signals.append(args))
    monkeypatch.setattr(driver, "_final_context_audit", lambda *args, **kwargs: None)
    def clean(argv, **kwargs):
        cleanup.append((argv, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(driver.subprocess, "run", clean)
    calls = []
    def selected(name, returned):
        def invoke(**kwargs):
            calls.append(name)
            if name == stage:
                raise LeaseTimeoutError("primary lease refusal")
            return returned
        return invoke
    monkeypatch.setattr(driver.preparation, "prepare", selected("prepare", {"intent_preplanning": {}}))
    monkeypatch.setattr(driver.preparation, "initial_context", selected("initial_context", {}))
    monkeypatch.setattr(driver.preparation, "plan", selected("planning", {}))
    def broken(*args, **kwargs):
        raise ValueError("PRIVATE_DIAGNOSTIC_MESSAGE")
    if diagnostic_failure == "resources":
        monkeypatch.setattr(resources, "collect_proof_host_resources", broken)
    elif diagnostic_failure == "helper":
        monkeypatch.setattr(driver, "_failure_diagnostics", broken)
    state = tmp_path / "state" / "run"
    report = driver.run(instruction=Path("/unused-authored-instruction"), state=state,
                        arm="full", source384_config=Path("/unused-authored-config"))
    assert calls == ["prepare", "initial_context", "planning"][:calls.index(stage) + 1]
    assert report["error"] == {"type": "LeaseTimeoutError", "message": "primary lease refusal"}
    assert report["error_phase"] == stage
    assert report["task_completed"] is False
    assert report["official_reward"] is None
    assert report["production_activation"] is report["benchmark_advantage_claimed"] is False
    assert report["provider_invocations"] == []
    assert report["remaining_processes"] is None
    assert report["max_total_agent_seconds"] == 285
    assert report["reserved_cleanup_seconds"] == 40
    assert report["work_cutoff_seconds"] == 245
    assert report["worker_cleanup_returncode"] == 0
    assert len(cleanup) == 1 and cleanup[0][0][-1] == "--cleanup"
    assert (driver.signal.ITIMER_REAL, 0) in signals
    assert json.loads((state.parent / "run-result.json").read_text()) == report
    assert "PRIVATE_DIAGNOSTIC_MESSAGE" not in json.dumps(report)
    if diagnostic_failure == "helper":
        assert report["failure_diagnostics_error"] == "ValueError"
    else:
        assert report["error_traceback"]["frames"][-1]["function"] == "invoke"
        if diagnostic_failure == "resources":
            assert report["failure_resource_error"] == "ValueError"
        else:
            assert report["failure_resources"] == _expected_resources()


def test_post_unwind_sample_does_not_copy_or_export_optional_metadata(resource_sample, monkeypatch):
    class PrivateMetadata:
        def __deepcopy__(self, memo):
            pytest.fail('scalar diagnostics must not traverse optional metadata')
    sample = replace(resource_sample, pressure_sources=PrivateMetadata())
    monkeypatch.setattr(resources, 'collect_proof_host_resources', lambda: sample)
    result = driver._failure_diagnostics(RuntimeError('authored control'), phase='prepare')
    assert result['failure_resources'] == _expected_resources()
    assert 'pressure_sources' not in result['failure_resources']
