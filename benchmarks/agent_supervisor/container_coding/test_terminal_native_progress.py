"""Real bounded-file observations; authored state doubles grant no authority."""
import copy
import json
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_native_progress as progress


def task(**updates):
    return SimpleNamespace(**{"task_cid": "task:fixture", "status": "in_progress", "revision": 2,
                              "body": {}, **updates})


def heartbeat(tmp_path, **updates):
    value = {"schema": progress.HEARTBEAT_SCHEMA, "sequence": 1, "write_count": 0, "unchanged": True,
        "process_birth": {"pid": 123, "start_time_ticks": 789, "boot_id": "boot:fixture"},
        "process_instance_id": "process:fixture", "owner_session_id": "owner:fixture",
        "selection_idle_reason": "no_ready_tasks", "active_task_id": "", "claimed_task_cid": "", **updates}
    path = tmp_path / "run/admitted_database_daemon_pass_heartbeat.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def sample(recorder, tmp_path, current=None):
    return recorder.sample(task=current or task(), task_cid="task:fixture", state=tmp_path, now=101.)


@pytest.mark.parametrize("reason", sorted(progress.QUIESCENT_STOP_REASONS))
def test_exact_quiescence_requires_two_advancing_same_generation_passes(tmp_path, reason):
    recorder = progress.NativeProgress(started=100.)
    heartbeat(tmp_path, selection_idle_reason=reason)
    assert sample(recorder, tmp_path) is None
    assert sample(recorder, tmp_path) is None  # Re-reading one file is no new pass.
    heartbeat(tmp_path, selection_idle_reason=reason, sequence=2)
    assert sample(recorder, tmp_path) == reason
    assert all(recorder.report[name] is False for name in
               ("completion_authority", "settlement_authority", "retry_authority"))
    assert recorder.report["transition_count"] == 1
    assert recorder.report["latest"]["heartbeat"]["sequence"] == 2


@pytest.mark.parametrize("change", ["generation", "revision", "active", "claimed", "write", "malformed", "missing", "regression"])
def test_quiescence_does_not_stitch_discontinuous_or_active_observations(tmp_path, change):
    recorder = progress.NativeProgress(started=100.)
    reason = "unsettled_portal_failure_quarantine"
    path = heartbeat(tmp_path, selection_idle_reason=reason, sequence=3)
    assert sample(recorder, tmp_path) is None
    updates = {"selection_idle_reason": reason, "sequence": 4}
    current = task()
    if change == "generation": updates["process_instance_id"] = "new-process"
    elif change == "revision": current.revision = 3
    elif change == "active": updates["active_task_id"] = "TASK-001"
    elif change == "claimed": updates["claimed_task_cid"] = "task:fixture"
    elif change == "write": updates["write_count"] = 1
    elif change == "regression": updates["sequence"] = 2
    heartbeat(tmp_path, **updates)
    if change == "malformed": path.write_text("{")
    if change == "missing": path.unlink()
    assert sample(recorder, tmp_path, current) is None


@pytest.mark.parametrize("reason", ["no_ready_tasks", "completion_evidence_unavailable",
    "task_fence_evidence_unavailable", "provider_capacity_backoff", "custom-private-reason"])
def test_transient_and_unknown_idle_reasons_never_stop_run(tmp_path, reason):
    recorder = progress.NativeProgress(started=100.)
    for sequence in range(1, 5):
        heartbeat(tmp_path, selection_idle_reason=reason, sequence=sequence)
        assert sample(recorder, tmp_path) is None
    assert "custom-private-reason" not in json.dumps(recorder.report)


def test_expired_custody_retains_existing_early_stop(tmp_path):
    recorder = progress.NativeProgress(started=100.)
    heartbeat(tmp_path, selection_idle_reason="expired_attempt_settlement_unavailable")
    assert sample(recorder, tmp_path) == "expired_attempt_settlement_unavailable"


@pytest.mark.parametrize("status", sorted(progress.TERMINAL_STATUSES))
def test_only_canonical_terminal_status_is_recorded_as_terminal(tmp_path, status):
    recorder = progress.NativeProgress(started=100.)
    assert sample(recorder, tmp_path, task(status=status)) == "native_task_" + status
    assert recorder.report["completion_authority"] is False


def test_completion_receipt_diagnostic_projection_keeps_ids_counts_and_closed_findings(tmp_path):
    receipt = {"operation": "database_portal_retry_budget_exhausted", "attempt_id": "private-attempt",
        "claim_id": "private-claim", "attempt_number": 1, "max_task_attempts": 1,
        "provider_invocation_count": 1, "effect_claim_count": 0,
        "provider_dispatched": True, "attempt_consumed": True, "reason": "proposal_gate_failed",
        "finding_codes": ["python_syntax_error", "private_finding"],
        "output": "private_model_body", "prompt": "private_prompt", "argv": ["private_command"]}
    current = task(status="blocked", body={"completion_receipt": receipt, "source": "private_source"})
    original = copy.deepcopy(current.body)
    recorder = progress.NativeProgress(started=100.)
    assert sample(recorder, tmp_path, current) == "native_task_blocked"
    result = recorder.report["latest"]["task"]["completion_receipt"]
    assert result["reason"] == "proposal_gate_failed"
    assert result["finding_codes"] == ["python_syntax_error"]
    assert result["unrecognized_finding_count"] == 1
    assert len(result["attempt_id_sha256"]) == 64
    assert result["provider_invocation_count"] == 1
    assert current.body == original
    assert "private_" not in json.dumps(recorder.report) and "private-" not in json.dumps(recorder.report)


@pytest.mark.parametrize("bad", ["symlink", "oversized", "duplicate", "nonfinite", "array", "boolean", "birth", "schema"])
def test_malformed_heartbeat_never_terminates_or_exports_body(tmp_path, bad):
    path = heartbeat(tmp_path, selection_idle_reason="unsettled_portal_failure_quarantine")
    if bad == "symlink":
        saved = path.with_suffix(".saved"); path.rename(saved); path.symlink_to(saved)
    elif bad == "oversized": path.write_bytes(b"x" * (progress.MAX_HEARTBEAT_BYTES + 1))
    elif bad == "duplicate": path.write_text('{"private":0,"private":1}')
    elif bad == "nonfinite": path.write_text('{"private":NaN}')
    elif bad == "array": path.write_text('[]')
    elif bad == "boolean": heartbeat(tmp_path, sequence=True)
    elif bad == "birth": heartbeat(tmp_path, process_birth={"private": "body"})
    elif bad == "schema": heartbeat(tmp_path, schema="private_schema")
    assert progress.read_heartbeat(tmp_path) == {"availability": "unavailable"}


def test_transcript_is_bounded_and_retains_latest_task_state(tmp_path):
    recorder = progress.NativeProgress(started=100.)
    for revision in range(1, 101):
        assert sample(recorder, tmp_path, task(revision=revision)) is None
    assert recorder.report["transition_count"] == 100
    assert recorder.report["transitions_omitted"] == 100 - progress.MAX_TRANSITIONS
    assert len(recorder.report["transitions"]) == progress.MAX_TRANSITIONS
    assert recorder.report["latest"]["task"]["revision"] == 100


@pytest.mark.parametrize("phase", ["prepare", "planning", "native_execution"])
def test_final_metadata_separates_provider_timeout_from_unattributed_timeout(phase):
    recorder = progress.NativeProgress(started=100.)
    recorder.finish({"start": {"status": "succeeded", "request_id": "private-request", "data": "private"},
        "stop": {"status": "succeeded"}, "error_phase": phase,
        "error": {"type": "TimeoutError", "message": "private"},
        "provider_invocations": [{"phase": "coding", "status": "failed", "timeout_seconds": 300,
            "seconds": 301.5, "router_calls": 1, "invocation_id": "private-invocation",
            "usage": {"timed_out": True, "exit_code": 124, "prompt": "private"}, "output": "private"}]})
    assert recorder.report["stop_reason"] == "timeout_observed_without_terminal_state"
    assert recorder.report["provider_outcomes"][0]["timed_out"] is True
    assert recorder.report["provider_outcomes"][0]["timeout_seconds"] == 300
    assert recorder.report["stop"]["status"] == "succeeded"
    assert "private" not in json.dumps(recorder.report)


@pytest.mark.parametrize("outcome", ["quarantine", "completed", "observation_failure"])
def test_driver_retains_stop_cleanup_and_incomplete_status_on_quiescence(tmp_path, monkeypatch, outcome):
    """Exercise the real driver loop/cleanup with authored owner/runtime doubles."""
    from contextlib import nullcontext
    import sys
    from types import ModuleType
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
    now = [1000.]
    stops = []
    options = []
    state = tmp_path / "state/benchmark"
    state.mkdir(parents=True)
    (state / "admission.json").write_text("{}")
    launch = state / "launch"
    launch.mkdir()
    reason = "unsettled_portal_failure_quarantine"
    heartbeat(launch, selection_idle_reason=reason)
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda *args: None)
    monkeypatch.setattr(driver.time, "monotonic", lambda: now[0])
    def tick(_):
        now[0] += .5
        assert now[0] <= 1001, "quiescent driver should not spin to its work deadline"
        heartbeat(launch, selection_idle_reason=reason, sequence=2)
    monkeypatch.setattr(driver.time, "sleep", tick)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver, "_failure_diagnostics", lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, "_final_context_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "_native_diagnostics", lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, "_refresh_completed_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: {"intent_preplanning": {}})
    monkeypatch.setattr(driver.preparation, "initial_context", lambda **kwargs: {})
    monkeypatch.setattr(driver.preparation, "plan", lambda **kwargs: {"qualified": True})
    monkeypatch.setattr(driver.preparation, "context", lambda **kwargs: {"context_bundle": {}})
    monkeypatch.setattr(driver, "verify_local_benchmark_admission", lambda *args, **kwargs: {
        "graph": SimpleNamespace(tasks=[SimpleNamespace(task_cid="task:fixture", task_key="TASK-001")]),
        "manifest": {"repository_cid": "repository:fixture", "sources": {
            driver.preparation.INSTRUCTION: {"sha256": "b" * 64}}}})
    monkeypatch.setattr(doctor, "prepare_terminal_doctor_dispatch", lambda **kwargs: {
        "route": "model_router", "status": "residual", "provider_calls": 0})
    current = task(status="completed" if outcome == "completed" else "in_progress")
    source = SimpleNamespace(get_task=lambda _: current)
    def observe():
        if outcome == "observation_failure":
            raise RuntimeError("authored failure after stop decision")
        return {}
    def create(*args, **kwargs):
        options.append(kwargs)
        return SimpleNamespace(state=launch, profile=object(),
            start=lambda: SimpleNamespace(to_dict=lambda: {"status": "succeeded"}),
            observe=observe, startup_diagnostics=lambda: {},
            stop=lambda: stops.append("stop") or SimpleNamespace(to_dict=lambda: {"status": "succeeded"}),
            close=lambda: stops.append("close"),
            process=SimpleNamespace(snapshot=lambda _: SimpleNamespace(members=[])))
    modules = {
        "benchmarks.agent_supervisor.container_coding.native_quack_qualification": {
            "open_existing_native_owner": lambda **kwargs: nullcontext(SimpleNamespace(server=object(), source=source))},
        "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime": {
            "AdmittedBenchmarkRuntime": SimpleNamespace(create=create)},
        "ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy": {"GROK_CODEX_EXECUTION_MODE": "fixture"},
        "ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction": {
            "prepare_public_instruction_context": lambda **kwargs: {"artifact": "fixture", "sha256": "a" * 64}},
    }
    for name, attrs in modules.items():
        module = ModuleType(name); vars(module).update(attrs); monkeypatch.setitem(sys.modules, name, module)
    report = driver.run(instruction=tmp_path / "instruction.md", state=state, arm="full")
    assert stops == ["stop", "close"]
    assert report["task_completed"] is (outcome == "completed")
    assert report["remaining_processes"] == 0 and report["worker_cleanup_returncode"] == 0
    assert report["native_progress"]["stop"]["status"] == "succeeded"
    assert report["native_progress"]["stop_reason"] == ("native_task_completed" if outcome == "completed" else reason)
    assert options[0]["implementation_timeout_seconds"] == 220
    assert report["implementation_timeout_seconds"] == 220
    assert report["provider_invocations"] == []
    assert json.loads((state.parent / "benchmark-result.json").read_text()) == report
