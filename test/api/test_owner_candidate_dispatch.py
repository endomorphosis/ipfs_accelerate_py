"""Actual failed child, cleanup, native snapshot and one sealed continuation.

Container UID/path and the pinned executable boundary are authored; Git/source
bindings, native IR production, descriptor sessions, waits and sealed stdin are
real. No provider, remote model, prover or benchmark hidden test is invoked.
"""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_prompt as prompts
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from test.api.test_owner_candidate_round_context import source_service, native, waited, CHANGED  # noqa: F401


def original_argv(service):
    return ["--semantic-repository", str(service.repository), "--source-view-profile",
        service._binding["profile"], "--timeout", "90", "--provider", "grok_cli",
        "--model", "authored-model", "--reasoning-effort", "high",
        "--max-output-tokens", "2048", "--grok-coding-max-turns", "7"]


@pytest.fixture
def captured_service(source_service, monkeypatch):
    service = source_service
    service.original_requirements = prompts.capture_native_only_requirements(source_broker=service)
    events = []
    def cleanup():
        assert all(signal.getsignal(number) is signal.SIG_IGN
                   for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP))
        events.append("cleanup")
        return subprocess.run([sys.executable, "-I", "-c", "pass"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=5)
    monkeypatch.setattr(dispatch, "_isolated_cleanup", cleanup)
    launch = dispatch.capture_original_coding_launch(source_broker=service,
        argv=original_argv(service), absolute_deadline=time.monotonic()+89)
    return launch, events


@pytest.fixture
def failed_retirement(captured_service):
    launch, events = captured_service
    service = launch.source_broker
    waited(service, close=False)
    (service.workspace / "mod.py").write_bytes(CHANGED)
    retired = dispatch.retire_coding_worker(launch=launch, process=service._process, source_broker=service)
    return launch, retired, events


@pytest.fixture
def prepared_candidate(failed_retirement):
    launch, retired, events = failed_retirement
    prepared = dispatch.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    try:
        yield launch, retired, prepared, events
    finally:
        child = prepared.source_broker._process
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        prepared.close()


def test_capture_pins_settings_and_first_command_before_any_child(captured_service):
    launch, _ = captured_service
    service = launch.source_broker
    args = original_argv(service)
    command = [*dispatch._PREFIX, *args, *service.extra_argv]
    launch.validate_first_command(command)
    settings = launch.revalidate()
    assert settings == dict(provider="grok_cli", model="authored-model", reasoning_effort="high",
        max_output_tokens=2048, budget_profile=None, grok_coding_max_turns=7)
    settings["model"] = "mutated"
    assert launch.launch_settings["model"] == "authored-model"
    command[command.index("authored-model")] = "other-model"
    with pytest.raises(ValueError, match="command differs"):
        launch.validate_first_command(command)
    with pytest.raises(FrozenInstanceError):
        launch.absolute_deadline += 1
    with pytest.raises(ValueError, match="already captured"):
        dispatch.capture_original_coding_launch(source_broker=service,
            argv=args, absolute_deadline=launch.absolute_deadline)


@pytest.mark.parametrize("extra", [
    ["--doctor-residual-artifact", "/unadmitted"], ["--doctor-candidate-artifact", "/unadmitted"],
    ["--finite-repository-artifact", "/unadmitted"], ["--finite-proof-query-context", "/unadmitted"],
    ["--numerical-obligation-descriptor", "/unadmitted"], ["--source-lookup-descriptor", "/unadmitted"],
    ["--candidate-rounds", "2"], ["--preflight"], ["--model=duplicate"],
    ["--purpose", "planning"], ["--source-edit-mode", "mapped-span@1"],
])
def test_unsupported_launch_route_refused_before_worker(source_service, extra):
    service = source_service
    service.original_requirements = prompts.capture_native_only_requirements(source_broker=service)
    with pytest.raises(ValueError):
        dispatch.capture_original_coding_launch(source_broker=service,
            argv=original_argv(service)+extra, absolute_deadline=time.monotonic()+89)
    assert not service._registered.is_set() and service._process is None


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, -1, 10**500])
def test_invalid_original_deadline_refused_before_worker(source_service, deadline):
    source_service.original_requirements = prompts.capture_native_only_requirements(source_broker=source_service)
    with pytest.raises(ValueError, match="deadline"):
        dispatch.capture_original_coding_launch(source_broker=source_service,
            argv=original_argv(source_service), absolute_deadline=deadline)


def test_cleanup_is_actual_fixed_command_without_caller_supplied_boolean(monkeypatch):
    calls = []
    def run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(dispatch, "subprocess", SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL))
    assert dispatch._isolated_cleanup().returncode == 0
    assert calls == [(list(dispatch._PREFIX)+["--cleanup"], dict(cwd="/", stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, start_new_session=True, timeout=15))]


def test_unknown_child_runs_cleanup_and_closes_broker_without_retirement(captured_service):
    launch, events = captured_service
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    with pytest.raises(ValueError, match="registered waited"):
        dispatch.retire_coding_worker(launch=launch, process=None, source_broker=launch.source_broker)
    assert events == ["cleanup"] and launch.source_broker._closed
    assert launch._state.initial_retirement is None
    assert all(signal.getsignal(number) is handler for number, handler in previous.items())


def test_failed_cleanup_cannot_mint_retirement(captured_service, monkeypatch):
    launch, _ = captured_service
    service = launch.source_broker
    waited(service, close=False)
    monkeypatch.setattr(dispatch, "_isolated_cleanup", lambda: subprocess.run(
        [sys.executable, "-I", "-c", "raise SystemExit(4)"], timeout=5))
    with pytest.raises(ValueError, match="cleanup failed"):
        dispatch.retire_coding_worker(launch=launch, process=service._process, source_broker=service)
    assert service._closed and launch._state.initial_retirement is None


def test_success_cannot_prepare_second_worker(captured_service):
    launch, _ = captured_service
    service = launch.source_broker
    waited(service, code=0, close=False)
    retired = dispatch.retire_coding_worker(launch=launch, process=service._process, source_broker=service)
    with pytest.raises(ValueError, match="positive first"):
        dispatch.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    assert launch._state.phase == "captured"


def test_killed_worker_cannot_prepare_second_worker(captured_service):
    launch, _ = captured_service
    service = launch.source_broker
    child = subprocess.Popen([sys.executable, "-I", "-c", "import time;time.sleep(30)"])
    service.note_spawned(child)
    child.kill()
    child.wait(timeout=5)
    retired = dispatch.retire_coding_worker(launch=launch, process=child, source_broker=service)
    with pytest.raises(ValueError, match="positive first"):
        dispatch.prepare_second_coding_worker(launch=launch, retired_worker=retired)


def test_native_preparation_is_one_use_and_does_not_promote_advisory_objects(prepared_candidate):
    launch, retired, prepared, events = prepared_candidate
    assert events == ["cleanup"]
    assert prepared.snapshot.metadata["next_round_admitted"] is False
    assert prepared.prompt.dispatch_usable is False
    assert prepared.snapshot._absolute_deadline == launch.absolute_deadline
    binding = prepared.admission.revalidate(snapshot=prepared.snapshot, prompt=prepared.prompt,
        source_broker=prepared.source_broker)
    assert binding["launch_settings"] == launch.launch_settings
    assert binding["prior_session_id"] == launch.source_broker.descriptor["session_id"]
    assert binding["session_id"] != binding["prior_session_id"]
    assert binding["deadline_monotonic_ns"] == int(launch.absolute_deadline * 1e9)
    assert prepared.source_broker.descriptor["expires_monotonic_ns"] <= binding["deadline_monotonic_ns"]
    with pytest.raises(ValueError, match="claim already consumed"):
        dispatch.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    with pytest.raises(ValueError, match="live captured"):
        dispatch.prepare_second_coding_worker(launch=launch, retired_worker={"cleanup_succeeded": True})


def test_second_child_gets_sealed_stdin_pinned_settings_and_distinct_custody(prepared_candidate, monkeypatch):
    launch, retired, prepared, events = prepared_candidate
    commands = []
    target = launch.source_broker.workspace / "authored-second-input.json"
    script = """import fcntl,hashlib,json,pathlib,sys,time
raw=sys.stdin.buffer.read()
result={'sha256':hashlib.sha256(raw).hexdigest(),'seals':fcntl.fcntl(0,1034)}
pathlib.Path(sys.argv[1]).write_text(json.dumps(result))
time.sleep(.1)
raise SystemExit(5)
"""
    def spawn(command, *, stdin, workspace):
        commands.append(command)
        return subprocess.Popen([sys.executable, "-I", "-c", script, str(target)], stdin=stdin, cwd=workspace)
    monkeypatch.setattr(dispatch, "_spawn_worker", spawn)
    running = prepared.start()
    assert running.process.wait(timeout=running.remaining_seconds) == 5
    second = dispatch.retire_coding_worker(launch=launch, process=running.process,
        source_broker=running.source_broker)
    assert second.round_number == 2 and second.process is not retired.process
    assert json.loads(target.read_text()) == dict(sha256=hashlib.sha256(prepared.prompt.provider_prompt).hexdigest(), seals=15)
    command = commands[0]
    assert command[:6] == list(dispatch._PREFIX)
    assert command[command.index("--model")+1] == "authored-model"
    assert command[command.index("--grok-coding-max-turns")+1] == "7"
    assert 0 < int(command[command.index("--timeout")+1]) < 90
    assert "--candidate-round-descriptor" in command and "--candidate-dispatch-handoff" in command
    assert "--source-view-descriptor" not in command and "--source-formalization-handoff" not in command
    assert events == ["cleanup", "cleanup"]
    with pytest.raises(ValueError, match="already consumed"):
        prepared.start()
    prepared.close()
    prepared.close()
    assert not prepared.handoff.path.exists()


def test_candidate_drift_prevents_spawn_and_removes_prepared_transport(prepared_candidate, monkeypatch):
    launch, _, prepared, events = prepared_candidate
    (launch.source_broker.workspace / "mod.py").write_bytes(CHANGED+b"\n# drift\n")
    monkeypatch.setattr(dispatch, "_spawn_worker", lambda *a, **kw: pytest.fail("stale candidate spawned"))
    with pytest.raises(ValueError):
        prepared.start()
    assert prepared.source_broker._closed and not prepared.handoff.path.exists()
    assert events == ["cleanup", "cleanup"]


def test_deadline_exhausted_after_preparation_prevents_spawn(prepared_candidate, monkeypatch):
    launch, _, prepared, _ = prepared_candidate
    monkeypatch.setattr(dispatch, "time", SimpleNamespace(monotonic=lambda: launch.absolute_deadline+1))
    monkeypatch.setattr(dispatch, "_spawn_worker", lambda *a, **kw: pytest.fail("expired lease spawned"))
    with pytest.raises(ValueError, match="deadline exhausted"):
        prepared.start()
    assert prepared.source_broker._closed and not prepared.handoff.path.exists()


def test_registration_failure_reaps_actual_partial_child_and_consumes_claim(prepared_candidate, monkeypatch):
    _, _, prepared, events = prepared_candidate
    children = []
    def spawn(command, *, stdin, workspace):
        child = subprocess.Popen([sys.executable, "-I", "-c", "import time;time.sleep(30)"], stdin=stdin, cwd=workspace)
        children.append(child)
        return child
    def refused(child):
        raise ValueError("authored registration refusal")
    original_terminate = dispatch._terminate
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    def interrupted_reap(child):
        assert all(signal.getsignal(number) is signal.SIG_IGN for number in previous)
        original_terminate(child)
        raise InterruptedError("authored secondary teardown interruption")
    monkeypatch.setattr(dispatch, "_spawn_worker", spawn)
    monkeypatch.setattr(dispatch, "_terminate", interrupted_reap)
    monkeypatch.setattr(prepared.source_broker, "note_spawned", refused)
    with pytest.raises(ValueError, match="registration refusal"):
        prepared.start()
    assert len(children) == 1 and children[0].poll() is not None
    assert prepared.source_broker._closed and not prepared.handoff.path.exists()
    assert events == ["cleanup", "cleanup"]
    assert all(signal.getsignal(number) is handler for number, handler in previous.items())
    with pytest.raises(ValueError, match="already consumed"):
        prepared.start()


def test_unused_preparation_close_blocks_spawn(prepared_candidate):
    _, _, prepared, _ = prepared_candidate
    prepared.close()
    with pytest.raises(ValueError, match="already consumed"):
        prepared.start()
    assert prepared.source_broker._closed and not prepared.handoff.path.exists()


def test_retirement_replay_cannot_trigger_global_cleanup_of_next_round(prepared_candidate):
    launch, retired, _, events = prepared_candidate
    with pytest.raises(ValueError, match="already retired"):
        dispatch.retire_coding_worker(launch=launch, process=retired.process,
            source_broker=launch.source_broker)
    assert events == ["cleanup"]


def test_waited_second_child_requires_real_retirement_before_close(prepared_candidate, monkeypatch):
    launch, _, prepared, events = prepared_candidate
    monkeypatch.setattr(dispatch, "_spawn_worker", lambda command, *, stdin, workspace:
        subprocess.Popen([sys.executable, "-I", "-c", "import time;time.sleep(.1)"], stdin=stdin, cwd=workspace))
    running = prepared.start()
    running.process.wait(timeout=5)
    with pytest.raises(ValueError, match="still requires isolated"):
        prepared.close()
    assert not prepared.source_broker._closed and prepared.handoff.path.exists()
    dispatch.retire_coding_worker(launch=launch, process=running.process, source_broker=running.source_broker)
    prepared.close()
    assert events == ["cleanup", "cleanup"]


def test_preparation_failure_consumes_original_broker_claim(failed_retirement):
    launch, retired, _ = failed_retirement
    path = launch.source_broker.workspace / "mod.py"
    path.unlink()
    with pytest.raises((ValueError, OSError)):
        dispatch.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    path.write_bytes(CHANGED)
    with pytest.raises(ValueError, match="claim already consumed"):
        dispatch.prepare_second_coding_worker(launch=launch, retired_worker=retired)


def test_owner_captures_real_initial_formalization_handoff_without_worker_identity(source_service, monkeypatch):
    """Run the owner factory and sealed-prompt broker before owner capture.

    Only an authored native selection and container boundary are substituted;
    the worker UID validator is expressly forbidden in this owner-side path.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as brokers
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import _validation_memfd_create
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
    service = source_service
    wire, end = json.JSONDecoder().raw_decode(service.prompt)
    sources = {name: hashlib.sha256((service.repository/name).read_bytes()).hexdigest()
        for name in ("mod.py", "harness.py")}
    summary = dict(schema="authored-owner-summary@1", proof_authority=False)
    context = dict(kind="source384-candidate-context",
        receipt_reference=dict(schema="authored-owner-reference@1", output="/authored/selection",
            receipt_sha256="a"*64, receipt_bytes=200),
        source_hashes_sha256=hashlib.sha256(guard._json(sources).encode()).hexdigest(),
        summary_sha256=hashlib.sha256(guard._json(summary).encode()).hexdigest(), summary=summary)
    payload = dict(schema=guard.WORKER_EVIDENCE_SCHEMA, repository=str(service.repository),
        task_cid="task:authored", task_id=wire["objective_id"],
        context_bundle=dict(artifact=".runtime/authored-formalization.json", sha256="b"*64),
        contexts=[context], proof_authority=False, formalization_authority=False,
        execution_authority=False, completion_authority=False, planning_authority=False,
        dispatch_authority=False, source_semantics_verified=False, whole_program_semantics_verified=False,
        source_obligations_discharged=0)
    references = build_text_context_references(guard._json(payload), reference_prefix="source-formalization",
        kind=guard.KIND, path=payload["context_bundle"]["artifact"], repository_id=wire["repository_id"],
        tree_id=wire["tree_id"], required=True, chunk_bytes=6144)
    wire["evidence"].extend(row.to_record() for row in references)
    combined = guard._json(wire)+service.prompt[end:]
    monkeypatch.setattr(guard, "load_task_context_selection", lambda **kwargs: dict(
        worker_evidence=payload, source384_context=dict(source_hashes=sources)))
    monkeypatch.setattr(guard, "_worker_boundary", lambda *a, **kw: pytest.fail("owner entered worker UID gate"))
    service.close()
    fd = _validation_memfd_create("authored-owner-initial", 0)
    os.write(fd, combined.encode())
    os.lseek(fd, 0, os.SEEK_SET)
    initial = None
    fresh = None
    try:
        initial = guard.prepare_container_source_formalization_dispatch(argv=original_argv(service),
            boundary=service.boundary, workspace=service.workspace, input_fd=fd)
        fresh = brokers.prepare_worker_source_broker(argv=initial["argv"], boundary=service.boundary,
            workspace=service.workspace, input_fd=initial["stdin_fd"])
        launch = dispatch.capture_original_coding_launch(source_broker=fresh, argv=initial["argv"],
            absolute_deadline=time.monotonic()+89)
        launch.validate_first_command([*dispatch._PREFIX, *initial["argv"], *fresh.extra_argv])
        assert launch.source_broker.prompt == combined
        assert "--source-formalization-handoff" in launch._argv
    finally:
        os.close(fd)
        if initial is not None:
            os.close(initial["stdin_fd"])
        if fresh is not None:
            fresh.close()
