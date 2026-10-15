"""Real host children under the native candidate gate, without a provider.

Native task/claim/grant, allocation/lifecycle, DS context, run lease and owner
event files are real. Container admission and separate-UID cleanup are authored
host seams. These inert children read stdin; they never execute target source.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import native_candidate_round_worker as worker
from ipfs_accelerate_py.agent_supervisor.runtime import owner_worker_attempt as attempts
from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_round_broker import WorkerCandidateRoundBroker
from test.api.test_native_candidate_round_admission import candidate_case, capture  # noqa: F401


CHILD = (
    "import hashlib,json,sys;from pathlib import Path;"
    "raw=sys.stdin.buffer.read();"
    "Path(sys.argv[1]).write_text('input released');"
    "print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())),flush=True);"
    "raise SystemExit(int(sys.argv[2]))"
)


def _options(case, tmp_path, *, code=0, cleanup_result=True):
    gate = capture(case)
    broker = WorkerCandidateRoundBroker(case.snapshot)
    observed = []
    marker = tmp_path / "authored-input-release"

    def cleanup(process):
        assert process is gate.process and type(process) is subprocess.Popen
        assert process.returncode is not None
        # The authored child creates no descendants. Its actual process group
        # must be absent after wait; this is not a Docker UID cleanup proof.
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
        observed.append(process)
        return cleanup_result

    options = dict(admission=gate, source_broker=broker,
        command=[sys.executable, "-I", "-c", CHILD, str(marker), str(code)],
        environment={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
        cleanup=cleanup)
    return options, marker, observed


def _retired(case):
    event = case.worker_attempt._state.retired_event
    assert event is not None
    return json.loads(event.raw)


@pytest.mark.parametrize("code", [0, 7])
def test_fenced_real_second_child_gets_exact_candidate_input_and_retires(candidate_case, tmp_path, code):
    case = candidate_case
    options, marker, cleaned = _options(case, tmp_path, code=code)
    log = tmp_path / "authored-child.log"
    with log.open("wb") as stream:
        result = worker.run_native_candidate_round_worker(**options, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == code
    actual = options["admission"].process
    assert len(cleaned) == 1 and cleaned[0] is actual
    assert actual.pid != case.first_process.pid
    assert marker.read_text() == "input released"
    assert json.loads(log.read_bytes()) == dict(bytes=len(case.candidate_prompt.provider_prompt),
        sha256=hashlib.sha256(case.candidate_prompt.provider_prompt).hexdigest())
    event = _retired(case)
    assert event["waited"] is event["cleanup_succeeded"] is event["brokers_closed"] is True
    assert event["child_returncode"] == code
    assert event["native_prompt_sha256"] == hashlib.sha256(case.candidate_prompt.provider_prompt).hexdigest()
    assert event["absolute_deadline_ns"] == json.loads(case.prior_worker_attempt._state.prepared.raw)["absolute_deadline_ns"]
    assert not options["source_broker"].socket_path.exists()
    assert not options["source_broker"].descriptor_path.exists()
    assert case.native.source.get_task(case.native.task_cid).status == "in_progress"
    assert case.candidate_prompt.dispatch_usable is False
    # A second call cannot clean, retire or replace the earlier child.
    with pytest.raises(ValueError, match="unused"):
        worker.run_native_candidate_round_worker(**options)
    assert cleaned == [actual]


def test_closing_native_refusal_reaps_actual_child_without_releasing_task_input(candidate_case, tmp_path, monkeypatch):
    case = candidate_case
    options, marker, cleaned = _options(case, tmp_path)
    broker = options["source_broker"]
    original = broker.note_spawned

    def revoke(process):
        original(process)
        case.native.server._command_gateway.revoke_grant(case.native.daemon_grant_id)

    monkeypatch.setattr(broker, "note_spawned", revoke)
    with pytest.raises((ValueError, RuntimeError)):
        worker.run_native_candidate_round_worker(**options)
    assert not marker.exists()
    process = options["admission"].process
    assert cleaned == [process] and process.returncode is not None
    event = _retired(case)
    assert event["waited"] is False and event["child_returncode"] is None
    assert event["cleanup_succeeded"] is event["brokers_closed"] is True


def test_failed_descendant_cleanup_preserves_actual_exit_and_refuses_success(candidate_case, tmp_path):
    options, _, cleaned = _options(candidate_case, tmp_path, cleanup_result=False)
    result = worker.run_native_candidate_round_worker(**options, stdout=subprocess.DEVNULL)
    assert result.returncode == 1 and len(cleaned) == 1
    event = _retired(candidate_case)
    assert event["child_returncode"] == 0 and event["waited"] is True
    assert event["cleanup_succeeded"] is False and event["brokers_closed"] is True


def test_late_source_observer_cannot_revoke_authority_then_release_input(candidate_case, tmp_path, monkeypatch):
    case = candidate_case
    options, marker, cleaned = _options(case, tmp_path)
    gate, broker = options["admission"], options["source_broker"]
    original = broker._current_context

    def closing(**kwargs):
        result = original(**kwargs)
        if gate.process is not None and gate._state.spawn_completed:
            case.native.server._command_gateway.revoke_grant(case.native.daemon_grant_id)
        return result

    monkeypatch.setattr(broker, "_current_context", closing)
    with pytest.raises((ValueError, RuntimeError)):
        worker.run_native_candidate_round_worker(**options)
    assert not marker.exists() and len(cleaned) == 1
    assert _retired(case)["waited"] is False


def test_original_remaining_budget_bounds_wait_and_reaps_timed_out_child(candidate_case, tmp_path, monkeypatch):
    options, marker, cleaned = _options(candidate_case, tmp_path)
    options["command"] = [sys.executable, "-I", "-c",
        "import sys,time;sys.stdin.buffer.read();time.sleep(30)"]
    # Author only the remaining wait allowance; native lease/source clocks
    # retain their real time and original deadline throughout admission.
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: candidate_case.deadline - .05))
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run_native_candidate_round_worker(**options)
    assert len(cleaned) == 1 and cleaned[0].returncode is not None
    assert not marker.exists()
    event = _retired(candidate_case)
    assert event["waited"] is False and event["child_returncode"] is None


def test_retirement_publication_failure_preserves_birth_and_original_error(candidate_case, tmp_path, monkeypatch):
    case = candidate_case
    options, marker, cleaned = _options(case, tmp_path)
    original = attempts.OwnerWorkerAttempt._publish

    def refuse_retirement(self, value, **kwargs):
        if value["event"] == "retired":
            raise OSError("authored retirement publication failure")
        return original(self, value, **kwargs)

    def refuse_register(process):
        raise RuntimeError("authored original child registration failure")

    monkeypatch.setattr(attempts.OwnerWorkerAttempt, "_publish", refuse_retirement)
    monkeypatch.setattr(options["source_broker"], "note_spawned", refuse_register)
    with pytest.raises(RuntimeError, match="original child registration failure"):
        worker.run_native_candidate_round_worker(**options)
    assert len(cleaned) == 1 and not marker.exists()
    assert case.worker_attempt._state.retired is True
    assert case.worker_attempt._state.retired_event is None
    assert case.worker_attempt._state.spawned is not None


def test_explicit_child_environment_cannot_inherit_database_authority(candidate_case, tmp_path):
    options, marker, cleaned = _options(candidate_case, tmp_path)
    options["environment"]["IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN"] = "authored-denied-token"
    try:
        with pytest.raises(ValueError, match="database authority"):
            worker.run_native_candidate_round_worker(**options)
        assert options["admission"].process is None and cleaned == [] and not marker.exists()
    finally:
        options["source_broker"].close()


def test_unbounded_output_capture_is_refused_before_birth(candidate_case, tmp_path):
    options, marker, cleaned = _options(candidate_case, tmp_path)
    try:
        with pytest.raises(ValueError, match="unbounded pipe"):
            worker.run_native_candidate_round_worker(**options, stdout=subprocess.PIPE)
        assert options["admission"].process is None and not marker.exists() and not cleaned
    finally:
        options["source_broker"].close()


def test_close_return_without_thread_retirement_keeps_custody_incomplete(candidate_case, tmp_path, monkeypatch):
    options, _, cleaned = _options(candidate_case, tmp_path)
    broker = options["source_broker"]
    close = broker.close
    original_thread = broker._thread
    release = threading.Event()
    stalled = threading.Thread(target=release.wait)
    stalled.start()

    def incomplete_close():
        close()
        # Author only a still-live service thread after a normal close return.
        # The production close, original thread and private paths retire first.
        broker._thread = stalled

    monkeypatch.setattr(broker, "close", incomplete_close)
    try:
        with pytest.raises(ValueError, match="did not fully retire"):
            worker.run_native_candidate_round_worker(**options, stdout=subprocess.DEVNULL)
        assert len(cleaned) == 1
        event = _retired(candidate_case)
        assert event["waited"] is event["cleanup_succeeded"] is True
        assert event["brokers_closed"] is False
    finally:
        release.set()
        stalled.join(timeout=5)
        broker._thread = original_thread
