"""Real scheduler capacity joined to the existing finite native candidate route.

Task/claim/grant, Portal/lifecycle, source context, actual Popen, Unix transport
and shared scheduler lineage are genuine. Existing host fixtures author only
deployment boundary/UID admission. No provider model or target source executes;
the dispatch test uses the read/check client rather than a provider call.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import native_candidate_round_resources as resources
from ipfs_accelerate_py.agent_supervisor.runtime import native_candidate_round_worker as worker
from ipfs_accelerate_py.agent_supervisor.runtime import owner_worker_attempt as attempts
from ipfs_accelerate_py.agent_supervisor.runtime.candidate_dispatch_broker import CandidateDispatchBroker
from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_round_broker import WorkerCandidateRoundBroker
from test.api.candidate_resource_test_support import lease_record, reservation_options, resource_parent
from test.api.test_native_candidate_round_admission import candidate_case, capture, spawn_child
from test.api.test_candidate_dispatch import CHILD, POLICY, _reply, _row, _uid_seam


THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
READ_INPUT = (
    "import hashlib,json,os,sys;from pathlib import Path;"
    "raw=sys.stdin.buffer.read();"
    "Path(sys.argv[1]).write_text(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),"
    "threads={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})));"
    "raise SystemExit(0)"
)


def cleanup_actual(process):
    assert type(process) is subprocess.Popen and process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
    return True


@pytest.fixture
def candidate_resources(candidate_case, resource_parent):
    case = candidate_case
    binding = resources.reserve_native_candidate_round_resources(**reservation_options(
        resource_parent, case.worker_attempt, repository_id=case.native.server.identity.repository_id))
    value = SimpleNamespace(case=case, owner=resource_parent, binding=binding, gate=None, broker=None)
    try:
        yield value
    finally:
        if not binding.native_lease.released:
            actual = None if value.gate is None else value.gate.process
            if actual is None:
                if value.broker is not None:
                    value.broker.close()
                binding.abort_unstarted()
            else:
                if actual.poll() is None:
                    actual.kill()
                actual.wait(timeout=5)
                if value.broker is not None:
                    value.broker.close()
                binding.recover_retirement(cleanup=cleanup_actual)


def options(value, tmp_path, *, cleanup=cleanup_actual):
    case, binding = value.case, value.binding
    gate = capture(case, resources=binding)
    broker = WorkerCandidateRoundBroker(case.snapshot)
    value.gate, value.broker = gate, broker
    marker = tmp_path / "actual-candidate-input.json"
    return dict(admission=gate, source_broker=broker,
        command=[sys.executable, "-I", "-c", READ_INPUT, str(marker)],
        environment={"PATH":os.defpath, "LANG":"C.UTF-8", "PYTHONDONTWRITEBYTECODE":"1",
            **{key:"1" for key in THREAD_KEYS}}, cleanup=cleanup), marker


def test_real_resource_bound_worker_receives_exact_input_threads_and_retires(candidate_resources, tmp_path):
    value = candidate_resources
    selected, marker = options(value, tmp_path)
    leaf = value.binding.native_lease
    assert value.gate.revalidate()["quantitative_resource_reservation_verified"] is True
    result = worker.run_native_candidate_round_worker(**selected, stdout=subprocess.DEVNULL)
    assert result.returncode == 0 and value.gate.process.returncode == 0
    observed = json.loads(marker.read_bytes())
    assert observed["bytes"] == len(value.case.candidate_prompt.provider_prompt)
    assert observed["sha256"] == hashlib.sha256(value.case.candidate_prompt.provider_prompt).hexdigest()
    assert all(observed["threads"][key] == "1" for key in THREAD_KEYS)
    assert leaf.released and lease_record(value.owner, leaf.lease_id) is None
    assert not value.owner.parent.released
    assert value.case.native.source.get_task(value.case.native.task_cid).status == "in_progress"


def test_failed_cleanup_keeps_leaf_until_explicit_recovery_without_rewriting_false_census(candidate_resources, tmp_path):
    value = candidate_resources
    selected, _marker = options(value, tmp_path, cleanup=lambda process: cleanup_actual(process) and False)
    leaf = value.binding.native_lease
    with pytest.raises((ValueError, RuntimeError)):
        worker.run_native_candidate_round_worker(**selected, stdout=subprocess.DEVNULL)
    original = value.case.worker_attempt._state.retired_event
    assert original is not None and json.loads(original.raw)["cleanup_succeeded"] is False
    assert not leaf.released and lease_record(value.owner, leaf.lease_id) is not None
    called = []
    def cleanup(process):
        assert process is value.gate.process
        called.append(process)
        return cleanup_actual(process)
    receipt = value.binding.recover_retirement(cleanup=cleanup)
    assert receipt["released"] is True and receipt["retirement_mode"] == "owner-recovered"
    assert called == [value.gate.process] and value.case.worker_attempt._state.retired_event.raw == original.raw
    assert json.loads(original.raw)["cleanup_succeeded"] is False
    assert lease_record(value.owner, leaf.lease_id) is None
    with pytest.raises((ValueError, RuntimeError)):
        value.binding.recover_retirement(cleanup=cleanup)


def test_original_retirement_publication_failure_retains_leaf_for_separate_owner_recovery(candidate_resources, tmp_path, monkeypatch):
    value = candidate_resources
    selected, _marker = options(value, tmp_path)
    leaf = value.binding.native_lease
    publish = attempts.OwnerWorkerAttempt._publish
    def refuse(self, row, **kwargs):
        if self is value.case.worker_attempt and row["event"] == "retired":
            raise OSError("authored original census publication failure")
        return publish(self, row, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(attempts.OwnerWorkerAttempt, "_publish", refuse)
        with pytest.raises(OSError, match="original census publication"):
            worker.run_native_candidate_round_worker(**selected, stdout=subprocess.DEVNULL)
    assert value.gate.process.poll() is not None
    assert value.case.worker_attempt._state.retired_event is None
    assert not leaf.released and lease_record(value.owner, leaf.lease_id) is not None
    recovered = value.binding.recover_retirement(cleanup=cleanup_actual)
    assert recovered["released"] is True and recovered["retirement_mode"] == "owner-recovered"
    assert value.case.worker_attempt._state.retired_event is None
    assert lease_record(value.owner, leaf.lease_id) is None


def test_partial_resource_audit_is_preserved_and_recovery_publishes_distinct_witness(candidate_resources, tmp_path, monkeypatch):
    value = candidate_resources
    selected, _marker = options(value, tmp_path)
    leaf = value.binding.native_lease
    resource_id = json.loads(value.binding.metadata_bytes)["resource_id"]
    parent = value.case.worker_attempt._parent
    prefix = "worker-resource-" + resource_id + "-retired-"
    refused = []
    original_fsync = os.fsync
    def resource_file_fsync(descriptor):
        path = Path(os.readlink("/proc/self/fd/" + str(descriptor)))
        if (not refused and path.parent == parent and path.name.startswith(prefix)
                and path.name.endswith(".json")):
            # Only the freshly written resource audit fails. Native storage,
            # original census publication and directory fsync remain genuine.
            refused.append(path)
            raise OSError("authored partial resource audit fsync failure")
        return original_fsync(descriptor)
    with monkeypatch.context() as patch:
        patch.setattr(resources, "os", SimpleNamespace(**(dict(vars(os)) | {"fsync":resource_file_fsync})))
        with pytest.raises(OSError, match="partial resource audit"):
            worker.run_native_candidate_round_worker(**selected, stdout=subprocess.DEVNULL)
    assert len(refused) == 1 and refused[0].exists()
    partial = refused[0].read_bytes()
    original_census = value.case.worker_attempt._state.retired_event.raw
    assert not leaf.released and lease_record(value.owner, leaf.lease_id) is not None
    before = set(parent.glob(prefix + "*.json"))
    assert before == {refused[0]}
    receipt = value.binding.recover_retirement(cleanup=cleanup_actual)
    after = set(parent.glob(prefix + "*.json"))
    created = after - before
    assert len(created) == 1 and refused[0].read_bytes() == partial
    new_path = created.pop()
    new_raw = new_path.read_bytes()
    assert hashlib.sha256(new_raw).hexdigest() == receipt["retirement_event_sha256"]
    assert json.loads(new_raw)["retirement_mode"] == "owner-recovered"
    assert json.loads(new_raw)["original_census_flags_unchanged"] is True
    assert value.case.worker_attempt._state.retired_event.raw == original_census
    assert receipt["released"] is True and lease_record(value.owner, leaf.lease_id) is None


def test_suppressed_recursive_recovery_invalidates_outer_cleanup_and_keeps_leaf(candidate_resources, tmp_path):
    value = candidate_resources
    selected, _marker = options(value, tmp_path, cleanup=lambda process: cleanup_actual(process) and False)
    leaf = value.binding.native_lease
    with pytest.raises((ValueError, RuntimeError)):
        worker.run_native_candidate_round_worker(**selected, stdout=subprocess.DEVNULL)
    original = value.case.worker_attempt._state.retired_event.raw
    assert json.loads(original)["cleanup_succeeded"] is False and not leaf.released
    caught = []
    def recursive_cleanup(process):
        assert process is value.gate.process and cleanup_actual(process) is True
        try:
            value.binding.recover_retirement(cleanup=cleanup_actual)
        except ValueError:
            caught.append(True)
        return True
    with pytest.raises(ValueError, match="recursive retirement"):
        value.binding.recover_retirement(cleanup=recursive_cleanup)
    assert caught == [True] and not leaf.released
    assert lease_record(value.owner, leaf.lease_id) is not None
    assert value.case.worker_attempt._state.retired_event.raw == original
    receipt = value.binding.recover_retirement(cleanup=cleanup_actual)
    assert receipt["released"] is True and receipt["retirement_mode"] == "owner-recovered"
    assert value.case.worker_attempt._state.retired_event.raw == original
    assert lease_record(value.owner, leaf.lease_id) is None


def test_native_leaf_cancellation_interrupts_communicate_without_resending_input(candidate_resources, tmp_path, monkeypatch):
    value = candidate_resources
    selected, marker = options(value, tmp_path)
    selected["command"][3] = READ_INPUT.replace("raise SystemExit(0)", "import time;time.sleep(30)")
    communications = []
    actual_register = value.broker.note_spawned
    def register(process):
        actual_register(process)
        communicate = process.communicate
        def observed(*args, **kwargs):
            communications.append(kwargs.get("input"))
            return communicate(*args, **kwargs)
        process.communicate = observed
    monkeypatch.setattr(value.broker, "note_spawned", register)
    cancelled, errors = threading.Event(), []
    def cancel_after_actual_input():
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists(), "actual candidate stdin was never released"
            assert value.binding.native_lease.cancel()
            cancelled.set()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=cancel_after_actual_input)
    thread.start()
    try:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            worker.run_native_candidate_round_worker(**selected, stdout=subprocess.DEVNULL)
    finally:
        thread.join(timeout=12)
    assert not thread.is_alive() and not errors and cancelled.is_set()
    assert [raw for raw in communications if raw is not None] == [value.case.candidate_prompt.provider_prompt]
    assert value.gate.process.poll() is not None
    assert value.binding.native_lease.released
    assert lease_record(value.owner, value.binding.native_lease.lease_id) is None


@pytest.mark.parametrize("drift", ["run_lease", "grant"])
def test_real_native_refusal_avoids_birth_and_releases_unstarted_leaf(candidate_resources, tmp_path, drift):
    value = candidate_resources
    selected, marker = options(value, tmp_path)
    if drift == "run_lease":
        value.case.run_coordinator.release(value.case.run_lease, reason="authored resource pipeline drift")
    else:
        value.case.native.server._command_gateway.revoke_grant(value.case.native.daemon_grant_id)
    with pytest.raises((ValueError, RuntimeError)):
        worker.run_native_candidate_round_worker(**selected, stdout=subprocess.DEVNULL)
    assert value.gate.process is None and not marker.exists()
    assert value.binding.native_lease.released
    assert lease_record(value.owner, value.binding.native_lease.lease_id) is None


def test_final_run_clock_resource_cancel_after_birth_keeps_child_and_withholds_input(candidate_resources, monkeypatch):
    value, state = candidate_resources, dict(closed=False, cancelled=False)
    case = value.case
    actual_clock = case.run_coordinator._clock_ms
    def closing_clock():
        result = actual_clock()
        if state["closed"] and not state["cancelled"]:
            assert value.binding.native_lease.cancel()
            state["cancelled"] = True
        return result
    # Capture this actual original clock object; later replacement would test
    # custody drift rather than cancellation at the closing native boundary.
    monkeypatch.setattr(case.run_coordinator, "_clock_ms", closing_clock)
    gate = capture(case, resources=value.binding)
    broker = WorkerCandidateRoundBroker(case.snapshot)
    value.gate, value.broker = gate, broker
    value.binding.bind_broker(broker)
    coordinator = case.native.daemon.coordinator
    original_fence = coordinator.execute_with_task_claim_fence
    def native_closing(*args, **kwargs):
        result = original_fence(*args, **kwargs)
        state["closed"] = True
        return result
    monkeypatch.setattr(coordinator, "execute_with_task_claim_fence", native_closing)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        gate.execute_spawn(lambda: spawn_child(case), broker.note_spawned)
    assert state["cancelled"] is True
    assert gate.process is case.children[-1] and gate.process.poll() is None
    assert gate._state.spawn_completed is False and gate._state.consumed is True
    assert not value.binding.native_lease.released
    # No task input has been sent. Explicit recovery stops the retained child,
    # checks actual group disappearance, closes the broker, then releases.
    receipt = value.binding.recover_retirement(cleanup=cleanup_actual)
    assert receipt["released"] is True and gate.process.poll() is not None


def test_real_dispatch_check_cannot_ignore_cancelled_native_resource_leaf(candidate_resources, monkeypatch):
    value, case = candidate_resources, candidate_resources.case
    gate = capture(case, resources=value.binding)
    broker = CandidateDispatchBroker(admission=gate, **POLICY)
    value.gate, value.broker = gate, broker
    value.binding.bind_broker(broker)
    _uid_seam(broker, monkeypatch)
    def birth():
        child = subprocess.Popen([sys.executable, "-I", "-c", CHILD,
            str(Path(__file__).resolve().parents[2]), str(broker.descriptor_path),
            broker.descriptor_sha256, str(broker.worker_uid)], cwd=case.native.workspace,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
        case.children.append(child)
        return child
    child = gate.execute_spawn(birth, broker.note_spawned)
    assert _reply(child, _row(case))["ok"] is True
    assert value.binding.native_lease.cancel()
    reply = _reply(child, dict(operation="authorize", invocation_id=uuid.uuid4().hex))
    assert reply["ok"] is False
    assert not value.binding.native_lease.released
    assert _reply(child, dict(operation="stop"))["stopped"] is True
    child.wait(timeout=5)
    broker.close()
    case.worker_attempt.finish(waited=True, cleanup_succeeded=cleanup_actual(child), brokers_closed=True)
    receipt = value.binding.finish_retirement()
    assert receipt["released"] is True and lease_record(value.owner, value.binding.native_lease.lease_id) is None
