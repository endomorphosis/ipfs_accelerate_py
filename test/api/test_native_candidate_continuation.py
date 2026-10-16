"""One owner composes the existing native second-child path, without a provider.

Native claims, grants, allocation, run leases, private census, source generation,
Unix endpoints, quantitative subleases and Popen births are real. The inherited
fixture authors deployment admission and this same-UID child's empty process
group cleanup. These tests do not qualify a container or transfer ownership.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import native_candidate_continuation as continuation
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_broker
from ipfs_accelerate_py.agent_supervisor.runtime import owner_worker_attempt as attempts
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_artifacts
from test.api.candidate_resource_test_support import lease_record, resource_parent
from test.api.test_native_candidate_round_admission import _candidate_case, candidate_case


POLICY = dict(provider="codex_cli", model="authored-continuation-no-provider", timeout=30,
    max_output_tokens=128, reasoning_effort="high")
READ_INPUT = (
    "import hashlib,json,os,sys;from pathlib import Path;"
    "raw=sys.stdin.buffer.read();"
    "Path(sys.argv[1]).write_text(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),"
    "argv=sys.argv[3:],threads={key:os.environ.get(key) for key in "
    "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})));"
    "raise SystemExit(int(sys.argv[2]))"
)
ENVIRONMENT = dict(PATH=os.defpath, LANG="C.UTF-8", PYTHONDONTWRITEBYTECODE="1")


def arguments(case, owner, **changes):
    return dict(allocation=case.allocation, run_coordinator=case.run_coordinator,
        run_lease=case.run_lease, parent_lease=owner.parent, source_broker=case.source_broker,
        prior_worker_attempt=case.prior_worker_attempt, requirements=case.original_requirements,
        absolute_deadline=case.deadline, max_worker_children=2, cpu_slots=1, memory_mb=512,
        child_process_slots=1, threads_per_process=1, **POLICY) | changes


def cleanup_actual(process):
    assert type(process) is subprocess.Popen and process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
    return True


def children(owner):
    """Inspect only this test's quantitative children, excluding private keys."""
    records = json.loads(owner.scheduler.state_path.read_bytes())["leases"]
    return {name: {key: value for key, value in record.items() if key != "lease_key"}
        for name, record in records.items() if record.get("parent_lease_id") == owner.parent.lease_id}


def release_retained(session):
    binding = session.resources
    if binding is None or binding._observation()["released"]:
        return
    gate = session.admission
    actual = None if gate is None else gate.process
    if actual is None:
        session.abort_unstarted()
    else:
        if actual.poll() is None:
            actual.kill()
        actual.wait(timeout=5)
        session.recover_retirement(cleanup=cleanup_actual)


@contextmanager
def prepared(case, owner, **changes):
    session = continuation.prepare_native_candidate_continuation(**arguments(case, owner, **changes))
    try:
        yield session
    finally:
        release_retained(session)


def run_options(session, tmp_path, *, code=0, cleanup=cleanup_actual):
    marker = tmp_path / "actual-continuation-input.json"
    return dict(command=[sys.executable, "-I", "-c", READ_INPUT, str(marker), str(code)],
        environment=dict(ENVIRONMENT), cleanup=cleanup, stdout=subprocess.DEVNULL), marker


def assert_generation_not_sampled(monkeypatch):
    monkeypatch.setattr(worker_candidate_artifacts, "build_candidate_artifacts",
        lambda **_kwargs: pytest.fail("invalid continuation sampled candidate generation"))


@pytest.mark.parametrize("code", [0, 7])
def test_real_composed_second_child_preserves_input_census_and_parent_capacity(
        candidate_case, resource_parent, tmp_path, code):
    case, owner = candidate_case, resource_parent
    parent_before = lease_record(owner, owner.parent.lease_id)
    with prepared(case, owner) as session:
        assert session.source_broker is case.source_broker
        assert session.prior_worker_attempt is case.prior_worker_attempt
        assert session.worker_attempt is not case.worker_attempt
        assert session.admission.resources is session.resources
        assert session.admission.candidate_prompt is session.candidate_prompt
        assert session.broker._admission is session.admission
        assert session.candidate_prompt.dispatch_usable is False
        prompt = session.candidate_prompt.provider_prompt
        payload = json.loads(prompt)
        assert payload["native_prompt"] == case.native_prompt
        assert session.resources.native_lease.parent_lease_id == owner.parent.lease_id
        assert len(children(owner)) == 1
        selected, marker = run_options(session, tmp_path, code=code)
        completed = session.run(**selected)
        assert completed.returncode == code
        actual = session.admission.process
        assert actual.pid != case.first_process.pid and actual.poll() == code
        received = json.loads(marker.read_bytes())
        assert received["bytes"] == len(prompt)
        assert received["sha256"] == hashlib.sha256(prompt).hexdigest()
        assert all(value == "1" for value in received["threads"].values())
        assert received["argv"] == session.extra_argv
        assert received["argv"].count("--owner-worker-attempt-id") == 1
        assert received["argv"][-1] == session.worker_attempt.attempt_id
        retired = json.loads(session.worker_attempt._state.retired_event.raw)
        prior = json.loads(case.prior_worker_attempt._state.prepared.raw)
        assert retired["absolute_deadline_ns"] == prior["absolute_deadline_ns"]
        assert retired["child_returncode"] == code
        assert retired["waited"] is retired["cleanup_succeeded"] is retired["brokers_closed"] is True
        assert session.resources.native_lease.released and children(owner) == {}
        assert case.native.source.get_task(case.native.task_cid).status == "in_progress"
        with pytest.raises((ValueError, RuntimeError)):
            session.run(**selected)
        with pytest.raises((ValueError, RuntimeError)):
            continuation.prepare_native_candidate_continuation(**arguments(case, owner))
    parent_after = lease_record(owner, owner.parent.lease_id)
    for field in ("cpu_slots", "memory_mb", "child_process_slots", "parent_lease_id", "owner_pid"):
        assert parent_after[field] == parent_before[field]
    assert not owner.parent.released


def test_failed_cleanup_retains_composed_capacity_until_explicit_recovery(
        candidate_case, resource_parent, tmp_path):
    with prepared(candidate_case, resource_parent) as session:
        selected, _marker = run_options(session, tmp_path,
            cleanup=lambda process: cleanup_actual(process) and False)
        with pytest.raises((ValueError, RuntimeError)):
            session.run(**selected)
        original = session.worker_attempt._state.retired_event.raw
        assert json.loads(original)["cleanup_succeeded"] is False
        assert not session.resources.native_lease.released
        assert len(children(resource_parent)) == 1
        receipt = session.recover_retirement(cleanup=cleanup_actual)
        assert receipt["released"] is True
        assert session.worker_attempt._state.retired_event.raw == original
        assert children(resource_parent) == {}
        with pytest.raises((ValueError, RuntimeError)):
            session.run(**selected)


def test_composed_prebirth_grant_refusal_releases_leaf_and_keeps_run_spent(
        candidate_case, resource_parent, tmp_path):
    with prepared(candidate_case, resource_parent) as session:
        selected, marker = run_options(session, tmp_path)
        candidate_case.native.server._command_gateway.revoke_grant(candidate_case.native.daemon_grant_id)
        with pytest.raises((ValueError, RuntimeError)):
            session.run(**selected)
        assert session.admission.process is None and not marker.exists()
        assert session.resources.native_lease.released and children(resource_parent) == {}
        with pytest.raises((ValueError, RuntimeError)):
            session.run(**selected)


@pytest.mark.parametrize("drift", ["grant", "run_lease"])
def test_stale_native_authority_refuses_before_candidate_generation(
        candidate_case, resource_parent, monkeypatch, drift):
    if drift == "grant":
        candidate_case.native.server._command_gateway.revoke_grant(candidate_case.native.daemon_grant_id)
    else:
        candidate_case.run_coordinator.release(candidate_case.run_lease,
            reason="authored continuation pre-generation refusal")
    assert_generation_not_sampled(monkeypatch)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        continuation.prepare_native_candidate_continuation(**arguments(candidate_case, resource_parent))
    assert children(resource_parent) == {}


@pytest.mark.parametrize("flags", [None, dict(waited=False), dict(cleanup_succeeded=False), dict(brokers_closed=False)])
def test_unretired_or_incomplete_first_child_refuses_before_candidate_generation(
        tmp_path, monkeypatch, resource_parent, flags):
    with _candidate_case(tmp_path, monkeypatch, prior_flags=flags) as case:
        assert_generation_not_sampled(monkeypatch)
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            continuation.prepare_native_candidate_continuation(**arguments(case, resource_parent))
        assert children(resource_parent) == {}


@pytest.mark.parametrize("budget", [1, 3, True])
def test_exact_explicit_two_child_budget_is_required_before_generation(
        candidate_case, resource_parent, monkeypatch, budget):
    assert_generation_not_sampled(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        continuation.prepare_native_candidate_continuation(
            **arguments(candidate_case, resource_parent, max_worker_children=budget))
    assert children(resource_parent) == {}


@pytest.mark.parametrize("offset", [-1.0, 1.0])
def test_original_absolute_deadline_is_retained_without_shortening_or_restart(
        candidate_case, resource_parent, monkeypatch, offset):
    assert_generation_not_sampled(monkeypatch)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        continuation.prepare_native_candidate_continuation(
            **arguments(candidate_case, resource_parent, absolute_deadline=candidate_case.deadline + offset))
    assert children(resource_parent) == {}


def test_generation_failure_spends_preparation_and_preserves_original_error(
        candidate_case, resource_parent, monkeypatch):
    original_error = RuntimeError("authored continuation generation failure")
    def refused(**_kwargs):
        raise original_error
    with monkeypatch.context() as patch:
        patch.setattr(worker_candidate_artifacts, "build_candidate_artifacts", refused)
        with pytest.raises(RuntimeError) as observed:
            continuation.prepare_native_candidate_continuation(**arguments(candidate_case, resource_parent))
        assert observed.value is original_error
    session = observed.value.candidate_continuation
    session.abort_unstarted()
    assert children(resource_parent) == {}
    with pytest.raises((ValueError, RuntimeError)):
        continuation.prepare_native_candidate_continuation(**arguments(candidate_case, resource_parent))


@pytest.mark.parametrize("stage", ["pre_listener", "descriptor_publication", "running_thread"])
def test_endpoint_preparation_failure_retains_actual_capacity_for_explicit_abort(
        candidate_case, resource_parent, monkeypatch, stage):
    original_error = RuntimeError("authored candidate endpoint initialization failure")
    original_constructor = continuation.CandidateDispatchBroker.__init__
    original_write = numerical_obligation_broker._write_new
    def refused(self, **_kwargs):
        raise original_error
    def refused_after_service(self, **kwargs):
        original_constructor(self, **kwargs)
        assert self._thread.is_alive() and self.socket_path.exists()
        raise original_error
    def refused_descriptor(path, value, mode):
        if path.name.startswith("candidate-dispatch-") and not path.name.endswith("-audit.json"):
            raise original_error
        return original_write(path, value, mode)
    with monkeypatch.context() as patch:
        if stage == "pre_listener":
            patch.setattr(continuation.CandidateDispatchBroker, "__init__", refused)
        elif stage == "running_thread":
            patch.setattr(continuation.CandidateDispatchBroker, "__init__", refused_after_service)
        else:
            patch.setattr(numerical_obligation_broker, "_write_new", refused_descriptor)
        with pytest.raises(RuntimeError) as observed:
            continuation.prepare_native_candidate_continuation(**arguments(candidate_case, resource_parent))
        assert observed.value is original_error
    session = observed.value.candidate_continuation
    try:
        assert session.admission.process is None
        assert session.resources.native_lease.released is False
        assert session.metadata["resource_capacity_held"] is True
        assert len(children(resource_parent)) == 1
        if stage == "running_thread":
            assert session.broker._thread.is_alive()
        elif stage == "descriptor_publication":
            assert session.broker._thread is None
            # The existing broker closes transport before its incomplete
            # audit raises. That first failure must retain capacity; a second
            # explicit abort can establish positive, already-closed custody.
            with pytest.raises(AttributeError) as abort_error:
                session.abort_unstarted()
            assert abort_error.value.candidate_continuation is session
            assert original_error.candidate_continuation is session
            assert session.broker._closed and session.broker._stop.is_set()
            assert not session.broker.socket_path.exists()
            assert not session.broker.descriptor_path.exists()
            assert session.resources._observation()["released"] is False
            assert session.metadata["resource_capacity_held"] is True
            assert len(children(resource_parent)) == 1
        session.abort_unstarted()
        assert session.resources.native_lease.released and children(resource_parent) == {}
        if stage != "pre_listener":
            assert session.broker._closed and session.broker._stop.is_set()
            assert session.broker._thread is None or not session.broker._thread.is_alive()
            assert not session.broker.socket_path.exists()
            assert not session.broker.descriptor_path.exists()
        with pytest.raises((ValueError, RuntimeError)):
            continuation.prepare_native_candidate_continuation(**arguments(candidate_case, resource_parent))
    finally:
        release_retained(session)


def test_generation_callback_cannot_replace_the_original_parent_budget(
        candidate_case, resource_parent, monkeypatch):
    original_builder = continuation.contexts.prepare_candidate_round_context
    original_memory = resource_parent.parent.memory_mb
    sampled = []
    def changed_parent(**kwargs):
        snapshot = original_builder(**kwargs)
        sampled.append(True)
        resource_parent.parent.memory_mb = original_memory + 1
        return snapshot
    def never_reserve(**_kwargs):
        pytest.fail("changed parent budget reached child resource reservation")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(continuation.contexts, "prepare_candidate_round_context", changed_parent)
            patch.setattr(continuation.reservations, "reserve_native_candidate_round_resources", never_reserve)
            with pytest.raises((ValueError, RuntimeError, TimeoutError)) as observed:
                continuation.prepare_native_candidate_continuation(**arguments(candidate_case, resource_parent))
            assert sampled == [True]
    finally:
        resource_parent.parent.memory_mb = original_memory
    assert children(resource_parent) == {}
    observed.value.candidate_continuation.abort_unstarted()
    with pytest.raises((ValueError, RuntimeError)):
        continuation.prepare_native_candidate_continuation(**arguments(candidate_case, resource_parent))


def test_local_release_acknowledgement_does_not_clear_retained_native_capacity(
        candidate_case, resource_parent):
    with prepared(candidate_case, resource_parent) as session:
        leaf = session.resources.native_lease
        original = leaf._released
        try:
            # A local acknowledgement can precede verified removal of a native
            # row. Preserve this actual row and the resource kernel's custody.
            leaf._released = True
            assert session.resources._observation()["released"] is False
            assert lease_record(resource_parent, leaf.lease_id) is not None
            session._keep_pending()
            assert session.metadata["resource_capacity_held"] is True
            assert continuation._PENDING[session.metadata["session_id"]] is session
        finally:
            leaf._released = original
        session.abort_unstarted()
        assert children(resource_parent) == {}
        assert session.metadata["resource_capacity_held"] is False
        assert session.metadata["session_id"] not in continuation._PENDING


def test_recursive_preparation_is_refused_without_replacing_outer_session(
        candidate_case, resource_parent, monkeypatch):
    original_builder = worker_candidate_artifacts.build_candidate_artifacts
    rejected = []
    def recursive(**kwargs):
        with pytest.raises((ValueError, RuntimeError)):
            continuation.prepare_native_candidate_continuation(**arguments(candidate_case, resource_parent))
        rejected.append(True)
        return original_builder(**kwargs)
    monkeypatch.setattr(worker_candidate_artifacts, "build_candidate_artifacts", recursive)
    with prepared(candidate_case, resource_parent) as session:
        assert rejected == [True]
        assert session.admission.process is None
        session.abort_unstarted()
        assert children(resource_parent) == {}


def test_recursive_run_is_refused_before_another_child_or_cleanup(
        candidate_case, resource_parent, tmp_path):
    with prepared(candidate_case, resource_parent) as session:
        selected, _marker = run_options(session, tmp_path)
        cleaned = []
        def cleanup(process):
            with pytest.raises((ValueError, RuntimeError)):
                session.run(**selected)
            cleaned.append(process)
            return cleanup_actual(process)
        completed = session.run(**(selected | dict(cleanup=cleanup)))
        assert completed.returncode == 0
        assert cleaned == [session.admission.process]
        assert children(resource_parent) == {}


@pytest.mark.parametrize("binding", ["--owner-worker-attempt-id", "--owner-provider-deadline-ns",
    "--candidate-dispatch-descriptor", "--source-lookup-descriptor"])
def test_caller_cannot_override_generated_continuation_bindings(
        candidate_case, resource_parent, tmp_path, binding):
    with prepared(candidate_case, resource_parent) as session:
        selected, marker = run_options(session, tmp_path)
        selected["command"] += [binding, "authored-forbidden-binding"]
        with pytest.raises((ValueError, RuntimeError)):
            session.run(**selected)
        assert session.admission.process is None and not marker.exists()
        if not session.resources.native_lease.released:
            session.abort_unstarted()
        assert children(resource_parent) == {}
        # Refusing a reserved binding never refunds this session's run permit.
        with pytest.raises((ValueError, RuntimeError)):
            session.run(**(selected | dict(command=selected["command"][:-2])))


def test_candidate_prompt_descriptor_is_sealed_and_closed_after_preparation(
        candidate_case, resource_parent, monkeypatch):
    original_publish = attempts.OwnerWorkerAttempt._publish
    existing = {}
    for path in Path("/proc/self/fd").iterdir():
        try:
            info = path.stat()
            existing[int(path.name)] = (info.st_dev, info.st_ino)
        except FileNotFoundError:
            pass
    observed = []
    def record_descriptor(self, value, *, deadline):
        if value["event"] == "prepared":
            for path in Path("/proc/self/fd").iterdir():
                if not path.name.isdigit():
                    continue
                descriptor = int(path.name)
                try:
                    info = path.stat()
                    if existing.get(descriptor) == (info.st_dev, info.st_ino):
                        continue
                    target = os.readlink(path)
                    if not target.startswith("/memfd:"):
                        continue
                    raw = os.pread(descriptor, 262145, 0)
                    if hashlib.sha256(raw).hexdigest() != value["native_prompt_sha256"]:
                        continue
                    seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
                    assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals
                    observed.append((info.st_dev, info.st_ino))
                except FileNotFoundError:
                    continue
        return original_publish(self, value, deadline=deadline)
    monkeypatch.setattr(attempts.OwnerWorkerAttempt, "_publish", record_descriptor)
    with prepared(candidate_case, resource_parent):
        assert len(observed) == 1
        retained = set()
        for path in Path("/proc/self/fd").iterdir():
            try:
                info = os.stat(path)
                retained.add((info.st_dev, info.st_ino))
            except FileNotFoundError:
                pass
        assert observed[0] not in retained
