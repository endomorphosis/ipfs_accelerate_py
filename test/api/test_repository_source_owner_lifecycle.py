"""Explicit source-owner custody; authored root admission and real nested smoke.

Root admission is simulated only through declared scheduler/lease seams. Native
cases require the enclosing admitted runner's real borrowed parent and retain
it throughout. Container identities/peer ancestry remain authored, so these
tests do not qualify a deployed worker, a provider, or Terminal auto-wiring.
"""
import json
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_lifecycle as lifecycle
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from test.api.test_repository_source_owner_broker import (
    initialize, message, policy_digest, short_service_directory, tool,
)
from test.api.test_repository_source_parent_admission import native_parent


@pytest.fixture
def authored(tmp_path, monkeypatch):
    """No scheduler state, lease or native work is created by this seam."""
    state = SimpleNamespace(opens=[], acquisitions=[], leases=[], brokers=[], events=[],
        acquire_error=None, constructor_error=None, context_error=None, service_error=None,
        pending=False, close_error=None, release_error=None, ready=True, auto_renew=True,
        proof_safety=True,
        expected_digest="a" * 64)

    class Lease:
        def __init__(self, label, scheduler, parent=None):
            self.lease_id, self.lease_key = label, "authored-private-capability-" + label
            self._scheduler = scheduler
            self.owner_pid = os.getpid()
            self.cpu_slots, self.memory_mb, self.child_process_slots = 1, 512, 1
            self.gpu_memory_mb = self.unified_memory_mb = 0
            self.requires_gpu = False
            self.released = self.cancelled = False
            self.parent_lease_id = None if parent is None else parent.lease_id
            self.token = SimpleNamespace(lease_id=label, lease_key=self.lease_key,
                                         state_path=str(scheduler.state_path))
        def release(self):
            state.events.append(("release", self.lease_id))
            if state.release_error is not None and self.lease_id.startswith("source-parent"):
                raise state.release_error
            self.released = self.cancelled = True
            return True
        def cancel(self):
            self.cancelled = True
            return True
        def renew(self):
            return not self.released
        def to_dict(self):
            return dict(lease_id=self.lease_id, parent_lease_id=self.parent_lease_id,
                owner_pid=self.owner_pid, cpu_slots=self.cpu_slots, memory_mb=self.memory_mb,
                child_process_slots=self.child_process_slots, released=self.released,
                cancelled=self.cancelled)

    class Scheduler:
        state_path = tmp_path / "native.json"
        @property
        def config(self):
            return SimpleNamespace(auto_renew_leases=state.auto_renew, proof_safety_enabled=state.proof_safety,
                lease_ttl_seconds=60, persisted_dict=lambda: {"proof_safety_enabled": True})
        def acquire(self, *args, **kwargs):
            state.acquisitions.append((args, kwargs))
            if state.acquire_error is not None:
                raise state.acquire_error
            parent = kwargs.get("parent_lease")
            lease = Lease("source-parent-" + str(len(state.leases)), self, parent=parent)
            state.leases.append(lease)
            return lease

    scheduler = Scheduler()
    class Broker:
        def __init__(self, **kwargs):
            state.brokers.append(self)
            self.options, self.spawned = kwargs, []
            self._ready, self._done, self._stop = threading.Event(), threading.Event(), threading.Event()
            self._service_error = state.service_error
            self._native = Lease("broker-child", scheduler, parent=kwargs["parent_lease"])
            self._server = object()
            self.alive = True
            self._thread = SimpleNamespace(is_alive=lambda: self.alive)
            self._closed = False
            self.extra_argv = [protocol.DESCRIPTOR_FLAG, str(tmp_path / "owner.json"),
                               protocol.DIGEST_FLAG, "b" * 64]
            self.descriptor = {"session_id": "c" * 32}
            if state.ready:
                self._ready.set()
            if state.constructor_error is not None:
                raise state.constructor_error
        def _current_context(self):
            if state.context_error is not None:
                raise state.context_error
        def note_spawned(self, process):
            if self.spawned or process.poll() is not None:
                raise ValueError("authored duplicate or exited worker")
            self.spawned.append(process)
        def close(self, *, timeout_seconds=3.):
            state.events.append(("broker_close", timeout_seconds))
            if state.close_error is not None:
                raise state.close_error
            if state.pending:
                return dict(cleanup_complete=False, cleanup_pending=True, service_stopped=False,
                    native_child_acquired=True, native_child_released=False)
            self.alive = False
            self._done.set()
            self._native.release()
            return dict(cleanup_complete=True, cleanup_pending=False, service_stopped=True,
                native_child_acquired=True, native_child_released=True, prepared_mcp_usage=None)

    def open_existing(path, digest):
        state.opens.append((path, digest))
        if digest != state.expected_digest:
            raise ValueError("authored exact policy mismatch")
        return scheduler

    monkeypatch.setattr(lifecycle, "ResourceLease", Lease)
    monkeypatch.setattr(lifecycle, "open_existing_scheduler", open_existing)
    monkeypatch.setattr(lifecycle, "OwnerRepositoryMCPBroker", Broker)
    state.scheduler, state.Lease, state.Broker = scheduler, Lease, Broker
    state.options = dict(repository=tmp_path / "repository", workspace=tmp_path / "workspace",
        boundary=tmp_path / "boundary.json", scheduler_state_path=scheduler.state_path,
        expected_policy_sha256=state.expected_digest, deadline_monotonic=time.monotonic() + 30,
        admission_timeout_seconds=1)
    def create(**changes):
        return lifecycle.RepositorySourceOwnerLifecycle(**{**state.options, **changes})
    state.create = create
    return state


def test_constructor_does_not_open_or_admit_and_close_before_start_is_safe(authored):
    owner = authored.create()
    assert authored.opens == authored.acquisitions == authored.brokers == []
    with pytest.raises((ValueError, RuntimeError)):
        _ = owner.extra_argv
    result = owner.close()
    assert result["cleanup_complete"] and result["native_parent_acquired"] is False
    assert result["native_parent_released"] is None
    assert authored.opens == authored.acquisitions == []


@pytest.mark.parametrize("changes", [
    {"scheduler_state_path": "relative.json"}, {"expected_policy_sha256": "A" * 64},
    {"expected_policy_sha256": "a" * 63}, {"expected_policy_sha256": None},
    {"deadline_monotonic": True}, {"deadline_monotonic": float("nan")},
    {"deadline_monotonic": float("inf")}, {"admission_timeout_seconds": True},
    {"admission_timeout_seconds": 0}, {"admission_timeout_seconds": -1},
    {"admission_timeout_seconds": 31}, {"admission_timeout_seconds": float("nan")},
    {"representation": "minified"}, {"cancel_event": object()}, {"parent_lease": object()},
])
def test_invalid_configuration_cannot_access_scheduler(authored, changes):
    with pytest.raises(ValueError):
        authored.create(**changes)
    assert authored.opens == authored.acquisitions == authored.brokers == []


def test_root_mode_is_only_authored_and_exact_source_demand_is_forwarded(authored):
    owner = authored.create()
    assert owner.start() is owner
    assert len(authored.opens) == len(authored.acquisitions) == len(authored.brokers) == 1
    path, digest = authored.opens[0]
    assert Path(path) == authored.scheduler.state_path and digest == authored.expected_digest
    _, request = authored.acquisitions[0]
    assert request.get("parent_lease") is None
    assert {key: request[key] for key in broker.DEMAND} == broker.DEMAND
    assert 0 < request["timeout"] <= 1
    assert callable(request["cancel_event"].is_set)
    assert owner._broker.options["parent_lease"] is owner._lease
    assert owner._broker.options["deadline_monotonic"] == authored.options["deadline_monotonic"]
    assert owner._broker.options["expected_policy_sha256"] == authored.expected_digest
    arguments = owner.extra_argv
    arguments.append("injected")
    assert "injected" not in owner.extra_argv
    result = owner.close()
    assert result["cleanup_complete"] and result["native_parent_released"] is True
    first_close = next(index for index, event in enumerate(authored.events) if event[0] == "broker_close")
    assert first_close < authored.events.index(("release", owner._lease.lease_id))


def test_parent_delegation_preserves_upstream_ownership(authored):
    parent = authored.Lease("caller-parent", authored.scheduler)
    owner = authored.create(parent_lease=parent)
    owner.start()
    supplied = authored.acquisitions[0][1]["parent_lease"]
    assert supplied is parent or supplied is parent.token
    owner.close()
    assert not parent.released and not parent.cancelled


@pytest.mark.parametrize("field,value", [
    ("owner_pid", -1), ("released", True), ("cancelled", True),
    ("cpu_slots", 0), ("memory_mb", 511), ("child_process_slots", 0),
])
def test_invalid_parent_cannot_admit(authored, field, value):
    parent = authored.Lease("caller-parent", authored.scheduler)
    setattr(parent, field, value)
    with pytest.raises((ValueError, lifecycle.RepositorySourceOwnerLifecycleError)):
        authored.create(parent_lease=parent).start()
    assert authored.acquisitions == authored.brokers == []


def test_foreign_parent_scheduler_is_refused(authored):
    parent = authored.Lease("caller-parent", authored.scheduler)
    parent.token.state_path = "/foreign/native.json"
    with pytest.raises((ValueError, lifecycle.RepositorySourceOwnerLifecycleError)):
        authored.create(parent_lease=parent).start()
    assert authored.acquisitions == authored.brokers == []


def test_wrong_selected_policy_never_reaches_admission(authored):
    owner = authored.create(expected_policy_sha256="d" * 64)
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError) as error:
        owner.start()
    assert error.value.owner is owner
    assert authored.acquisitions == authored.brokers == []


@pytest.mark.parametrize("setting", [False, None, 1])
def test_selected_policy_must_provide_actual_native_renewal(authored, setting):
    authored.auto_renew = setting
    owner = authored.create()
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError):
        owner.start()
    assert authored.acquisitions == authored.brokers == []


@pytest.mark.parametrize("setting", [False, None, 1])
def test_selected_policy_must_preserve_native_pressure_safety(authored, setting):
    authored.proof_safety = setting
    owner = authored.create()
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError):
        owner.start()
    assert authored.acquisitions == authored.brokers == []


def test_already_cancelled_never_admits(authored):
    signal = threading.Event()
    signal.set()
    owner = authored.create(cancel_event=signal)
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError) as error:
        owner.start()
    assert error.value.owner is owner
    assert authored.acquisitions == authored.brokers == []


@pytest.mark.parametrize("kind", ["timeout", "cancelled"])
def test_native_admission_failure_retains_reachable_lifecycle(authored, kind):
    authored.acquire_error = (lifecycle.LeaseTimeoutError("authored timeout") if kind == "timeout"
                              else lifecycle.LeaseCancelledError("authored cancellation"))
    owner = authored.create()
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError) as error:
        owner.start()
    assert error.value.owner is owner and isinstance(error.value.reason_code, str)
    assert not authored.brokers
    assert "authored" not in str(error.value)
    result = owner.close()
    assert result["native_parent_acquired"] is False and result["native_parent_released"] is None
    assert result["cleanup_complete"] and not result["admission_uncertain"]


@pytest.mark.parametrize("error", [RuntimeError("authored acquisition interrupted"), KeyboardInterrupt()])
def test_unexpected_admission_error_preserves_unknown_outcome(authored, error):
    authored.acquire_error = error
    owner = authored.create()
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError) as caught:
        owner.start()
    assert caught.value.owner is owner
    result = owner.close()
    assert result["native_parent_acquired"] is result["native_parent_released"] is None
    assert result["admission_uncertain"] and result["cleanup_pending"] and result["retained_custody"]
    assert not result["cleanup_complete"] and not authored.brokers


def test_close_during_inflight_admission_reports_unknown_until_joined(authored, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    acquire = authored.scheduler.acquire
    def blocked(*args, **kwargs):
        entered.set()
        assert finish.wait(3), "authored admission release was not signalled"
        return acquire(*args, **kwargs)
    monkeypatch.setattr(authored.scheduler, "acquire", blocked)
    owner = authored.create()
    failures = []
    def start():
        try:
            owner.start()
        except BaseException as error:
            failures.append(error)
    worker = threading.Thread(target=start, name="authored-source-admission")
    worker.start()
    try:
        assert entered.wait(2)
        result = owner.close(timeout_seconds=.01)
        assert result["native_parent_acquired"] is None and result["native_parent_released"] is None
        assert result["cleanup_pending"] and result["retained_custody"] and not result["ready"]
        assert not result["cleanup_complete"]
    finally:
        finish.set()
        worker.join(2)
    assert not worker.is_alive() and len(failures) == 1
    assert isinstance(failures[0], lifecycle.RepositorySourceOwnerLifecycleError)
    assert failures[0].owner is owner and not authored.brokers
    assert owner.close()["cleanup_complete"] and owner._lease.released


def test_expired_before_start_cannot_open_or_admit(authored, monkeypatch):
    owner = authored.create()
    monkeypatch.setattr(lifecycle, "time", SimpleNamespace(monotonic=lambda: owner.deadline_monotonic + 1))
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError):
        owner.start()
    assert authored.opens == authored.acquisitions == authored.brokers == []
    assert owner.close()["cleanup_complete"]


def test_insufficient_remaining_broker_lifetime_refuses_before_admission(authored):
    with pytest.raises((ValueError, lifecycle.RepositorySourceOwnerLifecycleError)):
        authored.create(deadline_monotonic=time.monotonic() + .05).start()
    assert authored.acquisitions == authored.brokers == []


@pytest.mark.parametrize("failure", ["constructor", "service", "context"])
def test_partial_startup_failure_returns_custody_without_dispatch(authored, failure):
    if failure == "constructor":
        authored.constructor_error = RuntimeError("authored private constructor failure")
    elif failure == "service":
        authored.service_error = "AuthoredNativeFailure"
    else:
        authored.context_error = ValueError("authored stale context")
    owner = authored.create()
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError) as error:
        owner.start()
    assert error.value.owner is owner
    with pytest.raises((ValueError, RuntimeError)):
        _ = owner.extra_argv
    result = owner.close()
    assert result["cleanup_complete"] and result["native_parent_released"] is True


def test_readiness_timeout_cannot_publish_worker_arguments(authored):
    authored.ready = False
    owner = authored.create(admission_timeout_seconds=.02)
    with pytest.raises(lifecycle.RepositorySourceOwnerLifecycleError) as error:
        owner.start()
    assert error.value.owner is owner
    with pytest.raises((ValueError, RuntimeError)):
        _ = owner.extra_argv
    assert owner.close()["cleanup_complete"]


def test_cleanup_pending_retains_owned_parent_until_broker_drains(authored):
    owner = authored.create().start()
    authored.pending = True
    first = owner.close(timeout_seconds=.01)
    assert first["cleanup_pending"] and not first["cleanup_complete"]
    assert first["retained_custody"] and not owner._lease.released
    assert not any(event == ("release", owner._lease.lease_id) for event in authored.events)
    authored.pending = False
    final = owner.close()
    assert final["cleanup_complete"] and owner._lease.released


def test_broker_close_exception_cannot_release_parent(authored):
    owner = authored.create().start()
    authored.close_error = RuntimeError("authored private close failure")
    result = owner.close(timeout_seconds=.01)
    assert result["cleanup_pending"] and result["retained_custody"] and not owner._lease.released
    authored.close_error = None
    assert owner.close()["cleanup_complete"]


def test_parent_release_failure_retains_custody_for_retry(authored):
    owner = authored.create().start()
    authored.release_error = RuntimeError("authored private release failure")
    result = owner.close()
    assert result["cleanup_pending"] and result["retained_custody"] and not owner._lease.released
    assert result["renewal_unproven"]
    authored.release_error = None
    assert owner.close()["cleanup_complete"] and owner._lease.released


@pytest.mark.parametrize("field", ["cleanup_complete", "service_stopped", "native_child_released"])
def test_release_requires_each_broker_cleanup_fact(authored, monkeypatch, field):
    owner = authored.create().start()
    real_close = owner._broker.close
    def uncertain(**kwargs):
        result = dict(cleanup_complete=True, service_stopped=True,
                      native_child_acquired=True, native_child_released=True)
        result[field] = None
        return result
    monkeypatch.setattr(owner._broker, "close", uncertain)
    assert owner.close()["cleanup_pending"] and not owner._lease.released
    monkeypatch.setattr(owner._broker, "close", real_close)
    assert owner.close()["cleanup_complete"]


@pytest.mark.parametrize("timeout", [True, float("nan"), float("inf"), -1, 3.01, 10**400])
def test_invalid_close_bound_does_not_stop_ready_service(authored, timeout):
    owner = authored.create().start()
    with pytest.raises(ValueError):
        owner.close(timeout_seconds=timeout)
    assert owner.extra_argv and not owner._lease.released
    assert owner.close()["cleanup_complete"]


def test_duplicate_start_spawn_and_post_close_access_are_refused(authored):
    owner = authored.create().start()
    with pytest.raises((ValueError, RuntimeError)):
        owner.start()
    process = SimpleNamespace(pid=12345, poll=lambda: None)
    owner.note_spawned(process)
    with pytest.raises((ValueError, RuntimeError)):
        owner.note_spawned(process)
    result = owner.close()
    assert result["cleanup_complete"]
    with pytest.raises((ValueError, RuntimeError)):
        _ = owner.extra_argv
    assert owner.close()["cleanup_complete"]
    assert sum(event == ("release", owner._lease.lease_id) for event in authored.events) == 1


def test_close_receipt_contains_no_native_capability_or_private_exception(authored):
    owner = authored.create().start()
    public = json.dumps(owner.close())
    assert "authored-private-capability" not in public and "lease_key" not in public
    assert "parent_token" not in public


@pytest.fixture
def native_owner(tmp_path, monkeypatch, native_parent, short_service_directory):
    repository, workspace = tmp_path / "repository", tmp_path / "workspace"
    repository.mkdir()
    workspace.mkdir()
    source = workspace / "calc.py"
    source.write_text("# Authored source, not executed.\ndef add(value):\n    return value + 1\n")
    source.chmod(0o640)
    before = source.read_bytes(), source.stat().st_mode
    boundary = short_service_directory / "boundary.json"
    boundary.write_text("authored boundary, not a deployed container")
    (short_service_directory / "worker-observations").mkdir(mode=0o750)
    monkeypatch.setattr(broker.guard, "_workspace", lambda *args: (repository, workspace))
    monkeypatch.setattr(broker.guard, "_owner_boundary", lambda path: (
        dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1,
             allowed_worktree_roots=[str(tmp_path)]), "b" * 64))
    monkeypatch.setattr(broker.guard, "_protected_directory", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker.OwnerRepositoryMCPBroker, "_peer", lambda self, channel: None)
    value = lifecycle.RepositorySourceOwnerLifecycle(repository=repository, workspace=workspace,
        boundary=boundary, scheduler_state_path=native_parent.token.state_path,
        expected_policy_sha256=policy_digest(native_parent), deadline_monotonic=time.monotonic() + 60,
        admission_timeout_seconds=10, parent_lease=native_parent)
    try:
        value.start()
        yield value
    finally:
        result = value.close()
        assert result["cleanup_complete"], result
        assert not native_parent.released and not native_parent.cancelled
        assert (source.read_bytes(), source.stat().st_mode) == before


def test_real_nested_owner_read_preserves_upstream_and_exact_lease_chain(native_owner, native_parent):
    owner = native_owner
    assert owner._lease.parent_lease_id == native_parent.lease_id
    assert owner._broker._native.parent_lease_id == owner._lease.lease_id
    assert owner._lease.owner_pid == owner._broker._native.owner_pid == os.getpid()
    assert owner._lease._scheduler.state_path == native_parent._scheduler.state_path
    initialize(owner._broker)
    generation = tool(owner._broker, "repository_status")["generation"]
    read = tool(owner._broker, "repository_read", generation=generation, path="calc.py")
    assert read["content"].encode() == (owner._broker.workspace / "calc.py").read_bytes()
    assert not owner._lease.released and not native_parent.released
    result = owner.close()
    assert result["cleanup_complete"] and result["native_parent_released"]
    assert result["broker_audit"]["prepared_mcp_usage"]["successful_reads"] == 1
    assert not native_parent.released


def test_real_blocked_handler_keeps_both_native_children_charged(native_owner, native_parent, monkeypatch):
    owner = native_owner
    entered, finish = threading.Event(), threading.Event()
    original = broker.mcp.RepositoryMCPServer.handle
    def blocked(self, request):
        entered.set()
        assert finish.wait(10), "authored handler release was not signalled"
        return original(self, request)
    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "handle", blocked)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.connect(str(owner._broker.socket_path))
        protocol.send_frame(channel, protocol.make_request(owner._broker.descriptor["session_id"], message("ping")),
                            deadline_monotonic=time.monotonic() + 2)
        try:
            assert entered.wait(2)
            result = owner.close(timeout_seconds=.01)
            assert result["cleanup_pending"] and result["retained_custody"]
            assert not owner._lease.released and not owner._broker._native.released
            assert not native_parent.released and not native_parent.cancelled
        finally:
            finish.set()
    result = owner.close()
    assert result["cleanup_complete"] and owner._lease.released and owner._broker._native.released
    assert not native_parent.released
