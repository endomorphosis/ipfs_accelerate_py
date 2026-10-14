"""Live native source custody around a real, inert direct child birth.

Git/CAS, native claims/grants, Portal allocation/lifecycle and kernel Popen
birth/CWD observations are real. Quack listener capability and deployment UID
metadata retain the authored host seams of the existing allocation fixture.
The child is isolated Python reading stdin; it never imports or executes the
target source. Owner-side retrieval and same-owner socket refusal do not
qualify a separate worker UID, provider/model, compiled context or formal proof.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_client as protocol
from test.api.test_current_source_task_admission import SOURCE, _payload, _update_body
from test.api.test_current_source_worktree_allocation import _allocation_case, _record_change
from test.api.test_native_nominated_source_allocation import bind
from test.api.test_native_nominated_source_scope import OTHER, SUPPORT
from test.api.test_source_lookup_worker_channel import (
    SOURCE as LEGACY_SOURCE, _native_case as _legacy_case, _payload as _legacy_payload,
)


CHILD = (
    "import json,os,sys; "
    "print(json.dumps(dict(pid=os.getpid(),parent_pid=os.getppid(),"
    "uid=os.geteuid(),cwd=os.getcwd())),flush=True); "
    "sys.stdin.buffer.read(1)"
)
DRIFTS = (
    "released_claim", "daemon_grant", "task_body", "allocation_lifecycle",
    "canonical_selected", "canonical_omitted", "allocated_selected",
    "allocated_omitted", "nomination_replaced",
)


@pytest.fixture
def spawned_source(tmp_path, monkeypatch):
    files = {b"module.py": SOURCE, b"instruction.md": SUPPORT, b"other.py": OTHER}
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
                          source_files=files) as case:
        bind(case)
        case.native.session = broker.start_source_lookup_worker_broker(**case.native.factory)
        yield case


@contextmanager
def inert_child(workspace):
    # No owner credential environment, target imports, helper process or model.
    process = subprocess.Popen([sys.executable, "-I", "-c", CHILD], cwd=workspace,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={"PATH": os.defpath, "LANG": "C.UTF-8"}, close_fds=True)
    try:
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready, "inert direct child did not report its birth"
        observed = json.loads(process.stdout.readline())
        yield process, observed
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
        assert process.returncode is not None


@contextmanager
def drift(case, monkeypatch, change):
    native = case.native
    if change == "allocation_lifecycle":
        with _record_change(case, lambda value: value.update(state="settling")):
            yield
        return
    if change == "nomination_replaced":
        with monkeypatch.context() as altered:
            altered.setattr(case.guard, "_nominated_source_scope", object())
            yield
        return
    if change == "task_body":
        before = deepcopy(dict(native.native_task.body))
        _update_body(native, before | {"title": "changed at the direct-child boundary"})
        try:
            yield
        finally:
            _update_body(native, before)
        return
    if change == "released_claim":
        claim = native.daemon.coordinator.get_task_claim(native.attempt.claim_id)
        lease = native.daemon.coordinator.protect_task_claim(claim)
        native.daemon.coordinator.release(lease, reason="authored pre/post child claim release")
    elif change == "daemon_grant":
        native.server._command_gateway.revoke_grant(native.daemon_grant_id)
    else:
        root, selected = change.split("_")
        path = (native.canonical if root == "canonical" else native.workspace) / (
            "module.py" if selected == "selected" else "other.py")
        before = path.read_bytes()
        path.write_bytes(before + b"# changed at the child boundary\n")
        try:
            yield
        finally:
            path.write_bytes(before)
        return
    yield


def assert_refused_child(case, process):
    session = case.native.session
    assert session._process is process
    assert session._registered.is_set() is False
    assert session._closed is True
    assert session._grant.grant_id in case.native.server._command_gateway._revoked_grants
    assert not session.socket_path.exists() and not session.descriptor_path.exists()
    # Refusal preserves the child's handle and the caller's reaping duty.
    with pytest.raises((ValueError, RuntimeError)):
        session.require_before_spawn()
    with pytest.raises((ValueError, RuntimeError)):
        session.note_spawned(process)


def test_actual_direct_birth_keeps_native_join_and_owner_side_source(spawned_source):
    case, native = spawned_source, spawned_source.native
    session = native.session
    assert session.require_before_spawn() is None
    with inert_child(native.workspace) as (process, child):
        session.note_spawned(process)
        birth = protocol.process_identity(process.pid)
        assert session._process is process and session._registered.is_set() is True
        assert session._child == birth
        assert birth["pid"] == child["pid"] == process.pid
        assert birth["parent_pid"] == child["parent_pid"] == os.getpid()
        assert birth["start_time_ticks"] > 0
        assert birth["uid"] == child["uid"] == os.geteuid() == session.descriptor["owner_uid"]
        assert Path(os.readlink(f"/proc/{process.pid}/cwd")) == native.workspace
        assert child["cwd"] == str(native.workspace)
        assert case.guard.require_nominated_source_scope()["allocation_ownership_verified"] is True
        source = session._read_source(_payload(native))
        assert source["content"].encode() == SOURCE and source["read_only"] is True
        assert source["completion_authority"] is source["proof_authority"] is False
        # An actual host socket peer is still the owner, not the distinct worker.
        left, right = socket.socketpair()
        with left, right:
            request = dict(schema=protocol.REQUEST_SCHEMA,
                session_id=session.descriptor["session_id"], request_id=uuid.uuid4().hex,
                payload=_payload(native))
            answer = session._answer(request, right)
        assert answer["ok"] is False and answer["result"] is None
        assert answer["reason_code"] == "peer_refused"
        protocol.validate_response(answer, descriptor=session.descriptor,
            payload=request["payload"], request_id=request["request_id"])
        assert native.daemon.provider_invocation_recorded(native.attempt.attempt_id,
            idempotency_key="provider:" + native.attempt.attempt_id) is None


@pytest.mark.parametrize("change", DRIFTS)
def test_prelaunch_native_or_complete_source_drift_never_calls_popen(
        spawned_source, monkeypatch, change):
    case, session = spawned_source, spawned_source.native.session
    calls = []

    def forbidden_popen(*_args, **_kwargs):
        calls.append("Popen")
        pytest.fail("refused native source state reached child creation")

    # Observe only the launcher's child creation. Native freshness replay
    # legitimately runs Git, so do not replace subprocess.Popen globally.
    launcher = SimpleNamespace(Popen=forbidden_popen)
    with drift(case, monkeypatch, change):
        with pytest.raises((ValueError, RuntimeError)):
            # Trusted launcher ordering: the check must return before Popen.
            session.require_before_spawn()
            launcher.Popen([sys.executable, "-I", "-c", CHILD], cwd=case.native.workspace)
    assert calls == []
    assert session._process is None and session._registered.is_set() is False
    assert session._closed is False


@pytest.mark.parametrize("change", DRIFTS)
def test_postbirth_native_or_complete_source_drift_revokes_without_registering(
        spawned_source, monkeypatch, change):
    case, session = spawned_source, spawned_source.native.session
    session.require_before_spawn()
    with inert_child(case.native.workspace) as (process, _child):
        with drift(case, monkeypatch, change):
            with pytest.raises((ValueError, RuntimeError)):
                session.note_spawned(process)
            assert_refused_child(case, process)
            assert process.poll() is None
    assert process.poll() is not None


def test_native_child_without_successful_prepare_is_closed_and_caller_reaped(spawned_source):
    case = spawned_source
    with inert_child(case.native.workspace) as (process, _child):
        with pytest.raises((ValueError, RuntimeError)):
            case.native.session.note_spawned(process)
        assert_refused_child(case, process)
        assert process.poll() is None
    assert process.poll() is not None


def test_duck_typed_child_cannot_replace_actual_popen(spawned_source):
    case, session = spawned_source, spawned_source.native.session
    session.require_before_spawn()
    with inert_child(case.native.workspace) as (process, _child):
        fake = SimpleNamespace(pid=process.pid, poll=process.poll)
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(fake)
        assert session._registered.is_set() is False and session._process is None
        assert process.poll() is None


def test_actual_direct_child_in_foreign_cwd_is_not_the_allocation(spawned_source):
    case, session = spawned_source, spawned_source.native.session
    session.require_before_spawn()
    with inert_child(case.native.canonical) as (process, child):
        assert child["cwd"] == str(case.native.canonical)
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(process)
        assert_refused_child(case, process)


def test_one_successful_prepare_and_child_cannot_be_replaced(spawned_source):
    case, session = spawned_source, spawned_source.native.session
    session.require_before_spawn()
    with pytest.raises((ValueError, RuntimeError)):
        session.require_before_spawn()
    with inert_child(case.native.workspace) as (process, _child):
        session.note_spawned(process)
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(process)
        assert session._registered.is_set() is True and session._process is process


def test_detached_allocation_observation_cannot_prepare_a_native_child(spawned_source, monkeypatch):
    case, session = spawned_source, spawned_source.native.session
    observed = case.guard.require_nominated_source_scope()
    with monkeypatch.context() as altered:
        altered.setattr(session, "_worktree_admission", observed)
        with pytest.raises((ValueError, RuntimeError)):
            session.require_before_spawn()
    assert session._registered.is_set() is False and session._process is None


def test_removing_all_nomination_tuples_cannot_downgrade_spawn(spawned_source, monkeypatch):
    case, session = spawned_source, spawned_source.native.session
    with monkeypatch.context() as altered:
        for handle in (case.guard, case.native.handle):
            altered.setattr(handle, "_nominated_source_scope", None)
            altered.setattr(handle, "_original_nominated_source_scope", None)
        with pytest.raises((ValueError, RuntimeError)):
            session.require_before_spawn()
    assert session._registered.is_set() is False and session._process is None


def test_restored_prelaunch_original_can_prepare_once_without_rebinding(spawned_source, monkeypatch):
    case, session = spawned_source, spawned_source.native.session
    binding = case.native.server._command_gateway._current_source_lookup
    with drift(case, monkeypatch, "allocated_selected"):
        with pytest.raises((ValueError, RuntimeError)):
            session.require_before_spawn()
    assert session.require_before_spawn() is None
    with inert_child(case.native.workspace) as (process, _child):
        session.note_spawned(process)
        assert session._registered.is_set() is True
        assert case.native.server._command_gateway._current_source_lookup is binding
        assert session._read_source(_payload(case.native))["content"].encode() == SOURCE


def test_native_legacy_binding_requires_prepare_without_inventing_nomination(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch) as case:
        session = case.native.session
        with pytest.raises((ValueError, RuntimeError)):
            case.guard.require_nominated_source_scope()
        assert session.require_before_spawn() is None
        with inert_child(case.native.workspace) as (process, _child):
            session.note_spawned(process)
            assert session._registered.is_set() is True
            assert session._read_source(_payload(case.native))["content"].encode() == SOURCE


def test_original_broker_without_native_handles_retains_legacy_birth_registration(tmp_path, monkeypatch):
    with _legacy_case(tmp_path, monkeypatch) as native:
        assert native.session._task_admission is native.session._worktree_admission is None
        with inert_child(native.workspace) as (process, _child):
            # Existing public API compatibility: no prepare is required here.
            native.session.note_spawned(process)
            assert native.session._registered.is_set() is True
            result = native.session._read_source(_legacy_payload(native, maximum=64))
            assert result["offset"] == 0 and 0 < result["end_offset"] <= 64
            assert result["content"].encode() == LEGACY_SOURCE[:result["end_offset"]]
            assert result["source_bytes"] == len(LEGACY_SOURCE)
            assert result["source_complete"] is False and result["next_continuation"]
